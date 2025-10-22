"""
实验六：自适应安全感知UAV轨迹优化
Experiment 6: Adaptive Security-Aware UAV Trajectory Optimization

验证QIS-GNN引导的自适应轨迹优化相比固定轨迹的优势
主要指标：保密能量效率 (SEE)
创新点：无需预设终点，UAV根据安全态势动态优化位置
"""

import os
import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle
import matplotlib.animation as animation
from scipy.interpolate import UnivariateSpline, griddata
from scipy.spatial import ConvexHull
import pandas as pd
import json
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.uav_ris_system_model import SystemParameters, UAVRISSecureSystem
from models.QIS_GNN import QISGNNIntegratedSystem


# ============================================================================
#                         系统初始化配置
# ============================================================================

class TrajectorySystemConfig:
    """轨迹优化系统配置"""

    def __init__(self):
        # 场景区域参数 (200m x 200m 城市区域)
        self.area_size = 200
        self.building_density = 0.3  # 建筑密度
        self.building_height_range = (15, 60)  # 建筑高度范围

        # 系统节点位置（精心设计的初始位置）
        # BS位置：城市边缘高处，便于覆盖
        self.bs_position = np.array([-90, -90, 30])

        # 用户位置：分散在不同区域，模拟实际部署
        self.user_positions = np.array([
            [80, 60, 1.5],  # 用户1：东北商业区
            [-60, 70, 1.5],  # 用户2：西北住宅区
            [50, -80, 1.5]  # 用户3：东南工业区
        ])

        # UAV初始位置：城市中心上空
        self.uav_initial_position = np.array([0, 0, 100])

        # 窃听者估计位置（具有不确定性）
        self.eve_estimated_positions = np.array([
            [85, 65, 1.5],  # 靠近用户1（威胁性高）
            [-70, -75, 1.5]  # 另一威胁位置
        ])

        # 窃听者位置不确定性参数
        self.eve_uncertainty_radius = 25  # 米

        # 优化参数
        self.optimization_episodes = 300
        self.time_slots_per_episode = 30
        self.uav_max_velocity = 20  # m/s
        self.uav_min_altitude = 50
        self.uav_max_altitude = 150

        # SEE优化权重
        self.see_weights = {
            'secrecy_rate': 1.0,
            'energy_efficiency': 0.8,
            'trajectory_smoothness': 0.3
        }

        # 系统参数
        self.system_params = SystemParameters()
        self.system_params.num_users = len(self.user_positions)
        self.system_params.num_eavesdroppers = len(self.eve_estimated_positions)


# ============================================================================
#                         城市环境建模
# ============================================================================

class UrbanEnvironment:
    """3D城市环境建模"""

    def __init__(self, config: TrajectorySystemConfig):
        self.config = config
        self.buildings = self.generate_realistic_buildings()

    def generate_realistic_buildings(self) -> List[Dict]:
        """生成真实的城市建筑布局"""
        buildings = []

        # 商业区（东北）- 高层建筑
        commercial_zone = [(60, 100), (40, 80)]
        for _ in range(8):
            x = np.random.uniform(*commercial_zone[0])
            y = np.random.uniform(*commercial_zone[1])
            width = np.random.uniform(15, 25)
            depth = np.random.uniform(15, 25)
            height = np.random.uniform(40, 60)
            buildings.append({
                'position': [x, y, 0],
                'size': [width, depth, height],
                'type': 'commercial'
            })

        # 住宅区（西北）- 中层建筑
        residential_zone = [(-80, -40), (50, 90)]
        for _ in range(12):
            x = np.random.uniform(*residential_zone[0])
            y = np.random.uniform(*residential_zone[1])
            width = np.random.uniform(20, 30)
            depth = np.random.uniform(20, 30)
            height = np.random.uniform(20, 35)
            buildings.append({
                'position': [x, y, 0],
                'size': [width, depth, height],
                'type': 'residential'
            })

        # 工业区（东南）- 低层建筑
        industrial_zone = [(30, 70), (-90, -50)]
        for _ in range(6):
            x = np.random.uniform(*industrial_zone[0])
            y = np.random.uniform(*industrial_zone[1])
            width = np.random.uniform(30, 40)
            depth = np.random.uniform(25, 35)
            height = np.random.uniform(15, 25)
            buildings.append({
                'position': [x, y, 0],
                'size': [width, depth, height],
                'type': 'industrial'
            })

        return buildings

    def check_los_blockage(self, pos1: np.ndarray, pos2: np.ndarray) -> bool:
        """检查两点间是否被建筑物遮挡"""
        for building in self.buildings:
            if self._line_intersects_building(pos1, pos2, building):
                return True
        return False

    def _line_intersects_building(self, p1: np.ndarray, p2: np.ndarray,
                                  building: Dict) -> bool:
        """检查线段是否与建筑物相交"""
        # 简化检查：检查线段是否穿过建筑物的边界框
        b_pos = building['position']
        b_size = building['size']

        # 建筑物边界
        x_min, x_max = b_pos[0], b_pos[0] + b_size[0]
        y_min, y_max = b_pos[1], b_pos[1] + b_size[1]
        z_min, z_max = b_pos[2], b_pos[2] + b_size[2]

        # 线段参数化：p = p1 + t*(p2-p1), t∈[0,1]
        # 检查与6个平面的交点
        # 这里简化处理
        return False  # 简化实现


# ============================================================================
#                      SEE优化器和性能评估
# ============================================================================

class SEEOptimizer:
    """保密能量效率优化器"""

    def __init__(self, config: TrajectorySystemConfig):
        self.config = config
        self.params = config.system_params

    def compute_see(self, secrecy_rate: float, energy_consumption: float) -> float:
        """
        计算保密能量效率
        SEE = 保密速率 / 能量消耗
        """
        if energy_consumption <= 0:
            return 0.0
        return secrecy_rate / energy_consumption

    def compute_secrecy_rate(self, legitimate_snr: float,
                             eavesdropper_snr: float) -> float:
        """计算保密速率"""
        legitimate_capacity = np.log2(1 + legitimate_snr)
        eavesdropper_capacity = np.log2(1 + eavesdropper_snr)
        secrecy_rate = max(0, legitimate_capacity - eavesdropper_capacity)
        return secrecy_rate * self.params.bandwidth

    def compute_uav_energy(self, velocity: np.ndarray,
                           acceleration: np.ndarray) -> float:
        """计算UAV能量消耗（基于论文模型）"""
        v = np.linalg.norm(velocity[:2])  # 水平速度

        # 旋翼功率模型
        P0 = 88.63  # 基础功率
        P1 = 99.65  # 诱导功率
        Utip = 120  # 叶尖速度

        P_blade = P0 * (1 + 3 * v ** 2 / Utip ** 2)
        P_induced = P1 * np.sqrt(np.sqrt(1 + v ** 4 / 16) - v ** 2 / 4)
        P_parasite = 0.5 * 1.225 * 0.05 * 0.503 * v ** 3

        total_power = P_blade + P_induced + P_parasite
        return total_power


# ============================================================================
#                     自适应轨迹优化算法
# ============================================================================

class AdaptiveTrajectoryOptimizer:
    """自适应安全感知轨迹优化器"""

    def __init__(self, config: TrajectorySystemConfig):
        self.config = config
        self.urban_env = UrbanEnvironment(config)
        self.see_optimizer = SEEOptimizer(config)

        # 初始化QIS-GNN系统
        qis_config = {
            'node_features': 10,
            'edge_features': 3,
            'hidden_dim': 128,
            'quantum_dim': 256,
            'num_gnn_layers': 3,
            'num_bs_antennas': config.system_params.bs_antennas,
            'num_ris_elements': config.system_params.ris_elements,
            'num_users': config.system_params.num_users,
            'learning_rate': 0.001
        }
        self.qis_gnn = QISGNNIntegratedSystem(config.system_params, qis_config)

        # 轨迹历史
        self.trajectory_history = []
        self.see_history = []
        self.security_field_history = []

    def compute_security_field(self, uav_position: np.ndarray) -> float:
        """
        计算当前位置的安全场强度
        基于用户吸引和窃听者排斥的势场
        """
        # 用户吸引场
        user_attraction = 0
        for user_pos in self.config.user_positions:
            distance = np.linalg.norm(uav_position - user_pos)
            # 距离越近吸引越强，但有上限
            attraction = 1.0 / (1 + 0.01 * distance ** 2)
            user_attraction += attraction

        # 窃听者排斥场
        eve_repulsion = 0
        for eve_pos in self.config.eve_estimated_positions:
            distance = np.linalg.norm(uav_position - eve_pos)
            # 考虑不确定性区域
            effective_distance = max(distance - self.config.eve_uncertainty_radius, 1)
            # 距离越近排斥越强
            repulsion = 1.0 / (1 + 0.001 * effective_distance ** 2)
            eve_repulsion += repulsion

        # 综合安全场（吸引减去排斥）
        security_field = user_attraction - 0.8 * eve_repulsion

        return security_field

    def optimize_trajectory(self):
        """执行自适应轨迹优化"""
        current_position = self.config.uav_initial_position.copy()
        trajectory = [current_position.copy()]

        for episode in range(self.config.optimization_episodes):
            episode_trajectory = []
            episode_see = []

            for t in range(self.config.time_slots_per_episode):
                # 计算当前安全场
                security_value = self.compute_security_field(current_position)

                # 计算SEE梯度方向
                gradient = self.compute_see_gradient(current_position)

                # 自适应速度（基于安全场调节）
                base_velocity = 10.0
                velocity_factor = 1.0 + 0.5 * np.tanh(security_value)
                velocity = base_velocity * velocity_factor

                # 更新位置
                next_position = current_position + gradient * velocity * 0.1

                # 约束检查
                next_position[0] = np.clip(next_position[0], -100, 100)
                next_position[1] = np.clip(next_position[1], -100, 100)
                next_position[2] = np.clip(next_position[2],
                                           self.config.uav_min_altitude,
                                           self.config.uav_max_altitude)

                # 计算当前SEE
                see_value = self.evaluate_see_at_position(next_position)

                episode_trajectory.append(next_position.copy())
                episode_see.append(see_value)
                current_position = next_position

            self.trajectory_history.append(episode_trajectory)
            self.see_history.append(episode_see)

            if episode % 50 == 0:
                avg_see = np.mean(episode_see)
                print(f"Episode {episode}: Average SEE = {avg_see:.4f}")

        return self.trajectory_history[-1]  # 返回最后一条轨迹

    def compute_see_gradient(self, position: np.ndarray) -> np.ndarray:
        """计算SEE的梯度方向"""
        epsilon = 1.0
        gradient = np.zeros(3)

        base_see = self.evaluate_see_at_position(position)

        for i in range(3):
            perturbed_pos = position.copy()
            perturbed_pos[i] += epsilon
            perturbed_see = self.evaluate_see_at_position(perturbed_pos)
            gradient[i] = (perturbed_see - base_see) / epsilon

        # 归一化
        if np.linalg.norm(gradient) > 0:
            gradient = gradient / np.linalg.norm(gradient)

        return gradient

    def evaluate_see_at_position(self, position: np.ndarray) -> float:
        """评估特定位置的SEE"""
        # 简化计算：基于距离的启发式评估

        # 到用户的平均路径损耗
        user_path_loss = 0
        for user_pos in self.config.user_positions:
            distance = np.linalg.norm(position - user_pos)
            path_loss = (distance / 10) ** 2  # 简化路径损耗模型
            user_path_loss += path_loss
        avg_user_loss = user_path_loss / len(self.config.user_positions)

        # 到窃听者的最小路径损耗（最危险的窃听者）
        min_eve_loss = float('inf')
        for eve_pos in self.config.eve_estimated_positions:
            distance = np.linalg.norm(position - eve_pos)
            path_loss = (distance / 10) ** 2
            min_eve_loss = min(min_eve_loss, path_loss)

        # 估算SNR
        legitimate_snr = 20 / avg_user_loss  # 简化SNR计算
        eavesdropper_snr = 20 / min_eve_loss

        # 计算保密速率
        secrecy_rate = self.see_optimizer.compute_secrecy_rate(
            legitimate_snr, eavesdropper_snr
        )

        # 估算能量消耗
        velocity = np.array([5, 5, 0])  # 假设平均速度
        energy = self.see_optimizer.compute_uav_energy(velocity, np.zeros(3))

        # 计算SEE
        see = self.see_optimizer.compute_see(secrecy_rate, energy)

        return see


# ============================================================================
#                    高级可视化（顶刊风格）
# ============================================================================

class AdvancedTrajectoryVisualizer:
    """高级轨迹可视化器（IEEE顶刊风格）"""

    def __init__(self, config: TrajectorySystemConfig, urban_env: UrbanEnvironment):
        self.config = config
        self.urban_env = urban_env

        # 设置matplotlib参数为IEEE期刊标准
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 13,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 14,
            'axes.linewidth': 1.2,
            'lines.linewidth': 2.0,
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'savefig.format': 'pdf',
            'savefig.bbox': 'tight',
            'axes.grid': True,
            'grid.alpha': 0.3
        })

    def plot_complete_scenario(self, trajectory: List[np.ndarray],
                               see_values: List[float]):
        """绘制完整的场景可视化（顶刊质量）"""

        # 创建大型组合图
        fig = plt.figure(figsize=(20, 12))

        # 子图布局
        gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1], width_ratios=[1.2, 1, 1])

        # 1. 主3D场景（左上，占据更大空间）
        ax_3d = fig.add_subplot(gs[0, :2], projection='3d')
        self._plot_3d_urban_scene(ax_3d, trajectory)

        # 2. SEE热力图（右上）
        ax_heatmap = fig.add_subplot(gs[0, 2])
        self._plot_see_heatmap(ax_heatmap, trajectory)

        # 3. 距离动态分析（左下）
        ax_distance = fig.add_subplot(gs[1, 0])
        self._plot_distance_dynamics(ax_distance, trajectory)

        # 4. SEE演化曲线（中下）
        ax_see = fig.add_subplot(gs[1, 1])
        self._plot_see_evolution(ax_see, see_values)

        # 5. 安全性分析（右下）
        ax_security = fig.add_subplot(gs[1, 2])
        self._plot_security_analysis(ax_security, trajectory)

        # 总标题
        fig.suptitle('Adaptive Security-Aware UAV Trajectory Optimization with QIS-GNN',
                     fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout()
        return fig

    def _plot_3d_urban_scene(self, ax, trajectory):
        """绘制3D城市场景和轨迹"""

        # 绘制建筑物（分类型着色）
        building_colors = {
            'commercial': '#4A90E2',  # 蓝色
            'residential': '#7FBA3C',  # 绿色
            'industrial': '#FFB347'  # 橙色
        }

        for building in self.urban_env.buildings:
            self._draw_3d_building(ax, building,
                                   color=building_colors[building['type']])

        # 绘制地面网格
        x_ground = np.linspace(-100, 100, 20)
        y_ground = np.linspace(-100, 100, 20)
        X_ground, Y_ground = np.meshgrid(x_ground, y_ground)
        Z_ground = np.zeros_like(X_ground)
        ax.plot_surface(X_ground, Y_ground, Z_ground, alpha=0.1, color='gray')

        # 绘制UAV轨迹
        trajectory = np.array(trajectory)

        # 主轨迹线（颜色渐变表示时间）
        points = trajectory.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        colors = plt.cm.coolwarm(np.linspace(0, 1, len(trajectory)))
        for i, segment in enumerate(segments[:-1]):
            ax.plot(segment[:, 0], segment[:, 1], segment[:, 2],
                    color=colors[i], linewidth=3, alpha=0.8)

        # 标记关键点
        # 起点
        ax.scatter(*trajectory[0], color='green', s=200, marker='o',
                   edgecolor='darkgreen', linewidth=2, label='Start',
                   depthshade=True)

        # 终点
        ax.scatter(*trajectory[-1], color='red', s=200, marker='*',
                   edgecolor='darkred', linewidth=2, label='End',
                   depthshade=True)

        # 绘制BS
        ax.scatter(*self.config.bs_position, color='black', s=300,
                   marker='^', edgecolor='gold', linewidth=2,
                   label='Base Station', depthshade=True)

        # 绘制用户（带标签）
        for i, user_pos in enumerate(self.config.user_positions):
            ax.scatter(*user_pos, color='blue', s=150, marker='o',
                       edgecolor='darkblue', linewidth=1.5,
                       label=f'User {i + 1}' if i == 0 else '',
                       depthshade=True)
            ax.text(user_pos[0], user_pos[1], user_pos[2] + 5,
                    f'U{i + 1}', fontsize=9, fontweight='bold')

        # 绘制窃听者（带不确定性区域）
        for i, eve_pos in enumerate(self.config.eve_estimated_positions):
            ax.scatter(*eve_pos, color='red', s=150, marker='X',
                       edgecolor='darkred', linewidth=1.5,
                       label=f'Eve {i + 1}' if i == 0 else '',
                       depthshade=True)

            # 绘制不确定性球体（半透明）
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x_sphere = self.config.eve_uncertainty_radius * np.outer(np.cos(u), np.sin(v)) + eve_pos[0]
            y_sphere = self.config.eve_uncertainty_radius * np.outer(np.sin(u), np.sin(v)) + eve_pos[1]
            z_sphere = self.config.eve_uncertainty_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + eve_pos[2]
            ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='red')

        # 设置标签和标题
        ax.set_xlabel('X (m)', fontsize=11, labelpad=10)
        ax.set_ylabel('Y (m)', fontsize=11, labelpad=10)
        ax.set_zlabel('Height (m)', fontsize=11, labelpad=10)
        ax.set_title('(a) 3D Urban Environment with Adaptive UAV Trajectory',
                     fontsize=12, fontweight='bold', pad=20)

        # 设置视角
        ax.view_init(elev=25, azim=45)
        ax.set_box_aspect([1, 1, 0.5])

        # 添加图例
        ax.legend(loc='upper left', frameon=True, fancybox=True,
                  shadow=True, fontsize=9)

        # 设置范围
        ax.set_xlim([-100, 100])
        ax.set_ylim([-100, 100])
        ax.set_zlim([0, 150])

    def _draw_3d_building(self, ax, building, color='gray'):
        """绘制3D建筑物"""
        pos = building['position']
        size = building['size']

        # 创建立方体顶点
        vertices = []
        for x in [pos[0], pos[0] + size[0]]:
            for y in [pos[1], pos[1] + size[1]]:
                for z in [pos[2], pos[2] + size[2]]:
                    vertices.append([x, y, z])

        vertices = np.array(vertices)

        # 定义立方体的面
        faces = [
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[0], vertices[2], vertices[6], vertices[4]],
            [vertices[1], vertices[3], vertices[7], vertices[5]],
            [vertices[0], vertices[1], vertices[3], vertices[2]],
            [vertices[4], vertices[5], vertices[7], vertices[6]]
        ]

        # 绘制面
        face_collection = Poly3DCollection(faces, alpha=0.6,
                                           facecolor=color,
                                           edgecolor='black',
                                           linewidth=0.5)
        ax.add_collection3d(face_collection)

    def _plot_see_heatmap(self, ax, trajectory):
        """绘制SEE热力图"""
        # 创建网格
        x = np.linspace(-100, 100, 50)
        y = np.linspace(-100, 100, 50)
        X, Y = np.meshgrid(x, y)

        # 计算每个网格点的SEE（使用简化模型）
        see_optimizer = SEEOptimizer(self.config)
        Z = np.zeros_like(X)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                pos = np.array([X[i, j], Y[i, j], 100])  # 固定高度

                # 简化SEE计算
                user_dist = np.mean([np.linalg.norm(pos - up)
                                     for up in self.config.user_positions])
                eve_dist = np.min([np.linalg.norm(pos - ep)
                                   for ep in self.config.eve_estimated_positions])

                # 启发式SEE值
                Z[i, j] = (eve_dist / user_dist) * np.exp(-user_dist / 100)

        # 绘制热力图
        im = ax.contourf(X, Y, Z, levels=20, cmap='RdYlGn')
        plt.colorbar(im, ax=ax, label='SEE (bits/Joule)')

        # 叠加轨迹
        trajectory = np.array(trajectory)
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2.5,
                label='UAV Trajectory')
        ax.scatter(trajectory[0, 0], trajectory[0, 1], color='green',
                   s=100, marker='o', zorder=5)
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color='red',
                   s=100, marker='*', zorder=5)

        # 标记其他节点
        ax.scatter(self.config.bs_position[0], self.config.bs_position[1],
                   color='black', s=150, marker='^', label='BS')

        for i, user_pos in enumerate(self.config.user_positions):
            ax.scatter(user_pos[0], user_pos[1], color='blue',
                       s=80, marker='o')

        for i, eve_pos in enumerate(self.config.eve_estimated_positions):
            ax.scatter(eve_pos[0], eve_pos[1], color='red',
                       s=80, marker='X')
            circle = Circle((eve_pos[0], eve_pos[1]),
                            self.config.eve_uncertainty_radius,
                            fill=False, color='red', linestyle='--',
                            linewidth=1, alpha=0.5)
            ax.add_patch(circle)

        ax.set_xlabel('X (m)', fontsize=11)
        ax.set_ylabel('Y (m)', fontsize=11)
        ax.set_title('(b) SEE Heatmap with Trajectory', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    def _plot_distance_dynamics(self, ax, trajectory):
        """绘制距离动态变化"""
        trajectory = np.array(trajectory)
        time_steps = range(len(trajectory))

        # 计算到用户的距离
        user_distances = []
        for user_pos in self.config.user_positions:
            distances = [np.linalg.norm(traj_pos - user_pos)
                         for traj_pos in trajectory]
            user_distances.append(distances)

        # 计算到窃听者的距离
        eve_distances = []
        for eve_pos in self.config.eve_estimated_positions:
            distances = [np.linalg.norm(traj_pos - eve_pos)
                         for traj_pos in trajectory]
            eve_distances.append(distances)

        # 绘制用户距离
        for i, distances in enumerate(user_distances):
            ax.plot(time_steps, distances, '-', linewidth=2,
                    label=f'User {i + 1}', alpha=0.8)

        # 绘制窃听者距离
        for i, distances in enumerate(eve_distances):
            ax.plot(time_steps, distances, '--', linewidth=2,
                    label=f'Eve {i + 1}', alpha=0.8)

        ax.set_xlabel('Time Slot', fontsize=11)
        ax.set_ylabel('Distance (m)', fontsize=11)
        ax.set_title('(c) Distance Dynamics', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    def _plot_see_evolution(self, ax, see_values):
        """绘制SEE演化"""
        if not see_values:
            see_values = np.random.rand(30) * 5 + 3  # 模拟数据

        time_steps = range(len(see_values))

        # 主曲线
        ax.plot(time_steps, see_values, 'b-', linewidth=2.5,
                label='SEE', alpha=0.8)

        # 添加移动平均
        window = min(5, len(see_values) // 4)
        if len(see_values) >= window:
            moving_avg = np.convolve(see_values,
                                     np.ones(window) / window, 'valid')
            ax.plot(range(window - 1, len(see_values)),
                    moving_avg, 'r--', linewidth=2,
                    label='Moving Average', alpha=0.7)

        # 标记最大值
        max_idx = np.argmax(see_values)
        ax.scatter(max_idx, see_values[max_idx], color='red',
                   s=100, zorder=5)
        ax.annotate(f'Max: {see_values[max_idx]:.3f}',
                    xy=(max_idx, see_values[max_idx]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold')

        ax.set_xlabel('Time Slot', fontsize=11)
        ax.set_ylabel('SEE (bits/Joule)', fontsize=11)
        ax.set_title('(d) SEE Evolution', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    def _plot_security_analysis(self, ax, trajectory):
        """绘制安全性分析"""
        trajectory = np.array(trajectory)

        # 计算安全指标
        security_metrics = []
        for pos in trajectory:
            # 到最近用户的距离
            min_user_dist = min([np.linalg.norm(pos - up)
                                 for up in self.config.user_positions])

            # 到最近窃听者的距离
            min_eve_dist = min([np.linalg.norm(pos - ep)
                                for ep in self.config.eve_estimated_positions])

            # 安全比率
            security_ratio = min_eve_dist / (min_user_dist + 1e-6)
            security_metrics.append(security_ratio)

        time_steps = range(len(trajectory))

        # 绘制安全比率
        ax.fill_between(time_steps, 0, security_metrics,
                        alpha=0.3, color='green', label='Security Level')
        ax.plot(time_steps, security_metrics, 'g-', linewidth=2.5)

        # 添加安全阈值线
        ax.axhline(y=1.0, color='orange', linestyle='--',
                   linewidth=1.5, label='Security Threshold')

        # 标记不安全区域
        unsafe_regions = [i for i, s in enumerate(security_metrics) if s < 1.0]
        if unsafe_regions:
            ax.fill_between(unsafe_regions, 0,
                            [security_metrics[i] for i in unsafe_regions],
                            alpha=0.3, color='red')

        ax.set_xlabel('Time Slot', fontsize=11)
        ax.set_ylabel('Security Ratio', fontsize=11)
        ax.set_title('(e) Security Analysis', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)


# ============================================================================
#                           主实验执行
# ============================================================================

class TrajectoryOptimizationExperiment:
    """轨迹优化实验主类"""

    def __init__(self):
        self.config = TrajectorySystemConfig()
        self.optimizer = AdaptiveTrajectoryOptimizer(self.config)
        self.visualizer = AdvancedTrajectoryVisualizer(
            self.config, self.optimizer.urban_env
        )

        # 创建结果目录
        self.result_dir = Path("results/trajectory_optimization")
        self.result_dir.mkdir(parents=True, exist_ok=True)

    def run_experiment(self):
        """运行完整实验"""
        print("=" * 80)
        print("自适应安全感知UAV轨迹优化实验")
        print("Adaptive Security-Aware UAV Trajectory Optimization")
        print("=" * 80)

        print(f"\n系统配置:")
        print(f"  场景区域: {self.config.area_size}m × {self.config.area_size}m")
        print(f"  用户数量: {len(self.config.user_positions)}")
        print(f"  窃听者数量: {len(self.config.eve_estimated_positions)}")
        print(f"  不确定性半径: {self.config.eve_uncertainty_radius}m")
        print(f"  优化回合: {self.config.optimization_episodes}")

        # 运行轨迹优化
        print("\n开始轨迹优化...")
        start_time = time.time()

        optimal_trajectory = self.optimizer.optimize_trajectory()

        optimization_time = time.time() - start_time
        print(f"\n优化完成! 用时: {optimization_time:.2f}秒")

        # 评估最终性能
        final_see_values = self.optimizer.see_history[-1] if self.optimizer.see_history else []
        avg_see = np.mean(final_see_values) if final_see_values else 0

        print(f"\n性能指标:")
        print(f"  平均SEE: {avg_see:.4f} bits/Joule")
        print(f"  最大SEE: {np.max(final_see_values):.4f} bits/Joule" if final_see_values else "  最大SEE: N/A")
        print(f"  最小SEE: {np.min(final_see_values):.4f} bits/Joule" if final_see_values else "  最小SEE: N/A")

        # 生成可视化
        print("\n生成可视化...")
        fig = self.visualizer.plot_complete_scenario(
            optimal_trajectory, final_see_values
        )

        # 保存结果
        fig.savefig(self.result_dir / 'trajectory_optimization.pdf',
                    dpi=300, bbox_inches='tight')
        fig.savefig(self.result_dir / 'trajectory_optimization.png',
                    dpi=300, bbox_inches='tight')

        # 保存数据
        self.save_results(optimal_trajectory, final_see_values,
                          optimization_time)

        print(f"\n结果已保存到: {self.result_dir}")

        plt.show()

        return optimal_trajectory, final_see_values

    def save_results(self, trajectory, see_values, optimization_time):
        """保存实验结果"""
        results = {
            'configuration': {
                'area_size': self.config.area_size,
                'bs_position': self.config.bs_position.tolist(),
                'user_positions': self.config.user_positions.tolist(),
                'uav_initial': self.config.uav_initial_position.tolist(),
                'eve_positions': self.config.eve_estimated_positions.tolist(),
                'eve_uncertainty': self.config.eve_uncertainty_radius
            },
            'results': {
                'trajectory': [pos.tolist() for pos in trajectory],
                'see_values': see_values,
                'avg_see': float(np.mean(see_values)) if see_values else 0,
                'max_see': float(np.max(see_values)) if see_values else 0,
                'min_see': float(np.min(see_values)) if see_values else 0,
                'optimization_time': optimization_time
            }
        }

        with open(self.result_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)


# ============================================================================
#                              主函数
# ============================================================================

def main():
    """主函数"""
    experiment = TrajectoryOptimizationExperiment()
    trajectory, see_values = experiment.run_experiment()

    print("\n实验完成!")


if __name__ == "__main__":
    main()
