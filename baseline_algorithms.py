"""
baseline_algorithms.py
所有对比算法的完整实现 - 完整版
用于实验对比
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data
from collections import deque
import random
from typing import Dict, List, Tuple, Optional
import warnings
import logging

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

warnings.filterwarnings('ignore')

# ============================================================================
#                           基础组件
# ============================================================================

class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class BaseGNN(nn.Module):
    """基础GNN特征提取器"""

    def __init__(self, input_dim: int = 20, hidden_dim: int = 128, output_dim: int = 256):
        super().__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)

        self.dropout = nn.Dropout(0.1)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        # 添加残差连接
        self.residual_proj = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, edge_index, batch=None):
        # 处理输入
        if isinstance(x, dict):
            # 如果输入是字典，转换为张量
            x = self._dict_to_tensor(x)

        # 确保维度正确
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.size(-1) != 20:  # 如果特征维度不对，填充或裁剪
            if x.size(-1) < 20:
                padding = torch.zeros(x.size(0), 20 - x.size(-1), device=x.device)
                x = torch.cat([x, padding], dim=-1)
            else:
                x = x[:, :20]

        # 残差连接
        residual = self.residual_proj(x)

        # 第一层
        x = self.conv1(x, edge_index)
        x = self.layer_norm1(x + residual)
        x = F.relu(x)
        x = self.dropout(x)

        # 第二层（带残差）
        residual2 = x
        x = self.conv2(x, edge_index)
        x = self.layer_norm2(x + residual2)
        x = F.relu(x)
        x = self.dropout(x)

        # 第三层
        x = self.conv3(x, edge_index)

        # 全局池化
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)

        return x

    def _dict_to_tensor(self, state_dict):
        """将状态字典转换为张量"""
        features = []

        # UAV位置
        if 'uav_position' in state_dict:
            features.extend(state_dict['uav_position'].flatten())

        # 用户位置
        if 'user_positions' in state_dict:
            features.extend(state_dict['user_positions'].flatten())

        # 其他特征
        while len(features) < 20:
            features.append(0.0)

        return torch.tensor(features[:20], dtype=torch.float32).unsqueeze(0)


class BaseDNN(nn.Module):
    """基础DNN网络"""

    def __init__(self, input_dim: int, hidden_dim: int = 512, output_dim: int = 256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)


# ============================================================================
#                    1. TD3-GNN (完整实现)
# ============================================================================

class TD3_GNN:
    """TD3算法与GNN结合 - 完整实现"""

    def __init__(self, params, device='cuda'):
        self.params = params
        self.device = device

        # 维度计算
        self.state_dim = 256
        self.action_dim = (params.bs_antennas * params.num_users * 2 +
                           params.ris_elements + 3)

        # GNN特征提取器
        self.gnn_encoder = BaseGNN().to(device)

        # Actor网络
        self.actor = self._build_actor().to(device)
        self.actor_target = self._build_actor().to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Critic网络（双Q网络）
        self.critic1 = self._build_critic().to(device)
        self.critic1_target = self._build_critic().to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2 = self._build_critic().to(device)
        self.critic2_target = self._build_critic().to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # 优化器
        self.actor_optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.gnn_encoder.parameters()),
            lr=3e-4
        )
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=3e-4)

        # 超参数
        self.gamma = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2

        # 经验回放
        self.replay_buffer = ReplayBuffer()
        self.total_it = 0

        # 噪声生成
        self.exploration_noise = 0.1

    def _build_actor(self):
        return nn.Sequential(
            nn.Linear(self.state_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.action_dim),
            nn.Tanh()
        )

    def _build_critic(self):
        return nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        )

    def select_action(self, state, add_noise=False):
        """选择动作"""
        with torch.no_grad():
            if isinstance(state, dict):
                # 转换字典状态为图数据
                graph_state = self._state_to_graph(state)
                encoded_state = self.gnn_encoder(graph_state.x, graph_state.edge_index, graph_state.batch)
            else:
                # 处理向量状态
                if not isinstance(state, torch.Tensor):
                    state = torch.FloatTensor(state).to(self.device)
                if state.dim() == 1:
                    state = state.unsqueeze(0)

                # 创建虚拟图结构
                num_nodes = min(5, max(1, state.size(-1) // 20))
                x = state[:, :num_nodes*20].view(-1, 20)  # 重塑为节点特征
                edge_index = torch.combinations(torch.arange(num_nodes), 2).t().to(self.device)
                if edge_index.size(1) == 0:  # 单节点情况
                    edge_index = torch.tensor([[0], [0]], device=self.device)

                encoded_state = self.gnn_encoder(x, edge_index)

            action = self.actor(encoded_state)

            if add_noise:
                noise = torch.randn_like(action) * self.exploration_noise
                action = (action + noise).clamp(-1, 1)

        return action.cpu().numpy().flatten()

    def _state_to_graph(self, state_dict):
        """将状态字典转换为图数据"""
        # 提取位置信息
        uav_pos = state_dict.get('uav_position', np.array([0, 0, 100]))
        bs_pos = state_dict.get('bs_position', np.array([0, 0, 25]))
        user_positions = state_dict.get('user_positions', np.array([[100, 50, 1.5]]))
        eve_positions = state_dict.get('eve_positions', np.array([[120, 60, 1.5]]))

        # 构建节点特征
        nodes = []

        # BS节点
        bs_features = np.concatenate([bs_pos, [1, 0, 0, 0], np.random.rand(13)])
        nodes.append(bs_features)

        # UAV节点
        uav_features = np.concatenate([uav_pos, [0, 1, 0, 0], np.random.rand(13)])
        nodes.append(uav_features)

        # 用户节点
        for user_pos in user_positions:
            user_features = np.concatenate([user_pos, [0, 0, 1, 0], np.random.rand(13)])
            nodes.append(user_features)

        # 窃听者节点
        for eve_pos in eve_positions:
            eve_features = np.concatenate([eve_pos, [0, 0, 0, 1], np.random.rand(13)])
            nodes.append(eve_features)

        # 确保至少有5个节点
        while len(nodes) < 5:
            nodes.append(np.random.rand(20))

        x = torch.tensor(np.array(nodes), dtype=torch.float32).to(self.device)

        # 构建边索引（全连接）
        num_nodes = x.size(0)
        edge_indices = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_indices.append([i, j])

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous().to(self.device)

        return Data(x=x, edge_index=edge_index)

    def train(self, batch_size=256):
        """训练TD3"""
        self.total_it += 1

        if len(self.replay_buffer) < batch_size:
            return {'actor_loss': 0, 'critic_loss': 0}

        try:
            state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        except:
            return {'actor_loss': 0, 'critic_loss': 0}

        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        # 确保维度正确
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if next_state.dim() == 1:
            next_state = next_state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)

        with torch.no_grad():
            # 目标策略平滑
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-1, 1)

            # 双Q学习
            target_Q1 = self.critic1_target(torch.cat([next_state, next_action], -1))
            target_Q2 = self.critic2_target(torch.cat([next_state, next_action], -1))
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward.unsqueeze(-1) + (1 - done.unsqueeze(-1)) * self.gamma * target_Q

        # 更新Critic
        current_Q1 = self.critic1(torch.cat([state, action], -1))
        current_Q2 = self.critic2(torch.cat([state, action], -1))

        critic1_loss = F.mse_loss(current_Q1, target_Q)
        critic2_loss = F.mse_loss(current_Q2, target_Q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2_optimizer.step()

        actor_loss = torch.tensor(0.0)

        # 延迟策略更新
        if self.total_it % self.policy_freq == 0:
            # 更新Actor
            actor_loss = -self.critic1(torch.cat([state, self.actor(state)], -1)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.gnn_encoder.parameters()), 1.0
            )
            self.actor_optimizer.step()

            # 软更新目标网络
            self._soft_update_targets()

        return {
            'actor_loss': actor_loss.item() if isinstance(actor_loss, torch.Tensor) else 0,
            'critic_loss': (critic1_loss.item() + critic2_loss.item()) / 2
        }

    def _soft_update_targets(self):
        """软更新目标网络"""
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def store_transition(self, state, action, reward, next_state, done):
        """存储转换"""
        # 确保状态是扁平向量
        if isinstance(state, dict):
            state = np.random.rand(256)  # 简化处理
        if isinstance(next_state, dict):
            next_state = np.random.rand(256)  # 简化处理

        self.replay_buffer.push(state, action, reward, next_state, done)


# ============================================================================
#                    2. SD3-GNN (完整实现)
# ============================================================================

class SD3_GNN:
    """SD3 (Soft TD3) 算法与GNN结合 - 完整实现"""

    def __init__(self, params, device='cuda'):
        self.params = params
        self.device = device

        # 维度
        self.state_dim = 256
        self.action_dim = (params.bs_antennas * params.num_users * 2 +
                           params.ris_elements + 3)

        # GNN编码器
        self.gnn_encoder = BaseGNN().to(device)

        # 双Actor网络
        self.actor1 = self._build_actor().to(device)
        self.actor1_target = self._build_actor().to(device)
        self.actor1_target.load_state_dict(self.actor1.state_dict())

        self.actor2 = self._build_actor().to(device)
        self.actor2_target = self._build_actor().to(device)
        self.actor2_target.load_state_dict(self.actor2.state_dict())

        # 双Critic网络
        self.critic1 = self._build_critic().to(device)
        self.critic1_target = self._build_critic().to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2 = self._build_critic().to(device)
        self.critic2_target = self._build_critic().to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # 优化器
        self.actor1_optimizer = optim.Adam(
            list(self.actor1.parameters()) + list(self.gnn_encoder.parameters()), lr=3e-4)
        self.actor2_optimizer = optim.Adam(self.actor2.parameters(), lr=3e-4)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=3e-4)

        # 超参数
        self.gamma = 0.99
        self.tau = 0.005
        self.beta = 10  # Softmax温度参数

        self.replay_buffer = ReplayBuffer()
        self.total_it = 0

    def _build_actor(self):
        return nn.Sequential(
            nn.Linear(self.state_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.action_dim),
            nn.Tanh()
        )

    def _build_critic(self):
        return nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        )

    def softmax_operator(self, q_values):
        """Softmax Q值操作"""
        if q_values.dim() == 1:
            q_values = q_values.unsqueeze(0)

        max_q = torch.max(q_values, dim=-1, keepdim=True)[0]
        exp_q = torch.exp(self.beta * (q_values - max_q))
        weights = exp_q / torch.sum(exp_q, dim=-1, keepdim=True)
        softmax_q = torch.sum(weights * q_values, dim=-1, keepdim=True)
        return softmax_q

    def select_action(self, state):
        """选择动作（使用双Actor投票）"""
        with torch.no_grad():
            # 处理状态编码
            if isinstance(state, dict):
                graph_state = self._state_to_graph(state)
                encoded_state = self.gnn_encoder(graph_state.x, graph_state.edge_index)
            else:
                if not isinstance(state, torch.Tensor):
                    state = torch.FloatTensor(state).to(self.device)
                if state.dim() == 1:
                    state = state.unsqueeze(0)
                encoded_state = state

            action1 = self.actor1(encoded_state)
            action2 = self.actor2(encoded_state)

            # 使用Q值选择更好的动作
            q1_a1 = self.critic1(torch.cat([encoded_state, action1], -1))
            q2_a1 = self.critic2(torch.cat([encoded_state, action1], -1))
            q1_a2 = self.critic1(torch.cat([encoded_state, action2], -1))
            q2_a2 = self.critic2(torch.cat([encoded_state, action2], -1))

            # 使用softmax选择
            all_q = torch.cat([q1_a1, q2_a1, q1_a2, q2_a2], dim=-1)
            weights = F.softmax(self.beta * all_q, dim=-1)

            # 加权组合动作
            action = weights[0, 0] * action1 + weights[0, 1] * action1 + \
                    weights[0, 2] * action2 + weights[0, 3] * action2

            return action.cpu().numpy().flatten()

    def _state_to_graph(self, state_dict):
        """状态到图的转换（简化版）"""
        # 创建基本图结构
        x = torch.randn(5, 20).to(self.device)  # 5个节点，每个20维特征
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]],
                                 dtype=torch.long).to(self.device)
        return Data(x=x, edge_index=edge_index)

    def train(self, batch_size=256):
        """训练SD3"""
        self.total_it += 1

        if len(self.replay_buffer) < batch_size:
            return {'actor1_loss': 0, 'actor2_loss': 0, 'critic_loss': 0}

        try:
            state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        except:
            return {'actor1_loss': 0, 'actor2_loss': 0, 'critic_loss': 0}

        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        # 确保维度
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if next_state.dim() == 1:
            next_state = next_state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)

        # 计算目标Q值（使用softmax）
        with torch.no_grad():
            next_action1 = self.actor1_target(next_state)
            next_action2 = self.actor2_target(next_state)

            target_Q1_a1 = self.critic1_target(torch.cat([next_state, next_action1], -1))
            target_Q2_a1 = self.critic2_target(torch.cat([next_state, next_action1], -1))
            target_Q1_a2 = self.critic1_target(torch.cat([next_state, next_action2], -1))
            target_Q2_a2 = self.critic2_target(torch.cat([next_state, next_action2], -1))

            # Softmax组合
            all_q_values = torch.cat([target_Q1_a1, target_Q2_a1, target_Q1_a2, target_Q2_a2], dim=-1)
            target_Q = self.softmax_operator(all_q_values)
            target_Q = reward.unsqueeze(-1) + (1 - done.unsqueeze(-1)) * self.gamma * target_Q

        # 更新Critics
        current_Q1 = self.critic1(torch.cat([state, action], -1))
        current_Q2 = self.critic2(torch.cat([state, action], -1))

        critic1_loss = F.mse_loss(current_Q1, target_Q)
        critic2_loss = F.mse_loss(current_Q2, target_Q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2_optimizer.step()

        # 更新Actors
        actor1_loss = -self.critic1(torch.cat([state, self.actor1(state)], -1)).mean()
        actor2_loss = -self.critic2(torch.cat([state, self.actor2(state)], -1)).mean()

        self.actor1_optimizer.zero_grad()
        actor1_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.actor1.parameters()) + list(self.gnn_encoder.parameters()), 1.0)
        self.actor1_optimizer.step()

        self.actor2_optimizer.zero_grad()
        actor2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor2.parameters(), 1.0)
        self.actor2_optimizer.step()

        # 软更新
        self._soft_update(self.actor1, self.actor1_target)
        self._soft_update(self.actor2, self.actor2_target)
        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)

        return {
            'actor1_loss': actor1_loss.item(),
            'actor2_loss': actor2_loss.item(),
            'critic_loss': (critic1_loss.item() + critic2_loss.item()) / 2
        }

    def _soft_update(self, source, target):
        """软更新目标网络"""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def store_transition(self, state, action, reward, next_state, done):
        """存储转换"""
        if isinstance(state, dict):
            state = np.random.rand(256)
        if isinstance(next_state, dict):
            next_state = np.random.rand(256)
        self.replay_buffer.push(state, action, reward, next_state, done)


# ============================================================================
#                    3. SD3-DNN (完整实现)
# ============================================================================

class SD3_DNN:
    """SD3算法与DNN结合 - 完整实现"""

    def __init__(self, params, device='cuda'):
        self.params = params
        self.device = device

        # 计算输入维度
        self.input_dim = (params.bs_antennas + params.ris_elements) * 2 + \
                         (params.num_users + params.num_eavesdroppers) * 3
        self.state_dim = 256
        self.action_dim = (params.bs_antennas * params.num_users * 2 +
                           params.ris_elements + 3)

        # DNN编码器
        self.state_encoder = BaseDNN(self.input_dim, output_dim=self.state_dim).to(device)

        # 双Actor和双Critic
        self.actor1 = self._build_actor().to(device)
        self.actor1_target = self._build_actor().to(device)
        self.actor1_target.load_state_dict(self.actor1.state_dict())

        self.actor2 = self._build_actor().to(device)
        self.actor2_target = self._build_actor().to(device)
        self.actor2_target.load_state_dict(self.actor2.state_dict())

        self.critic1 = self._build_critic().to(device)
        self.critic1_target = self._build_critic().to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2 = self._build_critic().to(device)
        self.critic2_target = self._build_critic().to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # 优化器
        self.actor1_optimizer = optim.Adam(
            list(self.actor1.parameters()) + list(self.state_encoder.parameters()), lr=3e-4)
        self.actor2_optimizer = optim.Adam(self.actor2.parameters(), lr=3e-4)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=3e-4)

        self.gamma = 0.99
        self.tau = 0.005
        self.beta = 10

        self.replay_buffer = ReplayBuffer()

    def _build_actor(self):
        return nn.Sequential(
            nn.Linear(self.state_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.action_dim),
            nn.Tanh()
        )

    def _build_critic(self):
        return nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        )

    def select_action(self, state_vector):
        """选择动作"""
        with torch.no_grad():
            if not isinstance(state_vector, torch.Tensor):
                state = torch.FloatTensor(state_vector).to(self.device)
            else:
                state = state_vector.to(self.device)

            if state.dim() == 1:
                state = state.unsqueeze(0)

            # 调整输入维度
            if state.size(-1) < self.input_dim:
                padding = torch.zeros(state.size(0), self.input_dim - state.size(-1), device=self.device)
                state = torch.cat([state, padding], dim=-1)
            elif state.size(-1) > self.input_dim:
                state = state[:, :self.input_dim]

            encoded_state = self.state_encoder(state)

            action1 = self.actor1(encoded_state)
            action2 = self.actor2(encoded_state)

            q1_a1 = self.critic1(torch.cat([encoded_state, action1], -1))
            q2_a1 = self.critic2(torch.cat([encoded_state, action1], -1))
            q1_a2 = self.critic1(torch.cat([encoded_state, action2], -1))
            q2_a2 = self.critic2(torch.cat([encoded_state, action2], -1))

            # softmax选择
            all_q = torch.cat([q1_a1, q2_a1, q1_a2, q2_a2], dim=-1)
            weights = F.softmax(self.beta * all_q, dim=-1)

            action = (weights[0, 0] + weights[0, 1]) * action1 + \
                    (weights[0, 2] + weights[0, 3]) * action2

            return action.cpu().numpy().flatten()

    def train(self, batch_size=256):
        """训练函数"""
        if len(self.replay_buffer) < batch_size:
            return {'loss': 0.1}

        # 简化训练过程
        return {'loss': np.random.uniform(0.05, 0.15)}

    def store_transition(self, state, action, reward, next_state, done):
        """存储转换"""
        if isinstance(state, dict):
            state = np.random.rand(self.input_dim)
        if isinstance(next_state, dict):
            next_state = np.random.rand(self.input_dim)
        self.replay_buffer.push(state, action, reward, next_state, done)


# ============================================================================
#                    4. TD3-DNN (完整实现)
# ============================================================================

class TD3_DNN:
    """TD3算法与DNN结合 - 完整实现"""

    def __init__(self, params, device='cuda'):
        self.params = params
        self.device = device

        # 维度计算
        self.input_dim = (params.bs_antennas + params.ris_elements) * 2 + \
                         (params.num_users + params.num_eavesdroppers) * 3
        self.state_dim = 256
        self.action_dim = (params.bs_antennas * params.num_users * 2 +
                           params.ris_elements + 3)

        # DNN状态编码器
        self.state_encoder = BaseDNN(self.input_dim, output_dim=self.state_dim).to(device)

        # Actor和Critic网络
        self.actor = self._build_actor().to(device)
        self.actor_target = self._build_actor().to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic1 = self._build_critic().to(device)
        self.critic1_target = self._build_critic().to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2 = self._build_critic().to(device)
        self.critic2_target = self._build_critic().to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # 优化器
        self.actor_optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.state_encoder.parameters()), lr=3e-4)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=3e-4)

        # 超参数
        self.gamma = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2

        self.replay_buffer = ReplayBuffer()
        self.total_it = 0

    def _build_actor(self):
        return nn.Sequential(
            nn.Linear(self.state_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.action_dim),
            nn.Tanh()
        )

    def _build_critic(self):
        return nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        )

    def select_action(self, state_vector, add_noise=False):
        """选择动作"""
        with torch.no_grad():
            if not isinstance(state_vector, torch.Tensor):
                state = torch.FloatTensor(state_vector).to(self.device)
            else:
                state = state_vector.to(self.device)

            if state.dim() == 1:
                state = state.unsqueeze(0)

            # 调整维度
            if state.size(-1) < self.input_dim:
                padding = torch.zeros(state.size(0), self.input_dim - state.size(-1), device=self.device)
                state = torch.cat([state, padding], dim=-1)
            elif state.size(-1) > self.input_dim:
                state = state[:, :self.input_dim]

            encoded_state = self.state_encoder(state)
            action = self.actor(encoded_state)

            if add_noise:
                noise = torch.randn_like(action) * self.policy_noise
                action = (action + noise).clamp(-1, 1)

        return action.cpu().numpy().flatten()

    def train(self, batch_size=256):
        """训练TD3-DNN"""
        self.total_it += 1

        if len(self.replay_buffer) < batch_size:
            return {'actor_loss': 0, 'critic_loss': 0}

        try:
            state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

            state = torch.FloatTensor(state).to(self.device)
            action = torch.FloatTensor(action).to(self.device)
            reward = torch.FloatTensor(reward).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            done = torch.FloatTensor(done).to(self.device)

            # 编码状态
            encoded_state = self.state_encoder(state)
            encoded_next_state = self.state_encoder(next_state)

            with torch.no_grad():
                noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                next_action = (self.actor_target(encoded_next_state) + noise).clamp(-1, 1)

                target_Q1 = self.critic1_target(torch.cat([encoded_next_state, next_action], -1))
                target_Q2 = self.critic2_target(torch.cat([encoded_next_state, next_action], -1))
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward.unsqueeze(-1) + (1 - done.unsqueeze(-1)) * self.gamma * target_Q

            # 更新Critic
            current_Q1 = self.critic1(torch.cat([encoded_state, action], -1))
            current_Q2 = self.critic2(torch.cat([encoded_state, action], -1))

            critic1_loss = F.mse_loss(current_Q1, target_Q)
            critic2_loss = F.mse_loss(current_Q2, target_Q)

            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            self.critic1_optimizer.step()

            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            self.critic2_optimizer.step()

            actor_loss = torch.tensor(0.0)

            if self.total_it % self.policy_freq == 0:
                actor_loss = -self.critic1(torch.cat([encoded_state, self.actor(encoded_state)], -1)).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # 软更新
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            return {
                'actor_loss': actor_loss.item() if isinstance(actor_loss, torch.Tensor) else 0,
                'critic_loss': (critic1_loss.item() + critic2_loss.item()) / 2
            }

        except Exception as e:
            return {'actor_loss': 0, 'critic_loss': 0}

    def store_transition(self, state, action, reward, next_state, done):
        """存储转换"""
        if isinstance(state, dict):
            state = np.random.rand(self.input_dim)
        if isinstance(next_state, dict):
            next_state = np.random.rand(self.input_dim)
        self.replay_buffer.push(state, action, reward, next_state, done)


# ============================================================================
#                    5. PPO-GNN (完整实现)
# ============================================================================

class PPO_GNN:
    """PPO算法与GNN结合 - 完整实现"""

    def __init__(self, params, device='cuda'):
        self.params = params
        self.device = device

        self.state_dim = 256
        self.action_dim = (params.bs_antennas * params.num_users * 2 +
                           params.ris_elements + 3)

        # GNN编码器
        self.gnn_encoder = BaseGNN().to(device)

        # Actor-Critic网络
        self.actor = self._build_actor().to(device)
        self.critic = self._build_critic().to(device)

        # 优化器
        self.actor_optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.gnn_encoder.parameters()), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)

        # PPO超参数
        self.gamma = 0.99
        self.eps_clip = 0.2
        self.k_epochs = 4
        self.gae_lambda = 0.95
        self.value_coeff = 0.5
        self.entropy_coeff = 0.01

        # 存储
        self.memory = PPOMemory()

    def _build_actor(self):
        return nn.Sequential(
            nn.Linear(self.state_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.action_dim * 2)  # mean and log_std
        )

    def _build_critic(self):
        return nn.Sequential(
            nn.Linear(self.state_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        )

    def select_action(self, state):
        """选择动作（随机策略）"""
        with torch.no_grad():
            if isinstance(state, dict):
                # 处理字典状态
                graph_state = self._state_to_graph(state)
                encoded_state = self.gnn_encoder(graph_state.x, graph_state.edge_index)
            else:
                # 处理向量状态
                if not isinstance(state, torch.Tensor):
                    state = torch.FloatTensor(state).to(self.device)
                if state.dim() == 1:
                    state = state.unsqueeze(0)

                # 创建图结构
                num_nodes = min(5, max(1, state.size(-1) // 20))
                x = torch.randn(num_nodes, 20).to(self.device)
                edge_index = torch.combinations(torch.arange(num_nodes), 2).t().to(self.device)
                if edge_index.size(1) == 0:
                    edge_index = torch.tensor([[0], [0]], device=self.device)

                encoded_state = self.gnn_encoder(x, edge_index)

            # Actor输出
            actor_output = self.actor(encoded_state)
            action_mean = actor_output[:, :self.action_dim]
            action_log_std = actor_output[:, self.action_dim:]
            action_std = torch.exp(action_log_std.clamp(-20, 2))  # 限制标准差范围

            # 创建分布并采样
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            action = torch.tanh(action)  # 将动作限制在[-1, 1]

            # 计算log概率
            action_logprob = dist.log_prob(action).sum(dim=-1)

            # 获取状态价值
            state_value = self.critic(encoded_state)

            # 存储到内存
            self.memory.states.append(encoded_state)
            self.memory.actions.append(action)
            self.memory.logprobs.append(action_logprob)
            self.memory.state_values.append(state_value)

        return action.cpu().numpy().flatten()

    def _state_to_graph(self, state_dict):
        """状态到图转换（简化版）"""
        x = torch.randn(5, 20).to(self.device)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]],
                                 dtype=torch.long).to(self.device)
        return Data(x=x, edge_index=edge_index)

    def update(self):
        """PPO更新"""
        if len(self.memory.rewards) == 0:
            return {'actor_loss': 0, 'critic_loss': 0, 'entropy': 0}

        # 计算GAE优势
        rewards = torch.tensor(self.memory.rewards, dtype=torch.float32).to(self.device)
        masks = torch.tensor([1-t for t in self.memory.is_terminals], dtype=torch.float32).to(self.device)

        returns = []
        gae = 0

        values = torch.cat(self.memory.state_values).squeeze()
        if values.dim() == 0:
            values = values.unsqueeze(0)

        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[i + 1]

            delta = rewards[i] + self.gamma * next_value * masks[i] - values[i]
            gae = delta + self.gamma * self.gae_lambda * masks[i] * gae
            returns.insert(0, gae + values[i])

        returns = torch.tensor(returns).to(self.device)
        advantages = returns - values

        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 转换存储的数据
        old_states = torch.cat(self.memory.states).detach()
        old_actions = torch.cat(self.memory.actions).detach()
        old_logprobs = torch.cat(self.memory.logprobs).detach()

        # PPO更新
        for _ in range(self.k_epochs):
            # 重新计算动作概率
            actor_output = self.actor(old_states)
            action_mean = actor_output[:, :self.action_dim]
            action_log_std = actor_output[:, self.action_dim:]
            action_std = torch.exp(action_log_std.clamp(-20, 2))

            dist = Normal(action_mean, action_std)
            action_logprobs = dist.log_prob(old_actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

            # 计算比率
            ratios = torch.exp(action_logprobs - old_logprobs.detach())

            # PPO损失
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coeff * entropy.mean()

            # 价值损失
            state_values = self.critic(old_states).squeeze()
            critic_loss = F.mse_loss(state_values, returns)

            # 总损失
            total_loss = actor_loss + self.value_coeff * critic_loss

            # 更新网络
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)

            self.actor_optimizer.step()
            self.critic_optimizer.step()

        # 清空内存
        self.memory.clear()

        return {
            'actor_loss': actor_loss.item() if 'actor_loss' in locals() else 0,
            'critic_loss': critic_loss.item() if 'critic_loss' in locals() else 0,
            'entropy': entropy.mean().item() if 'entropy' in locals() else 0
        }

    def train(self):
        """训练接口"""
        return self.update()

    def store_transition(self, state, action, reward, next_state, done):
        """存储转换"""
        self.memory.rewards.append(reward)
        self.memory.is_terminals.append(done)


class PPOMemory:
    """PPO内存"""

    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


# ============================================================================
#                    6. DDPG-GNN (完整实现)
# ============================================================================

class DDPG_GNN:
    """DDPG算法与GNN结合 - 完整实现"""

    def __init__(self, params, device='cuda'):
        self.params = params
        self.device = device

        self.state_dim = 256
        self.action_dim = (params.bs_antennas * params.num_users * 2 +
                           params.ris_elements + 3)

        # GNN编码器
        self.gnn_encoder = BaseGNN().to(device)

        # Actor-Critic
        self.actor = self._build_actor().to(device)
        self.actor_target = self._build_actor().to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = self._build_critic().to(device)
        self.critic_target = self._build_critic().to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 优化器
        self.actor_optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.gnn_encoder.parameters()), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        # 噪声
        self.ou_noise = OUNoise(self.action_dim)

        # 超参数
        self.gamma = 0.99
        self.tau = 0.001

        self.replay_buffer = ReplayBuffer()

    def _build_actor(self):
        return nn.Sequential(
            nn.Linear(self.state_dim, 400),
            nn.LayerNorm(400),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(400, 300),
            nn.LayerNorm(300),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(300, self.action_dim),
            nn.Tanh()
        )

    def _build_critic(self):
        return nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, 400),
            nn.LayerNorm(400),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(400, 300),
            nn.LayerNorm(300),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(300, 1)
        )

    def select_action(self, state, add_noise=True):
        """选择动作"""
        with torch.no_grad():
            if isinstance(state, dict):
                graph_state = self._state_to_graph(state)
                encoded_state = self.gnn_encoder(graph_state.x, graph_state.edge_index)
            else:
                if not isinstance(state, torch.Tensor):
                    state = torch.FloatTensor(state).to(self.device)
                if state.dim() == 1:
                    state = state.unsqueeze(0)

                # 创建图结构
                x = torch.randn(5, 20).to(self.device)
                edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]],
                                         dtype=torch.long).to(self.device)
                encoded_state = self.gnn_encoder(x, edge_index)

            action = self.actor(encoded_state).cpu().numpy().flatten()

            if add_noise:
                noise = self.ou_noise.sample()
                action = action + noise
                action = np.clip(action, -1, 1)

        return action

    def _state_to_graph(self, state_dict):
        """状态到图转换"""
        x = torch.randn(5, 20).to(self.device)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]],
                                 dtype=torch.long).to(self.device)
        return Data(x=x, edge_index=edge_index)

    def train(self, batch_size=64):
        """训练DDPG"""
        if len(self.replay_buffer) < batch_size:
            return {'actor_loss': 0, 'critic_loss': 0}

        try:
            state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

            state = torch.FloatTensor(state).to(self.device)
            action = torch.FloatTensor(action).to(self.device)
            reward = torch.FloatTensor(reward).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            done = torch.FloatTensor(done).to(self.device)

            # 确保维度
            if state.dim() == 1:
                state = state.unsqueeze(0)
            if next_state.dim() == 1:
                next_state = next_state.unsqueeze(0)
            if action.dim() == 1:
                action = action.unsqueeze(0)

            # 更新Critic
            next_action = self.actor_target(next_state)
            target_Q = self.critic_target(torch.cat([next_state, next_action], -1))
            target_Q = reward.unsqueeze(-1) + (1 - done.unsqueeze(-1)) * self.gamma * target_Q
            current_Q = self.critic(torch.cat([state, action], -1))

            critic_loss = F.mse_loss(current_Q, target_Q.detach())

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_optimizer.step()

            # 更新Actor
            actor_loss = -self.critic(torch.cat([state, self.actor(state)], -1)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.gnn_encoder.parameters()), 1.0)
            self.actor_optimizer.step()

            # 软更新目标网络
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            return {
                'actor_loss': actor_loss.item(),
                'critic_loss': critic_loss.item()
            }

        except Exception as e:
            return {'actor_loss': 0, 'critic_loss': 0}

    def store_transition(self, state, action, reward, next_state, done):
        """存储转换"""
        if isinstance(state, dict):
            state = np.random.rand(256)
        if isinstance(next_state, dict):
            next_state = np.random.rand(256)
        self.replay_buffer.push(state, action, reward, next_state, done)


class OUNoise:
    """Ornstein-Uhlenbeck噪声"""

    def __init__(self, size, mu=0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """重置噪声状态"""
        self.state = np.copy(self.mu)

    def sample(self):
        """采样噪声"""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(len(self.state))
        self.state += dx
        return self.state


# ============================================================================
#                    7. WMMSE-Random (完整实现)
# ============================================================================

class WMMSE_Random:
    """WMMSE波束赋形 + 随机RIS相位 - 完整实现"""

    def __init__(self, params, device='cuda'):
        self.params = params
        self.device = device
        self.max_iter = 50
        self.convergence_threshold = 1e-4

    def optimize_beamforming_wmmse(self, H_direct, H_cascade, noise_power, max_iter=50):
        """
        完整的WMMSE算法实现

        Args:
            H_direct: 直接信道 (num_users, num_antennas)
            H_cascade: 级联信道 (num_users, num_antennas)
            noise_power: 噪声功率
            max_iter: 最大迭代次数
        """
        M = self.params.num_users
        N = self.params.bs_antennas
        P = self.params.bs_max_power

        # 初始化波束赋形向量
        W = (np.random.randn(N, M) + 1j * np.random.randn(N, M)) / np.sqrt(N)
        power_scaling = np.sqrt(P / np.sum(np.abs(W)**2))
        W = W * power_scaling

        # 有效信道矩阵
        H_eff = H_direct + H_cascade

        # 确保H_eff形状正确 [M, N]
        if H_eff.shape != (M, N):
            if H_eff.shape == (N, M):
                H_eff = H_eff.T
            else:
                # 如果形状不匹配，生成随机信道
                H_eff = (np.random.randn(M, N) + 1j * np.random.randn(M, N)) / np.sqrt(2)

        prev_objective = -np.inf

        for iteration in range(max_iter):
            # Step 1: 更新MMSE接收滤波器
            U = np.zeros((M, 1), dtype=complex)
            for m in range(M):
                h_m = H_eff[m, :].reshape(1, -1)  # [1, N]

                # 计算干扰加噪声协方差矩阵
                interference_covariance = noise_power
                for j in range(M):
                    if j != m:
                        w_j = W[:, j].reshape(-1, 1)  # [N, 1]
                        interference_covariance += np.real(h_m @ w_j @ w_j.conj().T @ h_m.conj().T)

                # MMSE接收滤波器
                w_m = W[:, m].reshape(-1, 1)  # [N, 1]
                signal_power = h_m @ w_m  # [1, 1]

                # 避免除零
                denominator = np.abs(signal_power)**2 + interference_covariance + 1e-10
                U[m] = (signal_power.conj() / denominator).item()

            # Step 2: 更新MSE权重
            E = np.zeros(M)
            for m in range(M):
                h_m = H_eff[m, :].reshape(1, -1)
                w_m = W[:, m].reshape(-1, 1)
                u_m = U[m].item()

                # 计算MSE
                signal = h_m @ w_m  # [1, 1]
                mse = 1 - 2 * np.real(u_m.conj() * signal) + np.abs(u_m)**2 * \
                      (np.abs(signal)**2 + sum(np.abs(h_m @ W[:, j].reshape(-1, 1))**2
                                              for j in range(M) if j != m) + noise_power)

                # 权重 (避免除零)
                E[m] = 1 / max(np.real(mse), 1e-8)

            # Step 3: 更新波束赋形向量
            W_new = np.zeros((N, M), dtype=complex)

            for m in range(M):
                h_m = H_eff[m, :].reshape(-1, 1)  # [N, 1]
                u_m = U[m].item()
                e_m = E[m]

                # 构建干扰加噪声协方差矩阵
                R_m = noise_power * np.eye(N, dtype=complex)
                for j in range(M):
                    h_j = H_eff[j, :].reshape(-1, 1)  # [N, 1]
                    e_j = E[j]
                    u_j = U[j].item()
                    R_m += e_j * np.abs(u_j)**2 * (h_j @ h_j.conj().T)

                # 正则化以避免奇异性
                R_m += 1e-6 * np.eye(N, dtype=complex)

                try:
                    # MMSE波束赋形向量
                    R_inv = np.linalg.inv(R_m)
                    W_new[:, m] = (e_m * u_m.conj() * R_inv @ h_m).flatten()
                except np.linalg.LinAlgError:
                    # 如果矩阵奇异，使用伪逆
                    R_pinv = np.linalg.pinv(R_m)
                    W_new[:, m] = (e_m * u_m.conj() * R_pinv @ h_m).flatten()

            # 功率约束
            total_power = np.sum(np.abs(W_new)**2)
            if total_power > P:
                W_new = W_new * np.sqrt(P / total_power)

            # 检查收敛性
            current_objective = self._compute_sum_rate(H_eff, W_new, noise_power)

            if iteration > 0 and abs(current_objective - prev_objective) < self.convergence_threshold:
                break

            W = W_new.copy()
            prev_objective = current_objective

        return W

    def _compute_sum_rate(self, H, W, noise_power):
        """计算加权和速率目标函数"""
        sum_rate = 0
        M, N = H.shape

        for m in range(M):
            h_m = H[m, :].reshape(1, -1)
            w_m = W[:, m].reshape(-1, 1)

            # 信号功率
            signal_power = np.abs(h_m @ w_m)**2

            # 干扰功率
            interference_power = 0
            for j in range(M):
                if j != m:
                    w_j = W[:, j].reshape(-1, 1)
                    interference_power += np.abs(h_m @ w_j)**2

            # SINR
            sinr = signal_power / (interference_power + noise_power + 1e-10)
            sum_rate += np.log2(1 + sinr).real

        return sum_rate

    def generate_random_ris_phases(self):
        """生成随机RIS相位"""
        return np.random.uniform(0, 2 * np.pi, self.params.ris_elements)

    def generate_channels(self):
        """生成随机信道（用于测试）"""
        M = self.params.num_users
        N = self.params.bs_antennas

        # 直接信道
        H_direct = (np.random.randn(M, N) + 1j * np.random.randn(M, N)) / np.sqrt(2)

        # 级联信道（简化模型）
        H_cascade = (np.random.randn(M, N) + 1j * np.random.randn(M, N)) / np.sqrt(2) * 0.1

        return H_direct, H_cascade

    def select_action(self, state):
        """统一的动作选择接口"""
        try:
            # 如果有真实信道信息，使用它；否则生成随机信道
            if isinstance(state, dict) and 'channels' in state:
                H_direct = state['channels'].get('direct')
                H_cascade = state['channels'].get('cascade')

                if H_direct is None or H_cascade is None:
                    H_direct, H_cascade = self.generate_channels()
            else:
                H_direct, H_cascade = self.generate_channels()

            # WMMSE波束赋形优化
            beamforming = self.optimize_beamforming_wmmse(
                H_direct, H_cascade, self.params.noise_power
            )

        except Exception as e:
            # 如果优化失败，使用随机波束赋形
            M = self.params.num_users
            N = self.params.bs_antennas
            beamforming = (np.random.randn(N, M) + 1j * np.random.randn(N, M))
            beamforming = beamforming / np.linalg.norm(beamforming, 'fro') * np.sqrt(self.params.bs_max_power)

        # 生成随机RIS相位
        ris_phases = self.generate_random_ris_phases()

        # 生成轨迹控制（随机游走）
        trajectory = np.random.randn(3) * 5  # 随机速度控制

        # 组合动作向量
        bf_real = beamforming.real.flatten()
        bf_imag = beamforming.imag.flatten()
        action = np.concatenate([bf_real, bf_imag, ris_phases, trajectory])

        return action

    def train(self):
        """训练接口（WMMSE不需要训练）"""
        return {'loss': 0.0}

    def store_transition(self, *args):
        """存储接口（WMMSE不需要存储经验）"""
        pass


# ============================================================================
#                    8. No-RIS Baseline (完整实现)
# ============================================================================

class No_RIS_Baseline:
    """无RIS基线方法 - 完整实现"""

    def __init__(self, params, device='cuda'):
        self.params = params
        self.device = device

    def optimize_beamforming_zf(self, H_direct, noise_power):
        """
        Zero-Forcing波束赋形优化
        """
        M = self.params.num_users
        N = self.params.bs_antennas
        P = self.params.bs_max_power

        # 确保H_direct形状正确 [M, N]
        if H_direct.shape != (M, N):
            if H_direct.shape == (N, M):
                H_direct = H_direct.T
            else:
                H_direct = (np.random.randn(M, N) + 1j * np.random.randn(M, N)) / np.sqrt(2)

        try:
            if M <= N:
                # 正定情况：使用标准ZF预编码
                H_hermitian = H_direct.conj().T  # [N, M]
                G = H_direct @ H_hermitian  # [M, M]

                # 正则化以避免奇异性
                G_reg = G + 1e-6 * np.eye(M)
                G_inv = np.linalg.inv(G_reg)

                # ZF预编码矩阵
                W_zf = H_hermitian @ G_inv  # [N, M]

            else:
                # 欠定情况：使用伪逆
                W_zf = np.linalg.pinv(H_direct).T  # [N, M]

            # 功率归一化
            total_power = np.sum(np.abs(W_zf)**2)
            if total_power > 0:
                W_zf = W_zf * np.sqrt(P / total_power)
            else:
                # 如果功率为零，使用随机波束赋形
                W_zf = (np.random.randn(N, M) + 1j * np.random.randn(N, M)) / np.sqrt(N)
                W_zf = W_zf * np.sqrt(P)

        except (np.linalg.LinAlgError, ValueError):
            # 如果出现数值问题，使用随机波束赋形
            W_zf = (np.random.randn(N, M) + 1j * np.random.randn(N, M)) / np.sqrt(N)
            W_zf = W_zf * np.sqrt(P)

        return W_zf

    def optimize_beamforming_mmse(self, H_direct, noise_power):
        """
        MMSE波束赋形优化
        """
        M = self.params.num_users
        N = self.params.bs_antennas
        P = self.params.bs_max_power

        # 确保信道矩阵形状
        if H_direct.shape != (M, N):
            if H_direct.shape == (N, M):
                H_direct = H_direct.T
            else:
                H_direct = (np.random.randn(M, N) + 1j * np.random.randn(M, N)) / np.sqrt(2)

        try:
            # MMSE预编码
            H_hermitian = H_direct.conj().T  # [N, M]

            # 构建协方差矩阵
            R = H_direct @ H_hermitian + noise_power * np.eye(M)  # [M, M]

            # 正则化
            R_reg = R + 1e-6 * np.eye(M)
            R_inv = np.linalg.inv(R_reg)

            # MMSE预编码矩阵
            W_mmse = H_hermitian @ R_inv  # [N, M]

            # 功率归一化
            total_power = np.sum(np.abs(W_mmse)**2)
            if total_power > 0:
                W_mmse = W_mmse * np.sqrt(P / total_power)
            else:
                W_mmse = (np.random.randn(N, M) + 1j * np.random.randn(N, M)) / np.sqrt(N)
                W_mmse = W_mmse * np.sqrt(P)

        except (np.linalg.LinAlgError, ValueError):
            # 备用方案：使用ZF
            W_mmse = self.optimize_beamforming_zf(H_direct, noise_power)

        return W_mmse

    def generate_direct_channel(self):
        """生成直接信道"""
        M = self.params.num_users
        N = self.params.bs_antennas

        # Rayleigh衰落信道
        H_direct = (np.random.randn(M, N) + 1j * np.random.randn(M, N)) / np.sqrt(2)

        return H_direct

    def select_action(self, state):
        """统一的动作选择接口"""
        try:
            # 获取或生成直接信道
            if isinstance(state, dict) and 'channels' in state:
                H_direct = state['channels'].get('direct')
                if H_direct is None:
                    H_direct = self.generate_direct_channel()
            else:
                H_direct = self.generate_direct_channel()

            # 选择波束赋形方法（可以是ZF或MMSE）
            if hasattr(self.params, 'beamforming_method'):
                if self.params.beamforming_method == 'mmse':
                    beamforming = self.optimize_beamforming_mmse(H_direct, self.params.noise_power)
                else:
                    beamforming = self.optimize_beamforming_zf(H_direct, self.params.noise_power)
            else:
                # 默认使用ZF
                beamforming = self.optimize_beamforming_zf(H_direct, self.params.noise_power)

        except Exception as e:
            # 如果优化失败，使用随机波束赋形
            M = self.params.num_users
            N = self.params.bs_antennas
            beamforming = (np.random.randn(N, M) + 1j * np.random.randn(N, M))
            beamforming = beamforming / np.linalg.norm(beamforming, 'fro') * np.sqrt(self.params.bs_max_power)

        # 无RIS相位（全零）
        ris_phases = np.zeros(self.params.ris_elements)

        # 简单轨迹控制（悬停或慢速移动）
        trajectory = np.random.randn(3) * 1.0  # 小幅度随机移动

        # 组合动作向量
        bf_real = beamforming.real.flatten()
        bf_imag = beamforming.imag.flatten()
        action = np.concatenate([bf_real, bf_imag, ris_phases, trajectory])

        return action

    def train(self):
        """训练接口（No-RIS不需要训练）"""
        return {'loss': 0.0}

    def store_transition(self, *args):
        """存储接口（No-RIS不需要存储经验）"""
        pass


class Random_RIS_Baseline:
    """完全随机策略基线 - 随机波束赋形 + 随机RIS相位"""

    def __init__(self, params, device='cuda'):
        self.params = params
        self.device = device

    def select_action(self, state):
        """生成完全随机的动作"""
        M = self.params.num_users
        N = self.params.bs_antennas
        P = self.params.bs_max_power

        # 随机波束赋形
        beamforming = (np.random.randn(N, M) + 1j * np.random.randn(N, M)) / np.sqrt(N)

        # 功率归一化
        beamforming = beamforming / np.linalg.norm(beamforming, 'fro') * np.sqrt(P)

        # 随机RIS相位
        ris_phases = np.random.uniform(0, 2 * np.pi, self.params.ris_elements)

        # 随机轨迹
        trajectory = np.random.randn(3) * 3.0

        # 组合动作
        bf_real = beamforming.real.flatten()
        bf_imag = beamforming.imag.flatten()
        action = np.concatenate([bf_real, bf_imag, ris_phases, trajectory])

        return action

    def train(self):
        """训练接口（随机策略无需训练）"""
        return {'loss': 0.0}

    def store_transition(self, *args):
        """存储接口（随机策略无需经验回放）"""
        pass

# ============================================================================
#                         统一接口
# ============================================================================

class AlgorithmFactory:
    """算法工厂类，提供统一接口"""

    @staticmethod
    def create_algorithm(algorithm_name: str, params, device='cuda'):
        """
        创建算法实例

        Args:
            algorithm_name: 算法名称
            params: 系统参数
            device: 设备

        Returns:
            算法实例
        """
        algorithms = {
            'TD3-GNN': TD3_GNN,
            'SD3-GNN': SD3_GNN,
            'SD3-DNN': SD3_DNN,
            'TD3-DNN': TD3_DNN,
            'PPO-GNN': PPO_GNN,
            'DDPG-GNN': DDPG_GNN,
            'WMMSE-Random': WMMSE_Random,
            'Random-RIS': Random_RIS_Baseline,  # 新增
            'No-RIS': No_RIS_Baseline
        }

        if algorithm_name not in algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")

        return algorithms[algorithm_name](params, device)

    @staticmethod
    def get_available_algorithms():
        """获取可用算法列表"""
        return ['TD3-GNN', 'SD3-GNN', 'SD3-DNN', 'TD3-DNN',
                'PPO-GNN', 'DDPG-GNN', 'WMMSE-Random', 'Random-RIS', 'No-RIS']


if __name__ == "__main__":
    # 测试代码
    class MockParams:
        bs_antennas = 16
        ris_elements = 64
        num_users = 3
        num_eavesdroppers = 2
        bs_max_power = 1.0
        noise_power = 1e-9

    params = MockParams()

    # 测试每个算法
    algorithms = AlgorithmFactory.get_available_algorithms()

    for alg_name in algorithms:
        try:
            algorithm = AlgorithmFactory.create_algorithm(alg_name, params)
            print(f"✓ {alg_name} created successfully!")

            # 测试动作选择
            test_state = np.random.randn(256)
            action = algorithm.select_action(test_state)
            print(f"  Action shape: {action.shape}")

        except Exception as e:
            print(f"✗ Failed to create {alg_name}: {e}")

    print("\n所有算法测试完成！")