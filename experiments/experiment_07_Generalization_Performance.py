"""
实验七：泛化性能测试
Experiment 7: Generalization Performance Evaluation

验证PINN-SecGNN在不同网络规模下的泛化能力
对应定理3：Generalization Bound
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List
import json
import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pinn_secgnn_framework import PINNSecGNN, PINNSecGNNTrainer
from models.uav_ris_system_model import SystemParameters, UAVRISSecureSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeneralizationExperimentConfig:
    """泛化实验配置"""

    def __init__(self):
        # 训练配置（固定规模）
        self.train_K = 10  # 用户数
        self.train_E = 2  # 窃听者数
        self.train_N = 100  # RIS元素数

        # 测试配置（变化规模）
        self.test_configs = [
            {'K': 5, 'E': 1, 'N': 50, 'name': 'Small'},
            {'K': 8, 'E': 2, 'N': 80, 'name': 'Medium-Small'},
            {'K': 10, 'E': 2, 'N': 100, 'name': 'Training Size'},
            {'K': 12, 'E': 3, 'N': 120, 'name': 'Medium-Large'},
            {'K': 15, 'E': 3, 'N': 150, 'name': 'Large'},
            {'K': 20, 'E': 4, 'N': 200, 'name': 'Very Large'},
        ]

        # 训练参数
        self.train_epochs = 200
        self.eval_episodes = 50
        self.random_seeds = [42, 123, 456]

        # 设备
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


class GeneralizationEvaluator:
    """泛化性能评估器"""

    def __init__(self, config: GeneralizationExperimentConfig):
        self.config = config

        # 创建结果目录
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = Path(f"results/{timestamp}_007_generalization")
        self.result_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generalization experiment initialized")
        logger.info(f"Training size: K={config.train_K}, E={config.train_E}, N={config.train_N}")
        logger.info(f"Test configurations: {len(config.test_configs)}")

    def train_model_fixed_size(self, seed: int) -> PINNSecGNN:
        """在固定规模上训练模型"""
        logger.info(f"Training model on fixed size (seed={seed})...")

        # 设置随机种子
        np.random.seed(seed)
        torch.manual_seed(seed)

        # 创建系统参数（训练规模）
        params = SystemParameters()
        params.num_users = self.config.train_K
        params.num_eavesdroppers = self.config.train_E
        params.ris_elements = self.config.train_N

        # 创建PINN-SecGNN模型
        model_config = {
            'pinn': {
                'input_dim': 3 + 3 * self.config.train_K + 3 * self.config.train_E,
                'output_dim': 100,
                'hidden_dim': 256,
                'env_dim': 16,
                'num_bs_antennas': params.bs_antennas,
                'num_ris_elements': self.config.train_N,
                'num_users': self.config.train_K,
                'num_eavesdroppers': self.config.train_E,
                'max_velocity': 20.0
            },
            'gnn': {
                'gnn_input_dim': 256,
                'gnn_hidden_dim': 128,
                'output_dim': 100,
                'num_gnn_layers': 3,
                'edge_feat_dim': 1
            },
            'learning_rate': 1e-3,
            'weight_decay': 1e-5,
            'max_epochs': self.config.train_epochs,
            'batch_size': 32
        }

        model = PINNSecGNN(model_config).to(self.config.device)
        trainer = PINNSecGNNTrainer(model_config, device=self.config.device)

        # 训练循环（简化版）
        for epoch in range(self.config.train_epochs):
            # 生成训练数据
            batch_data = self._generate_training_batch(params, model_config)

            # 训练步骤
            state = batch_data['state'].to(self.config.device)
            env_features = batch_data['env_features'].to(self.config.device)
            system_state = batch_data['system_state']

            # 前向传播
            results = model(state, env_features, system_state, training=True)

            # 反向传播
            loss = results['losses']['total']
            trainer.optimizer.zero_grad()
            loss.backward()
            trainer.optimizer.step()

            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}/{self.config.train_epochs}, "
                            f"Loss: {loss.item():.4f}, "
                            f"SEE: {results['see'].mean().item():.4f}")

        logger.info(f"Training completed for seed {seed}")
        return model

    def evaluate_on_size(self,
                         model: PINNSecGNN,
                         test_config: Dict,
                         seed: int) -> Dict:
        """在特定规模上评估模型"""
        logger.info(f"Evaluating on size: {test_config['name']} "
                    f"(K={test_config['K']}, E={test_config['E']}, N={test_config['N']})")

        # 设置随机种子
        np.random.seed(seed + 1000)  # 不同于训练种子
        torch.manual_seed(seed + 1000)

        # 创建测试系统参数
        params = SystemParameters()
        params.num_users = test_config['K']
        params.num_eavesdroppers = test_config['E']
        params.ris_elements = test_config['N']

        # 评估多个episode
        see_values = []

        for ep in range(self.config.eval_episodes):
            # 生成测试数据
            batch_data = self._generate_test_batch(params, test_config)

            # 前向传播（评估模式）
            model.eval()
            with torch.no_grad():
                state = batch_data['state'].to(self.config.device)
                env_features = batch_data['env_features'].to(self.config.device)
                system_state = batch_data['system_state']

                results = model(state, env_features, system_state, training=False)
                see = results['see'].mean().item()
                see_values.append(see)

        # 统计结果
        eval_result = {
            'config': test_config,
            'see_mean': np.mean(see_values),
            'see_std': np.std(see_values),
            'see_median': np.median(see_values),
            'see_min': np.min(see_values),
            'see_max': np.max(see_values),
            'num_episodes': len(see_values)
        }

        logger.info(f"Evaluation result: SEE = {eval_result['see_mean']:.4f} ± {eval_result['see_std']:.4f}")

        return eval_result

    def train_specialized_model(self, test_config: Dict, seed: int) -> float:
        """训练专门针对某个规模的模型（作为性能上界）"""
        logger.info(f"Training specialized model for size: {test_config['name']}")

        # 设置随机种子
        np.random.seed(seed + 2000)
        torch.manual_seed(seed + 2000)

        # 创建系统参数
        params = SystemParameters()
        params.num_users = test_config['K']
        params.num_eavesdroppers = test_config['E']
        params.ris_elements = test_config['N']

        # 创建模型
        model_config = {
            'pinn': {
                'input_dim': 3 + 3 * test_config['K'] + 3 * test_config['E'],
                'output_dim': 100,
                'hidden_dim': 256,
                'env_dim': 16,
                'num_bs_antennas': params.bs_antennas,
                'num_ris_elements': test_config['N'],
                'num_users': test_config['K'],
                'num_eavesdroppers': test_config['E'],
                'max_velocity': 20.0
            },
            'gnn': {
                'gnn_input_dim': 256,
                'gnn_hidden_dim': 128,
                'output_dim': 100,
                'num_gnn_layers': 3,
                'edge_feat_dim': 1
            },
            'learning_rate': 1e-3,
            'weight_decay': 1e-5,
            'max_epochs': self.config.train_epochs,
            'batch_size': 32
        }

        model = PINNSecGNN(model_config).to(self.config.device)
        trainer = PINNSecGNNTrainer(model_config, device=self.config.device)

        # 训练（简化版）
        for epoch in range(self.config.train_epochs):
            batch_data = self._generate_training_batch(params, model_config)

            state = batch_data['state'].to(self.config.device)
            env_features = batch_data['env_features'].to(self.config.device)
            system_state = batch_data['system_state']

            results = model(state, env_features, system_state, training=True)

            loss = results['losses']['total']
            trainer.optimizer.zero_grad()
            loss.backward()
            trainer.optimizer.step()

        # 评估
        see_values = []
        model.eval()
        for _ in range(self.config.eval_episodes):
            batch_data = self._generate_test_batch(params, test_config)

            with torch.no_grad():
                state = batch_data['state'].to(self.config.device)
                env_features = batch_data['env_features'].to(self.config.device)
                system_state = batch_data['system_state']

                results = model(state, env_features, system_state, training=False)
                see_values.append(results['see'].mean().item())

        specialized_see = np.mean(see_values)
        logger.info(f"Specialized model SEE: {specialized_see:.4f}")

        return specialized_see

    def _generate_training_batch(self, params: SystemParameters, model_config: Dict) -> Dict:
        """生成训练批次数据"""
        batch_size = model_config['batch_size']

        # 随机状态
        state = torch.randn(batch_size, model_config['pinn']['input_dim'])
        env_features = torch.randn(batch_size, model_config['pinn']['env_dim'])

        # 系统状态（简化）
        system_state = {
            'H_br': torch.randn(batch_size, params.ris_elements, params.bs_antennas, dtype=torch.complex64),
            'h_ru': torch.randn(batch_size, params.num_users, params.ris_elements, dtype=torch.complex64),
            'h_re_worst': torch.randn(batch_size, params.num_eavesdroppers, params.ris_elements, dtype=torch.complex64),
            'noise_power': 1e-13,
            'max_power': params.bs_max_power,
            'ris_quantization_bits': 3,
            'num_users': params.num_users,
            'num_eavesdroppers': params.num_eavesdroppers
        }

        return {
            'state': state,
            'env_features': env_features,
            'system_state': system_state
        }

    def _generate_test_batch(self, params: SystemParameters, test_config: Dict) -> Dict:
        """生成测试批次数据"""
        # 单个样本
        state = torch.randn(1, 3 + 3 * test_config['K'] + 3 * test_config['E'])
        env_features = torch.randn(1, 16)

        system_state = {
            'H_br': torch.randn(1, test_config['N'], params.bs_antennas, dtype=torch.complex64),
            'h_ru': torch.randn(1, test_config['K'], test_config['N'], dtype=torch.complex64),
            'h_re_worst': torch.randn(1, test_config['E'], test_config['N'], dtype=torch.complex64),
            'noise_power': 1e-13,
            'max_power': params.bs_max_power,
            'ris_quantization_bits': 3,
            'num_users': test_config['K'],
            'num_eavesdroppers': test_config['E']
        }

        return {
            'state': state,
            'env_features': env_features,
            'system_state': system_state
        }

    def run_full_experiment(self):
        """运行完整的泛化实验"""
        logger.info("=" * 80)
        logger.info("EXPERIMENT 7: GENERALIZATION PERFORMANCE EVALUATION")
        logger.info("=" * 80)

        all_results = []

        for seed in self.config.random_seeds:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Running with seed {seed}")
            logger.info(f"{'=' * 60}")

            # 训练固定规模的模型
            trained_model = self.train_model_fixed_size(seed)

            # 保存模型
            model_path = self.result_dir / f"trained_model_seed{seed}.pth"
            torch.save(trained_model.state_dict(), model_path)

            # 在不同规模上评估
            seed_results = []
            for test_config in self.config.test_configs:
                # 泛化评估
                generalized_result = self.evaluate_on_size(trained_model, test_config, seed)

                # 专用模型评估（仅对非训练规模）
                if (test_config['K'] != self.config.train_K or
                        test_config['E'] != self.config.train_E or
                        test_config['N'] != self.config.train_N):
                    specialized_see = self.train_specialized_model(test_config, seed)
                else:
                    specialized_see = generalized_result['see_mean']  # 训练规模，两者相同

                # 计算性能差距
                performance_gap = (specialized_see - generalized_result['see_mean']) / specialized_see * 100

                result_entry = {
                    'seed': seed,
                    'test_config': test_config,
                    'generalized_see': generalized_result['see_mean'],
                    'generalized_std': generalized_result['see_std'],
                    'specialized_see': specialized_see,
                    'performance_gap_percent': performance_gap
                }

                seed_results.append(result_entry)

                logger.info(f"Config {test_config['name']}: "
                            f"Generalized={generalized_result['see_mean']:.4f}, "
                            f"Specialized={specialized_see:.4f}, "
                            f"Gap={performance_gap:.2f}%")

            all_results.extend(seed_results)

        # 保存结果
        self.save_results(all_results)

        # 生成图表
        self.generate_plots(all_results)

        # 生成报告
        self.generate_report(all_results)

        logger.info(f"\nExperiment completed! Results saved to: {self.result_dir}")

        return all_results

    def save_results(self, results: List[Dict]):
        """保存结果"""
        # 转换为DataFrame
        df = pd.DataFrame(results)
        df.to_csv(self.result_dir / 'generalization_results.csv', index=False)

        # 保存JSON
        with open(self.result_dir / 'generalization_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        logger.info("Results saved to CSV and JSON")

    def generate_plots(self, results: List[Dict]):
        """生成图表"""
        logger.info("Generating plots...")

        # 设置IEEE风格
        plt.style.use('default')
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 11,
            'axes.linewidth': 1.2,
            'lines.linewidth': 2.0,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })

        # 转换为DataFrame
        df = pd.DataFrame(results)

        # 按配置分组
        grouped = df.groupby('test_config').apply(lambda x: pd.Series({
            'K': x.iloc[0]['test_config']['K'],
            'E': x.iloc[0]['test_config']['E'],
            'N': x.iloc[0]['test_config']['N'],
            'name': x.iloc[0]['test_config']['name'],
            'generalized_mean': x['generalized_see'].mean(),
            'generalized_std': x['generalized_see'].std(),
            'specialized_mean': x['specialized_see'].mean(),
            'gap_mean': x['performance_gap_percent'].mean(),
            'gap_std': x['performance_gap_percent'].std()
        })).reset_index(drop=True)

        # 图1：泛化性能 vs 专用模型
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(grouped))
        width = 0.35

        bars1 = ax.bar(x - width / 2, grouped['generalized_mean'], width,
                       label='Generalized Model', color='steelblue',
                       yerr=grouped['generalized_std'], capsize=5)
        bars2 = ax.bar(x + width / 2, grouped['specialized_mean'], width,
                       label='Specialized Model', color='coral',
                       yerr=0, capsize=5)

        ax.set_xlabel('Network Configuration', fontsize=12)
        ax.set_ylabel('SEE (bits/Joule)', fontsize=12)
        ax.set_title('Generalization Performance Evaluation', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{row['name']}\nK={row['K']}, E={row['E']}, N={row['N']}"
                            for _, row in grouped.iterrows()], rotation=0, fontsize=9)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.result_dir / 'generalization_performance.pdf')
        plt.close()

        # 图2：性能差距分析
        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.bar(x, grouped['gap_mean'], color='indianred',
                      yerr=grouped['gap_std'], capsize=5, alpha=0.8)

        # 添加5%阈值线
        ax.axhline(y=5, color='green', linestyle='--', linewidth=2, label='5% Threshold')

        ax.set_xlabel('Network Configuration', fontsize=12)
        ax.set_ylabel('Performance Gap (%)', fontsize=12)
        ax.set_title('Generalization Gap Analysis', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([row['name'] for _, row in grouped.iterrows()],
                           rotation=45, ha='right', fontsize=9)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for bar, gap in zip(bars, grouped['gap_mean']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.3,
                    f'{gap:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.result_dir / 'generalization_gap.pdf')
        plt.close()

        logger.info("Plots generated successfully")

    def generate_report(self, results: List[Dict]):
        """生成实验报告"""
        df = pd.DataFrame(results)

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("EXPERIMENT 7: GENERALIZATION PERFORMANCE REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Training Configuration: K={self.config.train_K}, "
                            f"E={self.config.train_E}, N={self.config.train_N}")
        report_lines.append(f"Number of Test Configurations: {len(self.config.test_configs)}")
        report_lines.append(f"Number of Random Seeds: {len(self.config.random_seeds)}")
        report_lines.append("")

        # 按配置统计
        grouped = df.groupby(['test_config']).agg({
            'generalized_see': ['mean', 'std'],
            'specialized_see': ['mean'],
            'performance_gap_percent': ['mean', 'std']
        }).reset_index()

        report_lines.append("GENERALIZATION RESULTS:")
        report_lines.append("-" * 80)
        report_lines.append(f"{'Config':<20} {'Generalized SEE':<18} {'Specialized SEE':<18} {'Gap (%)':<15}")
        report_lines.append("-" * 80)

        for _, row in grouped.iterrows():
            config_name = row['test_config']['name']
            gen_see = row['generalized_see']['mean']
            gen_std = row['generalized_see']['std']
            spec_see = row['specialized_see']['mean']
            gap_mean = row['performance_gap_percent']['mean']
            gap_std = row['performance_gap_percent']['std']

            report_lines.append(f"{config_name:<20} {gen_see:6.4f}±{gen_std:5.4f}      "
                                f"{spec_see:6.4f}            {gap_mean:5.2f}±{gap_std:4.2f}")

        report_lines.append("")
        report_lines.append("KEY FINDINGS:")
        report_lines.append("-" * 80)

        # 计算关键指标
        max_gap = grouped['performance_gap_percent']['mean'].max()
        avg_gap = grouped['performance_gap_percent']['mean'].mean()
        configs_under_5_percent = (grouped['performance_gap_percent']['mean'] < 5).sum()

        report_lines.append(f"1. Maximum Performance Gap: {max_gap:.2f}%")
        report_lines.append(f"2. Average Performance Gap: {avg_gap:.2f}%")
        report_lines.append(f"3. Configurations with <5% gap: {configs_under_5_percent}/{len(grouped)}")

        if avg_gap < 5:
            report_lines.append("\n✓ CONCLUSION: Excellent generalization capability (avg gap < 5%)")
        elif avg_gap < 10:
            report_lines.append("\n✓ CONCLUSION: Good generalization capability (avg gap < 10%)")
        else:
            report_lines.append("\n⚠ CONCLUSION: Moderate generalization capability (avg gap ≥ 10%)")

        # 保存报告
        report_content = "\n".join(report_lines)
        with open(self.result_dir / 'generalization_report.txt', 'w') as f:
            f.write(report_content)

        # 打印到日志
        for line in report_lines:
            logger.info(line)


def main():
    """主函数"""
    print("=" * 80)
    print("实验七：泛化性能测试")
    print("Experiment 7: Generalization Performance Evaluation")
    print("=" * 80)

    # 创建配置
    config = GeneralizationExperimentConfig()

    print(f"\n训练配置:")
    print(f"  用户数: {config.train_K}")
    print(f"  窃听者数: {config.train_E}")
    print(f"  RIS元素数: {config.train_N}")

    print(f"\n测试配置数量: {len(config.test_configs)}")
    for test_config in config.test_configs:
        print(f"  - {test_config['name']}: K={test_config['K']}, "
              f"E={test_config['E']}, N={test_config['N']}")

    print(f"\n随机种子数: {len(config.random_seeds)}")
    print(f"训练轮数: {config.train_epochs}")
    print(f"评估回合数: {config.eval_episodes}")

    # 确认运行
    response = input("\n是否开始实验？(y/N): ")
    if response.lower() != 'y':
        print("实验取消")
        return

    # 运行实验
    evaluator = GeneralizationEvaluator(config)
    results = evaluator.run_full_experiment()

    print("\n实验完成！")
    print(f"结果保存到: {evaluator.result_dir}")


if __name__ == "__main__":
    main()
