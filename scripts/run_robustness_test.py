#!/usr/bin/env python
"""
鲁棒性测试Runner v3.0

功能:
1. Walk-Forward测试 (WF-1 ~ WF-6)
2. 参数敏感性测试
3. 成本敏感性测试
4. 输出汇总报告

使用方法:
    # 运行全部测试
    python scripts/run_robustness_test.py

    # 快速模式 (小参数网格)
    python scripts/run_robustness_test.py --fast

    # 只跑Walk-Forward
    python scripts/run_robustness_test.py --walk-forward-only

    # 指定输出目录
    python scripts/run_robustness_test.py --output results/robustness/
"""

import os
import sys
import json
import yaml
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
from itertools import product
import logging
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RobustnessTestRunner:
    """鲁棒性测试Runner"""

    # Walk-Forward配置
    WALK_FORWARD_CONFIGS = [
        {'id': 'WF-1', 'train_start': '2015-01-01', 'train_end': '2018-12-31', 'test': '2019'},
        {'id': 'WF-2', 'train_start': '2016-01-01', 'train_end': '2019-12-31', 'test': '2020'},
        {'id': 'WF-3', 'train_start': '2017-01-01', 'train_end': '2020-12-31', 'test': '2021'},
        {'id': 'WF-4', 'train_start': '2018-01-01', 'train_end': '2021-12-31', 'test': '2022'},
        {'id': 'WF-5', 'train_start': '2019-01-01', 'train_end': '2022-12-31', 'test': '2023'},
        {'id': 'WF-6', 'train_start': '2020-01-01', 'train_end': '2023-12-31', 'test': '2024'},
    ]

    # 参数敏感性网格 (完整版)
    PARAM_GRID_FULL = {
        'top_boost': [1.10, 1.20, 1.30, 1.40, 1.50],
        'bottom_penalty': [0.50, 0.60, 0.70, 0.80],
        'max_weight_bull': [0.08, 0.10, 0.12, 0.15],
        'max_weight_bear': [0.04, 0.06, 0.08],
        'roe_threshold': [4.0, 6.0, 8.0, 10.0],
        'drawdown_threshold': [20, 25, 30],
        'tilt_strength': [0.8, 1.0, 1.2, 1.5],
    }

    # 参数敏感性网格 (快速版)
    PARAM_GRID_FAST = {
        'top_boost': [1.20, 1.30, 1.40],
        'bottom_penalty': [0.60, 0.70, 0.80],
        'max_weight_bull': [0.10, 0.12],
        'max_weight_bear': [0.05, 0.06],
        'roe_threshold': [5.0, 6.0, 8.0],
        'drawdown_threshold': [22, 25, 28],
        'tilt_strength': [0.9, 1.0, 1.1],
    }

    # 成本敏感性
    SLIPPAGE_VALUES = [0.0005, 0.001, 0.002, 0.003]

    # KPI门槛
    PASS_CRITERIA = {
        'annual_return': {'min': 0.0, 'target': 0.15},
        'max_drawdown': {'max': -0.25, 'target': -0.15},
        'sharpe': {'min': 0.5, 'target': 1.0},
        'monthly_turnover': {'max': 0.10, 'target': 0.06},
    }

    # Gate-4: KPI阈值定义
    KPI_THRESHOLDS = {
        'annual_return': {'pass_min': 0.05, 'fail_max': -0.05, 'weight': 1.0},
        'max_drawdown': {'pass_min': -0.25, 'fail_max': -0.35, 'weight': 0.8},
        'sharpe': {'pass_min': 0.5, 'fail_max': 0.0, 'weight': 1.0},
        'monthly_turnover': {'pass_max': 0.10, 'fail_min': 0.15, 'weight': 0.5},
    }

    # Gate-6: 名单固化阈值定义
    LIST_FIXATION_THRESHOLDS = {
        'holdings_jaccard_12m_avg': {'max': 0.75, 'message': '12期平均Jaccard相似度过高'},
        'top_holdings_stickiness': {'max': 0.80, 'message': 'Top10持仓保持率过高'},
        'turnover_scale_avg_12m': {'min': 0.5, 'message': '换手缩放因子过低'},
    }

    def __init__(self,
                 output_dir: str = 'results/robustness',
                 fast_mode: bool = False,
                 strategy_type: str = 'aggressive'):
        """
        初始化Runner

        Args:
            output_dir: 输出目录
            fast_mode: 快速模式(小参数网格)
            strategy_type: 策略类型
        """
        self.output_dir = output_dir
        self.fast_mode = fast_mode
        self.strategy_type = strategy_type
        self.param_grid = self.PARAM_GRID_FAST if fast_mode else self.PARAM_GRID_FULL

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 结果存储
        self.results = []
        self.failed_tests = []

    def run_walk_forward_test(self,
                               backtest_func,
                               portfolio: pd.DataFrame,
                               price_data: pd.DataFrame) -> List[Dict]:
        """
        运行Walk-Forward测试

        Args:
            backtest_func: 回测函数
            portfolio: 持仓DataFrame
            price_data: 价格数据

        Returns:
            测试结果列表
        """
        logger.info("=" * 60)
        logger.info("开始 Walk-Forward 测试")
        logger.info("=" * 60)

        results = []

        for wf_config in self.WALK_FORWARD_CONFIGS:
            logger.info(f"\n运行 {wf_config['id']}: 训练 {wf_config['train_start']}~{wf_config['train_end']}, "
                       f"测试 {wf_config['test']}")

            # 获取测试年份数据
            test_year = wf_config['test']
            test_start = f"{test_year}-01-01"
            test_end = f"{test_year}-12-31"

            try:
                # 运行回测
                result = backtest_func(
                    portfolio=portfolio,
                    start_date=test_start,
                    end_date=test_end,
                    strategy_type=self.strategy_type
                )

                # 评估结果
                metrics = result.get('enhanced', {})
                passed, fail_reasons = self._evaluate_pass_criteria(metrics)

                # Gate-4: 添加PASS/FAIL字段
                results.append({
                    'test_id': wf_config['id'],
                    'test_type': 'walk_forward',
                    'test_year': test_year,
                    **metrics,
                    'passed': passed,
                    'fail_reason_top3': self._get_fail_reason_top3(metrics),  # Gate-4
                    'regime_bucket': self._determine_regime_bucket(metrics)   # Gate-4
                })

                status = "✓ PASS" if passed else "✗ FAIL"
                logger.info(f"  {wf_config['id']}: 年化{metrics.get('annual_return', 0)*100:.2f}%, "
                           f"回撤{metrics.get('max_drawdown', 0)*100:.2f}%, "
                           f"夏普{metrics.get('sharpe', 0):.2f} - {status}")

            except Exception as e:
                logger.error(f"  {wf_config['id']} 失败: {e}")
                self.failed_tests.append({
                    'test_id': wf_config['id'],
                    'test_type': 'walk_forward',
                    'error': str(e)
                })

        return results

    def run_parameter_sensitivity_test(self,
                                        backtest_func,
                                        portfolio: pd.DataFrame,
                                        start_date: str = '2020-01-01',
                                        end_date: str = '2024-12-31',
                                        max_tests: int = 50) -> List[Dict]:
        """
        运行参数敏感性测试

        Args:
            backtest_func: 回测函数 (需支持 **kwargs 覆盖参数)
            portfolio: 持仓DataFrame
            start_date: 开始日期
            end_date: 结束日期
            max_tests: 最大测试数量

        Returns:
            测试结果列表
        """
        logger.info("=" * 60)
        logger.info("开始参数敏感性测试")
        logger.info("=" * 60)

        results = []

        # 生成参数组合
        param_names = list(self.param_grid.keys())
        param_values = [self.param_grid[k] for k in param_names]
        combinations = list(product(*param_values))

        # 限制测试数量
        if len(combinations) > max_tests:
            np.random.seed(42)
            indices = np.random.choice(len(combinations), max_tests, replace=False)
            combinations = [combinations[i] for i in indices]

        logger.info(f"参数组合数: {len(combinations)}")

        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))

            try:
                # 运行回测 (使用覆盖参数)
                result = backtest_func(
                    portfolio=portfolio,
                    start_date=start_date,
                    end_date=end_date,
                    strategy_type=self.strategy_type,
                    **params
                )

                metrics = result.get('enhanced', {})
                passed, fail_reasons = self._evaluate_pass_criteria(metrics)

                # Gate-4: 添加PASS/FAIL字段
                results.append({
                    'test_id': f'PARAM-{i+1}',
                    'test_type': 'parameter_sensitivity',
                    **params,
                    **metrics,
                    'passed': passed,
                    'fail_reason_top3': self._get_fail_reason_top3(metrics),  # Gate-4
                    'regime_bucket': self._determine_regime_bucket(metrics)   # Gate-4
                })

                if (i + 1) % 10 == 0:
                    logger.info(f"  完成 {i+1}/{len(combinations)}")

            except Exception as e:
                self.failed_tests.append({
                    'test_id': f'PARAM-{i+1}',
                    'test_type': 'parameter_sensitivity',
                    'params': params,
                    'error': str(e)
                })

        return results

    def run_cost_sensitivity_test(self,
                                   backtest_func,
                                   portfolio: pd.DataFrame,
                                   start_date: str = '2020-01-01',
                                   end_date: str = '2024-12-31') -> List[Dict]:
        """
        运行成本敏感性测试

        Args:
            backtest_func: 回测函数
            portfolio: 持仓DataFrame
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            测试结果列表
        """
        logger.info("=" * 60)
        logger.info("开始成本敏感性测试")
        logger.info("=" * 60)

        results = []

        for slippage in self.SLIPPAGE_VALUES:
            logger.info(f"\n测试滑点: {slippage*100:.2f}%")

            try:
                result = backtest_func(
                    portfolio=portfolio,
                    start_date=start_date,
                    end_date=end_date,
                    strategy_type=self.strategy_type,
                    slippage=slippage
                )

                metrics = result.get('enhanced', {})
                passed, fail_reasons = self._evaluate_pass_criteria(metrics)

                # Gate-4: 添加PASS/FAIL字段
                results.append({
                    'test_id': f'COST-{slippage*100:.1f}bp',
                    'test_type': 'cost_sensitivity',
                    'slippage': slippage,
                    **metrics,
                    'passed': passed,
                    'fail_reason_top3': self._get_fail_reason_top3(metrics),  # Gate-4
                    'regime_bucket': self._determine_regime_bucket(metrics)   # Gate-4
                })

                logger.info(f"  滑点{slippage*100:.2f}%: 年化{metrics.get('annual_return', 0)*100:.2f}%, "
                           f"夏普{metrics.get('sharpe', 0):.2f}")

            except Exception as e:
                self.failed_tests.append({
                    'test_id': f'COST-{slippage*100:.1f}bp',
                    'test_type': 'cost_sensitivity',
                    'slippage': slippage,
                    'error': str(e)
                })

        return results

    def _evaluate_pass_criteria(self, metrics: Dict) -> Tuple[bool, List[str]]:
        """
        Gate-4 & Gate-6: 评估是否通过门槛，返回失败原因

        Args:
            metrics: 指标字典

        Returns:
            (是否通过, 失败原因列表)
        """
        fail_reasons = []

        # Gate-4: 基础KPI检查
        for key, criteria in self.PASS_CRITERIA.items():
            value = metrics.get(key, 0)

            if 'min' in criteria and value < criteria['min']:
                fail_reasons.append(f"{key}={value:.2%} < 门槛{criteria['min']:.2%}")
            if 'max' in criteria and value > criteria['max']:
                fail_reasons.append(f"{key}={value:.2%} > 上限{criteria['max']:.2%}")

        # Gate-6: 名单固化检查
        list_fixation_failed = False
        for key, criteria in self.LIST_FIXATION_THRESHOLDS.items():
            value = metrics.get(key)
            if value is None:
                continue

            if 'max' in criteria and value > criteria['max']:
                fail_reasons.append(f"list_fixation:{key}={value:.2%} > 上限{criteria['max']:.2%}")
                list_fixation_failed = True
            if 'min' in criteria and value < criteria['min']:
                fail_reasons.append(f"list_fixation:{key}={value:.2%} < 下限{criteria['min']:.2%}")
                list_fixation_failed = True

        passed = len(fail_reasons) == 0
        return passed, fail_reasons

    def _check_list_fixation(self, metrics: Dict) -> Tuple[bool, List[str]]:
        """
        Gate-6: 专门的名单固化检查

        Args:
            metrics: 指标字典

        Returns:
            (是否通过, 失败原因列表)
        """
        fail_reasons = []

        for key, criteria in self.LIST_FIXATION_THRESHOLDS.items():
            value = metrics.get(key)
            if value is None:
                continue

            if 'max' in criteria and value > criteria['max']:
                fail_reasons.append(f"{criteria['message']}: {key}={value:.2%}")
            if 'min' in criteria and value < criteria['min']:
                fail_reasons.append(f"{criteria['message']}: {key}={value:.2%}")

        passed = len(fail_reasons) == 0
        return passed, fail_reasons

    def _get_fail_reason_top3(self, metrics: Dict) -> List[str]:
        """
        Gate-4: 获取Top3失败原因

        Args:
            metrics: 指标字典

        Returns:
            Top3失败原因列表
        """
        _, fail_reasons = self._evaluate_pass_criteria(metrics)
        return fail_reasons[:3]

    def _determine_regime_bucket(self, metrics: Dict) -> str:
        """
        Gate-4: 确定市场环境桶

        Args:
            metrics: 指标字典

        Returns:
            市场环境类型
        """
        annual_return = metrics.get('annual_return', 0)
        max_drawdown = metrics.get('max_drawdown', 0)
        sharpe = metrics.get('sharpe', 0)

        # 简单分类逻辑
        if annual_return > 0.15 and sharpe > 1.0:
            return 'bull_market'
        elif annual_return < 0 or max_drawdown < -0.20:
            return 'bear_market'
        else:
            return 'sideways_market'

    def generate_report(self, all_results: List[Dict]) -> Tuple[str, str]:
        """
        生成报告

        Args:
            all_results: 所有测试结果

        Returns:
            (CSV路径, Markdown路径)
        """
        # 生成CSV
        df = pd.DataFrame(all_results)
        csv_path = os.path.join(self.output_dir, 'robustness_summary.csv')
        df.to_csv(csv_path, index=False)

        # 生成Markdown报告
        md_content = self._generate_markdown_report(df)
        md_path = os.path.join(self.output_dir, 'robustness_report.md')
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        return csv_path, md_path

    def _generate_markdown_report(self, df: pd.DataFrame) -> str:
        """生成Markdown报告内容"""
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 统计
        total_tests = len(df)
        passed_tests = df['passed'].sum() if 'passed' in df.columns else 0
        pass_rate = passed_tests / total_tests * 100 if total_tests > 0 else 0

        # 按测试类型分组
        wf_results = df[df['test_type'] == 'walk_forward'] if 'test_type' in df.columns else pd.DataFrame()
        param_results = df[df['test_type'] == 'parameter_sensitivity'] if 'test_type' in df.columns else pd.DataFrame()
        cost_results = df[df['test_type'] == 'cost_sensitivity'] if 'test_type' in df.columns else pd.DataFrame()

        report = f"""# 鲁棒性测试报告

生成时间: {now}

## 1. 测试概览

| 指标 | 值 |
|------|-----|
| 总测试数 | {total_tests} |
| 通过数 | {passed_tests} |
| 通过率 | {pass_rate:.1f}% |
| 测试模式 | {'快速' if self.fast_mode else '完整'} |

## 2. Walk-Forward 测试结果

"""
        if not wf_results.empty:
            report += "| 测试ID | 测试年份 | 年化收益 | 最大回撤 | 夏普 | 通过 |\n"
            report += "|--------|----------|----------|----------|------|------|\n"
            for _, row in wf_results.iterrows():
                report += f"| {row.get('test_id', '')} | {row.get('test_year', '')} | " \
                         f"{row.get('annual_return', 0)*100:.2f}% | " \
                         f"{row.get('max_drawdown', 0)*100:.2f}% | " \
                         f"{row.get('sharpe', 0):.2f} | " \
                         f"{'✓' if row.get('passed', False) else '✗'} |\n"
        else:
            report += "*无Walk-Forward测试结果*\n"

        report += "\n## 3. 参数敏感性测试结果\n\n"
        if not param_results.empty:
            report += f"测试组合数: {len(param_results)}\n\n"
            report += f"- 年化收益范围: {param_results['annual_return'].min()*100:.2f}% ~ {param_results['annual_return'].max()*100:.2f}%\n"
            report += f"- 最大回撤范围: {param_results['max_drawdown'].min()*100:.2f}% ~ {param_results['max_drawdown'].max()*100:.2f}%\n"
            report += f"- 夏普比率范围: {param_results['sharpe'].min():.2f} ~ {param_results['sharpe'].max():.2f}\n"

            # 通过率统计
            param_pass_rate = param_results['passed'].sum() / len(param_results) * 100
            report += f"- 通过率: {param_pass_rate:.1f}%\n"
        else:
            report += "*无参数敏感性测试结果*\n"

        report += "\n## 4. 成本敏感性测试结果\n\n"
        if not cost_results.empty:
            report += "| 滑点 | 年化收益 | 夏普 | 通过 |\n"
            report += "|------|----------|------|------|\n"
            for _, row in cost_results.iterrows():
                report += f"| {row.get('slippage', 0)*100:.2f}% | " \
                         f"{row.get('annual_return', 0)*100:.2f}% | " \
                         f"{row.get('sharpe', 0):.2f} | " \
                         f"{'✓' if row.get('passed', False) else '✗'} |\n"
        else:
            report += "*无成本敏感性测试结果*\n"

        # 失败测试
        if self.failed_tests:
            report += "\n## 5. 失败测试\n\n"
            for fail in self.failed_tests[:10]:  # 最多显示10条
                report += f"- {fail.get('test_id', '')}: {fail.get('error', 'Unknown error')}\n"

        report += "\n## 6. 结论\n\n"
        if pass_rate >= 80:
            report += "✅ 策略鲁棒性良好，大部分测试通过\n"
        elif pass_rate >= 60:
            report += "⚠️ 策略鲁棒性一般，部分场景表现不佳\n"
        else:
            report += "❌ 策略鲁棒性不足，需进一步优化\n"

        # Gate-6: 名单固化诊断
        report += self._generate_list_fixation_section(df)

        return report

    def _generate_list_fixation_section(self, df: pd.DataFrame) -> str:
        """生成名单固化诊断章节"""
        section = "\n## 7. Gate-6 名单固化诊断\n\n"

        if 'holdings_jaccard_12m_avg' not in df.columns:
            section += "*缺少名单固化指标*\n"
            return section

        # 统计固化情况
        jaccard_avg = df['holdings_jaccard_12m_avg'].mean()
        stickiness_avg = df['top_holdings_stickiness'].mean() if 'top_holdings_stickiness' in df.columns else 0

        # 判断严重程度
        if jaccard_avg > 0.85 or stickiness_avg > 0.90:
            severity = "🔴 硬失败 (Hard Fail)"
        elif jaccard_avg > 0.75 or stickiness_avg > 0.80:
            severity = "🟡 软失败 (Soft Fail) - 建议启用反固化"
        elif jaccard_avg > 0.70 or stickiness_avg > 0.75:
            severity = "🟠 警告 (Warning) - 需关注"
        else:
            severity = "🟢 正常 (Pass)"

        section += f"### 整体评估\n\n"
        section += f"| 指标 | 平均值 | 阈值 | 状态 |\n"
        section += f"|------|--------|------|------|\n"
        section += f"| Jaccard 12期均值 | {jaccard_avg:.2%} | ≤75% | {'✓' if jaccard_avg <= 0.75 else '✗'} |\n"
        section += f"| Top10粘性 | {stickiness_avg:.2%} | ≤80% | {'✓' if stickiness_avg <= 0.80 else '✗'} |\n"
        section += f"\n**严重程度**: {severity}\n"

        # 统计 list_fixation 失败
        if 'fail_reason_top3' in df.columns:
            list_fixation_fails = df[
                df['fail_reason_top3'].astype(str).str.contains('list_fixation', na=False)
            ]
            if len(list_fixation_fails) > 0:
                section += f"\n**名单固化失败数**: {len(list_fixation_fails)} / {len(df)}\n"

        # 改进建议
        if jaccard_avg > 0.75:
            section += "\n### 改进建议\n\n"
            section += "```yaml\n"
            section += "anti_fixation:\n"
            section += "  enabled: true\n"
            section += "  soft_diversify:\n"
            section += "    enabled: true\n"
            section += "```\n"

        return section

    def run_diagnosis(self, output_path: str = None) -> Dict:
        """
        运行诊断分析

        Args:
            output_path: 诊断报告输出路径

        Returns:
            诊断结果
        """
        from src.optimization.anti_fixation import diagnose_list_fixation

        csv_path = os.path.join(self.output_dir, 'robustness_summary.csv')
        if output_path is None:
            output_path = os.path.join(self.output_dir, 'list_fixation_diagnosis.md')

        return diagnose_list_fixation(csv_path, output_path)

    def run_ablation(self,
                     backtest_func,
                     portfolio: pd.DataFrame,
                     start_date: str,
                     end_date: str) -> Dict:
        """
        运行消融实验

        Args:
            backtest_func: 回测函数
            portfolio: 持仓数据
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            对比结果
        """
        from src.optimization.anti_fixation import run_ablation_study

        return run_ablation_study(
            backtest_func,
            portfolio,
            start_date,
            end_date,
            output_dir=self.output_dir
        )


def main():
    parser = argparse.ArgumentParser(description='鲁棒性测试Runner')
    parser.add_argument('--fast', action='store_true', help='快速模式')
    parser.add_argument('--walk-forward-only', action='store_true', help='只跑Walk-Forward')
    parser.add_argument('--output', default='results/robustness', help='输出目录')
    parser.add_argument('--strategy', default='aggressive', choices=['stable', 'aggressive'], help='策略类型')
    parser.add_argument('--diagnosis', action='store_true', help='运行诊断分析')
    parser.add_argument('--ablation', action='store_true', help='运行消融实验')
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("鲁棒性测试Runner v3.0")
    logger.info(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"模式: {'快速' if args.fast else '完整'}")
    logger.info("=" * 70)

    # 创建Runner
    runner = RobustnessTestRunner(
        output_dir=args.output,
        fast_mode=args.fast,
        strategy_type=args.strategy
    )

    # 这里需要实际的回测函数
    # 由于我们是在独立脚本中，这里用占位函数
    def mock_backtest_func(portfolio, start_date, end_date, strategy_type, **kwargs):
        """模拟回测函数 (实际使用时替换为真实回测)"""
        np.random.seed(42)
        days = 252 * 2
        returns = np.random.normal(0.001, 0.02, days)
        cum = np.cumprod(1 + returns)
        total = cum[-1] - 1
        annual = (1 + total) ** (1/2) - 1

        return {
            'enhanced': {
                'annual_return': annual + np.random.uniform(-0.02, 0.02),
                'max_drawdown': -0.15 + np.random.uniform(-0.05, 0.05),
                'sharpe': 1.0 + np.random.uniform(-0.2, 0.2),
            }
        }

    all_results = []

    # 1. Walk-Forward测试
    wf_results = runner.run_walk_forward_test(
        backtest_func=mock_backtest_func,
        portfolio=pd.DataFrame(),  # 实际使用时传入真实portfolio
        price_data=pd.DataFrame()
    )
    all_results.extend(wf_results)

    # 2. 参数敏感性测试
    if not args.walk_forward_only:
        param_results = runner.run_parameter_sensitivity_test(
            backtest_func=mock_backtest_func,
            portfolio=pd.DataFrame(),
            max_tests=20 if args.fast else 50
        )
        all_results.extend(param_results)

        # 3. 成本敏感性测试
        cost_results = runner.run_cost_sensitivity_test(
            backtest_func=mock_backtest_func,
            portfolio=pd.DataFrame()
        )
        all_results.extend(cost_results)

    # 生成报告
    csv_path, md_path = runner.generate_report(all_results)

    # Gate-6: 诊断分析
    if args.diagnosis or args.ablation:
        logger.info("\n" + "=" * 70)
        logger.info("Gate-6 诊断分析")
        logger.info("=" * 70)

        if args.diagnosis:
            diagnosis = runner.run_diagnosis()
            logger.info(f"诊断报告: {os.path.join(args.output, 'list_fixation_diagnosis.md')}")

        if args.ablation:
            logger.info("运行消融实验...")
            # ablation_results = runner.run_ablation(mock_backtest_func, pd.DataFrame(), '2020-01-01', '2024-12-31')
            logger.info("消融实验需要真实回测函数，请集成后运行")

    logger.info("\n" + "=" * 70)
    logger.info("测试完成!")
    logger.info(f"CSV报告: {csv_path}")
    logger.info(f"Markdown报告: {md_path}")
    if args.diagnosis:
        logger.info(f"诊断报告: {os.path.join(args.output, 'list_fixation_diagnosis.md')}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
