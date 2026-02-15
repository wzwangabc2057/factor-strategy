"""
回测指标计算模块 v3.0

计算关键指标:
- 收益指标: 年化收益、总收益、超额收益
- 风险指标: 最大回撤、波动率、VaR
- 风险调整收益: 夏普比率、卡尔玛比率、索提诺比率
- 换手指标: 月换手、日换手、总换手
- 成本指标: 实现成本、滑点估计、冲击成本
- 集中度指标: Top10权重、有效持仓数
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """指标计算器"""

    def __init__(self):
        self.metrics = {}

    def calculate_returns_metrics(self,
                                   returns: np.ndarray,
                                   benchmark_returns: np.ndarray = None) -> Dict:
        """
        计算收益指标

        Args:
            returns: 日收益率数组
            benchmark_returns: 基准日收益率数组

        Returns:
            收益指标字典
        """
        n_days = len(returns)
        n_years = n_days / 252

        # 累计净值
        cum_values = np.cumprod(1 + returns)
        total_return = cum_values[-1] - 1

        # 年化收益
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        # 超额收益
        alpha = 0
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            benchmark_cum = np.cumprod(1 + benchmark_returns)
            benchmark_total = benchmark_cum[-1] - 1
            benchmark_annual = (1 + benchmark_total) ** (1 / n_years) - 1 if n_years > 0 else 0
            alpha = annual_return - benchmark_annual

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'alpha': alpha,
            'final_nav': cum_values[-1],
            'n_days': n_days,
            'n_years': n_years
        }

    def calculate_risk_metrics(self,
                                returns: np.ndarray,
                                cum_values: np.ndarray = None) -> Dict:
        """
        计算风险指标

        Args:
            returns: 日收益率数组
            cum_values: 累计净值数组

        Returns:
            风险指标字典
        """
        if cum_values is None:
            cum_values = np.cumprod(1 + returns)

        # 最大回撤
        peak = np.maximum.accumulate(cum_values)
        drawdown = (cum_values - peak) / peak
        max_drawdown = drawdown.min()

        # 波动率
        volatility = np.std(returns) * np.sqrt(252)

        # VaR (95%)
        var_95 = np.percentile(returns, 5)

        # CVaR (Expected Shortfall)
        cvar_95 = np.mean(returns[returns <= var_95]) if np.any(returns <= var_95) else var_95

        # 下行波动率 (负收益)
        negative_returns = returns[returns < 0]
        downside_vol = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 0

        return {
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'downside_volatility': downside_vol
        }

    def calculate_risk_adjusted_metrics(self,
                                         returns: np.ndarray,
                                         benchmark_returns: np.ndarray = None,
                                         max_drawdown: float = None) -> Dict:
        """
        计算风险调整收益指标

        Args:
            returns: 日收益率数组
            benchmark_returns: 基准日收益率数组
            max_drawdown: 最大回撤

        Returns:
            风险调整收益指标字典
        """
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        # 夏普比率
        sharpe = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0

        # 卡尔玛比率
        annual_return = mean_return * 252
        calmar = annual_return / abs(max_drawdown) if max_drawdown and max_drawdown != 0 else 0

        # 索提诺比率
        negative_returns = returns[returns < 0]
        downside_std = np.std(negative_returns) if len(negative_returns) > 0 else 0.001
        sortino = mean_return / downside_std * np.sqrt(252)

        # 信息比率
        information_ratio = 0
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            active_returns = returns - benchmark_returns
            tracking_error = np.std(active_returns)
            information_ratio = np.mean(active_returns) / tracking_error * np.sqrt(252) if tracking_error > 0 else 0

        return {
            'sharpe': sharpe,
            'calmar': calmar,
            'sortino': sortino,
            'information_ratio': information_ratio
        }

    def calculate_turnover_metrics(self,
                                    turnovers: List[float],
                                    trade_counts: List[int] = None) -> Dict:
        """
        计算换手指标

        Args:
            turnovers: 每期换手率列表
            trade_counts: 每期交易次数列表

        Returns:
            换手指标字典
        """
        turnovers = np.array(turnovers)

        return {
            'avg_monthly_turnover': np.mean(turnovers) if len(turnovers) > 0 else 0,
            'max_monthly_turnover': np.max(turnovers) if len(turnovers) > 0 else 0,
            'min_monthly_turnover': np.min(turnovers) if len(turnovers) > 0 else 0,
            'total_turnover': np.sum(turnovers),
            'turnover_std': np.std(turnovers) if len(turnovers) > 0 else 0,
            'trade_count': sum(trade_counts) if trade_counts else 0
        }

    def calculate_cost_metrics(self,
                                costs: List[Dict],
                                total_trade_value: float = None) -> Dict:
        """
        计算成本指标

        Args:
            costs: 成本列表 [{total_cost, commission, slippage, impact, spread}, ...]
            total_trade_value: 总交易金额

        Returns:
            成本指标字典
        """
        if not costs:
            return {
                'total_cost': 0,
                'commission': 0,
                'slippage': 0,
                'impact_cost': 0,
                'spread': 0,
                'cost_ratio': 0,
                'impact_fallback_count': 0
            }

        total_cost = sum(c.get('total_cost', 0) for c in costs)
        commission = sum(c.get('commission', 0) for c in costs)
        slippage = sum(c.get('slippage', 0) for c in costs)
        impact = sum(c.get('impact', 0) for c in costs)
        spread = sum(c.get('spread', 0) for c in costs)
        fallback_count = sum(1 for c in costs if c.get('impact_fallback', False))

        cost_ratio = total_cost / total_trade_value if total_trade_value else 0

        return {
            'total_cost': total_cost,
            'commission': commission,
            'slippage': slippage,
            'impact_cost': impact,
            'spread': spread,
            'cost_ratio': cost_ratio,
            'impact_fallback_count': fallback_count
        }

    def calculate_concentration_metrics(self,
                                         weights: Dict[str, float],
                                         max_single_weight: float = None) -> Dict:
        """
        计算集中度指标

        Args:
            weights: 权重字典
            max_single_weight: 单股上限

        Returns:
            集中度指标字典
        """
        if not weights:
            return {
                'max_single_weight': 0,
                'top10_weight': 0,
                'top20_weight': 0,
                'effective_n': 0,
                'n_positions': 0
            }

        sorted_weights = sorted(weights.values(), reverse=True)

        # 最大单股权重
        max_weight = sorted_weights[0] if sorted_weights else 0

        # Top10/Top20权重和
        top10_weight = sum(sorted_weights[:10])
        top20_weight = sum(sorted_weights[:20])

        # 有效持仓数 (Herfindahl指数的倒数)
        hhi = sum(w ** 2 for w in sorted_weights)
        effective_n = 1 / hhi if hhi > 0 else 0

        return {
            'max_single_weight': max_weight,
            'top10_weight': top10_weight,
            'top20_weight': top20_weight,
            'effective_n': effective_n,
            'n_positions': len(weights)
        }

    def calculate_all_metrics(self,
                               returns: np.ndarray,
                               benchmark_returns: np.ndarray = None,
                               turnovers: List[float] = None,
                               costs: List[Dict] = None,
                               weights: Dict[str, float] = None,
                               risk_tier_counts: Dict = None,
                               factor_ic: List[float] = None) -> Dict:
        """
        计算所有指标

        Args:
            returns: 日收益率数组
            benchmark_returns: 基准日收益率数组
            turnovers: 换手率列表
            costs: 成本列表
            weights: 最终权重
            risk_tier_counts: 风控档位计数
            factor_ic: 因子IC列表

        Returns:
            完整指标字典
        """
        cum_values = np.cumprod(1 + returns)

        # 收益指标
        returns_metrics = self.calculate_returns_metrics(returns, benchmark_returns)

        # 风险指标
        risk_metrics = self.calculate_risk_metrics(returns, cum_values)

        # 风险调整收益
        risk_adj_metrics = self.calculate_risk_adjusted_metrics(
            returns, benchmark_returns, risk_metrics['max_drawdown']
        )

        # 换手指标
        turnover_metrics = self.calculate_turnover_metrics(turnovers or [])

        # 成本指标
        cost_metrics = self.calculate_cost_metrics(costs or [])

        # 集中度指标
        concentration_metrics = self.calculate_concentration_metrics(weights or {})

        # 因子IC
        ic_metrics = {}
        if factor_ic:
            ic_metrics = {
                'factor_ic_avg': np.mean(factor_ic),
                'factor_ic_std': np.std(factor_ic),
                'factor_ic_ir': np.mean(factor_ic) / np.std(factor_ic) if np.std(factor_ic) > 0 else 0,
                'factor_ic_positive_ratio': sum(1 for ic in factor_ic if ic > 0) / len(factor_ic)
            }

        # 风控指标
        risk_tier_metrics = risk_tier_counts or {}

        # 合并所有指标
        all_metrics = {
            **returns_metrics,
            **risk_metrics,
            **risk_adj_metrics,
            **{f'turnover_{k}': v for k, v in turnover_metrics.items()},
            **{f'cost_{k}': v for k, v in cost_metrics.items()},
            **concentration_metrics,
            **ic_metrics,
            'risk_tier_counts': risk_tier_metrics,
            'calculation_time': datetime.now().isoformat()
        }

        self.metrics = all_metrics
        return all_metrics


def generate_summary_table(metrics: Dict) -> str:
    """
    生成指标汇总表格

    Args:
        metrics: 指标字典

    Returns:
        Markdown格式表格
    """
    table = """
## 核心指标

| 指标 | 值 |
|------|-----|
| 年化收益 | {annual_return:.2%} |
| 最大回撤 | {max_drawdown:.2%} |
| 夏普比率 | {sharpe:.2f} |
| 卡尔玛比率 | {calmar:.2f} |
| 月换手率 | {turnover_avg_monthly_turnover:.2%} |
| 成本占比 | {cost_cost_ratio:.4%} |
| 最大单股权重 | {max_single_weight:.2%} |
| Top10权重和 | {top10_weight:.2%} |
| 有效持仓数 | {effective_n:.1f} |
""".format(**metrics)

    # 添加风控信息
    if metrics.get('risk_tier_counts'):
        tier_counts = metrics['risk_tier_counts']
        table += f"""
## 风控统计

| 档位 | 次数 |
|------|------|
| Tier1 (趋势) | {tier_counts.get('tier1_trend', 0)} |
| Tier2 (震荡) | {tier_counts.get('tier2_volatile', 0)} |
| Tier3 (急跌) | {tier_counts.get('tier3_crash', 0)} |
"""

    return table


if __name__ == '__main__':
    # 测试
    calculator = MetricsCalculator()

    # 模拟数据
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 252)
    benchmark = np.random.normal(0.0008, 0.015, 252)

    metrics = calculator.calculate_all_metrics(
        returns=returns,
        benchmark_returns=benchmark,
        turnovers=[0.05, 0.06, 0.04, 0.07, 0.05],
        costs=[{'total_cost': 100, 'commission': 50, 'slippage': 30, 'impact': 10, 'spread': 10}],
        weights={'A': 0.1, 'B': 0.08, 'C': 0.07, 'D': 0.06, 'E': 0.05}
    )

    print(generate_summary_table(metrics))
