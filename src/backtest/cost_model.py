"""
交易成本模型 v3.0

包含:
- 基础佣金成本
- 滑点成本
- 冲击成本 (基于成交额/ADV)
- 买卖价差

使用方法:
    from src.backtest.cost_model import CostModel

    cost_model = CostModel(config_path='config/cost_model.yaml')
    trade_cost = cost_model.calculate_cost(
        trade_value=100000,
        trade_type='buy',
        market_cap=300,  # 亿
        adv=5000000,     # 平均日成交额
    )
"""

import os
import yaml
import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class CostModel:
    """交易成本模型"""

    def __init__(self, config_path: str = None, config: Dict = None):
        """
        初始化成本模型

        Args:
            config_path: 配置文件路径
            config: 直接传入配置字典
        """
        if config is not None:
            self.config = config
        elif config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)['cost_model']
        else:
            # 使用默认配置
            self.config = self._default_config()

        self.enabled = self.config.get('enabled', True)
        self.base_cost = self.config.get('base_cost', {})
        self.impact_config = self.config.get('impact_cost', {})
        self.spread_config = self.config.get('spread', {})

        # 记录是否使用降级模式
        self.impact_fallback = False

    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            'enabled': True,
            'base_cost': {
                'commission': {'buy': 0.00026, 'sell': 0.00126},
                'slippage': {'default': 0.001}
            },
            'impact_cost': {
                'enabled': True,
                'coefficient': 0.1,
                'fallback_mode': 'zero'
            },
            'spread': {
                'enabled': True,
                'default_spread': 0.002
            }
        }

    def calculate_commission(self, trade_value: float, trade_type: str) -> float:
        """
        计算佣金成本

        Args:
            trade_value: 交易金额
            trade_type: 'buy' 或 'sell'

        Returns:
            佣金成本(金额)
        """
        commission_config = self.base_cost.get('commission', {})
        rate = commission_config.get(trade_type, 0.001 if trade_type == 'sell' else 0.0003)
        return trade_value * rate

    def calculate_slippage(self, trade_value: float, slippage: float = None) -> float:
        """
        计算滑点成本

        Args:
            trade_value: 交易金额
            slippage: 滑点比例，默认使用配置值

        Returns:
            滑点成本(金额)
        """
        if slippage is None:
            slippage = self.base_cost.get('slippage', {}).get('default', 0.001)
        return trade_value * slippage

    def calculate_impact_cost(self,
                               trade_value: float,
                               adv: float = None,
                               market_cap: float = None) -> Tuple[float, bool]:
        """
        计算冲击成本

        公式: impact = k * sqrt(trade_value / ADV)

        Args:
            trade_value: 交易金额
            adv: 平均日成交额 (Average Daily Volume)
            market_cap: 市值(亿)，用于日志

        Returns:
            (冲击成本, 是否使用降级模式)
        """
        if not self.impact_config.get('enabled', True):
            return 0.0, False

        coefficient = self.impact_config.get('coefficient', 0.1)

        if adv is None or adv <= 0:
            # 降级模式
            fallback_mode = self.impact_config.get('fallback_mode', 'zero')
            if fallback_mode == 'fixed_rate':
                fallback_rate = self.impact_config.get('fallback_rate', 0.001)
                return trade_value * fallback_rate, True
            else:
                # zero模式，返回0
                return 0.0, True

        # 正常计算: impact = k * sqrt(trade_value / ADV)
        impact_ratio = coefficient * np.sqrt(trade_value / adv)
        return trade_value * impact_ratio, False

    def calculate_spread(self, trade_value: float, market_cap: float = None) -> float:
        """
        计算买卖价差成本

        Args:
            trade_value: 交易金额
            market_cap: 市值(亿)

        Returns:
            价差成本(金额)
        """
        if not self.spread_config.get('enabled', True):
            return 0.0

        # 根据市值分档
        if market_cap is not None:
            tiers = self.spread_config.get('tiers', [])
            for tier in tiers:
                cap_range = tier.get('market_cap_range', [0, None])
                if cap_range[0] is not None and market_cap < cap_range[0]:
                    continue
                if cap_range[1] is not None and market_cap >= cap_range[1]:
                    continue
                spread = tier.get('spread', 0.002)
                return trade_value * spread

        # 默认价差
        default_spread = self.spread_config.get('default_spread', 0.002)
        return trade_value * default_spread

    def calculate_cost(self,
                       trade_value: float,
                       trade_type: str = 'buy',
                       slippage: float = None,
                       adv: float = None,
                       market_cap: float = None) -> Dict:
        """
        计算完整交易成本

        Args:
            trade_value: 交易金额
            trade_type: 'buy' 或 'sell'
            slippage: 滑点比例
            adv: 平均日成交额
            market_cap: 市值(亿)

        Returns:
            成本明细字典
        """
        if not self.enabled:
            return {
                'total_cost': 0.0,
                'commission': 0.0,
                'slippage': 0.0,
                'impact': 0.0,
                'spread': 0.0,
                'cost_ratio': 0.0,
                'impact_fallback': False
            }

        commission = self.calculate_commission(trade_value, trade_type)
        slippage_cost = self.calculate_slippage(trade_value, slippage)
        impact_cost, impact_fallback = self.calculate_impact_cost(trade_value, adv, market_cap)
        spread_cost = self.calculate_spread(trade_value, market_cap)

        total_cost = commission + slippage_cost + impact_cost + spread_cost
        cost_ratio = total_cost / trade_value if trade_value > 0 else 0

        # 记录降级模式
        if impact_fallback and not self.impact_fallback:
            self.impact_fallback = True
            logger.warning("冲击成本计算使用降级模式(无ADV数据)")

        return {
            'total_cost': total_cost,
            'commission': commission,
            'slippage': slippage_cost,
            'impact': impact_cost,
            'spread': spread_cost,
            'cost_ratio': cost_ratio,
            'impact_fallback': impact_fallback
        }

    def calculate_rebalance_cost(self,
                                  trades: list,
                                  slippage: float = None) -> Dict:
        """
        计算调仓总成本

        Args:
            trades: 交易列表 [{code, trade_value, trade_type, adv, market_cap}, ...]
            slippage: 滑点比例

        Returns:
            调仓成本汇总
        """
        total_cost = 0.0
        total_commission = 0.0
        total_slippage = 0.0
        total_impact = 0.0
        total_spread = 0.0
        total_trade_value = 0.0
        impact_fallback_count = 0

        for trade in trades:
            result = self.calculate_cost(
                trade_value=trade.get('trade_value', 0),
                trade_type=trade.get('trade_type', 'buy'),
                slippage=slippage,
                adv=trade.get('adv'),
                market_cap=trade.get('market_cap')
            )

            total_cost += result['total_cost']
            total_commission += result['commission']
            total_slippage += result['slippage']
            total_impact += result['impact']
            total_spread += result['spread']
            total_trade_value += trade.get('trade_value', 0)
            if result['impact_fallback']:
                impact_fallback_count += 1

        return {
            'total_cost': total_cost,
            'commission': total_commission,
            'slippage': total_slippage,
            'impact': total_impact,
            'spread': total_spread,
            'total_trade_value': total_trade_value,
            'cost_ratio': total_cost / total_trade_value if total_trade_value > 0 else 0,
            'trade_count': len(trades),
            'impact_fallback_count': impact_fallback_count
        }


def run_cost_sensitivity_test(base_cost_func, trade_value: float, config_path: str = None) -> pd.DataFrame:
    """
    运行成本敏感性测试

    Args:
        base_cost_func: 基础成本计算函数
        trade_value: 测试交易金额
        config_path: 配置文件路径

    Returns:
        敏感性测试结果DataFrame
    """
    import pandas as pd

    config_path = config_path or 'config/cost_model.yaml'
    cost_model = CostModel(config_path=config_path)

    results = []
    slippage_values = cost_model.config.get('base_cost', {}).get('slippage', {}).get(
        'sensitivity_test_values', [0.0005, 0.001, 0.002, 0.003]
    )

    for slippage in slippage_values:
        result = cost_model.calculate_cost(
            trade_value=trade_value,
            trade_type='buy',
            slippage=slippage
        )
        results.append({
            'slippage': slippage,
            'total_cost': result['total_cost'],
            'cost_ratio': result['cost_ratio']
        })

    return pd.DataFrame(results)


if __name__ == '__main__':
    # 测试
    model = CostModel('config/cost_model.yaml')

    # 测试买入
    result = model.calculate_cost(
        trade_value=100000,
        trade_type='buy',
        market_cap=300,
        adv=5000000
    )
    print(f"买入成本: {result}")

    # 测试卖出
    result = model.calculate_cost(
        trade_value=100000,
        trade_type='sell',
        market_cap=300,
        adv=5000000
    )
    print(f"卖出成本: {result}")
