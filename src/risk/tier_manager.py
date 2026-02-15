"""
风控三档管理器 v3.0

实现三档分层风控:
- Tier 1: 趋势市 (正常运行)
- Tier 2: 震荡市 (降仓观望)
- Tier 3: 急跌市 (防御模式)

使用方法:
    from src.risk.tier_manager import RiskTierManager

    manager = RiskTierManager(config_path='config/risk_control.yaml')
    tier, actions = manager.detect_tier(market_data)
"""

import os
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class RiskTierManager:
    """风控档位管理器"""

    TIER_NAMES = ['tier1_trend', 'tier2_volatile', 'tier3_crash']

    def __init__(self, config_path: str = None, config: Dict = None):
        """
        初始化风控管理器

        Args:
            config_path: 配置文件路径
            config: 直接传入配置字典
        """
        if config is not None:
            self.config = config.get('risk_control', config)
        elif config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f).get('risk_control', {})
        else:
            self.config = self._default_config()

        self.enabled = self.config.get('enabled', True)
        self.tier_configs = {
            'tier1_trend': self.config.get('tier1_trend', {}),
            'tier2_volatile': self.config.get('tier2_volatile', {}),
            'tier3_crash': self.config.get('tier3_crash', {}),
        }

        # 状态跟踪
        self.current_tier = 'tier1_trend'
        self.tier_history = []
        self.tier_counts = {'tier1_trend': 0, 'tier2_volatile': 0, 'tier3_crash': 0}

    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            'enabled': True,
            'tier1_trend': {
                'detect': {'score_threshold': 1.0},
                'action': {
                    'position_ratio': 1.0,
                    'max_single_weight': 0.12,
                    'allow_buy': True,
                    'cash_ratio_min': 0.02
                }
            },
            'tier2_volatile': {
                'detect': {'score_threshold': 1.0},
                'action': {
                    'position_ratio': 0.85,
                    'max_single_weight': 0.08,
                    'allow_buy': True,
                    'cash_ratio_min': 0.10
                }
            },
            'tier3_crash': {
                'detect': {'score_threshold': 1.5},
                'action': {
                    'position_ratio': 0.5,
                    'max_single_weight': 0.05,
                    'allow_buy': False,
                    'cash_ratio_min': 0.40,
                    'trigger_stop_loss': True
                }
            }
        }

    def calculate_market_indicators(self, price_series: pd.Series) -> Dict:
        """
        计算市场指标

        Args:
            price_series: 价格序列 (最近的收盘价)

        Returns:
            市场指标字典
        """
        if len(price_series) < 60:
            return {
                'ma20': np.nan,
                'ma60': np.nan,
                'adx': np.nan,
                'volatility_20d': np.nan,
                'index_return_20d': 0.0,
                'index_return_60d': 0.0,
                'drawdown_from_peak': 0.0
            }

        # 均线
        ma20 = price_series.rolling(20).mean().iloc[-1]
        ma60 = price_series.rolling(60).mean().iloc[-1]

        # 收益率
        index_return_20d = price_series.iloc[-1] / price_series.iloc[-20] - 1
        index_return_60d = price_series.iloc[-1] / price_series.iloc[-60] - 1

        # 波动率
        returns = price_series.pct_change().dropna().tail(20)
        volatility_20d = returns.std() * np.sqrt(252) if len(returns) > 0 else 0

        # ADX (简化版)
        adx = self._calculate_adx(price_series)

        # 从高点回撤
        peak = price_series.expanding().max().iloc[-1]
        drawdown_from_peak = (price_series.iloc[-1] - peak) / peak if peak > 0 else 0

        return {
            'ma20': ma20,
            'ma60': ma60,
            'adx': adx,
            'volatility_20d': volatility_20d,
            'index_return_20d': index_return_20d,
            'index_return_60d': index_return_60d,
            'drawdown_from_peak': drawdown_from_peak,
            'current_price': price_series.iloc[-1]
        }

    def _calculate_adx(self, price_series: pd.Series, period: int = 14) -> float:
        """
        计算ADX (Average Directional Index)
        简化版本
        """
        if len(price_series) < period * 2:
            return 25.0  # 默认中等值

        # 简化: 用波动率和趋势强度近似
        returns = price_series.pct_change().dropna()
        if len(returns) < period:
            return 25.0

        # 趋势强度 = abs(收益) / 波动
        recent_return = returns.tail(period).sum()
        recent_vol = returns.tail(period).std() * np.sqrt(period)

        if recent_vol > 0:
            trend_strength = abs(recent_return) / recent_vol
            adx = min(50, trend_strength * 30)  # 映射到0-50
        else:
            adx = 25.0

        return adx

    def evaluate_tier_conditions(self, indicators: Dict) -> Dict[str, float]:
        """
        评估各档位条件得分

        Args:
            indicators: 市场指标

        Returns:
            各档位得分
        """
        scores = {}

        # Tier 3: 急跌市
        tier3_score = 0.0
        if indicators['index_return_20d'] < -0.10:
            tier3_score += 2.0
        if not np.isnan(indicators['ma20']) and not np.isnan(indicators['ma60']):
            if indicators['ma20'] < indicators['ma60'] * 0.95:
                tier3_score += 1.0
        if indicators['drawdown_from_peak'] < -0.15:
            tier3_score += 1.5
        scores['tier3_crash'] = tier3_score

        # Tier 2: 震荡市
        tier2_score = 0.0
        if not np.isnan(indicators['ma20']) and not np.isnan(indicators['ma60']):
            if abs(indicators['ma20'] / indicators['ma60'] - 1) < 0.02:
                tier2_score += 1.0
        if indicators['adx'] < 20:
            tier2_score += 1.0
        if indicators['volatility_20d'] > 0.25:
            tier2_score += 0.5
        scores['tier2_volatile'] = tier2_score

        # Tier 1: 趋势市 (默认)
        tier1_score = 0.0
        if not np.isnan(indicators['ma20']) and not np.isnan(indicators['ma60']):
            if indicators['ma20'] > indicators['ma60']:
                tier1_score += 1.0
        if indicators['adx'] > 20:
            tier1_score += 0.5
        if indicators['index_return_60d'] > 0.05:
            tier1_score += 0.5
        scores['tier1_trend'] = tier1_score

        return scores

    def detect_tier(self,
                    market_data: pd.Series = None,
                    indicators: Dict = None) -> Tuple[str, Dict]:
        """
        检测当前风控档位

        Args:
            market_data: 价格序列
            indicators: 预计算的指标(可选)

        Returns:
            (档位名称, 动作配置)
        """
        if not self.enabled:
            return 'tier1_trend', self.tier_configs['tier1_trend'].get('action', {})

        # 计算指标
        if indicators is None and market_data is not None:
            indicators = self.calculate_market_indicators(market_data)
        elif indicators is None:
            indicators = {}

        # 评估条件
        scores = self.evaluate_tier_conditions(indicators)

        # 确定档位 (优先级: tier3 > tier2 > tier1)
        if scores['tier3_crash'] >= self.tier_configs['tier3_crash'].get('detect', {}).get('score_threshold', 1.5):
            new_tier = 'tier3_crash'
        elif scores['tier2_volatile'] >= self.tier_configs['tier2_volatile'].get('detect', {}).get('score_threshold', 1.0):
            new_tier = 'tier2_volatile'
        else:
            new_tier = 'tier1_trend'

        # 记录状态变化
        if new_tier != self.current_tier:
            logger.info(f"风控档位变化: {self.current_tier} -> {new_tier}")
            self.tier_history.append({
                'timestamp': datetime.now().isoformat(),
                'from_tier': self.current_tier,
                'to_tier': new_tier,
                'scores': scores,
                'indicators': indicators
            })
            self.current_tier = new_tier

        # 更新计数
        self.tier_counts[new_tier] = self.tier_counts.get(new_tier, 0) + 1

        # 返回动作配置
        actions = self.tier_configs[new_tier].get('action', {})

        return new_tier, actions

    def get_actions(self, tier: str) -> Dict:
        """获取指定档位的动作配置"""
        return self.tier_configs.get(tier, {}).get('action', {
            'position_ratio': 1.0,
            'max_single_weight': 0.12,
            'allow_buy': True,
            'cash_ratio_min': 0.02
        })

    def apply_position_ratio(self, weights: Dict[str, float], position_ratio: float) -> Dict[str, float]:
        """
        应用仓位比例调整

        Args:
            weights: 原始权重字典
            position_ratio: 仓位比例

        Returns:
            调整后的权重
        """
        adjusted = {}
        for code, weight in weights.items():
            adjusted[code] = weight * position_ratio

        # 归一化
        total = sum(adjusted.values())
        if total > 0:
            for code in adjusted:
                adjusted[code] /= total

        return adjusted

    def apply_max_weight_constraint(self,
                                     weights: Dict[str, float],
                                     max_single_weight: float) -> Dict[str, float]:
        """
        应用单股上限约束

        Args:
            weights: 权重字典
            max_single_weight: 单股最大权重

        Returns:
            调整后的权重
        """
        adjusted = {}
        for code, weight in weights.items():
            adjusted[code] = min(weight, max_single_weight)

        # 归一化
        total = sum(adjusted.values())
        if total > 0:
            for code in adjusted:
                adjusted[code] /= total

        return adjusted

    def get_summary(self) -> Dict:
        """获取风控状态摘要"""
        return {
            'enabled': self.enabled,
            'current_tier': self.current_tier,
            'tier_counts': self.tier_counts,
            'tier_changes': len(self.tier_history),
            'tier3_dates': [
                h['timestamp'] for h in self.tier_history
                if h['to_tier'] == 'tier3_crash'
            ]
        }


if __name__ == '__main__':
    # 测试
    import numpy as np

    manager = RiskTierManager('config/risk_control.yaml')

    # 模拟价格序列
    np.random.seed(42)
    prices = pd.Series(100 * np.cumprod(1 + np.random.normal(0.001, 0.02, 100)))

    tier, actions = manager.detect_tier(prices)
    print(f"当前档位: {tier}")
    print(f"动作配置: {actions}")
    print(f"状态摘要: {manager.get_summary()}")
