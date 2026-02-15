"""
双层组合构建器 v3.0

Core Layer: 核心池（低频、稳健）
- 从固化逻辑提取核心选择法
- 季度调整，低换手

Satellite Layer: 卫星池（高频、轮动）
- 因子轮动策略
- 月度调整

使用方法:
    from src.diagnostics.core_satellite_builder import CoreSatelliteBuilder

    builder = CoreSatelliteBuilder(config_path='config/core_satellite.yaml')
    core, satellite = builder.build(scores, holdings_history, current_core)
"""

import os
import yaml
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class CoreSatelliteBuilder:
    """双层组合构建器"""

    def __init__(self, config_path: str = None, config: Dict = None):
        """
        初始化

        Args:
            config_path: 配置路径
            config: 配置字典
        """
        if config is not None:
            self.config = config
        elif config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f).get('portfolio_structure', {})
        else:
            self.config = self._default_config()

        self.enabled = self.config.get('enabled', False)

        # 核心池配置
        core_config = self.config.get('core', {})
        self.core_k = core_config.get('k', 30)
        self.core_rebalance_freq = core_config.get('rebalance_frequency', 'quarterly')
        self.core_min_hold = core_config.get('min_hold_period', 60)
        self.core_max_weight_sum = core_config.get('max_core_weight_sum', 0.60)
        self.core_method = core_config.get('method', 'fixation_core')

        # 卫星池配置
        sat_config = self.config.get('satellite', {})
        self.satellite_n = sat_config.get('n', 40)
        self.satellite_rebalance_freq = sat_config.get('rebalance_frequency', 'monthly')
        self.satellite_weight_sum = sat_config.get('weight_sum', 0.40)
        self.satellite_exclude_core = sat_config.get('selection', {}).get('exclude_core', True)
        self.satellite_method = sat_config.get('method', 'factor_rotation')

        # 状态追踪
        self.current_core = []
        self.core_history = []
        self.last_core_rebalance = None
        self.last_satellite_rebalance = None

        # 统计
        self.stats = {
            'core_turnover_avg': 0,
            'core_jaccard_avg': 0,
            'satellite_turnover_avg': 0,
            'total_rebalances': 0
        }

    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            'enabled': False,
            'core': {
                'k': 30,
                'rebalance_frequency': 'quarterly',
                'min_hold_period': 60,
                'max_core_weight_sum': 0.60,
                'method': 'fixation_core'
            },
            'satellite': {
                'n': 40,
                'rebalance_frequency': 'monthly',
                'weight_sum': 0.40,
                'method': 'factor_rotation',
                'selection': {'exclude_core': True}
            }
        }

    def should_rebalance_core(self, current_date: str) -> bool:
        """检查是否应该调整核心池"""
        if self.last_core_rebalance is None:
            return True

        # 解析日期
        current = pd.to_datetime(current_date)
        last = pd.to_datetime(self.last_core_rebalance)

        if self.core_rebalance_freq == 'quarterly':
            # 季度调整：相隔约90天
            return (current - last).days >= 90
        elif self.core_rebalance_freq == 'bi_annual':
            return (current - last).days >= 180
        else:  # monthly
            return (current - last).days >= 30

    def should_rebalance_satellite(self, current_date: str) -> bool:
        """检查是否应该调整卫星池"""
        if self.last_satellite_rebalance is None:
            return True

        current = pd.to_datetime(current_date)
        last = pd.to_datetime(self.last_satellite_rebalance)

        return (current - last).days >= 30

    def select_core(self,
                    scores: pd.Series,
                    holdings_history: List[Dict[str, float]],
                    current_date: str) -> Tuple[List[str], Dict]:
        """
        选择核心池

        Args:
            scores: 因子得分
            holdings_history: 历史持仓
            current_date: 当前日期

        Returns:
            (核心池代码列表, 选择详情)
        """
        if not self.enabled:
            # 禁用时返回空核心池
            return [], {'enabled': False}

        # 检查是否需要调整
        if not self.should_rebalance_core(current_date):
            logger.info(f"[CoreSatellite] 核心池保持不变（距上次调整不足{self.core_rebalance_freq}）")
            return self.current_core, {'rebalanced': False, 'reason': 'frequency_constraint'}

        details = {
            'rebalanced': True,
            'method': self.core_method,
            'previous_core': self.current_core.copy() if self.current_core else []
        }

        if self.core_method == 'fixation_core':
            new_core = self._select_by_fixation_logic(scores, holdings_history)
        else:
            new_core = self._select_by_top_scores(scores)

        # 计算换手
        if self.current_core:
            old_set = set(self.current_core)
            new_set = set(new_core)
            turnover = len(old_set.symmetric_difference(new_set)) / (len(old_set) + len(new_set)) / 2
            jaccard = len(old_set & new_set) / len(old_set | new_set) if (old_set | new_set) else 0
        else:
            turnover = 1.0
            jaccard = 0

        details['turnover'] = turnover
        details['jaccard'] = jaccard

        # 更新状态
        self.current_core = new_core
        self.last_core_rebalance = current_date
        self.core_history.append({
            'date': current_date,
            'core': new_core,
            'turnover': turnover,
            'jaccard': jaccard
        })

        logger.info(f"[CoreSatellite] 核心池调整: {len(new_core)}只, 换手{turnover:.2%}, Jaccard{jaccard:.2%}")

        return new_core, details

    def _select_by_fixation_logic(self,
                                   scores: pd.Series,
                                   holdings_history: List[Dict[str, float]]) -> List[str]:
        """
        使用固化逻辑选择核心池

        策略：
        1. 计算历史出现频率
        2. 结合当前得分
        3. 选择稳定+高分的股票
        """
        if not holdings_history:
            # 无历史时，直接按分数选择
            return self._select_by_top_scores(scores)

        # 计算出现频率
        frequency = {}
        for h in holdings_history[-12:]:  # 最近12期
            for code in h.keys():
                frequency[code] = frequency.get(code, 0) + 1

        max_freq = max(frequency.values()) if frequency else 1

        # 综合得分 = 原始得分 * (1 + 频率加成)
        combined_scores = {}
        for code in scores.index:
            freq_bonus = frequency.get(code, 0) / max_freq * 0.2  # 频率加成最多20%
            combined_scores[code] = scores.loc[code] * (1 + freq_bonus)

        # 排序选择TopK
        sorted_codes = sorted(combined_scores.keys(),
                              key=lambda x: combined_scores[x],
                              reverse=True)

        return sorted_codes[:self.core_k]

    def _select_by_top_scores(self, scores: pd.Series) -> List[str]:
        """直接按分数选择TopK"""
        sorted_codes = scores.sort_values(ascending=False).index.tolist()
        return sorted_codes[:self.core_k]

    def select_satellite(self,
                         scores: pd.Series,
                         core: List[str],
                         current_date: str) -> Tuple[List[str], Dict]:
        """
        选择卫星池

        Args:
            scores: 因子得分
            core: 当前核心池
            current_date: 当前日期

        Returns:
            (卫星池代码列表, 选择详情)
        """
        if not self.enabled:
            return [], {'enabled': False}

        details = {
            'rebalanced': self.should_rebalance_satellite(current_date),
            'method': self.satellite_method,
            'exclude_core': self.satellite_exclude_core
        }

        # 排除核心池
        if self.satellite_exclude_core:
            available_scores = scores[~scores.index.isin(core)]
        else:
            available_scores = scores

        # 选择TopN
        sorted_codes = available_scores.sort_values(ascending=False).index.tolist()
        satellite = sorted_codes[:self.satellite_n]

        self.last_satellite_rebalance = current_date

        logger.info(f"[CoreSatellite] 卫星池选择: {len(satellite)}只")

        return satellite, details

    def build(self,
              scores: pd.Series,
              holdings_history: List[Dict[str, float]],
              current_date: str) -> Tuple[Dict[str, float], Dict]:
        """
        构建完整组合

        Args:
            scores: 因子得分
            holdings_history: 历史持仓
            current_date: 当前日期

        Returns:
            (最终权重, 构建详情)
        """
        if not self.enabled:
            # 禁用时返回等权
            weights = {code: 1/len(scores) for code in scores.index[:50]}
            return weights, {'enabled': False}

        # 1. 选择核心池
        core, core_details = self.select_core(scores, holdings_history, current_date)

        # 2. 选择卫星池
        satellite, sat_details = self.select_satellite(scores, core, current_date)

        # 3. 分配权重
        weights = {}

        # 核心池权重
        if core:
            core_weight_per_stock = self.core_max_weight_sum / len(core)
            for code in core:
                weights[code] = core_weight_per_stock

        # 卫星池权重
        if satellite:
            sat_weight_per_stock = self.satellite_weight_sum / len(satellite)
            for code in satellite:
                weights[code] = sat_weight_per_stock

        # 4. 归一化
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        # 汇总详情
        details = {
            'enabled': True,
            'core': {
                'codes': core,
                'count': len(core),
                'weight_sum': self.core_max_weight_sum,
                'details': core_details
            },
            'satellite': {
                'codes': satellite,
                'count': len(satellite),
                'weight_sum': self.satellite_weight_sum,
                'details': sat_details
            },
            'overlap': len(set(core) & set(satellite)),
            'total_positions': len(weights)
        }

        return weights, details

    def get_stats(self) -> Dict:
        """获取统计信息"""
        if self.core_history:
            turnovers = [h['turnover'] for h in self.core_history]
            jaccards = [h['jaccard'] for h in self.core_history]
            self.stats['core_turnover_avg'] = np.mean(turnovers)
            self.stats['core_jaccard_avg'] = np.mean(jaccards)

        return self.stats.copy()


def build_core_satellite_portfolio(scores: pd.Series,
                                    holdings_history: List[Dict[str, float]],
                                    current_date: str,
                                    config_path: str = 'config/core_satellite.yaml') -> Tuple[Dict[str, float], Dict]:
    """
    便捷函数：构建双层组合

    Args:
        scores: 因子得分
        holdings_history: 历史持仓
        current_date: 当前日期
        config_path: 配置路径

    Returns:
        (权重字典, 构建详情)
    """
    builder = CoreSatelliteBuilder(config_path=config_path)
    return builder.build(scores, holdings_history, current_date)


if __name__ == '__main__':
    # 测试
    print("=" * 60)
    print("双层组合构建器测试")
    print("=" * 60)

    builder = CoreSatelliteBuilder(config={'enabled': True})

    # 模拟得分
    np.random.seed(42)
    codes = [f'{i:06d}' for i in range(1, 101)]
    scores = pd.Series(np.random.uniform(30, 90, len(codes)), index=codes)

    # 模拟历史持仓
    history = [
        {f'{i:06d}': 1/30 for i in range(1, 31)},
        {f'{i:06d}': 1/30 for i in range(1, 31)},
        {f'{i:06d}': 1/30 for i in range(1, 31)},
    ]

    weights, details = builder.build(scores, history, '2024-01-01')

    print(f"\n核心池: {details['core']['count']}只")
    print(f"卫星池: {details['satellite']['count']}只")
    print(f"重叠: {details['overlap']}只")
    print(f"总持仓: {details['total_positions']}只")
