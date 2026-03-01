"""
Gate-0: 数据/信号熔断 v3.0

防止 scores 缺失时策略退化为等权随机

功能：
- 检测数据缺失率
- 熔断机制（freeze_rebalance / reduce_only）
- 统计与日志

使用方法:
    from src.reliability.gate0 import ReliabilityGate

    gate = ReliabilityGate(config_path='config/reliability.yaml')
    breach, action = gate.check(codes, scores, factors)
    if breach:
        target_weights = gate.apply_action(action, current_weights)
"""

import os
import yaml
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BreachEvent:
    """熔断事件"""
    timestamp: str
    breach_type: str
    value: float
    threshold: float
    action: str


class ReliabilityGate:
    """Gate-0: 数据/信号熔断"""

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
                self.config = yaml.safe_load(f).get('reliability', {})
        else:
            self.config = self._default_config()

        # 解析阈值
        thresholds = self.config.get('thresholds', {})
        self.score_missing_rate_max = thresholds.get('score_missing_rate_max', 0.02)
        self.factor_missing_rate_max = thresholds.get('factor_missing_rate_max', 0.05)
        self.data_freshness_max = thresholds.get('data_freshness_hours_max', 48)

        # 熔断行为
        on_breach = self.config.get('on_breach', {})
        self.default_action = on_breach.get('action', 'freeze_rebalance')
        self.reduce_only_ratio = on_breach.get('reduce_only_ratio', 0.8)
        self.log_prefix = on_breach.get('log_prefix', '[Gate-0]')

        # 恢复策略
        recovery = self.config.get('recovery', {})
        self.cooldown_periods = recovery.get('cooldown_periods', 2)
        self.gradual_recovery = recovery.get('gradual_recovery', True)
        self.recovery_ratio = recovery.get('recovery_ratio', 0.5)

        # 状态
        self.breach_history = []
        self.cooldown_counter = 0
        self.is_in_cooldown = False

        # 统计
        self.stats = {
            'gate0_triggered': False,
            'gate0_action': 'none',
            'gate0_trigger_count': 0,
            'freeze_rebalance_count': 0,
            'reduce_only_count': 0,
            'score_missing_rate': 0.0,
            'factor_missing_rate': 0.0,
            'avg_score_missing_rate': 0.0,
            'avg_factor_missing_rate': 0.0,
        }
        self.missing_rates_history = []

    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            'thresholds': {
                'score_missing_rate_max': 0.02,
                'factor_missing_rate_max': 0.05,
            },
            'on_breach': {
                'action': 'freeze_rebalance',
                'reduce_only_ratio': 0.8,
            },
            'recovery': {
                'cooldown_periods': 2,
                'gradual_recovery': True,
            }
        }

    def check(self,
              codes: List[str],
              scores: pd.Series = None,
              factors: pd.DataFrame = None,
              data_timestamp: datetime = None) -> Tuple[bool, str, Dict]:
        """
        检查数据可靠性

        Args:
            codes: 期望的股票代码列表
            scores: 得分Series (index=code)
            factors: 因子DataFrame
            data_timestamp: 数据时间戳

        Returns:
            (是否触发熔断, 熔断动作, 详情)
        """
        details = {
            'score_missing_rate': 0.0,
            'factor_missing_rate': 0.0,
            'data_freshness_hours': 0,
            'breach_type': None,
        }

        # 检查得分缺失率
        if scores is not None and len(codes) > 0:
            missing_codes = set(codes) - set(scores.index)
            score_missing_rate = len(missing_codes) / len(codes)
            details['score_missing_rate'] = score_missing_rate
        else:
            score_missing_rate = 1.0  # 完全缺失
            details['score_missing_rate'] = 1.0

        # 检查因子缺失率
        if factors is not None and len(codes) > 0:
            if 'code' in factors.columns:
                missing_codes = set(codes) - set(factors['code'].unique())
                factor_missing_rate = len(missing_codes) / len(codes)
            else:
                factor_missing_rate = 0.0
            details['factor_missing_rate'] = factor_missing_rate
        else:
            factor_missing_rate = 0.0

        # 检查数据新鲜度
        if data_timestamp:
            freshness_hours = (datetime.now() - data_timestamp).total_seconds() / 3600
            details['data_freshness_hours'] = freshness_hours
        else:
            freshness_hours = 0

        # 更新统计
        self.stats['score_missing_rate'] = score_missing_rate
        self.stats['factor_missing_rate'] = factor_missing_rate
        self.missing_rates_history.append({
            'score_missing': score_missing_rate,
            'factor_missing': factor_missing_rate
        })

        # 检查是否在冷却期
        if self.is_in_cooldown:
            self.cooldown_counter += 1
            if self.cooldown_counter >= self.cooldown_periods:
                self.is_in_cooldown = False
                self.cooldown_counter = 0
                logger.info(f"{self.log_prefix} 冷却期结束")
            else:
                logger.info(f"{self.log_prefix} 冷却期中 ({self.cooldown_counter}/{self.cooldown_periods})")
                return True, 'cooldown', details

        # 判断是否熔断
        breach = False
        breach_type = None
        action = 'none'

        if score_missing_rate > self.score_missing_rate_max:
            breach = True
            breach_type = 'score_missing'
            action = self.default_action
            logger.warning(f"{self.log_prefix} 得分缺失率 {score_missing_rate:.2%} > 阈值 {self.score_missing_rate_max:.0%}")

        if factor_missing_rate > self.factor_missing_rate_max:
            breach = True
            breach_type = breach_type or 'factor_missing'
            action = self.default_action
            logger.warning(f"{self.log_prefix} 因子缺失率 {factor_missing_rate:.2%} > 阈值 {self.factor_missing_rate_max:.0%}")

        if freshness_hours > self.data_freshness_max:
            breach = True
            breach_type = breach_type or 'data_stale'
            action = self.default_action
            logger.warning(f"{self.log_prefix} 数据陈旧 {freshness_hours:.0f}h > 阈值 {self.data_freshness_max}h")

        if breach:
            details['breach_type'] = breach_type
            self._record_breach(breach_type, details.get('score_missing_rate', 0), action)

        self.stats['gate0_triggered'] = breach
        self.stats['gate0_action'] = action

        return breach, action, details

    def _record_breach(self, breach_type: str, value: float, action: str):
        """记录熔断事件"""
        event = BreachEvent(
            timestamp=datetime.now().isoformat(),
            breach_type=breach_type,
            value=value,
            threshold=self.score_missing_rate_max if breach_type == 'score_missing' else 0,
            action=action
        )
        self.breach_history.append(event)
        self.stats['gate0_trigger_count'] += 1

        if action == 'freeze_rebalance':
            self.stats['freeze_rebalance_count'] += 1
        elif action == 'reduce_only':
            self.stats['reduce_only_count'] += 1

        self.is_in_cooldown = True
        self.cooldown_counter = 0

    def apply_action(self,
                     action: str,
                     current_weights: Dict[str, float],
                     target_weights: Dict[str, float] = None) -> Dict[str, float]:
        """
        应用熔断动作

        Args:
            action: 熔断动作
            current_weights: 当前权重
            target_weights: 目标权重

        Returns:
            调整后的权重
        """
        if action == 'none':
            return target_weights or current_weights

        if action == 'cooldown':
            # 冷却期保持当前仓位
            logger.info(f"{self.log_prefix} 冷却期：保持当前仓位")
            return current_weights

        if action == 'freeze_rebalance':
            # 冻结调仓：目标=当前
            logger.info(f"{self.log_prefix} 冻结调仓：保持当前仓位")
            return current_weights.copy()

        if action == 'reduce_only':
            # 仅减仓：不允许加仓，总仓位降到reduce_only_ratio
            logger.info(f"{self.log_prefix} 仅减仓模式：仓位降到{self.reduce_only_ratio:.0%}")
            adjusted = {}
            for code, weight in current_weights.items():
                # 只减不加
                if target_weights and code in target_weights:
                    new_weight = min(weight, target_weights[code])
                else:
                    new_weight = weight
                adjusted[code] = new_weight * self.reduce_only_ratio

            # 归一化
            total = sum(adjusted.values())
            if total > 0:
                adjusted = {k: v / total * self.reduce_only_ratio for k, v in adjusted.items()}

            return adjusted

        return current_weights.copy()

    def get_stats(self) -> Dict:
        """获取统计信息"""
        # 计算平均缺失率
        if self.missing_rates_history:
            self.stats['avg_score_missing_rate'] = np.mean([
                r['score_missing'] for r in self.missing_rates_history
            ])
            self.stats['avg_factor_missing_rate'] = np.mean([
                r['factor_missing'] for r in self.missing_rates_history
            ])

        return self.stats.copy()

    def get_breach_history(self) -> List[Dict]:
        """获取熔断历史"""
        return [
            {
                'timestamp': e.timestamp,
                'breach_type': e.breach_type,
                'value': e.value,
                'threshold': e.threshold,
                'action': e.action
            }
            for e in self.breach_history
        ]


def check_data_reliability(codes: List[str],
                            scores: pd.Series,
                            config_path: str = 'config/reliability.yaml') -> Tuple[bool, str, Dict]:
    """
    便捷函数：检查数据可靠性

    Args:
        codes: 期望的股票代码列表
        scores: 得分Series
        config_path: 配置路径

    Returns:
        (是否触发熔断, 熔断动作, 详情)
    """
    gate = ReliabilityGate(config_path=config_path)
    return gate.check(codes, scores)


def apply_circuit_breaker(action: str,
                           current_weights: Dict[str, float],
                           target_weights: Dict[str, float] = None,
                           config_path: str = 'config/reliability.yaml') -> Dict[str, float]:
    """
    便捷函数：应用熔断

    Args:
        action: 熔断动作
        current_weights: 当前权重
        target_weights: 目标权重
        config_path: 配置路径

    Returns:
        调整后的权重
    """
    gate = ReliabilityGate(config_path=config_path)
    return gate.apply_action(action, current_weights, target_weights)


if __name__ == '__main__':
    # 测试
    print("=" * 60)
    print("Gate-0 熔断测试")
    print("=" * 60)

    gate = ReliabilityGate()

    # 正常情况
    codes = [f'{i:06d}' for i in range(1, 101)]
    scores = pd.Series(np.random.uniform(30, 90, len(codes)), index=codes)

    breach, action, details = gate.check(codes, scores)
    print(f"\n正常情况: breach={breach}, action={action}")
    print(f"缺失率: score={details['score_missing_rate']:.2%}")

    # 缺失10%
    partial_scores = scores.iloc[:90]
    breach, action, details = gate.check(codes, partial_scores)
    print(f"\n缺失10%: breach={breach}, action={action}")
    print(f"缺失率: score={details['score_missing_rate']:.2%}")

    # 完全缺失
    breach, action, details = gate.check(codes, None)
    print(f"\n完全缺失: breach={breach}, action={action}")
    print(f"缺失率: score={details['score_missing_rate']:.2%}")

    # 测试熔断动作
    current = {'A': 0.3, 'B': 0.3, 'C': 0.2, 'D': 0.2}
    target = {'A': 0.2, 'B': 0.4, 'C': 0.2, 'E': 0.2}

    frozen = gate.apply_action('freeze_rebalance', current, target)
    print(f"\n冻结调仓: {frozen}")

    reduced = gate.apply_action('reduce_only', current, target)
    print(f"仅减仓: {reduced}")
