"""
反固化机制 v3.0

解决名单固化问题：
- 机制A: Soft Diversify (软分散约束)
- 机制B: Adaptive Threshold (自适应阈值)
- 诊断与建议

使用方法:
    from src.optimization.anti_fixation import AntiFixationEngine

    engine = AntiFixationEngine(config_path='config/anti_fixation.yaml')
    adjusted_scores = engine.apply_soft_diversify(scores, holdings_history)
"""

import os
import yaml
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)


class SoftDiversifyPenalty:
    """机制A: 软分散约束 - 出现频率惩罚"""

    def __init__(self, config: Dict = None):
        """
        初始化

        Args:
            config: 配置字典
        """
        self.config = config or {}
        soft_config = self.config.get('soft_diversify', {})
        penalty_config = soft_config.get('frequency_penalty', {})

        self.max_penalty_bps = penalty_config.get('max_penalty_bps', 50)
        self.lookback_periods = penalty_config.get('lookback_periods', 12)
        self.threshold_percentile = penalty_config.get('threshold_percentile', 0.8)
        self.method = penalty_config.get('method', 'linear')

        protection = soft_config.get('protection', {})
        self.preserve_top_n = protection.get('preserve_top_n', 5)
        self.min_score_to_penalize = protection.get('min_score_to_penalize', 60)
        self.max_weight_change = protection.get('max_weight_change', 0.02)

        # 记录惩罚详情
        self.penalty_log = []

    def calculate_frequency(self, holdings_history: List[Dict[str, float]]) -> Dict[str, float]:
        """
        计算各股票的出现频率

        Args:
            holdings_history: 历史持仓列表

        Returns:
            {code: frequency} 频率字典 (0-1)
        """
        if not holdings_history:
            return {}

        lookback = min(self.lookback_periods, len(holdings_history))
        recent_history = holdings_history[-lookback:]

        all_codes = set()
        for h in recent_history:
            all_codes.update(h.keys())

        frequency = {}
        for code in all_codes:
            count = sum(1 for h in recent_history if code in h and h.get(code, 0) > 0)
            frequency[code] = count / lookback

        return frequency

    def apply_penalty(self,
                      scores: pd.Series,
                      holdings_history: List[Dict[str, float]],
                      current_weights: Dict[str, float] = None) -> Tuple[pd.Series, Dict]:
        """
        应用频率惩罚

        Args:
            scores: 因子得分 Series (index=code)
            holdings_history: 历史持仓列表
            current_weights: 当前权重（用于保护TopN）

        Returns:
            (调整后得分, 惩罚详情)
        """
        if not holdings_history or len(holdings_history) < 2:
            return scores, {'penalty_applied': False, 'reason': 'insufficient_history'}

        self.penalty_log = []

        # 计算频率
        frequency = self.calculate_frequency(holdings_history)

        # 找出需要保护的TopN
        protected_codes = set()
        if current_weights:
            sorted_weights = sorted(current_weights.items(), key=lambda x: -x[1])
            protected_codes = set(c for c, _ in sorted_weights[:self.preserve_top_n])

        # 高频阈值
        freq_threshold = self.threshold_percentile

        # 应用惩罚
        adjusted_scores = scores.copy()
        penalty_details = {
            'penalty_applied': True,
            'total_penalized': 0,
            'max_penalty': 0,
            'protected_count': len(protected_codes),
            'penalty_list': []
        }

        for code in scores.index:
            if code in protected_codes:
                continue

            score = scores.loc[code]
            if score < self.min_score_to_penalize:
                continue

            freq = frequency.get(code, 0)
            if freq < freq_threshold:
                continue

            # 计算惩罚
            if self.method == 'linear':
                # 线性惩罚：频率越高，惩罚越大
                penalty_ratio = (freq - freq_threshold) / (1 - freq_threshold)
            else:
                # 指数惩罚
                penalty_ratio = ((freq - freq_threshold) / (1 - freq_threshold)) ** 2

            penalty = penalty_ratio * self.max_penalty_bps / 100  # 转换为分数

            # 应用惩罚
            adjusted_scores.loc[code] = max(0, score - penalty)

            # 记录
            self.penalty_log.append({
                'code': code,
                'original_score': score,
                'adjusted_score': adjusted_scores.loc[code],
                'penalty': penalty,
                'frequency': freq
            })

            penalty_details['total_penalized'] += 1
            penalty_details['max_penalty'] = max(penalty_details['max_penalty'], penalty)

        # 保留最近10条惩罚记录
        penalty_details['penalty_list'] = self.penalty_log[:10]

        logger.info(f"[SoftDiversify] 惩罚了 {penalty_details['total_penalized']} 只高频股票")

        return adjusted_scores, penalty_details


class AdaptiveThresholdAdjuster:
    """机制B: 自适应阈值调整"""

    def __init__(self, config: Dict = None):
        """
        初始化

        Args:
            config: 配置字典
        """
        self.config = config or {}
        adaptive_config = self.config.get('adaptive_threshold', {})

        self.enabled = adaptive_config.get('enabled', False)

        trigger = adaptive_config.get('trigger', {})
        self.jaccard_trigger = trigger.get('jaccard_trigger', 0.75)
        self.stickiness_trigger = trigger.get('stickiness_trigger', 0.80)
        self.hysteresis = trigger.get('hysteresis', 0.05)

        adjustments = adaptive_config.get('adjustments', {})
        self.top_boost_range = (adjustments.get('top_boost_decay', {}).get('from', 1.30),
                                 adjustments.get('top_boost_decay', {}).get('to', 1.15))
        self.bottom_penalty_range = (adjustments.get('bottom_penalty_relax', {}).get('from', 0.70),
                                      adjustments.get('bottom_penalty_relax', {}).get('to', 0.85))
        self.tilt_range = (adjustments.get('tilt_strength_decay', {}).get('from', 1.0),
                           adjustments.get('tilt_strength_decay', {}).get('to', 0.85))

        recovery = adaptive_config.get('recovery', {})
        self.cooldown_periods = recovery.get('cooldown_periods', 3)
        self.gradual_recovery = recovery.get('gradual_recovery', True)

        # 状态跟踪
        self.adjustment_level = 0  # 0=正常, 1-3=调整级别
        self.cooldown_counter = 0
        self.adjustment_log = []

    def check_and_adjust(self,
                         jaccard_12m: float,
                         stickiness: float,
                         current_params: Dict) -> Tuple[Dict, Dict]:
        """
        检查并调整参数

        Args:
            jaccard_12m: 12期平均Jaccard
            stickiness: Top10粘性
            current_params: 当前参数

        Returns:
            (调整后参数, 调整详情)
        """
        if not self.enabled:
            return current_params, {'adjustment_applied': False, 'reason': 'disabled'}

        details = {
            'adjustment_applied': False,
            'trigger_jaccard': jaccard_12m,
            'trigger_stickiness': stickiness,
            'old_level': self.adjustment_level,
            'new_level': self.adjustment_level,
            'params_changed': []
        }

        # 检查是否需要调整
        triggered = (jaccard_12m > self.jaccard_trigger + self.hysteresis or
                     stickiness > self.stickiness_trigger + self.hysteresis)

        # 检查是否可以恢复
        recovered = (jaccard_12m < self.jaccard_trigger - self.hysteresis and
                     stickiness < self.stickiness_trigger - self.hysteresis)

        new_params = current_params.copy()

        if triggered and self.adjustment_level < 3:
            # 增加调整级别
            self.adjustment_level = min(3, self.adjustment_level + 1)
            self.cooldown_counter = 0
            details['adjustment_applied'] = True

        elif recovered and self.adjustment_level > 0:
            # 冷却后恢复
            self.cooldown_counter += 1
            if self.cooldown_counter >= self.cooldown_periods:
                if self.gradual_recovery:
                    self.adjustment_level = max(0, self.adjustment_level - 1)
                else:
                    self.adjustment_level = 0
                self.cooldown_counter = 0
                details['adjustment_applied'] = True

        details['new_level'] = self.adjustment_level

        # 应用参数调整
        if self.adjustment_level > 0:
            level_ratio = self.adjustment_level / 3.0

            # Top boost 衰减
            old_top = new_params.get('top_10_pct_boost', 1.30)
            new_top = self.top_boost_range[0] + level_ratio * (self.top_boost_range[1] - self.top_boost_range[0])
            new_params['top_10_pct_boost'] = new_top
            if abs(old_top - new_top) > 0.01:
                details['params_changed'].append(f"top_boost: {old_top:.2f} -> {new_top:.2f}")

            # Bottom penalty 放宽
            old_bottom = new_params.get('bottom_10_pct_penalty', 0.70)
            new_bottom = self.bottom_penalty_range[0] + level_ratio * (self.bottom_penalty_range[1] - self.bottom_penalty_range[0])
            new_params['bottom_10_pct_penalty'] = new_bottom
            if abs(old_bottom - new_bottom) > 0.01:
                details['params_changed'].append(f"bottom_penalty: {old_bottom:.2f} -> {new_bottom:.2f}")

            # Tilt strength 衰减
            old_tilt = new_params.get('tilt_strength', 1.0)
            new_tilt = self.tilt_range[0] + level_ratio * (self.tilt_range[1] - self.tilt_range[0])
            new_params['tilt_strength'] = new_tilt
            if abs(old_tilt - new_tilt) > 0.01:
                details['params_changed'].append(f"tilt_strength: {old_tilt:.2f} -> {new_tilt:.2f}")

        if details['adjustment_applied']:
            logger.info(f"[AdaptiveThreshold] 级别调整: {details['old_level']} -> {details['new_level']}")
            self.adjustment_log.append({
                'timestamp': datetime.now().isoformat(),
                'jaccard': jaccard_12m,
                'stickiness': stickiness,
                'level': self.adjustment_level
            })

        return new_params, details


class AntiFixationEngine:
    """反固化引擎"""

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
                self.config = yaml.safe_load(f).get('anti_fixation', {})
        else:
            self.config = self._default_config()

        self.enabled = self.config.get('enabled', False)

        # 初始化子模块
        self.soft_diversify = SoftDiversifyPenalty(self.config)
        self.adaptive_threshold = AdaptiveThresholdAdjuster(self.config)

        # 诊断结果
        self.last_diagnosis = None

    def _default_config(self) -> Dict:
        """默认配置"""
        return {
            'enabled': False,
            'target': {
                'jaccard_12m_avg': 0.70,
                'max_jaccard_12m': 0.75,
                'stickiness_max': 0.80
            },
            'soft_diversify': {
                'enabled': True,
                'frequency_penalty': {
                    'max_penalty_bps': 50,
                    'lookback_periods': 12,
                    'threshold_percentile': 0.8
                },
                'protection': {
                    'preserve_top_n': 5,
                    'min_score_to_penalize': 60
                }
            },
            'adaptive_threshold': {
                'enabled': False
            }
        }

    def apply_soft_diversify(self,
                             scores: pd.Series,
                             holdings_history: List[Dict[str, float]],
                             current_weights: Dict[str, float] = None) -> Tuple[pd.Series, Dict]:
        """
        应用软分散惩罚

        Args:
            scores: 因子得分
            holdings_history: 历史持仓
            current_weights: 当前权重

        Returns:
            (调整后得分, 详情)
        """
        if not self.enabled:
            return scores, {'applied': False, 'reason': 'anti_fixation_disabled'}

        return self.soft_diversify.apply_penalty(scores, holdings_history, current_weights)

    def check_and_adapt(self,
                        jaccard_12m: float,
                        stickiness: float,
                        current_params: Dict) -> Tuple[Dict, Dict]:
        """
        检查并自适应调整

        Args:
            jaccard_12m: 12期平均Jaccard
            stickiness: Top10粘性
            current_params: 当前参数

        Returns:
            (调整后参数, 详情)
        """
        if not self.enabled:
            return current_params, {'applied': False, 'reason': 'anti_fixation_disabled'}

        return self.adaptive_threshold.check_and_adjust(jaccard_12m, stickiness, current_params)

    def get_severity_level(self, jaccard_12m: float, stickiness: float) -> Tuple[str, List[str]]:
        """
        获取Gate-6严重程度

        Args:
            jaccard_12m: 12期平均Jaccard
            stickiness: Top10粘性

        Returns:
            (严重程度, 原因列表)
        """
        gate6_config = self.config.get('gate6_severity', {})
        hard_fail = gate6_config.get('hard_fail', {})
        soft_fail = gate6_config.get('soft_fail', {})
        warning = gate6_config.get('warning', {})

        reasons = []
        severity = 'pass'

        # 检查硬失败
        hard_jaccard = hard_fail.get('jaccard_12m_avg', 0.85)
        hard_stickiness = hard_fail.get('top_holdings_stickiness', 0.90)

        if jaccard_12m > hard_jaccard:
            severity = 'hard_fail'
            reasons.append(f"jaccard_12m={jaccard_12m:.2%} > {hard_jaccard:.0%}")
        if stickiness > hard_stickiness:
            severity = 'hard_fail'
            reasons.append(f"stickiness={stickiness:.2%} > {hard_stickiness:.0%}")

        if severity == 'hard_fail':
            return severity, reasons

        # 检查软失败
        soft_jaccard = soft_fail.get('jaccard_12m_avg', 0.75)
        soft_stickiness = soft_fail.get('top_holdings_stickiness', 0.80)

        if jaccard_12m > soft_jaccard:
            severity = 'soft_fail'
            reasons.append(f"jaccard_12m={jaccard_12m:.2%} > {soft_jaccard:.0%}")
        if stickiness > soft_stickiness:
            severity = 'soft_fail'
            reasons.append(f"stickiness={stickiness:.2%} > {soft_stickiness:.0%}")

        if severity == 'soft_fail':
            return severity, reasons

        # 检查警告
        warn_jaccard = warning.get('jaccard_12m_avg', 0.70)
        warn_stickiness = warning.get('top_holdings_stickiness', 0.75)

        if jaccard_12m > warn_jaccard or stickiness > warn_stickiness:
            severity = 'warning'
            if jaccard_12m > warn_jaccard:
                reasons.append(f"jaccard_12m={jaccard_12m:.2%} approaching threshold")
            if stickiness > warn_stickiness:
                reasons.append(f"stickiness={stickiness:.2%} approaching threshold")

        return severity, reasons

    def get_recommendation(self, severity: str, metrics: Dict) -> Dict:
        """
        获取改进建议

        Args:
            severity: 严重程度
            metrics: 相关指标

        Returns:
            建议字典
        """
        recommendations = {
            'severity': severity,
            'actions': [],
            'config_changes': {}
        }

        if severity == 'pass':
            recommendations['actions'].append("当前名单轮动正常，无需调整")
            return recommendations

        if severity in ['soft_fail', 'hard_fail']:
            recommendations['actions'].append("建议启用 anti_fixation.enabled=true")

            # 检查是否universe过窄
            universe_size = metrics.get('universe_size', 0)
            if universe_size < 500:
                recommendations['actions'].append(f"universe_size={universe_size} 过小，建议扩大股票池")
                recommendations['config_changes']['universe.liquidity.top_n'] = 800

            # 检查换手锁仓
            turnover_scale = metrics.get('turnover_scale_avg_12m', 1.0)
            if turnover_scale < 0.7:
                recommendations['actions'].append(f"turnover_scale={turnover_scale:.2f} 过低，存在换手锁仓")
                recommendations['config_changes']['rebalance.turnover_limit'] = 0.10

            # 检查非线性加成
            top_boost = metrics.get('top_10_pct_boost', 1.30)
            if top_boost > 1.20:
                recommendations['actions'].append(f"top_10_pct_boost={top_boost} 过高，建议降低")
                recommendations['config_changes']['weight_tilt.top_10_pct_boost'] = 1.15

        if severity == 'hard_fail':
            recommendations['actions'].append("严重固化！必须立即采取反固化措施")

        return recommendations


def diagnose_list_fixation(robustness_csv: str,
                           output_path: str = 'results/list_fixation_diagnosis.md') -> Dict:
    """
    诊断名单固化问题

    Args:
        robustness_csv: robustness结果CSV路径
        output_path: 输出路径

    Returns:
        诊断结果字典
    """
    if not os.path.exists(robustness_csv):
        logger.warning(f"文件不存在: {robustness_csv}")
        return {'error': 'file_not_found'}

    df = pd.read_csv(robustness_csv)

    diagnosis = {
        'total_tests': len(df),
        'timestamp': datetime.now().isoformat(),
        'gate6_stats': {},
        'fail_reasons': {},
        'root_causes': [],
        'recommendations': []
    }

    # Gate-6 统计
    if 'passed' in df.columns:
        fail_count = len(df[df['passed'] == False])
        diagnosis['gate6_stats']['fail_rate'] = fail_count / len(df) if len(df) > 0 else 0
        diagnosis['gate6_stats']['fail_count'] = fail_count

    # Fail reason 分布
    if 'fail_reason_top3' in df.columns:
        all_reasons = []
        for reasons in df['fail_reason_top3'].dropna():
            if isinstance(reasons, str):
                try:
                    reason_list = eval(reasons)  # 安全风险：仅用于内部数据
                    all_reasons.extend(reason_list)
                except:
                    all_reasons.append(reasons)

        from collections import Counter
        reason_counter = Counter(all_reasons)
        diagnosis['fail_reasons'] = dict(reason_counter.most_common(10))

    # 分析固化特征
    fixation_df = df[df['passed'] == False] if 'passed' in df.columns else df

    if len(fixation_df) > 0:
        # 检查 list_fixation 相关失败
        list_fixation_fails = fixation_df[
            fixation_df['fail_reason_top3'].astype(str).str.contains('list_fixation', na=False)
        ]

        if len(list_fixation_fails) > 0:
            # 分析共性特征
            features = {
                'universe_size_avg': fixation_df.get('universe_size', pd.Series([0])).mean(),
                'selected_size_avg': fixation_df.get('selected_size', pd.Series([0])).mean(),
                'turnover_scale_avg': fixation_df.get('turnover_scale_avg_12m', pd.Series([1.0])).mean(),
                'jaccard_avg': fixation_df.get('holdings_jaccard_12m_avg', pd.Series([0])).mean(),
                'stickiness_avg': fixation_df.get('top_holdings_stickiness', pd.Series([0])).mean(),
            }

            # 根因分析
            root_causes = []

            if features['universe_size_avg'] < 500:
                root_causes.append({
                    'cause': 'universe_too_narrow',
                    'evidence': f"平均universe_size={features['universe_size_avg']:.0f} < 500",
                    'impact': 'high',
                    'recommendation': '扩大股票池 top_n 到 800+'
                })

            if features['turnover_scale_avg'] < 0.7:
                root_causes.append({
                    'cause': 'turnover_locking',
                    'evidence': f"平均turnover_scale={features['turnover_scale_avg']:.2f} < 0.7",
                    'impact': 'high',
                    'recommendation': '放宽换手限制或启用反固化机制'
                })

            if features['jaccard_avg'] > 0.80:
                root_causes.append({
                    'cause': 'extreme_fixation',
                    'evidence': f"平均jaccard={features['jaccard_avg']:.2%} > 80%",
                    'impact': 'critical',
                    'recommendation': '必须启用 anti_fixation 机制'
                })

            if features['stickiness_avg'] > 0.85:
                root_causes.append({
                    'cause': 'top_holdings_locked',
                    'evidence': f"平均stickiness={features['stickiness_avg']:.2%} > 85%",
                    'impact': 'high',
                    'recommendation': '降低非线性加成强度'
                })

            diagnosis['root_causes'] = root_causes[:3]  # 取前3个
            diagnosis['features'] = features

    # 生成报告
    _generate_diagnosis_report(diagnosis, output_path)

    return diagnosis


def _generate_diagnosis_report(diagnosis: Dict, output_path: str):
    """生成诊断报告"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    report = f"""# 名单固化诊断报告

生成时间: {diagnosis['timestamp']}

## 1. Gate-6 统计

| 指标 | 值 |
|------|-----|
| 总测试数 | {diagnosis['total_tests']} |
| 失败数 | {diagnosis['gate6_stats'].get('fail_count', 0)} |
| 失败率 | {diagnosis['gate6_stats'].get('fail_rate', 0):.1%} |

## 2. 失败原因分布 (Top10)

| 原因 | 出现次数 |
|------|----------|
"""
    for reason, count in diagnosis.get('fail_reasons', {}).items():
        report += f"| {reason[:50]} | {count} |\n"

    report += """
## 3. 根因分析 (Top3)

| 排名 | 根因 | 证据 | 影响程度 | 建议 |
|------|------|------|----------|------|
"""
    for i, cause in enumerate(diagnosis.get('root_causes', []), 1):
        report += f"| {i} | {cause['cause']} | {cause['evidence']} | {cause['impact']} | {cause['recommendation']} |\n"

    if not diagnosis.get('root_causes'):
        report += "| - | 无明显根因 | - | - | 建议收集更多数据 |\n"

    report += """
## 4. 固化特征统计

| 指标 | 平均值 | 状态 |
|------|--------|------|
"""
    features = diagnosis.get('features', {})
    metrics_status = [
        ('universe_size_avg', features.get('universe_size_avg', 0), 500, '>=500 OK'),
        ('selected_size_avg', features.get('selected_size_avg', 0), 40, '>=40 OK'),
        ('turnover_scale_avg', features.get('turnover_scale_avg', 1.0), 0.7, '>=0.7 OK'),
        ('jaccard_avg', features.get('jaccard_avg', 0), 0.75, '<=0.75 OK'),
        ('stickiness_avg', features.get('stickiness_avg', 0), 0.80, '<=0.80 OK'),
    ]

    for name, value, threshold, ok_msg in metrics_status:
        if value > 0:
            status = ok_msg if (name in ['jaccard_avg', 'stickiness_avg'] and value <= threshold) or \
                              (name not in ['jaccard_avg', 'stickiness_avg'] and value >= threshold) else '⚠️ 需关注'
            report += f"| {name} | {value:.2f} | {status} |\n"

    report += """
## 5. 改进建议

"""
    if diagnosis.get('root_causes'):
        report += "建议采取以下措施:\n\n"
        for cause in diagnosis['root_causes']:
            report += f"1. **{cause['recommendation']}**\n"
        report += "\n### 配置变更建议\n\n"
        report += "```yaml\nanti_fixation:\n  enabled: true\n```\n"
    else:
        report += "当前状态良好，建议持续监控。\n"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    logger.info(f"诊断报告已生成: {output_path}")


def run_ablation_study(backtest_func,
                        portfolio: pd.DataFrame,
                        start_date: str,
                        end_date: str,
                        output_dir: str = 'results') -> Dict:
    """
    运行消融实验（baseline vs anti_fixation）

    Args:
        backtest_func: 回测函数
        portfolio: 持仓数据
        start_date: 开始日期
        end_date: 结束日期
        output_dir: 输出目录

    Returns:
        对比结果
    """
    os.makedirs(output_dir, exist_ok=True)

    results = {
        'baseline': None,
        'improved': None,
        'comparison': {}
    }

    # Baseline（反固化关闭）
    logger.info("运行 Baseline (anti_fixation disabled)...")
    try:
        baseline_result = backtest_func(
            portfolio=portfolio,
            start_date=start_date,
            end_date=end_date,
            anti_fixation_enabled=False
        )
        results['baseline'] = {
            'jaccard_12m_avg': baseline_result.get('holdings_jaccard_12m_avg', 0),
            'stickiness': baseline_result.get('top_holdings_stickiness', 0),
            'annual_return': baseline_result.get('annual_return', 0),
            'sharpe': baseline_result.get('sharpe', 0),
            'max_drawdown': baseline_result.get('max_drawdown', 0),
            'monthly_turnover': baseline_result.get('turnover_avg_monthly_turnover', 0),
            'turnover_scale_avg_12m': baseline_result.get('turnover_audit_turnover_scale_avg', 1.0),
            'universe_size_avg': baseline_result.get('universe_size', 0),
            'selected_size_avg': baseline_result.get('selected_size', 0),
            'pass_kpi': baseline_result.get('passed', True)
        }
    except Exception as e:
        logger.warning(f"Baseline 运行失败: {e}")
        results['baseline'] = {'error': str(e)}

    # Improved（反固化开启）
    logger.info("运行 Improved (anti_fixation enabled)...")
    try:
        improved_result = backtest_func(
            portfolio=portfolio,
            start_date=start_date,
            end_date=end_date,
            anti_fixation_enabled=True
        )
        results['improved'] = {
            'jaccard_12m_avg': improved_result.get('holdings_jaccard_12m_avg', 0),
            'stickiness': improved_result.get('top_holdings_stickiness', 0),
            'annual_return': improved_result.get('annual_return', 0),
            'sharpe': improved_result.get('sharpe', 0),
            'max_drawdown': improved_result.get('max_drawdown', 0),
            'monthly_turnover': improved_result.get('turnover_avg_monthly_turnover', 0),
            'turnover_scale_avg_12m': improved_result.get('turnover_audit_turnover_scale_avg', 1.0),
            'universe_size_avg': improved_result.get('universe_size', 0),
            'selected_size_avg': improved_result.get('selected_size', 0),
            'pass_kpi': improved_result.get('passed', True)
        }
    except Exception as e:
        logger.warning(f"Improved 运行失败: {e}")
        results['improved'] = {'error': str(e)}

    # 生成对比报告
    _generate_ablation_report(results, output_dir)

    return results


def _generate_ablation_report(results: Dict, output_dir: str):
    """生成消融实验报告"""
    baseline = results.get('baseline', {})
    improved = results.get('improved', {})

    if 'error' in baseline or 'error' in improved:
        logger.warning("消融实验不完整，跳过报告生成")
        return

    # CSV输出
    csv_path = os.path.join(output_dir, 'list_fixation_ablation.csv')
    df = pd.DataFrame({
        'metric': ['jaccard_12m_avg', 'stickiness', 'annual_return', 'sharpe',
                   'max_drawdown', 'monthly_turnover', 'turnover_scale_avg_12m',
                   'universe_size_avg', 'selected_size_avg', 'pass_kpi'],
        'baseline': [
            baseline.get('jaccard_12m_avg', 0),
            baseline.get('stickiness', 0),
            baseline.get('annual_return', 0),
            baseline.get('sharpe', 0),
            baseline.get('max_drawdown', 0),
            baseline.get('monthly_turnover', 0),
            baseline.get('turnover_scale_avg_12m', 1.0),
            baseline.get('universe_size_avg', 0),
            baseline.get('selected_size_avg', 0),
            baseline.get('pass_kpi', True)
        ],
        'improved': [
            improved.get('jaccard_12m_avg', 0),
            improved.get('stickiness', 0),
            improved.get('annual_return', 0),
            improved.get('sharpe', 0),
            improved.get('max_drawdown', 0),
            improved.get('monthly_turnover', 0),
            improved.get('turnover_scale_avg_12m', 1.0),
            improved.get('universe_size_avg', 0),
            improved.get('selected_size_avg', 0),
            improved.get('pass_kpi', True)
        ]
    })
    df['delta'] = df['improved'] - df['baseline']
    df.to_csv(csv_path, index=False)

    # Markdown报告
    md_path = os.path.join(output_dir, 'list_fixation_ablation.md')
    report = f"""# 反固化消融实验报告

生成时间: {datetime.now().isoformat()}

## 对比结果

| 指标 | Baseline | Improved | 变化 |
|------|----------|----------|------|
| jaccard_12m_avg | {baseline.get('jaccard_12m_avg', 0):.2%} | {improved.get('jaccard_12m_avg', 0):.2%} | {(improved.get('jaccard_12m_avg', 0) - baseline.get('jaccard_12m_avg', 0)):.2%} |
| stickiness | {baseline.get('stickiness', 0):.2%} | {improved.get('stickiness', 0):.2%} | {(improved.get('stickiness', 0) - baseline.get('stickiness', 0)):.2%} |
| annual_return | {baseline.get('annual_return', 0):.2%} | {improved.get('annual_return', 0):.2%} | {(improved.get('annual_return', 0) - baseline.get('annual_return', 0)):.2%} |
| sharpe | {baseline.get('sharpe', 0):.2f} | {improved.get('sharpe', 0):.2f} | {(improved.get('sharpe', 0) - baseline.get('sharpe', 0)):.2f} |
| max_drawdown | {baseline.get('max_drawdown', 0):.2%} | {improved.get('max_drawdown', 0):.2%} | {(improved.get('max_drawdown', 0) - baseline.get('max_drawdown', 0)):.2%} |
| monthly_turnover | {baseline.get('monthly_turnover', 0):.2%} | {improved.get('monthly_turnover', 0):.2%} | {(improved.get('monthly_turnover', 0) - baseline.get('monthly_turnover', 0)):.2%} |

## 结论

"""
    jaccard_improved = improved.get('jaccard_12m_avg', 1) < baseline.get('jaccard_12m_avg', 1)
    sharpe_preserved = improved.get('sharpe', 0) >= baseline.get('sharpe', 0) * 0.95

    if jaccard_improved and sharpe_preserved:
        report += "✅ 反固化机制有效：Jaccard降低，收益未显著退化\n"
    elif jaccard_improved:
        report += "⚠️ 反固化机制部分有效：Jaccard降低，但收益有所下降\n"
    else:
        report += "❌ 反固化机制无效：Jaccard未降低\n"

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(report)

    logger.info(f"消融报告已生成: {md_path}")


if __name__ == '__main__':
    # 测试
    print("=" * 60)
    print("反固化机制测试")
    print("=" * 60)

    # 测试 SoftDiversify
    penalty = SoftDiversifyPenalty({
        'soft_diversify': {
            'frequency_penalty': {'max_penalty_bps': 50, 'lookback_periods': 5},
            'protection': {'preserve_top_n': 3}
        }
    })

    # 模拟数据
    scores = pd.Series({'A': 80, 'B': 75, 'C': 70, 'D': 65, 'E': 60})
    history = [
        {'A': 0.3, 'B': 0.25, 'C': 0.2, 'D': 0.15, 'E': 0.1},
        {'A': 0.28, 'B': 0.27, 'C': 0.2, 'D': 0.15, 'E': 0.1},
        {'A': 0.3, 'B': 0.25, 'C': 0.2, 'D': 0.15, 'F': 0.1},  # E被替换
    ]

    adjusted, details = penalty.apply_penalty(scores, history)
    print(f"原始得分: {scores.to_dict()}")
    print(f"调整后: {adjusted.to_dict()}")
    print(f"详情: {details}")
