"""
固化逻辑分析器 v3.0

目标：复现"固化名单产生过程"，给出可解释的因果链

输出：
- 固化名单出现频率表
- 固化贡献度拆解（按模块）
- Top2 根因及证据

使用方法:
    from src.diagnostics.fixation_analyzer import FixationAnalyzer

    analyzer = FixationAnalyzer()
    attribution = analyzer.analyze(backtest_history)
    analyzer.generate_report(attribution, 'results/fixation_attribution.md')
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import Counter
import logging

logger = logging.getLogger(__name__)


@dataclass
class FixationContribution:
    """固化贡献度"""
    name: str
    contribution_pct: float       # 贡献百分比 (0-100)
    evidence: str                 # 证据描述
    impact: str                   # high/medium/low
    remediation: str              # 修复建议


@dataclass
class FixationAttribution:
    """固化归因结果"""
    # TopN 出现频率
    top_n_frequency: Dict[str, Dict]  # {code: {frequency, first_period, last_period, avg_rank}}

    # 贡献度拆解
    contributions: List[FixationContribution]

    # 根因
    root_causes: List[Dict]

    # 关键指标
    metrics: Dict

    # 元数据
    analysis_time: str
    total_periods: int


class FixationAnalyzer:
    """固化逻辑分析器"""

    def __init__(self, config: Dict = None):
        """
        初始化

        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.top_n = self.config.get('top_n', 30)

        # 追踪数据
        self.period_data = []

    def record_period(self,
                      period_id: str,
                      universe: List[str],
                      filtered: List[str],
                      scores: pd.Series,
                      final_weights: Dict[str, float],
                      turnover_scale: float = 1.0,
                      filter_reasons: Dict[str, List[str]] = None):
        """
        记录单期数据

        Args:
            period_id: 期次标识
            universe: 候选股票列表
            filtered: 过滤后可交易列表
            scores: 因子得分 Series (index=code)
            final_weights: 最终权重
            turnover_scale: 换手缩放因子
            filter_reasons: 过滤原因 {code: [reason1, reason2]}
        """
        self.period_data.append({
            'period_id': period_id,
            'universe': universe,
            'filtered': filtered,
            'scores': scores.to_dict() if isinstance(scores, pd.Series) else scores,
            'final_weights': final_weights,
            'turnover_scale': turnover_scale,
            'filter_reasons': filter_reasons or {},
            'timestamp': datetime.now().isoformat()
        })

    def analyze(self,
                holdings_history: List[Dict[str, float]] = None) -> FixationAttribution:
        """
        分析固化原因

        Args:
            holdings_history: 历史持仓（可选，用于验证）

        Returns:
            固化归因结果
        """
        if len(self.period_data) < 2:
            logger.warning("数据期数不足，至少需要2期")
            return self._empty_attribution()

        contributions = []

        # Contribution-1: Universe/Filter Narrowing
        filter_contrib = self._analyze_filter_contribution()
        contributions.append(filter_contrib)

        # Contribution-2: Score Dominance
        score_contrib = self._analyze_score_dominance()
        contributions.append(score_contrib)

        # Contribution-3: Weight Concentration
        weight_contrib = self._analyze_weight_concentration()
        contributions.append(weight_contrib)

        # Contribution-4: Turnover Lock-in
        turnover_contrib = self._analyze_turnover_lockin()
        contributions.append(turnover_contrib)

        # 排序贡献度
        contributions.sort(key=lambda x: x.contribution_pct, reverse=True)

        # 计算TopN出现频率
        top_n_frequency = self._calculate_top_n_frequency()

        # 提取根因（Top2）
        root_causes = []
        for i, contrib in enumerate(contributions[:2]):
            if contrib.contribution_pct > 20:
                root_causes.append({
                    'rank': i + 1,
                    'cause': contrib.name,
                    'contribution': contrib.contribution_pct,
                    'evidence': contrib.evidence,
                    'remediation': contrib.remediation
                })

        # 汇总指标
        metrics = self._calculate_summary_metrics()

        return FixationAttribution(
            top_n_frequency=top_n_frequency,
            contributions=contributions,
            root_causes=root_causes,
            metrics=metrics,
            analysis_time=datetime.now().isoformat(),
            total_periods=len(self.period_data)
        )

    def _analyze_filter_contribution(self) -> FixationContribution:
        """分析过滤导致的固化"""
        universe_sizes = [len(p['universe']) for p in self.period_data]
        filtered_sizes = [len(p['filtered']) for p in self.period_data]

        avg_universe = np.mean(universe_sizes)
        avg_filtered = np.mean(filtered_sizes)
        reduction_rate = (avg_universe - avg_filtered) / avg_universe if avg_universe > 0 else 0

        # 收集过滤原因
        all_reasons = []
        for p in self.period_data:
            all_reasons.extend(p.get('filter_reasons', {}).values())

        # 估算贡献度：缩减率越高，贡献越大
        contribution = min(100, reduction_rate * 100)

        if reduction_rate > 0.5:
            impact = "high"
            evidence = f"过滤导致候选池缩减{reduction_rate:.1%} ({avg_universe:.0f}->{avg_filtered:.0f})"
            remediation = "放宽过滤条件，扩大股票池"
        elif reduction_rate > 0.3:
            impact = "medium"
            evidence = f"过滤导致候选池缩减{reduction_rate:.1%}"
            remediation = "检查过滤条件是否过严"
        else:
            impact = "low"
            evidence = f"过滤缩减率{reduction_rate:.1%}，影响较小"
            remediation = "保持现有过滤逻辑"

        return FixationContribution(
            name="Universe/Filter Narrowing",
            contribution_pct=contribution,
            evidence=evidence,
            impact=impact,
            remediation=remediation
        )

    def _analyze_score_dominance(self) -> FixationContribution:
        """分析分数优势导致的固化"""
        top_margins = []
        top_vs_mid_gaps = []

        for p in self.period_data:
            scores = p['scores']
            if not scores:
                continue

            sorted_scores = sorted(scores.values(), reverse=True)

            # Top10 vs Top11-20 的分差
            if len(sorted_scores) >= 20:
                top10_avg = np.mean(sorted_scores[:10])
                top20_avg = np.mean(sorted_scores[10:20])
                gap = top10_avg - top20_avg
                top_margins.append(gap)

            # Top30 vs Top31-60 的分差
            if len(sorted_scores) >= 60:
                top30_avg = np.mean(sorted_scores[:30])
                top60_avg = np.mean(sorted_scores[30:60])
                gap = top30_avg - top60_avg
                top_vs_mid_gaps.append(gap)

        avg_margin = np.mean(top_margins) if top_margins else 0
        avg_gap = np.mean(top_vs_mid_gaps) if top_vs_mid_gaps else 0

        # 贡献度：分差越大，固化越严重
        contribution = min(100, avg_margin * 5)  # 假设1分差=5%贡献

        if avg_margin > 10:
            impact = "high"
            evidence = f"Top10与Top20平均分差{avg_margin:.1f}分，分数优势明显"
            remediation = "降低非线性加成(top_boost)，平滑分数差异"
        elif avg_margin > 5:
            impact = "medium"
            evidence = f"Top10与Top20平均分差{avg_margin:.1f}分"
            remediation = "适当降低加成强度"
        else:
            impact = "low"
            evidence = f"分数分布较均匀，分差{avg_margin:.1f}分"
            remediation = "保持现有因子权重"

        return FixationContribution(
            name="Score Dominance",
            contribution_pct=contribution,
            evidence=evidence,
            impact=impact,
            remediation=remediation
        )

    def _analyze_weight_concentration(self) -> FixationContribution:
        """分析权重集中导致的固化"""
        top10_ratios = []
        max_weight_hits = []

        for p in self.period_data:
            weights = p['final_weights']
            if not weights:
                continue

            sorted_weights = sorted(weights.values(), reverse=True)
            total_weight = sum(weights.values())

            # Top10权重占比
            top10_sum = sum(sorted_weights[:10])
            top10_ratio = top10_sum / total_weight if total_weight > 0 else 0
            top10_ratios.append(top10_ratio)

            # 触顶次数（达到单票上限）
            max_weight = 0.08  # 假设上限8%
            hits = sum(1 for w in sorted_weights if w >= max_weight * 0.95)
            max_weight_hits.append(hits)

        avg_top10_ratio = np.mean(top10_ratios) if top10_ratios else 0
        avg_hits = np.mean(max_weight_hits) if max_weight_hits else 0

        # 贡献度
        contribution = avg_top10_ratio * 50 + avg_hits * 5

        if avg_top10_ratio > 0.6:
            impact = "high"
            evidence = f"Top10权重占比{avg_top10_ratio:.1%}，平均{avg_hits:.0f}只触顶"
            remediation = "降低单票上限，增加持仓分散度"
        elif avg_top10_ratio > 0.45:
            impact = "medium"
            evidence = f"Top10权重占比{avg_top10_ratio:.1%}"
            remediation = "适当降低权重集中度"
        else:
            impact = "low"
            evidence = f"权重分布较分散，Top10占比{avg_top10_ratio:.1%}"
            remediation = "保持现有权重约束"

        return FixationContribution(
            name="Weight Concentration",
            contribution_pct=min(100, contribution),
            evidence=evidence,
            impact=impact,
            remediation=remediation
        )

    def _analyze_turnover_lockin(self) -> FixationContribution:
        """分析换手锁仓导致的固化"""
        turnover_scales = [p['turnover_scale'] for p in self.period_data]

        avg_scale = np.mean(turnover_scales)
        low_scale_periods = sum(1 for s in turnover_scales if s < 0.8)
        low_scale_ratio = low_scale_periods / len(turnover_scales)

        # 贡献度：缩放因子越低，锁仓越严重
        contribution = (1 - avg_scale) * 100 + low_scale_ratio * 30

        if avg_scale < 0.6:
            impact = "high"
            evidence = f"平均换手缩放因子{avg_scale:.2f}，{low_scale_ratio:.0%}期低于0.8"
            remediation = "放宽换手限制(0.08->0.10)，减少锁仓"
        elif avg_scale < 0.85:
            impact = "medium"
            evidence = f"平均换手缩放因子{avg_scale:.2f}"
            remediation = "适当放宽换手限制"
        else:
            impact = "low"
            evidence = f"换手限制影响较小，缩放因子{avg_scale:.2f}"
            remediation = "保持现有换手限制"

        return FixationContribution(
            name="Turnover Lock-in",
            contribution_pct=min(100, contribution),
            evidence=evidence,
            impact=impact,
            remediation=remediation
        )

    def _calculate_top_n_frequency(self) -> Dict[str, Dict]:
        """计算TopN出现频率"""
        all_top_n = []

        for p in self.period_data:
            weights = p['final_weights']
            sorted_codes = sorted(weights.keys(), key=lambda x: weights.get(x, 0), reverse=True)
            top_n = sorted_codes[:self.top_n]
            all_top_n.append(set(top_n))

        # 统计每只股票的出现频率
        code_stats = {}
        total_periods = len(all_top_n)

        for period_idx, top_set in enumerate(all_top_n):
            for code in top_set:
                if code not in code_stats:
                    code_stats[code] = {
                        'frequency': 0,
                        'first_period': period_idx,
                        'last_period': period_idx,
                        'ranks': []
                    }
                code_stats[code]['frequency'] += 1
                code_stats[code]['last_period'] = period_idx

        # 计算频率百分比
        for code in code_stats:
            code_stats[code]['frequency_pct'] = code_stats[code]['frequency'] / total_periods

        # 按频率排序
        sorted_stats = dict(sorted(code_stats.items(),
                                    key=lambda x: x[1]['frequency'],
                                    reverse=True))

        return sorted_stats

    def _calculate_summary_metrics(self) -> Dict:
        """计算汇总指标"""
        turnover_scales = [p['turnover_scale'] for p in self.period_data]
        universe_sizes = [len(p['universe']) for p in self.period_data]
        filtered_sizes = [len(p['filtered']) for p in self.period_data]

        return {
            'total_periods': len(self.period_data),
            'avg_universe_size': np.mean(universe_sizes),
            'avg_filtered_size': np.mean(filtered_sizes),
            'avg_turnover_scale': np.mean(turnover_scales),
            'low_turnover_periods': sum(1 for s in turnover_scales if s < 0.8),
        }

    def _empty_attribution(self) -> FixationAttribution:
        """返回空归因结果"""
        return FixationAttribution(
            top_n_frequency={},
            contributions=[],
            root_causes=[],
            metrics={},
            analysis_time=datetime.now().isoformat(),
            total_periods=0
        )

    def generate_report(self,
                        attribution: FixationAttribution,
                        output_path: str) -> str:
        """
        生成固化归因报告

        Args:
            attribution: 归因结果
            output_path: 输出路径

        Returns:
            报告内容
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        report = f"""# 固化逻辑归因分析报告

生成时间: {attribution.analysis_time}
分析期数: {attribution.total_periods}

## 1. 固化名单 Top{self.top_n} 出现频率

| 排名 | 股票代码 | 出现次数 | 出现频率 | 首次期次 | 最后期次 |
|------|----------|----------|----------|----------|----------|
"""
        # Top30 出现频率
        for i, (code, stats) in enumerate(list(attribution.top_n_frequency.items())[:30]):
            report += f"| {i+1} | {code} | {stats['frequency']} | {stats['frequency_pct']:.1%} | {stats['first_period']} | {stats['last_period']} |\n"

        report += f"""
## 2. 固化贡献度拆解

| 模块 | 贡献度 | 影响程度 | 证据 |
|------|--------|----------|------|
"""
        for contrib in attribution.contributions:
            report += f"| {contrib.name} | {contrib.contribution_pct:.1f}% | {contrib.impact} | {contrib.evidence} |\n"

        report += f"""
## 3. Top2 根因分析

"""
        for rc in attribution.root_causes:
            report += f"""### 根因 {rc['rank']}: {rc['cause']}

- **贡献度**: {rc['contribution']:.1f}%
- **证据**: {rc['evidence']}
- **修复建议**: {rc['remediation']}

"""

        if not attribution.root_causes:
            report += "无明显根因，固化程度较轻。\n"

        report += f"""
## 4. 关键指标汇总

| 指标 | 值 |
|------|-----|
| 总期数 | {attribution.metrics.get('total_periods', 0)} |
| 平均候选池规模 | {attribution.metrics.get('avg_universe_size', 0):.0f} |
| 平均可交易池规模 | {attribution.metrics.get('avg_filtered_size', 0):.0f} |
| 平均换手缩放因子 | {attribution.metrics.get('avg_turnover_scale', 0):.2f} |
| 低换手期数 | {attribution.metrics.get('low_turnover_periods', 0)} |

## 5. 修复路径建议

根据根因分析，建议按以下优先级修复：

| 根因 | 修复路径 | 预期效果 |
|------|----------|----------|
| Universe/Filter Narrowing | 放宽过滤条件，扩大股票池 | 降低固化15-25% |
| Score Dominance | 降低top_boost，平滑分数 | 降低固化10-20% |
| Weight Concentration | 降低单票上限，增加分散 | 降低固化5-15% |
| Turnover Lock-in | 放宽换手限制 | 降低固化10-20% |

## 6. 与双层结构的关系

- **固化逻辑** → Core Layer（低频稳健的核心池）
- **轮动与分散** → Satellite Layer（高频轮动的卫星池）
- **惩罚机制** → Satellite Layer Optional（作为微调器）

建议启用 `portfolio_structure.enabled: true` 采用双层结构，
将固化逻辑转化为核心池选择机制，同时保持卫星层的灵活性。
"""

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)

        # 同时输出JSON
        json_path = output_path.replace('.md', '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'top_n_frequency': {k: {**v, 'frequency_pct': f"{v['frequency_pct']:.2%}"}
                                    for k, v in list(attribution.top_n_frequency.items())[:50]},
                'contributions': [asdict(c) for c in attribution.contributions],
                'root_causes': attribution.root_causes,
                'metrics': attribution.metrics,
                'analysis_time': attribution.analysis_time
            }, f, indent=2, ensure_ascii=False)

        logger.info(f"固化归因报告已生成: {output_path}")

        return report


def analyze_fixation_causes(backtest_history: List[Dict],
                             output_dir: str = 'results') -> FixationAttribution:
    """
    便捷函数：分析固化原因

    Args:
        backtest_history: 回测历史数据
        output_dir: 输出目录

    Returns:
        归因结果
    """
    analyzer = FixationAnalyzer()

    for i, period in enumerate(backtest_history):
        analyzer.record_period(
            period_id=period.get('period_id', f'period_{i}'),
            universe=period.get('universe', []),
            filtered=period.get('filtered', []),
            scores=period.get('scores', pd.Series()),
            final_weights=period.get('final_weights', {}),
            turnover_scale=period.get('turnover_scale', 1.0),
            filter_reasons=period.get('filter_reasons', {})
        )

    attribution = analyzer.analyze()
    analyzer.generate_report(attribution, os.path.join(output_dir, 'fixation_attribution.md'))

    return attribution


if __name__ == '__main__':
    # 测试
    print("=" * 60)
    print("固化逻辑分析器测试")
    print("=" * 60)

    analyzer = FixationAnalyzer()

    # 模拟5期数据
    np.random.seed(42)
    base_codes = [f'{i:06d}' for i in range(1, 101)]

    for i in range(5):
        # 模拟得分
        scores = pd.Series(
            np.random.uniform(30, 90, len(base_codes)),
            index=base_codes
        )

        # Top10 得分更高（模拟固化）
        for j in range(10):
            scores[f'{j+1:06d}'] = 80 + np.random.uniform(0, 10)

        # 模拟权重
        weights = {}
        for code in base_codes[:30]:
            weights[code] = 1/30

        analyzer.record_period(
            period_id=f'period_{i}',
            universe=base_codes,
            filtered=base_codes[:80],  # 过滤20%
            scores=scores,
            final_weights=weights,
            turnover_scale=0.7 if i > 2 else 1.0
        )

    attribution = analyzer.analyze()

    print(f"\nTop2 根因:")
    for rc in attribution.root_causes:
        print(f"  {rc['cause']}: {rc['contribution']:.1f}%")

    print(f"\n贡献度拆解:")
    for c in attribution.contributions:
        print(f"  {c.name}: {c.contribution_pct:.1f}% ({c.impact})")
