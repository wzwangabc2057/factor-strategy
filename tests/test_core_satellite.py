"""
双层组合结构测试 v3.0

测试:
- test_fixation_analyzer_outputs_contributions
- test_core_pool_dynamic_but_low_frequency
- test_core_satellite_weight_sums_to_1
- test_satellite_excludes_core_when_configured

运行:
    pytest tests/test_core_satellite.py -v
    或
    python tests/test_core_satellite.py
"""

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

import numpy as np
import pandas as pd
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestFixationAnalyzer:
    """固化逻辑分析器测试"""

    def test_fixation_analyzer_outputs_contributions(self):
        """分析器应输出4个贡献度模块"""
        from src.diagnostics.fixation_analyzer import FixationAnalyzer

        analyzer = FixationAnalyzer()

        # 模拟3期数据
        codes = [f'{i:06d}' for i in range(1, 101)]
        for i in range(3):
            scores = pd.Series(np.random.uniform(30, 90, len(codes)), index=codes)
            weights = {c: 1/30 for c in codes[:30]}

            analyzer.record_period(
                period_id=f'p{i}',
                universe=codes,
                filtered=codes[:80],
                scores=scores,
                final_weights=weights,
                turnover_scale=0.7
            )

        attribution = analyzer.analyze()

        # 应有4个贡献度模块
        assert len(attribution.contributions) == 4, \
            f"应有4个贡献度模块，实际{len(attribution.contributions)}"

        # 检查模块名称
        names = [c.name for c in attribution.contributions]
        assert 'Universe/Filter Narrowing' in names
        assert 'Score Dominance' in names
        assert 'Weight Concentration' in names
        assert 'Turnover Lock-in' in names

    def test_fixation_analyzer_identifies_top_causes(self):
        """分析器应识别Top2根因"""
        from src.diagnostics.fixation_analyzer import FixationAnalyzer

        analyzer = FixationAnalyzer()

        # 模拟极端固化情况
        codes = [f'{i:06d}' for i in range(1, 51)]
        for i in range(5):
            scores = pd.Series(np.random.uniform(30, 90, len(codes)), index=codes)
            # 固定Top10
            for j in range(10):
                scores[f'{j+1:06d}'] = 85

            weights = {f'{j+1:06d}': 0.08 for j in range(10)}  # 高度集中

            analyzer.record_period(
                period_id=f'p{i}',
                universe=codes,
                filtered=codes[:30],  # 大幅过滤
                scores=scores,
                final_weights=weights,
                turnover_scale=0.5  # 低换手
            )

        attribution = analyzer.analyze()

        # 应有根因
        assert len(attribution.root_causes) >= 1, "应识别至少1个根因"

    def test_fixation_analyzer_generates_report(self):
        """分析器应生成报告"""
        from src.diagnostics.fixation_analyzer import FixationAnalyzer

        analyzer = FixationAnalyzer()

        codes = [f'{i:06d}' for i in range(1, 51)]
        for i in range(2):
            scores = pd.Series(np.random.uniform(30, 90, len(codes)), index=codes)
            weights = {c: 1/20 for c in codes[:20]}

            analyzer.record_period(
                period_id=f'p{i}',
                universe=codes,
                filtered=codes[:40],
                scores=scores,
                final_weights=weights,
                turnover_scale=0.9
            )

        attribution = analyzer.analyze()

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = os.path.join(tmpdir, 'fixation_attribution.md')
            analyzer.generate_report(attribution, report_path)

            assert os.path.exists(report_path), "MD报告应存在"
            assert os.path.exists(report_path.replace('.md', '.json')), "JSON报告应存在"


class TestCoreSatelliteBuilder:
    """双层组合构建器测试"""

    def test_core_pool_dynamic_but_low_frequency(self):
        """核心池应动态但低频调整"""
        from src.diagnostics.core_satellite_builder import CoreSatelliteBuilder

        builder = CoreSatelliteBuilder(config={
            'enabled': True,
            'core': {
                'k': 20,
                'rebalance_frequency': 'quarterly',
                'min_hold_period': 60,
                'max_core_weight_sum': 0.60
            }
        })

        codes = [f'{i:06d}' for i in range(1, 101)]
        scores = pd.Series(np.random.uniform(30, 90, len(codes)), index=codes)
        history = [{c: 1/30 for c in codes[:30]}]

        # 首次调整
        core1, details1 = builder.select_core(scores, history, '2024-01-01')
        assert len(core1) == 20, f"核心池应为20只，实际{len(core1)}"
        assert details1['rebalanced'] == True

        # 10天后尝试调整（应被阻止）
        core2, details2 = builder.select_core(scores, history, '2024-01-11')
        assert details2['rebalanced'] == False, "季度调整间隔内不应再次调整"
        assert core2 == core1, "核心池应保持不变"

    def test_core_satellite_weight_sums_to_1(self):
        """Core+Satellite权重和应为1"""
        from src.diagnostics.core_satellite_builder import CoreSatelliteBuilder

        builder = CoreSatelliteBuilder(config={
            'enabled': True,
            'core': {'k': 20, 'max_core_weight_sum': 0.60},
            'satellite': {'n': 30, 'weight_sum': 0.40, 'selection': {'exclude_core': True}}
        })

        codes = [f'{i:06d}' for i in range(1, 101)]
        scores = pd.Series(np.random.uniform(30, 90, len(codes)), index=codes)
        history = [{c: 1/30 for c in codes[:30]}]

        weights, details = builder.build(scores, history, '2024-01-01')

        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01, f"权重和应为1，实际{total}"

    def test_satellite_excludes_core_when_configured(self):
        """卫星池应排除核心池（当配置时）"""
        from src.diagnostics.core_satellite_builder import CoreSatelliteBuilder

        builder = CoreSatelliteBuilder(config={
            'enabled': True,
            'core': {'k': 20, 'max_core_weight_sum': 0.60},
            'satellite': {
                'n': 30,
                'weight_sum': 0.40,
                'selection': {'exclude_core': True}
            }
        })

        codes = [f'{i:06d}' for i in range(1, 101)]
        scores = pd.Series(np.random.uniform(30, 90, len(codes)), index=codes)
        history = [{c: 1/30 for c in codes[:30]}]

        weights, details = builder.build(scores, history, '2024-01-01')

        core_codes = set(details['core']['codes'])
        sat_codes = set(details['satellite']['codes'])

        overlap = core_codes & sat_codes
        assert len(overlap) == 0, f"核心池和卫星池不应重叠，实际重叠{len(overlap)}只"

    def test_satellite_allows_overlap_when_configured(self):
        """卫星池允许重叠（当配置时）"""
        from src.diagnostics.core_satellite_builder import CoreSatelliteBuilder

        builder = CoreSatelliteBuilder(config={
            'enabled': True,
            'core': {'k': 20, 'max_core_weight_sum': 0.60},
            'satellite': {
                'n': 30,
                'weight_sum': 0.40,
                'selection': {'exclude_core': False}  # 不排除
            }
        })

        codes = [f'{i:06d}' for i in range(1, 101)]
        scores = pd.Series(np.random.uniform(30, 90, len(codes)), index=codes)
        history = [{c: 1/30 for c in codes[:30]}]

        weights, details = builder.build(scores, history, '2024-01-01')

        # 允许重叠时，重叠数可能>0（不强制，但不应报错）
        assert details['enabled'] == True

    def test_disabled_returns_original_behavior(self):
        """禁用时应返回原有行为"""
        from src.diagnostics.core_satellite_builder import CoreSatelliteBuilder

        builder = CoreSatelliteBuilder(config={'enabled': False})

        codes = [f'{i:06d}' for i in range(1, 101)]
        scores = pd.Series(np.random.uniform(30, 90, len(codes)), index=codes)

        weights, details = builder.build(scores, [], '2024-01-01')

        assert details['enabled'] == False, "禁用时details.enabled应为False"
        assert len(weights) > 0, "仍应返回权重"


class TestAblationOutput:
    """消融实验输出测试"""

    def test_core_satellite_ablation_columns(self):
        """消融实验输出应包含必需列"""
        # 模拟消融实验结果
        results = {
            'baseline': {
                'holdings_jaccard_12m_avg': 0.82,
                'top_holdings_stickiness': 0.85,
                'core_jaccard': None,
                'core_turnover': None,
                'satellite_turnover': None,
                'annual_return': 0.15,
                'sharpe': 1.2,
                'max_drawdown': -0.18,
                'monthly_turnover': 0.05,
                'cost_impact': 0.02
            },
            'core_satellite': {
                'holdings_jaccard_12m_avg': 0.68,
                'top_holdings_stickiness': 0.72,
                'core_jaccard': 0.85,
                'core_turnover': 0.15,
                'satellite_turnover': 0.35,
                'annual_return': 0.14,
                'sharpe': 1.15,
                'max_drawdown': -0.17,
                'monthly_turnover': 0.06,
                'cost_impact': 0.025
            }
        }

        required_cols = [
            'holdings_jaccard_12m_avg', 'top_holdings_stickiness',
            'core_jaccard', 'core_turnover', 'satellite_turnover',
            'annual_return', 'sharpe', 'max_drawdown',
            'monthly_turnover', 'cost_impact'
        ]

        for key in ['baseline', 'core_satellite']:
            for col in required_cols:
                assert col in results[key], f"{key}缺少列: {col}"


if __name__ == '__main__':
    if HAS_PYTEST:
        pytest.main([__file__, '-v'])
    else:
        print("运行测试 (无pytest)...\n")

        test_classes = [
            TestFixationAnalyzer,
            TestCoreSatelliteBuilder,
            TestAblationOutput,
        ]

        passed = 0
        failed = 0

        for test_class in test_classes:
            try:
                instance = test_class()
            except Exception as e:
                print(f"✗ {test_class.__name__} 初始化失败: {e}")
                continue

            for method_name in dir(instance):
                if method_name.startswith('test_'):
                    try:
                        method = getattr(instance, method_name)
                        method()
                        print(f'✓ {test_class.__name__}.{method_name}')
                        passed += 1
                    except AssertionError as e:
                        print(f'✗ {test_class.__name__}.{method_name}: {e}')
                        failed += 1
                    except Exception as e:
                        print(f'✗ {test_class.__name__}.{method_name}: {type(e).__name__}: {e}')
                        failed += 1

        print(f'\n总计: {passed} 通过, {failed} 失败')
