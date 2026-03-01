"""
反固化机制测试 v3.0

Gate-6 改进闭环测试:
- test_soft_diversify_penalty_monotonic - 频率越高扣分越多
- test_adaptive_thresholds_change_only_when_triggered - 仅触发时调整
- test_ablation_outputs_exist_and_have_required_columns - 消融输出验证
- test_gate6_severity_levels - 严重程度分级

运行:
    pytest tests/test_anti_fixation.py -v
    或
    python tests/test_anti_fixation.py
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

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSoftDiversify:
    """机制A: 软分散约束测试"""

    def test_soft_diversify_penalty_monotonic(self):
        """频率越高扣分越多（单调性）"""
        from src.optimization.anti_fixation import SoftDiversifyPenalty

        penalty = SoftDiversifyPenalty({
            'soft_diversify': {
                'frequency_penalty': {
                    'max_penalty_bps': 50,
                    'lookback_periods': 5,
                    'threshold_percentile': 0.6,
                    'method': 'linear'
                },
                'protection': {'preserve_top_n': 0, 'min_score_to_penalize': 0}
            }
        })

        # 模拟历史：A出现5/5次，B出现4/5次，C出现2/5次
        history = [
            {'A': 0.3, 'B': 0.25, 'C': 0.2, 'D': 0.1},
            {'A': 0.3, 'B': 0.25, 'C': 0.2, 'D': 0.1},
            {'A': 0.3, 'B': 0.25, 'C': 0.0, 'E': 0.2},
            {'A': 0.3, 'B': 0.25, 'F': 0.2, 'D': 0.1},
            {'A': 0.3, 'B': 0.0, 'C': 0.2, 'D': 0.1},
        ]

        scores = pd.Series({'A': 80, 'B': 80, 'C': 80, 'D': 80, 'E': 80, 'F': 80})
        adjusted, details = penalty.apply_penalty(scores, history)

        # A频率最高(1.0)，惩罚应该最大
        # B频率次之(0.8)，惩罚次之
        # C频率(0.6)，刚好在阈值
        penalty_a = 80 - adjusted.get('A', 80)
        penalty_b = 80 - adjusted.get('B', 80)

        # 验证单调性
        assert penalty_a >= penalty_b, \
            f"A(频率1.0)的惩罚应>=B(频率0.8): {penalty_a} vs {penalty_b}"

    def test_soft_diversify_preserves_top_n(self):
        """保护TopN不受惩罚"""
        from src.optimization.anti_fixation import SoftDiversifyPenalty

        penalty = SoftDiversifyPenalty({
            'soft_diversify': {
                'frequency_penalty': {'max_penalty_bps': 50, 'threshold_percentile': 0.5},
                'protection': {'preserve_top_n': 2, 'min_score_to_penalize': 0}
            }
        })

        history = [
            {'A': 0.4, 'B': 0.3, 'C': 0.2, 'D': 0.1},
            {'A': 0.4, 'B': 0.3, 'C': 0.2, 'D': 0.1},
        ]

        scores = pd.Series({'A': 80, 'B': 80, 'C': 80, 'D': 80})
        current_weights = {'A': 0.4, 'B': 0.3, 'C': 0.2, 'D': 0.1}

        adjusted, details = penalty.apply_penalty(scores, history, current_weights)

        # A和B应该被保护，不受惩罚
        assert adjusted['A'] == 80, "Top1 A应被保护"
        assert adjusted['B'] == 80, "Top2 B应被保护"

    def test_soft_diversify_respects_min_score(self):
        """低于阈值分数的股票不受惩罚"""
        from src.optimization.anti_fixation import SoftDiversifyPenalty

        penalty = SoftDiversifyPenalty({
            'soft_diversify': {
                'frequency_penalty': {'max_penalty_bps': 50, 'threshold_percentile': 0.5},
                'protection': {'preserve_top_n': 0, 'min_score_to_penalize': 70}
            }
        })

        history = [{'A': 0.5, 'B': 0.5}, {'A': 0.5, 'B': 0.5}]
        scores = pd.Series({'A': 80, 'B': 50})  # B分数低于70

        adjusted, _ = penalty.apply_penalty(scores, history)

        # B不应被惩罚
        assert adjusted['B'] == 50, "低于阈值的B不应被惩罚"


class TestAdaptiveThreshold:
    """机制B: 自适应阈值测试"""

    def test_adaptive_thresholds_change_only_when_triggered(self):
        """阈值仅在触发时调整"""
        from src.optimization.anti_fixation import AdaptiveThresholdAdjuster

        adjuster = AdaptiveThresholdAdjuster({
            'adaptive_threshold': {
                'enabled': True,
                'trigger': {'jaccard_trigger': 0.75, 'hysteresis': 0.05},
                'adjustments': {
                    'top_boost_decay': {'from': 1.30, 'to': 1.15}
                }
            }
        })

        current_params = {'top_10_pct_boost': 1.30}

        # 低于触发阈值，不应调整
        new_params, details = adjuster.check_and_adapt(0.65, 0.60, current_params)
        assert new_params['top_10_pct_boost'] == 1.30, "低于阈值不应调整"
        assert details['adjustment_applied'] == False

        # 高于触发阈值，应调整
        new_params, details = adjuster.check_and_adapt(0.85, 0.70, current_params)
        assert new_params['top_10_pct_boost'] < 1.30, "高于阈值应调整"

    def test_adaptive_threshold_hysteresis(self):
        """滞后机制防止抖动"""
        from src.optimization.anti_fixation import AdaptiveThresholdAdjuster

        adjuster = AdaptiveThresholdAdjuster({
            'adaptive_threshold': {
                'enabled': True,
                'trigger': {'jaccard_trigger': 0.75, 'hysteresis': 0.10},
                'recovery': {'cooldown_periods': 2, 'gradual_recovery': True}
            }
        })

        current_params = {'top_10_pct_boost': 1.30}

        # 触发
        adjuster.check_and_adapt(0.82, 0.60, current_params)
        assert adjuster.adjustment_level > 0, "应进入调整状态"

        # 在滞后区间内，不应立即恢复
        adjuster.check_and_adapt(0.76, 0.60, current_params)  # 76% 在滞后区间
        # 需要冷却期

    def test_adaptive_threshold_disabled_by_default(self):
        """默认禁用"""
        from src.optimization.anti_fixation import AdaptiveThresholdAdjuster

        adjuster = AdaptiveThresholdAdjuster({'adaptive_threshold': {'enabled': False}})

        current_params = {'top_10_pct_boost': 1.30}
        new_params, details = adjuster.check_and_adapt(0.90, 0.90, current_params)

        assert new_params['top_10_pct_boost'] == 1.30, "禁用时应保持原参数"


class TestGate6SeverityLevels:
    """Gate-6 严重程度分级测试"""

    def test_gate6_severity_levels(self):
        """测试严重程度分级"""
        from src.optimization.anti_fixation import AntiFixationEngine

        engine = AntiFixationEngine(config={'enabled': True})

        # Pass 级别
        severity, reasons = engine.get_severity_level(0.60, 0.60)
        assert severity == 'pass', f"Jaccard 0.60 应为 pass，实际为 {severity}"

        # Warning 级别
        severity, reasons = engine.get_severity_level(0.72, 0.60)
        assert severity == 'warning', f"Jaccard 0.72 应为 warning"

        # Soft fail 级别
        severity, reasons = engine.get_severity_level(0.78, 0.60)
        assert severity == 'soft_fail', f"Jaccard 0.78 应为 soft_fail"

        # Hard fail 级别
        severity, reasons = engine.get_severity_level(0.88, 0.60)
        assert severity == 'hard_fail', f"Jaccard 0.88 应为 hard_fail"

    def test_gate6_stickiness_severity(self):
        """测试粘性严重程度"""
        from src.optimization.anti_fixation import AntiFixationEngine

        engine = AntiFixationEngine(config={'enabled': True})

        # Stickiness 触发 hard fail
        severity, reasons = engine.get_severity_level(0.60, 0.92)
        assert severity == 'hard_fail', f"Stickiness 0.92 应为 hard_fail"

    def test_gate6_recommendation_generation(self):
        """测试建议生成"""
        from src.optimization.anti_fixation import AntiFixationEngine

        engine = AntiFixationEngine(config={'enabled': True})

        # Hard fail 应有建议
        rec = engine.get_recommendation('hard_fail', {'universe_size': 300})
        assert len(rec['actions']) > 0, "hard_fail 应有行动建议"
        assert 'config_changes' in rec


class TestAblationOutputs:
    """消融实验输出测试"""

    def test_ablation_outputs_exist_and_have_required_columns(self):
        """消融输出应包含必需列"""
        import tempfile
        import os

        from src.optimization.anti_fixation import run_ablation_study

        # 模拟回测函数
        def mock_backtest(portfolio, start_date, end_date, anti_fixation_enabled=False):
            base = {
                'holdings_jaccard_12m_avg': 0.82 if not anti_fixation_enabled else 0.68,
                'top_holdings_stickiness': 0.85 if not anti_fixation_enabled else 0.72,
                'annual_return': 0.15,
                'sharpe': 1.2,
                'max_drawdown': -0.18,
                'turnover_avg_monthly_turnover': 0.05,
                'turnover_audit_turnover_scale_avg': 0.8,
                'universe_size': 600,
                'selected_size': 50,
                'passed': not anti_fixation_enabled  # 开启反固化后通过
            }
            return base

        with tempfile.TemporaryDirectory() as tmpdir:
            portfolio = pd.DataFrame({'code': ['000001'], 'weight': [1.0]})
            results = run_ablation_study(
                mock_backtest,
                portfolio,
                '2020-01-01',
                '2024-12-31',
                output_dir=tmpdir
            )

            # 检查输出文件
            csv_path = os.path.join(tmpdir, 'list_fixation_ablation.csv')
            md_path = os.path.join(tmpdir, 'list_fixation_ablation.md')

            assert os.path.exists(csv_path), f"CSV文件应存在: {csv_path}"
            assert os.path.exists(md_path), f"MD文件应存在: {md_path}"

            # 检查必需列
            df = pd.read_csv(csv_path)
            required_cols = ['jaccard_12m_avg', 'stickiness', 'annual_return',
                             'sharpe', 'max_drawdown', 'monthly_turnover']
            for col in required_cols:
                assert col in df['metric'].values, f"缺少指标: {col}"


class TestDiagnosisReport:
    """诊断报告测试"""

    def test_diagnosis_generates_report(self):
        """诊断应生成报告"""
        import tempfile
        import os

        from src.optimization.anti_fixation import diagnose_list_fixation

        # 创建模拟CSV
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, 'robustness_summary.csv')

            # 模拟数据
            df = pd.DataFrame({
                'test_id': ['WF-1', 'WF-2', 'WF-3'],
                'passed': [True, False, False],
                'fail_reason_top3': [
                    '[]',
                    "['list_fixation:jaccard=0.82', 'sharpe<0.5']",
                    "['list_fixation:stickiness=0.90']"
                ],
                'holdings_jaccard_12m_avg': [0.65, 0.82, 0.78],
                'top_holdings_stickiness': [0.60, 0.75, 0.90],
                'universe_size': [600, 300, 400],
                'turnover_scale_avg_12m': [0.9, 0.5, 0.6]
            })
            df.to_csv(csv_path, index=False)

            # 运行诊断
            output_path = os.path.join(tmpdir, 'diagnosis.md')
            diagnosis = diagnose_list_fixation(csv_path, output_path)

            # 检查结果
            assert 'gate6_stats' in diagnosis
            assert 'root_causes' in diagnosis
            assert os.path.exists(output_path), "诊断报告应存在"


if __name__ == '__main__':
    if HAS_PYTEST:
        pytest.main([__file__, '-v'])
    else:
        print("运行测试 (无pytest)...\n")

        test_classes = [
            TestSoftDiversify,
            TestAdaptiveThreshold,
            TestGate6SeverityLevels,
            TestAblationOutputs,
            TestDiagnosisReport,
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
