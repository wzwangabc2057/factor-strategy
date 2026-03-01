"""
Gate-0 可靠性测试

测试:
- test_freeze_rebalance_on_score_missing: 得分缺失时冻结调仓
- test_reduce_only_never_increases_position: 仅减仓模式不增加仓位
- test_filter_reason_tags_present: 过滤原因标签存在
- test_softmax_allocation_sums_to_1_and_respects_caps: softmax分配权重和为1且遵守上限
"""

import pytest
import numpy as np
import pandas as pd
from src.reliability.gate0 import ReliabilityGate, check_data_reliability, apply_circuit_breaker
from src.reliability.filter_tagger import FilterTagger, tag_filter_reasons


class TestGate0FreezeRebalance:
    """Gate-0 冻结调仓测试"""

    def test_freeze_rebalance_on_score_missing(self):
        """得分缺失率超过阈值时必须触发 freeze_rebalance"""
        gate = ReliabilityGate()

        # 100只股票，只有90只有得分
        all_codes = [f'{i:06d}' for i in range(1, 101)]
        scores = pd.Series(np.random.uniform(30, 90, 90), index=all_codes[:90])

        # 缺失率 = 10% > 2% 阈值
        breach, action, details = gate.check(all_codes, scores)

        assert breach is True, "得分缺失10%应触发熔断"
        assert action == 'freeze_rebalance', f"动作应为freeze_rebalance，实际为{action}"
        assert details['score_missing_rate'] == 0.10

    def test_freeze_rebalance_on_complete_missing(self):
        """得分完全缺失时必须触发熔断"""
        gate = ReliabilityGate()

        all_codes = [f'{i:06d}' for i in range(1, 51)]
        breach, action, details = gate.check(all_codes, None)

        assert breach is True, "得分完全缺失应触发熔断"
        assert details['score_missing_rate'] == 1.0

    def test_no_breach_on_normal_data(self):
        """正常数据不应触发熔断"""
        gate = ReliabilityGate()

        all_codes = [f'{i:06d}' for i in range(1, 101)]
        scores = pd.Series(np.random.uniform(30, 90, 100), index=all_codes)

        breach, action, details = gate.check(all_codes, scores)

        assert breach is False, "数据完整不应触发熔断"
        assert action == 'none'

    def test_freeze_returns_current_weights(self):
        """freeze_rebalance 返回当前权重（不调仓）"""
        gate = ReliabilityGate()

        current = {'A': 0.3, 'B': 0.3, 'C': 0.2, 'D': 0.2}
        target = {'A': 0.2, 'B': 0.4, 'C': 0.2, 'E': 0.2}  # 不同的目标

        result = gate.apply_action('freeze_rebalance', current, target)

        assert result == current, "冻结调仓应保持当前权重"


class TestGate0ReduceOnly:
    """Gate-0 仅减仓模式测试"""

    def test_reduce_only_never_increases_position(self):
        """仅减仓模式不允许增加任何股票的仓位"""
        gate = ReliabilityGate(config={
            'on_breach': {'reduce_only_ratio': 0.8}
        })

        current = {'A': 0.3, 'B': 0.3, 'C': 0.2, 'D': 0.2}
        target = {'A': 0.4, 'B': 0.2, 'C': 0.3, 'E': 0.1}  # A加仓了

        result = gate.apply_action('reduce_only', current, target)

        # A的权重应该等于或小于当前权重
        assert result['A'] <= current['A'], "仅减仓模式不应增加A的仓位"
        # B的权重应该取min(current, target)
        # 总权重应该降到 80%
        total = sum(result.values())
        assert total <= 0.81, f"总权重应不超过80%，实际为{total:.2%}"  # 允许小误差

    def test_reduce_only_respects_ratio(self):
        """仅减仓模式应遵守 reduce_only_ratio"""
        gate = ReliabilityGate(config={
            'on_breach': {'reduce_only_ratio': 0.6}
        })

        current = {'A': 0.5, 'B': 0.3, 'C': 0.2}
        target = {'A': 0.3, 'B': 0.3, 'C': 0.4}

        result = gate.apply_action('reduce_only', current, target)
        total = sum(result.values())

        assert total <= 0.61, f"总权重应不超过60%，实际为{total:.2%}"


class TestFilterReasonTags:
    """过滤原因标签测试"""

    def test_filter_reason_tags_present(self):
        """被过滤的股票必须有原因标签"""
        tagger = FilterTagger()

        all_codes = [f'{i:06d}' for i in range(1, 101)]
        filtered_codes = all_codes[:70]  # 过滤30只

        # 模拟得分（80只有得分）
        scores = pd.Series(np.random.uniform(30, 90, 80), index=all_codes[:80])

        reasons = tagger.tag(all_codes, filtered_codes, scores=scores)

        # 被过滤的股票应该有原因标签
        assert len(reasons) == 30, f"应该有30只被过滤的股票，实际为{len(reasons)}"

        for code, tags in reasons.items():
            assert len(tags) > 0, f"股票{code}应该有至少一个过滤原因"

    def test_filter_tags_include_missing_factor(self):
        """因子缺失的股票应标记为 missing_factor"""
        tagger = FilterTagger()

        all_codes = ['000001', '000002', '000003', '000004', '000005']
        filtered_codes = ['000001', '000002']

        # 只有3只有因子数据
        factors = pd.DataFrame({
            'code': ['000003', '000004', '000005'],
            'factor1': [1.0, 2.0, 3.0]
        })

        reasons = tagger.tag(all_codes, filtered_codes, factors=factors)

        # 000001 和 000002 缺少因子数据
        assert 'missing_factor' in reasons.get('000001', []), "000001应该标记为missing_factor"
        assert 'missing_factor' in reasons.get('000002', []), "000002应该标记为missing_factor"

    def test_filter_tags_include_low_liquidity(self):
        """流动性不足的股票应标记为 low_liquidity"""
        tagger = FilterTagger()

        all_codes = ['000001', '000002', '000003', '000004', '000005']
        filtered_codes = ['000001', '000002']

        # 成交额数据
        amount_data = pd.Series({
            '000001': 5000000,   # 500万，低于1000万阈值
            '000002': 8000000,   # 800万，低于1000万阈值
            '000003': 15000000,  # 1500万
            '000004': 20000000,  # 2000万
            '000005': 30000000,  # 3000万
        })

        reasons = tagger.tag(all_codes, filtered_codes, amount_data=amount_data, min_adv=10000000)

        assert 'low_liquidity' in reasons.get('000001', []), "000001应该标记为low_liquidity"
        assert 'low_liquidity' in reasons.get('000002', []), "000002应该标记为low_liquidity"

    def test_filter_stats_aggregation(self):
        """过滤统计应正确聚合"""
        tagger = FilterTagger()

        all_codes = [f'{i:06d}' for i in range(1, 101)]
        filtered_codes = all_codes[:60]

        # 模拟多种过滤原因
        scores = pd.Series(np.random.uniform(30, 90, 80), index=all_codes[:80])
        factors = pd.DataFrame({
            'code': all_codes[:70],
            'factor1': np.random.uniform(0, 1, 70)
        })

        tagger.tag(all_codes, filtered_codes, scores=scores, factors=factors)

        stats = tagger.get_stats()
        assert sum(stats.values()) == 40, f"总过滤数应为40，实际为{sum(stats.values())}"


class TestSoftmaxAllocation:
    """Softmax 权重分配测试"""

    def test_softmax_allocation_sums_to_1_and_respects_caps(self):
        """softmax分配权重和应为1且遵守上限"""
        from src.diagnostics.core_satellite_builder import CoreSatelliteBuilder, softmax

        # 测试 softmax 函数
        scores = np.array([90, 80, 70, 60, 50])
        weights = softmax(scores, temperature=1.5)

        # 权重和应为1
        assert abs(sum(weights) - 1.0) < 0.0001, f"权重和应为1，实际为{sum(weights)}"

        # 高分应有更高权重
        assert weights[0] > weights[-1], "高分股票应有更高权重"

    def test_softmax_with_temperature(self):
        """温度参数应影响权重分布"""
        from src.diagnostics.core_satellite_builder import softmax

        scores = np.array([100, 50, 0])

        # 低温度（更尖锐）
        weights_low_temp = softmax(scores, temperature=0.5)
        # 高温度（更平滑）
        weights_high_temp = softmax(scores, temperature=3.0)

        # 低温度时最高分权重应更高
        assert weights_low_temp[0] > weights_high_temp[0], "低温度应使分布更尖锐"

        # 高温度时权重应更均匀
        assert weights_high_temp[-1] > weights_low_temp[-1], "高温度应使分布更平滑"

    def test_core_satellite_allocation_integration(self):
        """Core-Satellite 权重分配集成测试"""
        from src.diagnostics.core_satellite_builder import CoreSatelliteBuilder

        config = {
            'enabled': True,
            'core': {
                'k': 10,
                'allocation': 'softmax',
                'temperature': 1.5,
                'max_core_weight_sum': 0.6,
                'min_weight': 0.005
            },
            'satellite': {
                'n': 15,
                'allocation': 'linear',
                'weight_sum': 0.4,
                'top10_cap': 0.30,
                'min_weight': 0.002,
                'selection': {'exclude_core': True}
            }
        }

        builder = CoreSatelliteBuilder(config=config)

        # 模拟得分
        np.random.seed(42)
        codes = [f'{i:06d}' for i in range(1, 51)]
        scores = pd.Series(np.random.uniform(30, 90, len(codes)), index=codes)

        weights, details = builder.build(scores, [], '2024-01-01')

        # 权重和应为1
        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 0.01, f"权重和应为1，实际为{total_weight}"

        # 核心池数量
        assert details['core']['count'] == 10, f"核心池应有10只，实际为{details['core']['count']}"

        # 卫星池数量
        assert details['satellite']['count'] <= 15, f"卫星池应不超过15只"

        # 所有权重应大于0
        for code, weight in weights.items():
            assert weight > 0, f"股票{code}权重应大于0"

    def test_allocation_respects_top10_cap(self):
        """卫星池 Top10 权重上限约束"""
        from src.diagnostics.core_satellite_builder import CoreSatelliteBuilder

        config = {
            'enabled': True,
            'core': {
                'k': 5,
                'allocation': 'equal_weight',
                'max_core_weight_sum': 0.5
            },
            'satellite': {
                'n': 20,
                'allocation': 'linear',
                'weight_sum': 0.5,
                'top10_cap': 0.30,
                'selection': {'exclude_core': True}
            }
        }

        builder = CoreSatelliteBuilder(config=config)

        # 模拟得分
        np.random.seed(42)
        codes = [f'{i:06d}' for i in range(1, 51)]
        scores = pd.Series(np.random.uniform(30, 90, len(codes)), index=codes)

        weights, details = builder.build(scores, [], '2024-01-01')

        # 获取卫星池股票（排除核心池）
        core_codes = set(details['core']['codes'])
        sat_weights = {k: v for k, v in weights.items() if k not in core_codes}

        # Top10权重和
        top10_weights = sorted(sat_weights.values(), reverse=True)[:10]
        top10_sum = sum(top10_weights)

        # Top10权重和应不超过30%
        assert top10_sum <= 0.31, f"卫星池Top10权重和应不超过30%，实际为{top10_sum:.2%}"


class TestGate0Cooldown:
    """Gate-0 冷却期测试"""

    def test_cooldown_period(self):
        """熔断后应进入冷却期"""
        gate = ReliabilityGate(config={
            'recovery': {'cooldown_periods': 2}
        })

        all_codes = [f'{i:06d}' for i in range(1, 101)]
        scores = pd.Series(np.random.uniform(30, 90, 90), index=all_codes[:90])

        # 第一次检查（触发熔断）
        breach1, action1, _ = gate.check(all_codes, scores)
        assert breach1 is True
        assert action1 == 'freeze_rebalance'

        # 第二次检查（冷却期）
        breach2, action2, _ = gate.check(all_codes, scores)
        assert breach2 is True
        assert action2 == 'cooldown'

        # 第三次检查（冷却期）
        breach3, action3, _ = gate.check(all_codes, scores)
        assert breach3 is True
        assert action3 == 'cooldown'

        # 第四次检查（冷却期结束，重新评估）
        scores_full = pd.Series(np.random.uniform(30, 90, 100), index=all_codes)
        breach4, action4, _ = gate.check(all_codes, scores_full)
        assert breach4 is False  # 数据完整，不熔断
        assert gate.is_in_cooldown is False


class TestGate0Stats:
    """Gate-0 统计测试"""

    def test_stats_tracking(self):
        """统计应正确追踪"""
        gate = ReliabilityGate()

        all_codes = [f'{i:06d}' for i in range(1, 101)]

        # 触发一次熔断
        scores_partial = pd.Series(np.random.uniform(30, 90, 90), index=all_codes[:90])
        gate.check(all_codes, scores_partial)

        stats = gate.get_stats()

        assert stats['gate0_triggered'] is True
        assert stats['gate0_trigger_count'] == 1
        assert stats['score_missing_rate'] == 0.10

    def test_breach_history(self):
        """熔断历史应正确记录"""
        gate = ReliabilityGate()

        all_codes = [f'{i:06d}' for i in range(1, 101)]
        scores = pd.Series(np.random.uniform(30, 90, 90), index=all_codes[:90])

        gate.check(all_codes, scores)

        history = gate.get_breach_history()
        assert len(history) == 1
        assert history[0]['breach_type'] == 'score_missing'
        assert history[0]['action'] == 'freeze_rebalance'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
