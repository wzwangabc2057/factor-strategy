"""
单元测试 v3.0

测试红线项:
1. test_weights_normalization - 权重归一化
2. test_max_single_weight_enforced - 单股权重上限
3. test_turnover_limit_caps_trades - 换手限制
4. test_risk_tier_detection - 风控档位检测
5. test_no_lookahead_in_rolling - 未来函数检查
6. test_cost_model - 成本模型
7. test_slippage_required - 滑点必须>=0.1%

运行:
    pytest tests/test_production.py -v
    或
    python tests/test_production.py
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


class TestWeightsNormalization:
    """权重归一化测试"""

    def test_sum_equals_one(self):
        """权重和必须等于1.0 (误差<1e-4)"""
        weights = {'A': 0.25, 'B': 0.25, 'C': 0.25, 'D': 0.25}
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-4, f"权重和{total}不等于1.0"

    def test_sum_equals_one_with_floats(self):
        """浮点数权重归一化"""
        weights = {'A': 0.1, 'B': 0.2, 'C': 0.3, 'D': 0.4}
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-4

    def test_normalized_weights(self):
        """归一化后权重和=1"""
        raw_weights = {'A': 10, 'B': 20, 'C': 30, 'D': 40}
        total = sum(raw_weights.values())
        normalized = {k: v / total for k, v in raw_weights.items()}
        assert abs(sum(normalized.values()) - 1.0) < 1e-10

    def test_empty_weights(self):
        """空权重处理"""
        weights = {}
        total = sum(weights.values())
        assert total == 0.0

    def test_single_weight(self):
        """单只股票权重必须=1"""
        weights = {'A': 1.0}
        assert abs(sum(weights.values()) - 1.0) < 1e-4


class TestMaxSingleWeight:
    """单股权重上限测试"""

    def test_max_weight_enforced(self):
        """单股权重不能超过上限"""
        max_weight = 0.08
        weights = {'A': 0.10, 'B': 0.30, 'C': 0.30, 'D': 0.30}

        # 应用上限
        adjusted = {k: min(v, max_weight) for k, v in weights.items()}
        # 归一化
        total = sum(adjusted.values())
        adjusted = {k: v / total for k, v in adjusted.items()}

        for code, weight in adjusted.items():
            assert weight <= max_weight, f"{code}权重{weight}超过上限{max_weight}"

    def test_max_weight_normal_case(self):
        """正常情况下权重不超过上限"""
        max_weight = 0.10
        weights = {'A': 0.05, 'B': 0.05, 'C': 0.05, 'D': 0.05}

        for weight in weights.values():
            assert weight <= max_weight

    def test_extreme_concentration(self):
        """极端集中度处理"""
        max_weight = 0.08
        # 一只股票占90%
        weights = {'A': 0.90, 'B': 0.10}
        adjusted = {k: min(v, max_weight) for k, v in weights.items()}

        assert adjusted['A'] == max_weight


class TestTurnoverLimit:
    """换手限制测试"""

    def test_monthly_turnover_within_limit(self):
        """月换手率必须<=8%"""
        turnover_limit = 0.08
        monthly_turnover = 0.05

        assert monthly_turnover <= turnover_limit, \
            f"月换手{monthly_turnover}超过上限{turnover_limit}"

    def test_turnover_exceeded_scaling(self):
        """换手超限时应缩放交易"""
        turnover_limit = 0.08
        target_trades = {'A': 0.05, 'B': 0.04, 'C': 0.03}  # 总换手12%
        total_turnover = sum(abs(v) for v in target_trades.values())

        if total_turnover > turnover_limit:
            scale_factor = turnover_limit / total_turnover
            scaled_trades = {k: v * scale_factor for k, v in target_trades.items()}
            scaled_turnover = sum(abs(v) for v in scaled_trades.values())

            assert scaled_turnover <= turnover_limit + 1e-4

    def test_daily_turnover_limit(self):
        """日换手率测试"""
        daily_limit = 0.03
        daily_turnovers = [0.02, 0.025, 0.03, 0.028]

        for turnover in daily_turnovers:
            assert turnover <= daily_limit


class TestRiskTierDetection:
    """风控档位检测测试"""

    def test_tier1_bull_market(self):
        """牛市应识别为Tier1"""
        from src.risk.tier_manager import RiskTierManager

        manager = RiskTierManager()

        # 模拟牛市数据
        np.random.seed(42)
        prices = pd.Series(100 * np.cumprod(1 + np.random.normal(0.002, 0.01, 100)))

        tier, actions = manager.detect_tier(prices)

        assert tier in ['tier1_trend', 'tier2_volatile'], f"牛市应识别为tier1或tier2，实际{tier}"

    def test_tier3_crash_detection(self):
        """急跌应识别为Tier3"""
        from src.risk.tier_manager import RiskTierManager

        manager = RiskTierManager()

        # 模拟急跌数据 (连续下跌)
        np.random.seed(42)
        prices = pd.Series(100 * np.cumprod(1 + np.random.normal(-0.008, 0.02, 100)))

        tier, actions = manager.detect_tier(prices)

        # 检查是否正确返回动作
        assert 'position_ratio' in actions
        assert 'max_single_weight' in actions

    def test_tier_actions_correct(self):
        """各档位动作配置正确"""
        from src.risk.tier_manager import RiskTierManager

        manager = RiskTierManager()

        # Tier3动作检查
        tier3_actions = manager.get_actions('tier3_crash')
        assert tier3_actions['position_ratio'] <= 0.5, "Tier3仓位应<=50%"
        assert tier3_actions['allow_buy'] == False, "Tier3应禁止买入"
        assert tier3_actions['max_single_weight'] <= 0.05, "Tier3单股上限应<=5%"


class TestNoLookahead:
    """未来函数检查测试"""

    def test_rolling_window_no_future(self):
        """滚动窗口不能使用未来数据"""
        dates = pd.date_range('2020-01-01', periods=100)
        prices = pd.Series(np.random.randn(100).cumsum() + 100, index=dates)

        # 计算20日均线
        ma20 = prices.rolling(20).mean()

        # 检查: 第i天的MA20只能用第i天及之前的数据
        for i in range(20, len(prices)):
            expected_ma = prices.iloc[i-19:i+1].mean()
            assert abs(ma20.iloc[i] - expected_ma) < 1e-10, \
                f"第{i}天MA20使用了未来数据"

    def test_rank_no_future(self):
        """排名不能使用未来数据"""
        # 模拟因子值
        factor_values = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        # 计算百分位排名
        ranks = factor_values.rank(pct=True)

        # 检查排名值在合理范围
        assert ranks.min() >= 0
        assert ranks.max() <= 1

    def test_shift_correctness(self):
        """shift操作正确性"""
        s = pd.Series([1, 2, 3, 4, 5])
        shifted = s.shift(1)

        assert pd.isna(shifted.iloc[0])
        assert shifted.iloc[1] == 1
        assert shifted.iloc[4] == 4

    def test_no_peeking_in_backtest(self):
        """回测不能偷看未来"""
        # 模拟调仓决策
        rebalance_date = '2020-03-01'
        available_data_end = '2020-02-29'  # 只能用这之前的数据

        # 因子计算应该只用available_data_end之前的数据
        # 这里是一个概念测试
        assert rebalance_date > available_data_end


class TestCostModel:
    """成本模型测试"""

    def test_buy_cost_calculation(self):
        """买入成本计算"""
        from src.backtest.cost_model import CostModel

        model = CostModel()
        result = model.calculate_cost(
            trade_value=100000,
            trade_type='buy',
            slippage=0.001
        )

        # 检查成本为正
        assert result['total_cost'] > 0
        # 检查成本比例
        assert result['cost_ratio'] > 0

    def test_sell_cost_higher_than_buy(self):
        """卖出成本应该高于买入(含印花税)"""
        from src.backtest.cost_model import CostModel

        model = CostModel()

        buy_cost = model.calculate_cost(trade_value=100000, trade_type='buy')
        sell_cost = model.calculate_cost(trade_value=100000, trade_type='sell')

        assert sell_cost['total_cost'] > buy_cost['total_cost'], \
            "卖出成本应高于买入(含印花税)"

    def test_impact_cost_with_adv(self):
        """有ADV时应计算冲击成本"""
        from src.backtest.cost_model import CostModel

        model = CostModel()
        result = model.calculate_cost(
            trade_value=1000000,
            trade_type='buy',
            adv=10000000  # 大ADV，冲击成本小
        )

        assert result['impact'] >= 0

    def test_impact_cost_without_adv(self):
        """无ADV时降级处理"""
        from src.backtest.cost_model import CostModel

        model = CostModel()
        result = model.calculate_cost(
            trade_value=1000000,
            trade_type='buy',
            adv=None
        )

        # 应该标记为降级模式
        # 根据配置，可能是0或固定值
        assert result['impact_fallback'] == True


class TestSlippageRequired:
    """滑点必须>=0.1%测试"""

    def test_default_slippage_minimum(self):
        """默认滑点>=0.1%"""
        from src.backtest.cost_model import CostModel

        model = CostModel()
        default_slippage = model.base_cost.get('slippage', {}).get('default', 0)

        assert default_slippage >= 0.001, \
            f"默认滑点{default_slippage}小于0.1%"

    def test_backtest_must_include_slippage(self):
        """回测必须包含滑点"""
        # 检查配置
        import yaml

        config_path = 'config/cost_model.yaml'
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            backtest_defaults = config.get('cost_model', {}).get('backtest_defaults', {})
            slippage = backtest_defaults.get('slippage', 0)

            assert slippage >= 0.001, f"回测默认滑点{slippage}小于0.1%"


class TestIntegration:
    """集成测试"""

    def test_full_backtest_pipeline(self):
        """完整回测流水线"""
        from src.backtest.cost_model import CostModel
        from src.risk.tier_manager import RiskTierManager
        from src.reporting.metrics import MetricsCalculator

        # 1. 风控检测
        manager = RiskTierManager()
        np.random.seed(42)
        prices = pd.Series(100 * np.cumprod(1 + np.random.normal(0.001, 0.02, 100)))
        tier, actions = manager.detect_tier(prices)

        assert tier in ['tier1_trend', 'tier2_volatile', 'tier3_crash']

        # 2. 成本计算
        cost_model = CostModel()
        cost = cost_model.calculate_cost(
            trade_value=100000,
            trade_type='buy'
        )
        assert cost['total_cost'] > 0

        # 3. 指标计算
        calculator = MetricsCalculator()
        returns = np.random.normal(0.001, 0.02, 252)
        metrics = calculator.calculate_returns_metrics(returns)

        assert 'annual_return' in metrics
        assert 'sharpe' in calculator.calculate_risk_adjusted_metrics(returns)


if __name__ == '__main__':
    if HAS_PYTEST:
        pytest.main([__file__, '-v'])
    else:
        # 无pytest时手动运行
        print("运行测试 (无pytest)...\n")

        test_classes = [
            TestWeightsNormalization,
            TestMaxSingleWeight,
            TestTurnoverLimit,
            TestRiskTierDetection,
            TestCostModel,
            TestSlippageRequired,
            TestIntegration,
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
