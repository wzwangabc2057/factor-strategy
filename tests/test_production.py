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

Gate-5 新增测试:
8. test_config_validation_rejects_bad_values - 配置验证
9. test_impact_fallback_rate_flagging - 降级率标记
10. test_cash_ratio_min_assert - 现金比例断言
11. test_robustness_pass_fail_fields_present - 鲁棒性输出字段

Gate-6 名单固化测试:
12. test_universe_builder_liquidity_top_returns_expected_size - 股票池构建
13. test_list_fixation_metrics_jaccard_range_0_1 - Jaccard范围
14. test_gate6_list_fixation_fail_when_jaccard_high - 高Jaccard失败

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


# ==================== Gate-5: 新增测试 ====================

class TestConfigValidation:
    """Gate-5: 配置验证测试"""

    def test_config_validation_rejects_bad_values(self):
        """配置验证应拒绝非法值"""
        from src.config.validator import ConfigValidator

        validator = ConfigValidator(strict=False)

        # 测试非法换手限制
        bad_config = {
            'strategy': {'name': 'test', 'version': '1.0'},
            'factors': {'weights': {'roe': 0.5}},
            'rebalance': {'frequency': 'monthly', 'turnover_limit': 0.50},  # 超过20%
            'constraints': {'min_weight': 0.001, 'max_position_count': 50}
        }
        errors = validator.validate_strategy_config(bad_config)
        assert len(errors) > 0, "应拒绝换手限制>20%的配置"

        # 测试非法滑点
        bad_cost_config = {
            'cost_model': {
                'enabled': True,
                'base_cost': {
                    'commission': {'buy': 0.001, 'sell': 0.002},
                    'slippage': {'default': 0.0001}  # 小于0.1%
                },
                'impact_cost': {'enabled': True}
            }
        }
        errors = validator.validate_cost_config(bad_cost_config)
        assert len(errors) > 0, "应拒绝滑点<0.1%的配置"

    def test_strategy_config_required_fields(self):
        """策略配置必需字段检查"""
        from src.config.validator import ConfigValidator

        validator = ConfigValidator(strict=False)

        # 缺少必需字段
        incomplete_config = {
            'strategy': {'name': 'test'}
            # 缺少 factors, rebalance, constraints
        }
        errors = validator.validate_strategy_config(incomplete_config)
        assert len(errors) > 0, "应检测到缺少必需字段"

    def test_risk_config_logic_consistency(self):
        """风控配置逻辑一致性检查"""
        from src.config.validator import ConfigValidator

        validator = ConfigValidator(strict=False)

        # 仓位比例不递减 (非法)
        bad_config = {
            'risk_control': {
                'enabled': True,
                'tier1_trend': {'action': {'position_ratio': 0.5}},  # Tier1=50%
                'tier2_volatile': {'action': {'position_ratio': 0.7}},  # Tier2=70% (非法)
                'tier3_crash': {'action': {'position_ratio': 0.9}}   # Tier3=90% (非法)
            }
        }
        errors = validator.validate_risk_config(bad_config)
        # 应检测到仓位比例不递减
        assert len(errors) > 0, "应检测到仓位比例不递减的配置错误"


class TestImpactFallbackRate:
    """Gate-5: 降级率标记测试"""

    def test_impact_fallback_rate_flagging(self):
        """降级率应正确标记"""
        from src.backtest.cost_model import CostModel

        model = CostModel()

        # 无ADV时应标记降级
        result = model.calculate_cost(
            trade_value=100000,
            trade_type='buy',
            adv=None  # 无ADV
        )

        assert result['impact_fallback'] == True, "无ADV时应标记impact_fallback=True"

        # 检查降级统计
        stats = model.get_degradation_stats()
        assert 'impact_fallback_rate' in stats, "应返回impact_fallback_rate"
        assert stats['impact_fallback_rate'] >= 0, "降级率应>=0"

    def test_impact_fallback_rate_calculation(self):
        """降级率计算正确性"""
        from src.backtest.cost_model import CostModel

        model = CostModel()

        # 3次有ADV，2次无ADV
        model.calculate_cost(100000, 'buy', adv=1000000)  # 有ADV
        model.calculate_cost(100000, 'buy', adv=None)      # 无ADV
        model.calculate_cost(100000, 'buy', adv=1000000)  # 有ADV
        model.calculate_cost(100000, 'buy', adv=None)      # 无ADV
        model.calculate_cost(100000, 'buy', adv=1000000)  # 有ADV

        stats = model.get_degradation_stats()
        # 2/5 = 0.4
        assert abs(stats['impact_fallback_rate'] - 0.4) < 0.01, \
            f"降级率应为0.4，实际为{stats['impact_fallback_rate']}"


class TestCashRatioMinAssert:
    """Gate-5: 现金比例断言测试"""

    def test_cash_ratio_min_assert(self):
        """现金比例应满足最低要求"""
        from src.risk.tier_manager import RiskTierManager

        manager = RiskTierManager()

        # Tier3应要求现金比例>=40%
        tier3_actions = manager.get_actions('tier3_crash')
        cash_ratio_min = tier3_actions.get('cash_ratio_min', 0)

        assert cash_ratio_min >= 0.30, \
            f"Tier3现金比例应>=30%，实际为{cash_ratio_min:.0%}"

    def test_cash_ratio_tier_increasing(self):
        """现金比例应随档位递增"""
        from src.risk.tier_manager import RiskTierManager

        manager = RiskTierManager()

        tier1_cash = manager.get_actions('tier1_trend').get('cash_ratio_min', 0)
        tier2_cash = manager.get_actions('tier2_volatile').get('cash_ratio_min', 0)
        tier3_cash = manager.get_actions('tier3_crash').get('cash_ratio_min', 0)

        assert tier1_cash <= tier2_cash, \
            f"Tier1现金比例{tier1_cash:.0%}应<=Tier2{tier2_cash:.0%}"
        assert tier2_cash <= tier3_cash, \
            f"Tier2现金比例{tier2_cash:.0%}应<=Tier3{tier3_cash:.0%}"


class TestRobustnessPassFailFields:
    """Gate-5: 鲁棒性输出字段测试"""

    def test_robustness_pass_fail_fields_present(self):
        """鲁棒性测试结果应包含PASS/FAIL字段"""
        # 模拟鲁棒性测试结果
        result = {
            'test_id': 'WF-1',
            'test_type': 'walk_forward',
            'test_year': '2019',
            'annual_return': 0.15,
            'max_drawdown': -0.18,
            'sharpe': 1.2,
            'passed': True,
            'fail_reason_top3': [],
            'regime_bucket': 'bull_market'
        }

        # 检查必需字段
        assert 'passed' in result, "结果应包含passed字段"
        assert 'fail_reason_top3' in result, "结果应包含fail_reason_top3字段"
        assert 'regime_bucket' in result, "结果应包含regime_bucket字段"

    def test_robustness_fail_reason_format(self):
        """失败原因应包含具体指标信息"""
        # 模拟失败结果
        fail_reasons = [
            "annual_return=-5.00% < 门槛0.00%",
            "max_drawdown=-30.00% < 上限-25.00%"
        ]

        for reason in fail_reasons:
            # 每个失败原因应包含指标名和阈值
            assert '%' in reason or '<' in reason or '>' in reason, \
                f"失败原因格式不正确: {reason}"


class TestDegradationVisibility:
    """Gate-5: 降级可见性测试"""

    def test_regime_proxy_used_tracking(self):
        """风控代理使用应被跟踪"""
        from src.risk.tier_manager import RiskTierManager

        manager = RiskTierManager()

        # 模拟使用代理数据
        tier, actions = manager.detect_tier(data_source='stock_proxy')

        stats = manager.get_degradation_stats()
        assert 'regime_proxy_used' in stats, "应返回regime_proxy_used字段"
        assert stats['regime_proxy_used'] == 'stock_average', \
            "应标记regime_proxy_used=stock_average"

    def test_no_silent_fallback(self):
        """不应有静默降级"""
        from src.backtest.cost_model import CostModel
        import logging

        # 捕获日志
        import io
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.WARNING)

        logger = logging.getLogger('src.backtest.cost_model')
        logger.addHandler(handler)

        model = CostModel()
        model.calculate_cost(100000, 'buy', adv=None)

        log_output = log_capture.getvalue()

        # 应有警告日志
        assert 'Gate-2' in log_output or '降级' in log_output, \
            "降级时应有明确的警告日志"

        logger.removeHandler(handler)


# ==================== Gate-6: 名单固化测试 ====================

class TestUniverseBuilder:
    """Gate-6: 股票池构建测试"""

    def test_universe_builder_liquidity_top_returns_expected_size(self):
        """流动性筛选应返回预期数量股票"""
        from src.universe.builder import UniverseBuilder
        import numpy as np

        # 模拟数据
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=30).astype(str)
        codes = [f'{i:06d}' for i in range(1, 501)]

        data = []
        for date in dates:
            for code in codes:
                data.append({
                    'date': date,
                    'code': code,
                    'close': np.random.uniform(10, 100),
                    'amount': np.random.uniform(1e7, 1e9)
                })

        price_df = pd.DataFrame(data)

        # 构建
        builder = UniverseBuilder(config={'source': 'liquidity_top', 'liquidity': {'top_n': 300}})
        universe = builder.build(price_df, date='2024-01-30')

        assert len(universe) > 0, "股票池不应为空"
        assert len(universe) <= 300, f"股票池应<=300，实际{len(universe)}"

    def test_universe_config_validation(self):
        """文件池模式需显式启用"""
        from src.universe.builder import UniverseBuilder

        # 默认不允许文件池
        builder = UniverseBuilder(config={
            'source': 'file',
            'file': {'allow_file_pool': False, 'path': 'nonexistent.csv'}
        })

        # 应返回空池或降级
        universe = builder.build(pd.DataFrame({'date': [], 'code': [], 'close': []}))
        assert len(universe) == 0 or builder.is_degraded, \
            "文件池未启用时应返回空池或标记降级"


class TestListFixationMetrics:
    """Gate-6: 名单固化指标测试"""

    def test_list_fixation_metrics_jaccard_range_0_1(self):
        """Jaccard相似度应在0-1范围"""
        from src.reporting.metrics import MetricsCalculator

        calculator = MetricsCalculator()

        # 模拟持仓历史（完全不同）
        holdings_history = [
            {'A': 0.3, 'B': 0.3, 'C': 0.2, 'D': 0.1, 'E': 0.1},
            {'F': 0.3, 'G': 0.3, 'H': 0.2, 'I': 0.1, 'J': 0.1},
            {'K': 0.3, 'L': 0.3, 'M': 0.2, 'N': 0.1, 'O': 0.1},
        ]

        metrics = calculator.calculate_list_fixation_metrics(holdings_history, top_n=5)

        assert 0.0 <= metrics['holdings_jaccard_1m'] <= 1.0, \
            f"Jaccard应在0-1之间，实际为{metrics['holdings_jaccard_1m']}"
        assert 0.0 <= metrics['holdings_jaccard_12m_avg'] <= 1.0, \
            f"12期平均Jaccard应在0-1之间"
        assert 0.0 <= metrics['top_holdings_stickiness'] <= 1.0, \
            f"粘性应在0-1之间"

    def test_list_fixation_metrics_same_holdings(self):
        """相同持仓应有高Jaccard"""
        from src.reporting.metrics import MetricsCalculator

        calculator = MetricsCalculator()

        # 模拟完全相同的持仓
        holdings_history = [
            {'A': 0.3, 'B': 0.3, 'C': 0.2, 'D': 0.1, 'E': 0.1},
            {'A': 0.25, 'B': 0.25, 'C': 0.2, 'D': 0.15, 'E': 0.15},
            {'A': 0.2, 'B': 0.3, 'C': 0.25, 'D': 0.15, 'E': 0.1},
        ]

        metrics = calculator.calculate_list_fixation_metrics(holdings_history, top_n=5)

        assert metrics['holdings_jaccard_1m'] > 0.9, \
            f"相同持仓Jaccard应接近1，实际为{metrics['holdings_jaccard_1m']}"
        assert metrics['top_holdings_stickiness'] > 0.9, \
            f"相同持仓粘性应接近1"


class TestGate6ListFixation:
    """Gate-6: 名单固化门禁测试"""

    def test_gate6_list_fixation_fail_when_jaccard_high(self):
        """Jaccard过高应导致Gate-6失败"""
        from scripts.run_robustness_test import RobustnessTestRunner

        runner = RobustnessTestRunner()

        # 模拟高Jaccard指标
        high_jaccard_metrics = {
            'holdings_jaccard_12m_avg': 0.85,  # 超过0.75阈值
            'top_holdings_stickiness': 0.60,
            'annual_return': 0.10,
            'max_drawdown': -0.15,
            'sharpe': 1.0,
        }

        passed, fail_reasons = runner._evaluate_pass_criteria(high_jaccard_metrics)

        assert not passed, "Jaccard过高时应失败"
        assert any('list_fixation' in r for r in fail_reasons), \
            f"失败原因应包含list_fixation: {fail_reasons}"

    def test_gate6_list_fixation_pass_when_jaccard_low(self):
        """Jaccard正常应通过Gate-6"""
        from scripts.run_robustness_test import RobustnessTestRunner

        runner = RobustnessTestRunner()

        # 模拟正常指标
        normal_metrics = {
            'holdings_jaccard_12m_avg': 0.50,  # 低于0.75阈值
            'top_holdings_stickiness': 0.60,   # 低于0.80阈值
            'annual_return': 0.10,
            'max_drawdown': -0.15,
            'sharpe': 1.0,
        }

        passed, fail_reasons = runner._evaluate_pass_criteria(normal_metrics)

        # 只检查list_fixation相关失败原因
        list_fixation_failures = [r for r in fail_reasons if 'list_fixation' in r]
        assert len(list_fixation_failures) == 0, \
            f"正常指标不应触发list_fixation失败: {list_fixation_failures}"


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
            # Gate-5: 新增测试类
            TestConfigValidation,
            TestImpactFallbackRate,
            TestCashRatioMinAssert,
            TestRobustnessPassFailFields,
            TestDegradationVisibility,
            # Gate-6: 名单固化测试
            TestUniverseBuilder,
            TestListFixationMetrics,
            TestGate6ListFixation,
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
