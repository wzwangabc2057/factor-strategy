"""
增强版多因子策略 v2.0
核心改进：
1. 支持12个因子的动态权重调整
2. 分层权重调整（更精细的分位档）
3. 绩优股回撤买入机制
4. 市场环境自适应
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedStrategyV2:
    """增强版多因子策略"""

    # 权重调整系数 - 更精细的10档
    WEIGHT_MULTIPLIERS_10 = {
        'top_5_pct': 2.5,       # Top 5%: 大幅超配
        'top_10_pct': 2.0,      # 5-10%
        'top_20_pct': 1.6,      # 10-20%
        'top_30_pct': 1.3,      # 20-30%
        'top_40_pct': 1.1,      # 30-40%
        'mid_40_60_pct': 1.0,   # 40-60%: 保持原权重
        'bottom_40_pct': 0.9,   # 60-70%
        'bottom_30_pct': 0.75,  # 70-80%
        'bottom_20_pct': 0.5,   # 80-90%
        'bottom_10_pct': 0.3,   # Bottom 10%: 大幅低配
    }

    # 简化版5档
    WEIGHT_MULTIPLIERS_5 = {
        'top_20_pct': 1.8,
        'top_40_pct': 1.3,
        'mid_40_60_pct': 1.0,
        'bottom_40_pct': 0.7,
        'bottom_20_pct': 0.5,
    }

    def __init__(self,
                 weight_multipliers: Dict[str, float] = None,
                 max_weight: float = 0.05,
                 min_weight: float = 0.0005,
                 use_10_tier: bool = True,
                 enable_reversal_boost: bool = True):
        """
        初始化策略

        Args:
            weight_multipliers: 自定义权重调整系数
            max_weight: 单股最大权重 (默认5%)
            min_weight: 单股最小权重 (默认0.05%)
            use_10_tier: 是否使用10档权重调整（更精细）
            enable_reversal_boost: 是否启用绩优股回撤加成
        """
        if weight_multipliers is None:
            if use_10_tier:
                self.weight_multipliers = self.WEIGHT_MULTIPLIERS_10.copy()
            else:
                self.weight_multipliers = self.WEIGHT_MULTIPLIERS_5.copy()
        else:
            self.weight_multipliers = weight_multipliers

        self.max_weight = max_weight
        self.min_weight = min_weight
        self.use_10_tier = use_10_tier
        self.enable_reversal_boost = enable_reversal_boost

        logger.info("="*60)
        logger.info("增强版策略 V2.0 初始化")
        logger.info("="*60)
        logger.info(f"权重档位: {'10档' if use_10_tier else '5档'}")
        logger.info(f"单股权重范围: {min_weight:.4%} ~ {max_weight:.2%}")
        logger.info(f"绩优股回撤加成: {'启用' if enable_reversal_boost else '禁用'}")

    def get_multiplier_by_percentile(self, percentile: float) -> float:
        """
        根据百分位获取权重调整系数

        Args:
            percentile: 得分百分位 (0-1)

        Returns:
            调整系数
        """
        if self.use_10_tier:
            if percentile >= 0.95:
                return self.weight_multipliers['top_5_pct']
            elif percentile >= 0.90:
                return self.weight_multipliers['top_10_pct']
            elif percentile >= 0.80:
                return self.weight_multipliers['top_20_pct']
            elif percentile >= 0.70:
                return self.weight_multipliers['top_30_pct']
            elif percentile >= 0.60:
                return self.weight_multipliers['top_40_pct']
            elif percentile >= 0.40:
                return self.weight_multipliers['mid_40_60_pct']
            elif percentile >= 0.30:
                return self.weight_multipliers['bottom_40_pct']
            elif percentile >= 0.20:
                return self.weight_multipliers['bottom_30_pct']
            elif percentile >= 0.10:
                return self.weight_multipliers['bottom_20_pct']
            else:
                return self.weight_multipliers['bottom_10_pct']
        else:
            if percentile >= 0.80:
                return self.weight_multipliers['top_20_pct']
            elif percentile >= 0.60:
                return self.weight_multipliers['top_40_pct']
            elif percentile >= 0.40:
                return self.weight_multipliers['mid_40_60_pct']
            elif percentile >= 0.20:
                return self.weight_multipliers['bottom_40_pct']
            else:
                return self.weight_multipliers['bottom_20_pct']

    def apply_reversal_boost(self,
                            portfolio: pd.DataFrame,
                            factor_data: pd.DataFrame) -> pd.DataFrame:
        """
        应用绩优股回撤加成

        对于满足以下条件的股票额外加成：
        1. ROE > 12%
        2. 净利润增速 > 5%
        3. 近3个月跌幅 > 15%

        加成逻辑：在原有调整系数基础上 × 1.3
        """
        if not self.enable_reversal_boost:
            return portfolio

        logger.info("应用绩优股回撤加成...")

        # 合并因子数据
        df = portfolio.merge(
            factor_data[['code', 'roe', 'net_profit_yoy', 'momentum_3m']],
            on='code',
            how='left'
        )

        # 绩优股回撤条件
        quality_mask = (
            (df['roe'] > 12) &
            (df['net_profit_yoy'] > 5)
        )

        drawdown_mask = df['momentum_3m'] < -15 if 'momentum_3m' in df.columns else pd.Series(False, index=df.index)

        # 符合条件的加成1.3倍
        boost_mask = quality_mask & drawdown_mask
        n_boosted = boost_mask.sum()

        if n_boosted > 0:
            df.loc[boost_mask, 'multiplier'] *= 1.3
            logger.info(f"  发现 {n_boosted} 只绩优股回撤，已加成30%权重")

            # 显示这些股票
            boosted_stocks = df[boost_mask][['code', 'roe', 'net_profit_yoy', 'momentum_3m']].head(10)
            logger.info("  加成股票示例:")
            for _, row in boosted_stocks.iterrows():
                logger.info(f"    {row['code']}: ROE={row['roe']:.1f}%, "
                          f"增速={row['net_profit_yoy']:.1f}%, "
                          f"3个月={row['momentum_3m']:.1f}%")

        # 返回不包含合并列的portfolio
        return portfolio

    def adjust_weights(self,
                      original_portfolio: pd.DataFrame,
                      composite_scores: pd.DataFrame,
                      factor_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        根据因子得分调整权重

        Args:
            original_portfolio: 原始持仓 DataFrame [code, weight]
            composite_scores: 综合得分 DataFrame [code, composite_score, ...]
            factor_data: 原始因子数据（用于绩优股回撤判断）

        Returns:
            调整后的持仓 DataFrame
        """
        logger.info("="*60)
        logger.info("开始调整权重")
        logger.info("="*60)

        # 计算百分位
        scores = composite_scores.copy()
        scores['percentile'] = scores['composite_score'].rank(pct=True)

        # 获取调整系数
        scores['multiplier'] = scores['percentile'].apply(self.get_multiplier_by_percentile)

        # 合并到持仓
        portfolio = original_portfolio.copy()
        portfolio = portfolio.merge(
            scores[['code', 'composite_score', 'percentile', 'multiplier']],
            on='code',
            how='left'
        )

        # 缺失得分的股票，使用保守系数
        portfolio['multiplier'] = portfolio['multiplier'].fillna(0.8)
        portfolio['composite_score'] = portfolio['composite_score'].fillna(50)
        portfolio['percentile'] = portfolio['percentile'].fillna(0.5)

        # 应用绩优股回撤加成
        if factor_data is not None and self.enable_reversal_boost:
            portfolio = self.apply_reversal_boost(portfolio, factor_data)

        # 计算调整后权重
        portfolio['original_weight'] = portfolio['weight']
        portfolio['adjusted_weight'] = portfolio['weight'] * portfolio['multiplier']

        # 应用约束
        portfolio['adjusted_weight'] = portfolio['adjusted_weight'].clip(
            lower=self.min_weight,
            upper=self.max_weight
        )

        # 归一化
        total = portfolio['adjusted_weight'].sum()
        portfolio['adjusted_weight'] = portfolio['adjusted_weight'] / total

        # 统计
        self._log_adjustment_stats(portfolio)

        return portfolio

    def _log_adjustment_stats(self, portfolio: pd.DataFrame):
        """记录调整统计"""
        logger.info("\n权重调整统计:")

        # 分位分布
        if self.use_10_tier:
            tiers = [
                ('Top 5%', 0.95, 1.0),
                ('5-10%', 0.90, 0.95),
                ('10-20%', 0.80, 0.90),
                ('20-40%', 0.60, 0.80),
                ('40-60%', 0.40, 0.60),
                ('60-80%', 0.20, 0.40),
                ('80-90%', 0.10, 0.20),
                ('Bottom 10%', 0.0, 0.10),
            ]
        else:
            tiers = [
                ('Top 20%', 0.80, 1.0),
                ('20-40%', 0.60, 0.80),
                ('40-60%', 0.40, 0.60),
                ('60-80%', 0.20, 0.40),
                ('Bottom 20%', 0.0, 0.20),
            ]

        logger.info("  分位分布:")
        for name, low, high in tiers:
            count = len(portfolio[(portfolio['percentile'] >= low) & (portfolio['percentile'] < high)])
            weight = portfolio[(portfolio['percentile'] >= low) & (portfolio['percentile'] < high)]['adjusted_weight'].sum()
            logger.info(f"    {name}: {count}只, 总权重{weight:.1%}")

        # 权重变化
        portfolio['weight_change'] = portfolio['adjusted_weight'] / portfolio['original_weight'] - 1

        increased = len(portfolio[portfolio['weight_change'] > 0.1])
        decreased = len(portfolio[portfolio['weight_change'] < -0.1])
        at_max = len(portfolio[portfolio['adjusted_weight'] >= self.max_weight * 0.99])

        logger.info(f"\n  权重变化:")
        logger.info(f"    增加>10%: {increased}只")
        logger.info(f"    减少>10%: {decreased}只")
        logger.info(f"    触及上限: {at_max}只")

        # Top 10 权重最高
        top10 = portfolio.nlargest(10, 'adjusted_weight')
        logger.info(f"\n  Top 10 权重最高:")
        for _, row in top10.iterrows():
            change = row['weight_change'] * 100
            logger.info(f"    {row['code']}: {row['adjusted_weight']:.2%} "
                       f"(得分{row['composite_score']:.0f}, {change:+.0f}%)")

    def generate_rebalance_orders(self,
                                 current_holdings: Dict[str, float],
                                 target_portfolio: pd.DataFrame,
                                 prices: Dict[str, float],
                                 total_value: float,
                                 min_trade_value: float = 1000) -> Tuple[List[Dict], List[Dict]]:
        """
        生成调仓订单

        Args:
            current_holdings: 当前持仓 {code: shares}
            target_portfolio: 目标持仓 DataFrame [code, adjusted_weight]
            prices: 当前价格 {code: price}
            total_value: 总市值
            min_trade_value: 最小交易金额（过滤小额交易）

        Returns:
            (sell_orders, buy_orders)
        """
        logger.info("生成调仓订单...")

        sell_orders = []
        buy_orders = []

        # 目标持仓字典
        target_weights = dict(zip(target_portfolio['code'], target_portfolio['adjusted_weight']))

        # 当前持仓市值
        current_weights = {}
        for code, shares in current_holdings.items():
            if code in prices and shares > 0:
                value = shares * prices[code]
                current_weights[code] = value / total_value

        all_codes = set(current_weights.keys()) | set(target_weights.keys())

        for code in all_codes:
            current_w = current_weights.get(code, 0)
            target_w = target_weights.get(code, 0)
            diff_w = target_w - current_w

            if code not in prices:
                continue

            price = prices[code]
            diff_value = diff_w * total_value

            if abs(diff_value) < min_trade_value:
                continue

            if diff_value < 0:
                # 卖出
                shares_to_sell = int(abs(diff_value) / price / 100) * 100
                if shares_to_sell > 0:
                    sell_orders.append({
                        'code': code,
                        'shares': shares_to_sell,
                        'price': price,
                        'value': shares_to_sell * price,
                        'reason': f'减仓 {current_w:.2%} -> {target_w:.2%}'
                    })
            else:
                # 买入
                shares_to_buy = int(diff_value / price / 100) * 100
                if shares_to_buy > 0:
                    buy_orders.append({
                        'code': code,
                        'shares': shares_to_buy,
                        'price': price,
                        'value': shares_to_buy * price,
                        'target_weight': target_w,
                        'reason': f'加仓 {current_w:.2%} -> {target_w:.2%}'
                    })

        # 按交易金额排序
        sell_orders.sort(key=lambda x: -x['value'])
        buy_orders.sort(key=lambda x: -x['value'])

        logger.info(f"  卖出订单: {len(sell_orders)}笔, 总额{sum(o['value'] for o in sell_orders):,.0f}")
        logger.info(f"  买入订单: {len(buy_orders)}笔, 总额{sum(o['value'] for o in buy_orders):,.0f}")

        return sell_orders, buy_orders


class AdaptiveStrategy(EnhancedStrategyV2):
    """
    自适应策略
    根据市场环境动态调整因子权重和权重系数
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.market_regime = 'neutral'

    def detect_market_regime(self, market_data: pd.DataFrame) -> str:
        """
        检测市场环境

        Args:
            market_data: 市场数据 (需包含指数收益率)

        Returns:
            'bull' / 'bear' / 'volatile' / 'neutral'
        """
        if 'index_return_20d' in market_data.columns:
            ret_20d = market_data['index_return_20d'].iloc[-1]
            vol_20d = market_data['index_volatility_20d'].iloc[-1] if 'index_volatility_20d' in market_data.columns else 0.02

            if ret_20d > 0.05 and vol_20d < 0.025:
                return 'bull'
            elif ret_20d < -0.08:
                return 'bear'
            elif vol_20d > 0.03:
                return 'volatile'

        return 'neutral'

    def adjust_for_regime(self, regime: str):
        """
        根据市场环境调整策略参数
        """
        self.market_regime = regime

        if regime == 'bull':
            # 牛市：更激进，扩大超配
            self.weight_multipliers['top_5_pct'] = 3.0
            self.weight_multipliers['top_10_pct'] = 2.5
            self.weight_multipliers['bottom_10_pct'] = 0.4
            logger.info("市场环境: 牛市 - 扩大超配系数")

        elif regime == 'bear':
            # 熊市：保守，缩小差距，强化反转
            self.weight_multipliers['top_5_pct'] = 2.0
            self.weight_multipliers['top_10_pct'] = 1.7
            self.weight_multipliers['bottom_10_pct'] = 0.5
            self.enable_reversal_boost = True
            logger.info("市场环境: 熊市 - 保守配置，启用反转加成")

        elif regime == 'volatile':
            # 震荡市：降低波动，偏向质量
            self.weight_multipliers['top_5_pct'] = 2.2
            self.weight_multipliers['bottom_10_pct'] = 0.4
            logger.info("市场环境: 震荡市 - 偏向质量因子")


if __name__ == '__main__':
    # 测试
    print("="*60)
    print("测试增强版策略 V2")
    print("="*60)

    # 创建测试数据
    np.random.seed(42)
    n = 50

    # 原始持仓（等权）
    original = pd.DataFrame({
        'code': [f'{i:06d}' for i in range(n)],
        'weight': [1/n] * n
    })

    # 综合得分
    scores = pd.DataFrame({
        'code': [f'{i:06d}' for i in range(n)],
        'composite_score': np.random.uniform(30, 90, n)
    })

    # 因子数据（用于反转判断）
    factor_data = pd.DataFrame({
        'code': [f'{i:06d}' for i in range(n)],
        'roe': np.random.uniform(5, 25, n),
        'net_profit_yoy': np.random.uniform(-10, 50, n),
        'momentum_3m': np.random.uniform(-25, 20, n),
    })

    # 测试10档策略
    strategy = EnhancedStrategyV2(use_10_tier=True, enable_reversal_boost=True)
    adjusted = strategy.adjust_weights(original, scores, factor_data)

    print("\n调整后持仓示例:")
    print(adjusted[['code', 'original_weight', 'adjusted_weight', 'composite_score', 'multiplier']].head(10))
