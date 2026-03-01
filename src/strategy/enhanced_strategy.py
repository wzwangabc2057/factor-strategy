"""
多因子增强策略核心逻辑
根据因子得分动态调整个股权重
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedStrategy:
    """多因子增强策略"""

    def __init__(self,
                 weight_multipliers: Dict[str, float] = None,
                 max_weight: float = 0.05,
                 min_weight: float = 0.0005):
        """
        初始化策略

        Args:
            weight_multipliers: 权重调整系数
                {
                    'top_20_pct': 1.8,      # 得分前20%
                    'top_40_pct': 1.3,      # 得分20-40%
                    'mid_40_60_pct': 1.0,   # 得分40-60%
                    'bottom_40_pct': 0.7,   # 得分60-80%
                    'bottom_20_pct': 0.5    # 得分后20%
                }
            max_weight: 单股最大权重
            min_weight: 单股最小权重
        """
        if weight_multipliers is None:
            self.weight_multipliers = {
                'top_20_pct': 1.8,
                'top_40_pct': 1.3,
                'mid_40_60_pct': 1.0,
                'bottom_40_pct': 0.7,
                'bottom_20_pct': 0.5
            }
        else:
            self.weight_multipliers = weight_multipliers

        self.max_weight = max_weight
        self.min_weight = min_weight

        logger.info("策略参数配置:")
        logger.info(f"  权重调整系数: {self.weight_multipliers}")
        logger.info(f"  单股最大权重: {self.max_weight:.2%}")
        logger.info(f"  单股最小权重: {self.min_weight:.4%}")

    def get_multiplier_by_score(self, composite_scores: pd.DataFrame) -> pd.DataFrame:
        """
        根据综合得分确定调整系数

        Args:
            composite_scores: DataFrame with columns [code, composite_score]

        Returns:
            DataFrame with columns [code, composite_score, multiplier]
        """
        df = composite_scores.copy()

        # 计算百分位
        df['score_percentile'] = df['composite_score'].rank(pct=True)

        # 根据百分位确定调整系数
        def get_multiplier(percentile):
            if percentile >= 0.8:
                return self.weight_multipliers['top_20_pct']
            elif percentile >= 0.6:
                return self.weight_multipliers['top_40_pct']
            elif percentile >= 0.4:
                return self.weight_multipliers['mid_40_60_pct']
            elif percentile >= 0.2:
                return self.weight_multipliers['bottom_40_pct']
            else:
                return self.weight_multipliers['bottom_20_pct']

        df['multiplier'] = df['score_percentile'].apply(get_multiplier)

        # 统计分布
        distribution = {
            f'Top 20% (×{self.weight_multipliers["top_20_pct"]})': len(df[df['score_percentile'] >= 0.8]),
            f'20-40% (×{self.weight_multipliers["top_40_pct"]})': len(df[(df['score_percentile'] >= 0.6) & (df['score_percentile'] < 0.8)]),
            f'40-60% (×{self.weight_multipliers["mid_40_60_pct"]})': len(df[(df['score_percentile'] >= 0.4) & (df['score_percentile'] < 0.6)]),
            f'60-80% (×{self.weight_multipliers["bottom_40_pct"]})': len(df[(df['score_percentile'] >= 0.2) & (df['score_percentile'] < 0.4)]),
            f'Bottom 20% (×{self.weight_multipliers["bottom_20_pct"]})': len(df[df['score_percentile'] < 0.2])
        }

        logger.info("权重调整系数分布:")
        for category, count in distribution.items():
            logger.info(f"  {category}: {count}只")

        return df[['code', 'composite_score', 'multiplier']]

    def adjust_weights(self,
                      original_portfolio: pd.DataFrame,
                      composite_scores: pd.DataFrame) -> pd.DataFrame:
        """
        根据因子得分调整权重

        Args:
            original_portfolio: 原始持仓 DataFrame [code, weight]
            composite_scores: 综合得分 DataFrame [code, composite_score, ...]

        Returns:
            调整后的持仓 DataFrame [code, original_weight, adjusted_weight, multiplier, composite_score]
        """
        logger.info("="*60)
        logger.info("开始调整权重")
        logger.info("="*60)

        # 获取调整系数
        multipliers = self.get_multiplier_by_score(composite_scores)

        # 合并原始权重和调整系数
        portfolio = original_portfolio.copy()
        portfolio = portfolio.merge(multipliers, on='code', how='left')

        # 缺失调整系数的股票（可能因子数据缺失），使用默认系数0.7
        portfolio['multiplier'] = portfolio['multiplier'].fillna(0.7)
        portfolio['composite_score'] = portfolio['composite_score'].fillna(50)

        # 计算调整后权重
        portfolio['adjusted_weight'] = portfolio['weight'] * portfolio['multiplier']

        # 应用约束条件
        portfolio['adjusted_weight'] = portfolio['adjusted_weight'].clip(
            lower=self.min_weight,
            upper=self.max_weight
        )

        # 权重归一化
        total_weight = portfolio['adjusted_weight'].sum()
        portfolio['adjusted_weight'] = portfolio['adjusted_weight'] / total_weight

        # 重命名列
        portfolio = portfolio.rename(columns={'weight': 'original_weight'})

        # 统计调整效果
        logger.info(f"\n权重调整统计:")
        logger.info(f"  调整前权重标准差: {portfolio['original_weight'].std():.6f}")
        logger.info(f"  调整后权重标准差: {portfolio['adjusted_weight'].std():.6f}")
        logger.info(f"  权重增加>50%的股票: {len(portfolio[portfolio['adjusted_weight'] / portfolio['original_weight'] > 1.5])}只")
        logger.info(f"  权重减少>30%的股票: {len(portfolio[portfolio['adjusted_weight'] / portfolio['original_weight'] < 0.7])}只")
        logger.info(f"  触及最大权重限制: {len(portfolio[portfolio['adjusted_weight'] >= self.max_weight * 0.99])}只")

        # 显示调整幅度最大的前10只
        portfolio['weight_change_pct'] = (portfolio['adjusted_weight'] / portfolio['original_weight'] - 1) * 100
        top_changes = portfolio.nlargest(10, 'weight_change_pct')[
            ['code', 'original_weight', 'adjusted_weight', 'weight_change_pct', 'composite_score']
        ]

        logger.info(f"\n权重增加最多的前10只股票:")
        for idx, row in top_changes.iterrows():
            logger.info(f"  {row['code']}: {row['original_weight']:.4f} → {row['adjusted_weight']:.4f} "
                       f"({row['weight_change_pct']:+.1f}%, 得分{row['composite_score']:.1f})")

        logger.info("="*60)
        logger.info("权重调整完成")
        logger.info("="*60)

        return portfolio

    def generate_rebalance_orders(self,
                                 current_holdings: Dict[str, float],
                                 target_portfolio: pd.DataFrame,
                                 current_prices: Dict[str, float],
                                 total_value: float) -> Tuple[List[Dict], List[Dict]]:
        """
        生成调仓订单

        Args:
            current_holdings: 当前持仓 {code: shares}
            target_portfolio: 目标持仓 DataFrame [code, adjusted_weight]
            current_prices: 当前价格 {code: price}
            total_value: 总市值

        Returns:
            (sell_orders, buy_orders)
            sell_orders: [{code, shares, price, value}]
            buy_orders: [{code, shares, price, value, weight}]
        """
        logger.info("生成调仓订单...")

        sell_orders = []
        buy_orders = []

        # 1. 生成卖出订单（清仓所有当前持仓）
        for code, shares in current_holdings.items():
            if code in current_prices and shares > 0:
                price = current_prices[code]
                value = shares * price
                sell_orders.append({
                    'code': code,
                    'shares': shares,
                    'price': price,
                    'value': value
                })

        # 2. 生成买入订单（按目标权重买入）
        for _, row in target_portfolio.iterrows():
            code = row['code']
            weight = row['adjusted_weight']

            if code in current_prices:
                price = current_prices[code]
                if price > 0:
                    target_value = total_value * weight
                    shares = int(target_value / price / 100) * 100  # 买入整百股

                    if shares > 0:
                        buy_orders.append({
                            'code': code,
                            'shares': shares,
                            'price': price,
                            'value': shares * price,
                            'weight': weight
                        })

        logger.info(f"  生成 {len(sell_orders)} 个卖出订单")
        logger.info(f"  生成 {len(buy_orders)} 个买入订单")

        return sell_orders, buy_orders


if __name__ == '__main__':
    # 测试代码
    strategy = EnhancedStrategy()

    # 创建测试数据
    original_portfolio = pd.DataFrame({
        'code': ['000001', '000002', '600000', '600519', '601318'],
        'weight': [0.20, 0.20, 0.20, 0.20, 0.20]
    })

    composite_scores = pd.DataFrame({
        'code': ['000001', '000002', '600000', '600519', '601318'],
        'composite_score': [85, 60, 45, 25, 90]
    })

    # 调整权重
    adjusted_portfolio = strategy.adjust_weights(original_portfolio, composite_scores)

    print("\n调整后的持仓:")
    print(adjusted_portfolio[['code', 'original_weight', 'adjusted_weight', 'composite_score']])
