"""
因子计算模块
计算5大因子得分并合成综合得分
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FactorCalculator:
    """因子计算器"""

    def __init__(self, factor_weights: Dict[str, float] = None):
        """
        初始化因子计算器

        Args:
            factor_weights: 因子权重字典
                {
                    'dividend_yield': 0.15,
                    'roe': 0.20,
                    'profit_growth': 0.25,
                    'market_cap': 0.15,
                    'momentum': 0.25
                }
        """
        if factor_weights is None:
            # 默认权重
            self.factor_weights = {
                'dividend_yield': 0.15,
                'roe': 0.20,
                'profit_growth': 0.25,
                'market_cap': 0.15,
                'momentum': 0.25
            }
        else:
            self.factor_weights = factor_weights

        # 验证权重和为1
        total_weight = sum(self.factor_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"因子权重和必须为1，当前为{total_weight}")

        logger.info("因子权重配置:")
        for factor, weight in self.factor_weights.items():
            logger.info(f"  {factor}: {weight:.2%}")

    def calculate_dividend_yield_score(self, dividend_df: pd.DataFrame) -> pd.DataFrame:
        """
        计算股息率因子得分

        方法：
        - 得分 = (个股股息率 / 全市场90分位股息率) × 100
        - 上限100分

        Args:
            dividend_df: DataFrame with columns [code, dividend_yield]

        Returns:
            DataFrame with columns [code, dividend_score]
        """
        logger.info("计算股息率因子得分...")

        df = dividend_df.copy()

        # 计算90分位数作为基准
        percentile_90 = df['dividend_yield'].quantile(0.90)

        if percentile_90 == 0:
            percentile_90 = df['dividend_yield'].median()

        # 计算得分
        df['dividend_score'] = (df['dividend_yield'] / percentile_90) * 100
        df['dividend_score'] = df['dividend_score'].clip(0, 100)

        logger.info(f"  股息率中位数: {df['dividend_yield'].median():.2%}")
        logger.info(f"  股息率90分位: {percentile_90:.2%}")
        logger.info(f"  平均得分: {df['dividend_score'].mean():.1f}")

        return df[['code', 'dividend_score']]

    def calculate_roe_score(self, roe_df: pd.DataFrame) -> pd.DataFrame:
        """
        计算ROE因子得分

        方法：
        - 得分 = (个股ROE / 全市场90分位ROE) × 100
        - 上限100分

        Args:
            roe_df: DataFrame with columns [code, roe]

        Returns:
            DataFrame with columns [code, roe_score]
        """
        logger.info("计算ROE因子得分...")

        df = roe_df.copy()

        # 使用roe或roe_diluted（取较大值）
        if 'roe_diluted' in df.columns:
            df['roe_value'] = df[['roe', 'roe_diluted']].max(axis=1)
        else:
            df['roe_value'] = df['roe']

        # 计算90分位数
        percentile_90 = df['roe_value'].quantile(0.90)

        if percentile_90 == 0:
            percentile_90 = df['roe_value'].median()

        # 计算得分
        df['roe_score'] = (df['roe_value'] / percentile_90) * 100
        df['roe_score'] = df['roe_score'].clip(0, 100)

        logger.info(f"  ROE中位数: {df['roe_value'].median():.2f}")
        logger.info(f"  ROE 90分位: {percentile_90:.2f}")
        logger.info(f"  平均得分: {df['roe_score'].mean():.1f}")

        return df[['code', 'roe_score']]

    def calculate_profit_growth_score(self, profit_df: pd.DataFrame) -> pd.DataFrame:
        """
        计算净利润增长率因子得分

        方法：
        - 增长率 > 50%: 100分
        - 增长率 = 20%: 70分
        - 增长率 = 0%: 50分
        - 增长率 < -20%: 0分
        - 线性映射

        Args:
            profit_df: DataFrame with columns [code, net_profit_yoy]

        Returns:
            DataFrame with columns [code, profit_growth_score]
        """
        logger.info("计算净利润增长率因子得分...")

        df = profit_df.copy()

        def growth_to_score(growth):
            """增长率转得分"""
            if pd.isna(growth):
                return 50  # 缺失值默认50分

            if growth >= 50:
                return 100
            elif growth >= 20:
                # 20-50映射到70-100
                return 70 + (growth - 20) / 30 * 30
            elif growth >= 0:
                # 0-20映射到50-70
                return 50 + growth / 20 * 20
            elif growth >= -20:
                # -20-0映射到0-50
                return 50 + growth / 20 * 50
            else:
                return 0

        df['profit_growth_score'] = df['net_profit_yoy'].apply(growth_to_score)

        logger.info(f"  增长率中位数: {df['net_profit_yoy'].median():.2f}%")
        logger.info(f"  平均得分: {df['profit_growth_score'].mean():.1f}")

        return df[['code', 'profit_growth_score']]

    def calculate_market_cap_score(self, market_cap_df: pd.DataFrame) -> pd.DataFrame:
        """
        计算市值因子得分

        方法：
        - 超大盘 (>1000亿): 90分
        - 大盘 (300-1000亿): 100分 (最优)
        - 中盘 (100-300亿): 80分
        - 小盘 (<100亿): 60分

        Args:
            market_cap_df: DataFrame with columns [code, market_cap]（单位：亿元）

        Returns:
            DataFrame with columns [code, market_cap_score]
        """
        logger.info("计算市值因子得分...")

        df = market_cap_df.copy()

        def cap_to_score(cap):
            """市值转得分"""
            if pd.isna(cap):
                return 70  # 缺失值默认70分

            if cap > 1000:
                return 90
            elif cap >= 300:
                return 100
            elif cap >= 100:
                return 80
            else:
                return 60

        df['market_cap_score'] = df['market_cap'].apply(cap_to_score)

        # 统计分布
        distribution = {
            '超大盘(>1000亿)': len(df[df['market_cap'] > 1000]),
            '大盘(300-1000亿)': len(df[(df['market_cap'] >= 300) & (df['market_cap'] <= 1000)]),
            '中盘(100-300亿)': len(df[(df['market_cap'] >= 100) & (df['market_cap'] < 300)]),
            '小盘(<100亿)': len(df[df['market_cap'] < 100])
        }

        logger.info(f"  市值分布: {distribution}")
        logger.info(f"  平均得分: {df['market_cap_score'].mean():.1f}")

        return df[['code', 'market_cap_score']]

    def calculate_momentum_score(self, momentum_df: pd.DataFrame) -> pd.DataFrame:
        """
        计算动量因子得分

        方法：
        - 按百分位排名 × 100
        - Top 10%: 100分
        - Top 50%: 50分
        - Bottom 10%: 0分

        Args:
            momentum_df: DataFrame with columns [code, momentum]

        Returns:
            DataFrame with columns [code, momentum_score]
        """
        logger.info("计算动量因子得分...")

        df = momentum_df.copy()

        # 计算百分位排名
        df['momentum_rank'] = df['momentum'].rank(pct=True)
        df['momentum_score'] = df['momentum_rank'] * 100

        logger.info(f"  动量中位数: {df['momentum'].median():.2f}%")
        logger.info(f"  动量90分位: {df['momentum'].quantile(0.90):.2f}%")
        logger.info(f"  平均得分: {df['momentum_score'].mean():.1f}")

        return df[['code', 'momentum_score']]

    def calculate_composite_score(self,
                                  dividend_score: pd.DataFrame,
                                  roe_score: pd.DataFrame,
                                  profit_growth_score: pd.DataFrame,
                                  market_cap_score: pd.DataFrame,
                                  momentum_score: pd.DataFrame) -> pd.DataFrame:
        """
        计算综合得分

        综合得分 = Σ(因子得分 × 因子权重)

        Args:
            各因子得分DataFrame

        Returns:
            DataFrame with columns [code, composite_score, 各因子得分]
        """
        logger.info("计算综合得分...")

        # 合并所有因子得分
        result = dividend_score.copy()
        result = result.merge(roe_score, on='code', how='outer')
        result = result.merge(profit_growth_score, on='code', how='outer')
        result = result.merge(market_cap_score, on='code', how='outer')
        result = result.merge(momentum_score, on='code', how='outer')

        # 填充缺失值（默认50分）
        score_columns = ['dividend_score', 'roe_score', 'profit_growth_score',
                        'market_cap_score', 'momentum_score']
        result[score_columns] = result[score_columns].fillna(50)

        # 计算综合得分
        result['composite_score'] = (
            result['dividend_score'] * self.factor_weights['dividend_yield'] +
            result['roe_score'] * self.factor_weights['roe'] +
            result['profit_growth_score'] * self.factor_weights['profit_growth'] +
            result['market_cap_score'] * self.factor_weights['market_cap'] +
            result['momentum_score'] * self.factor_weights['momentum']
        )

        logger.info(f"  综合得分分布:")
        logger.info(f"    最小值: {result['composite_score'].min():.1f}")
        logger.info(f"    25分位: {result['composite_score'].quantile(0.25):.1f}")
        logger.info(f"    中位数: {result['composite_score'].median():.1f}")
        logger.info(f"    75分位: {result['composite_score'].quantile(0.75):.1f}")
        logger.info(f"    最大值: {result['composite_score'].max():.1f}")

        return result

    def calculate_all_scores(self, factor_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        计算所有因子得分

        Args:
            factor_data: 包含所有因子数据的字典
                {
                    'dividend': DataFrame,
                    'roe': DataFrame,
                    'profit': DataFrame,
                    'market_cap': DataFrame,
                    'momentum': DataFrame
                }

        Returns:
            DataFrame with all factor scores and composite score
        """
        logger.info("="*60)
        logger.info("开始计算所有因子得分")
        logger.info("="*60)

        # 1. 股息率得分
        dividend_score = self.calculate_dividend_yield_score(factor_data['dividend'])

        # 2. ROE得分
        roe_score = self.calculate_roe_score(factor_data['roe'])

        # 3. 净利润增长率得分
        profit_growth_score = self.calculate_profit_growth_score(factor_data['profit'])

        # 4. 市值得分
        market_cap_score = self.calculate_market_cap_score(factor_data['market_cap'])

        # 5. 动量得分
        momentum_score = self.calculate_momentum_score(factor_data['momentum'])

        # 6. 综合得分
        composite_scores = self.calculate_composite_score(
            dividend_score, roe_score, profit_growth_score,
            market_cap_score, momentum_score
        )

        logger.info("="*60)
        logger.info("所有因子得分计算完成")
        logger.info("="*60)

        return composite_scores


if __name__ == '__main__':
    # 测试代码
    calculator = FactorCalculator()

    # 创建测试数据
    test_data = {
        'dividend': pd.DataFrame({
            'code': ['000001', '000002', '600000', '600519'],
            'dividend_yield': [0.03, 0.05, 0.02, 0.01]
        }),
        'roe': pd.DataFrame({
            'code': ['000001', '000002', '600000', '600519'],
            'roe': [15, 20, 10, 30],
            'roe_diluted': [14, 19, 9, 29]
        }),
        'profit': pd.DataFrame({
            'code': ['000001', '000002', '600000', '600519'],
            'net_profit_yoy': [10, 25, -5, 50]
        }),
        'market_cap': pd.DataFrame({
            'code': ['000001', '000002', '600000', '600519'],
            'market_cap': [500, 200, 800, 1500]
        }),
        'momentum': pd.DataFrame({
            'code': ['000001', '000002', '600000', '600519'],
            'momentum': [5, 10, -3, 15]
        })
    }

    # 计算得分
    scores = calculator.calculate_all_scores(test_data)
    print("\n综合得分结果:")
    print(scores)
