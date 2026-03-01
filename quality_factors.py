"""
质量因子和成长因子计算模块

因子类别：
1. 盈利能力 (Profitability): ROE, ROA, 毛利率, 净利率
2. 经营效率 (Efficiency): 资产周转率, 存货周转率
3. 财务健康 (Financial Health): 负债率, 流动比率
4. 成长性 (Growth): 营收增速, 利润增速
5. 估值 (Valuation): PE, PB, PS, PEG
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QualityFactorCalculator:
    """质量因子计算器"""

    def __init__(self):
        """初始化"""
        logger.info("质量因子计算器初始化")

    # ==================== 盈利能力因子 ====================

    def calc_ROE(self, financial_df: pd.DataFrame) -> pd.Series:
        """
        ROE = 净利润 / 净资产
        净资产收益率
        """
        if 'net_profit' in financial_df.columns and 'equity' in financial_df.columns:
            return financial_df['net_profit'] / (financial_df['equity'] + 1e-12)
        elif 'roe' in financial_df.columns:
            return financial_df['roe']
        else:
            return pd.Series(np.nan, index=financial_df.index)

    def calc_ROA(self, financial_df: pd.DataFrame) -> pd.Series:
        """
        ROA = 净利润 / 总资产
        总资产收益率
        """
        if 'net_profit' in financial_df.columns and 'total_assets' in financial_df.columns:
            return financial_df['net_profit'] / (financial_df['total_assets'] + 1e-12)
        elif 'roa' in financial_df.columns:
            return financial_df['roa']
        else:
            return pd.Series(np.nan, index=financial_df.index)

    def calc_gross_margin(self, financial_df: pd.DataFrame) -> pd.Series:
        """
        毛利率 = (营收 - 营业成本) / 营收
        """
        if all(col in financial_df.columns for col in ['revenue', 'cost']):
            return (financial_df['revenue'] - financial_df['cost']) / (financial_df['revenue'] + 1e-12)
        elif 'gross_margin' in financial_df.columns:
            return financial_df['gross_margin']
        else:
            return pd.Series(np.nan, index=financial_df.index)

    def calc_net_margin(self, financial_df: pd.DataFrame) -> pd.Series:
        """
        净利率 = 净利润 / 营收
        """
        if 'net_profit' in financial_df.columns and 'revenue' in financial_df.columns:
            return financial_df['net_profit'] / (financial_df['revenue'] + 1e-12)
        elif 'net_margin' in financial_df.columns:
            return financial_df['net_margin']
        else:
            return pd.Series(np.nan, index=financial_df.index)

    def calc_operating_margin(self, financial_df: pd.DataFrame) -> pd.Series:
        """
        营业利润率 = 营业利润 / 营收
        """
        if 'operating_profit' in financial_df.columns and 'revenue' in financial_df.columns:
            return financial_df['operating_profit'] / (financial_df['revenue'] + 1e-12)
        elif 'operating_margin' in financial_df.columns:
            return financial_df['operating_margin']
        else:
            return pd.Series(np.nan, index=financial_df.index)

    # ==================== 经营效率因子 ====================

    def calc_asset_turnover(self, financial_df: pd.DataFrame) -> pd.Series:
        """
        资产周转率 = 营收 / 总资产
        """
        if 'revenue' in financial_df.columns and 'total_assets' in financial_df.columns:
            return financial_df['revenue'] / (financial_df['total_assets'] + 1e-12)
        elif 'asset_turnover' in financial_df.columns:
            return financial_df['asset_turnover']
        else:
            return pd.Series(np.nan, index=financial_df.index)

    def calc_inventory_turnover(self, financial_df: pd.DataFrame) -> pd.Series:
        """
        存货周转率 = 营业成本 / 平均存货
        """
        if all(col in financial_df.columns for col in ['cost', 'inventory']):
            return financial_df['cost'] / (financial_df['inventory'] + 1e-12)
        elif 'inventory_turnover' in financial_df.columns:
            return financial_df['inventory_turnover']
        else:
            return pd.Series(np.nan, index=financial_df.index)

    # ==================== 财务健康因子 ====================

    def calc_debt_ratio(self, financial_df: pd.DataFrame) -> pd.Series:
        """
        资产负债率 = 总负债 / 总资产
        """
        if all(col in financial_df.columns for col in ['total_liabilities', 'total_assets']):
            return financial_df['total_liabilities'] / (financial_df['total_assets'] + 1e-12)
        elif 'debt_ratio' in financial_df.columns:
            return financial_df['debt_ratio']
        else:
            return pd.Series(np.nan, index=financial_df.index)

    def calc_current_ratio(self, financial_df: pd.DataFrame) -> pd.Series:
        """
        流动比率 = 流动资产 / 流动负债
        """
        if all(col in financial_df.columns for col in ['current_assets', 'current_liabilities']):
            return financial_df['current_assets'] / (financial_df['current_liabilities'] + 1e-12)
        elif 'current_ratio' in financial_df.columns:
            return financial_df['current_ratio']
        else:
            return pd.Series(np.nan, index=financial_df.index)

    def calc_quick_ratio(self, financial_df: pd.DataFrame) -> pd.Series:
        """
        速动比率 = (流动资产 - 存货) / 流动负债
        """
        if all(col in financial_df.columns for col in ['current_assets', 'inventory', 'current_liabilities']):
            return (financial_df['current_assets'] - financial_df['inventory']) / \
                   (financial_df['current_liabilities'] + 1e-12)
        elif 'quick_ratio' in financial_df.columns:
            return financial_df['quick_ratio']
        else:
            return pd.Series(np.nan, index=financial_df.index)

    # ==================== 成长性因子 ====================

    def calc_revenue_growth(self, financial_df: pd.DataFrame) -> pd.Series:
        """
        营收同比增长率
        """
        if 'revenue_yoy' in financial_df.columns:
            return financial_df['revenue_yoy']
        elif 'revenue' in financial_df.columns:
            # 假设有历史数据可以计算同比
            return financial_df['revenue'].pct_change(periods=4)  # 假设季度数据
        else:
            return pd.Series(np.nan, index=financial_df.index)

    def calc_profit_growth(self, financial_df: pd.DataFrame) -> pd.Series:
        """
        净利润同比增长率
        """
        if 'net_profit_yoy' in financial_df.columns:
            return financial_df['net_profit_yoy']
        elif 'net_profit' in financial_df.columns:
            return financial_df['net_profit'].pct_change(periods=4)
        else:
            return pd.Series(np.nan, index=financial_df.index)

    def calc_eps_growth(self, financial_df: pd.DataFrame) -> pd.Series:
        """
        EPS同比增长率
        """
        if 'eps_yoy' in financial_df.columns:
            return financial_df['eps_yoy']
        elif 'eps' in financial_df.columns:
            return financial_df['eps'].pct_change(periods=4)
        else:
            return pd.Series(np.nan, index=financial_df.index)

    # ==================== 估值因子 ====================

    def calc_PE(self, valuation_df: pd.DataFrame) -> pd.Series:
        """市盈率"""
        if 'pe' in valuation_df.columns:
            return valuation_df['pe']
        elif all(col in valuation_df.columns for col in ['market_cap', 'net_profit']):
            return valuation_df['market_cap'] / (valuation_df['net_profit'] + 1e-12)
        else:
            return pd.Series(np.nan, index=valuation_df.index)

    def calc_PB(self, valuation_df: pd.DataFrame) -> pd.Series:
        """市净率"""
        if 'pb' in valuation_df.columns:
            return valuation_df['pb']
        elif all(col in valuation_df.columns for col in ['market_cap', 'equity']):
            return valuation_df['market_cap'] / (valuation_df['equity'] + 1e-12)
        else:
            return pd.Series(np.nan, index=valuation_df.index)

    def calc_PS(self, valuation_df: pd.DataFrame) -> pd.Series:
        """市销率"""
        if 'ps' in valuation_df.columns:
            return valuation_df['ps']
        elif all(col in valuation_df.columns for col in ['market_cap', 'revenue']):
            return valuation_df['market_cap'] / (valuation_df['revenue'] + 1e-12)
        else:
            return pd.Series(np.nan, index=valuation_df.index)

    def calc_PEG(self, valuation_df: pd.DataFrame, financial_df: pd.DataFrame) -> pd.Series:
        """
        PEG = PE / 净利润增长率
        """
        pe = self.calc_PE(valuation_df)
        growth = self.calc_profit_growth(financial_df)
        return pe / (growth + 1e-12)

    # ==================== 稳定性因子 ====================

    def calc_ROE_stability(self, roe_series: pd.Series, window: int = 8) -> pd.Series:
        """
        ROE稳定性 = -std(ROE)
        负的ROE标准差，稳定性越高得分越高
        """
        return -roe_series.rolling(window).std()

    def calc_earnings_stability(self, profit_series: pd.Series, window: int = 8) -> pd.Series:
        """
        盈利稳定性 = -std(净利润变化率)
        """
        change = profit_series.pct_change()
        return -change.rolling(window).std()

    # ==================== 综合因子 ====================

    def calc_quality_score(self,
                          financial_df: pd.DataFrame,
                          weights: Dict[str, float] = None) -> pd.Series:
        """
        综合质量得分

        Args:
            financial_df: 财务数据
            weights: 各因子权重

        Returns:
            综合得分
        """
        if weights is None:
            weights = {
                'roe': 0.25,
                'gross_margin': 0.15,
                'net_margin': 0.15,
                'asset_turnover': 0.10,
                'debt_ratio': -0.10,  # 负权重，负债率越低越好
                'current_ratio': 0.10,
                'revenue_growth': 0.15
            }

        scores = pd.Series(0, index=financial_df.index)
        total_weight = 0

        for factor, weight in weights.items():
            if factor == 'roe':
                value = self.calc_ROE(financial_df)
            elif factor == 'gross_margin':
                value = self.calc_gross_margin(financial_df)
            elif factor == 'net_margin':
                value = self.calc_net_margin(financial_df)
            elif factor == 'asset_turnover':
                value = self.calc_asset_turnover(financial_df)
            elif factor == 'debt_ratio':
                value = self.calc_debt_ratio(financial_df)
            elif factor == 'current_ratio':
                value = self.calc_current_ratio(financial_df)
            elif factor == 'revenue_growth':
                value = self.calc_revenue_growth(financial_df)
            else:
                continue

            # 标准化
            if not value.isna().all():
                value_normalized = (value - value.mean()) / (value.std() + 1e-12)
                scores += value_normalized.fillna(0) * abs(weight)
                total_weight += abs(weight)

        if total_weight > 0:
            scores = scores / total_weight

        return scores

    def calculate_all_quality_factors(self,
                                     financial_df: pd.DataFrame,
                                     valuation_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        计算所有质量因子

        Args:
            financial_df: 财务数据 DataFrame
            valuation_df: 估值数据 DataFrame（可选）

        Returns:
            因子DataFrame
        """
        logger.info(f"开始计算质量因子，数据量: {len(financial_df)}")

        factors = pd.DataFrame(index=financial_df.index)

        # 盈利能力因子
        factors['ROE'] = self.calc_ROE(financial_df)
        factors['ROA'] = self.calc_ROA(financial_df)
        factors['gross_margin'] = self.calc_gross_margin(financial_df)
        factors['net_margin'] = self.calc_net_margin(financial_df)
        factors['operating_margin'] = self.calc_operating_margin(financial_df)

        # 经营效率因子
        factors['asset_turnover'] = self.calc_asset_turnover(financial_df)
        factors['inventory_turnover'] = self.calc_inventory_turnover(financial_df)

        # 财务健康因子
        factors['debt_ratio'] = self.calc_debt_ratio(financial_df)
        factors['current_ratio'] = self.calc_current_ratio(financial_df)
        factors['quick_ratio'] = self.calc_quick_ratio(financial_df)

        # 成长性因子
        factors['revenue_growth'] = self.calc_revenue_growth(financial_df)
        factors['profit_growth'] = self.calc_profit_growth(financial_df)
        factors['eps_growth'] = self.calc_eps_growth(financial_df)

        # 估值因子（如果有估值数据）
        if valuation_df is not None:
            factors['PE'] = self.calc_PE(valuation_df)
            factors['PB'] = self.calc_PB(valuation_df)
            factors['PS'] = self.calc_PS(valuation_df)
            factors['PEG'] = self.calc_PEG(valuation_df, financial_df)

        # 综合质量得分
        factors['quality_score'] = self.calc_quality_score(financial_df)

        logger.info(f"质量因子计算完成，共 {len(factors.columns)} 个因子")

        return factors


class GrowthFactorCalculator:
    """成长因子计算器"""

    def __init__(self):
        """初始化"""
        logger.info("成长因子计算器初始化")

    def calc_qoq_growth(self, series: pd.Series) -> pd.Series:
        """环比增长率"""
        return series.pct_change()

    def calc_yoy_growth(self, series: pd.Series, periods: int = 4) -> pd.Series:
        """同比增长率（假设季度数据，periods=4）"""
        return series.pct_change(periods=periods)

    def calc_cagr(self, series: pd.Series, periods: int = 4) -> pd.Series:
        """
        复合年化增长率
        CAGR = (end_value / start_value)^(1/n) - 1
        """
        start = series.shift(periods)
        return (series / (start + 1e-12)) ** (1 / (periods / 4)) - 1

    def calc_acceleration(self, series: pd.Series) -> pd.Series:
        """
        增长加速度 = 二阶差分
        正值表示增长在加速
        """
        growth = series.pct_change()
        return growth.diff()

    def calc_consistency(self, series: pd.Series, window: int = 8) -> pd.Series:
        """
        增长一致性 = 增长为正的比例
        """
        growth = series.pct_change()
        return (growth > 0).rolling(window).mean()

    def calculate_all_growth_factors(self, financial_df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有成长因子

        Args:
            financial_df: 财务数据 DataFrame

        Returns:
            因子DataFrame
        """
        logger.info(f"开始计算成长因子，数据量: {len(financial_df)}")

        factors = pd.DataFrame(index=financial_df.index)

        # 营收增长
        if 'revenue' in financial_df.columns:
            factors['revenue_qoq'] = self.calc_qoq_growth(financial_df['revenue'])
            factors['revenue_yoy'] = self.calc_yoy_growth(financial_df['revenue'])
            factors['revenue_cagr'] = self.calc_cagr(financial_df['revenue'])
            factors['revenue_acceleration'] = self.calc_acceleration(financial_df['revenue'])
            factors['revenue_consistency'] = self.calc_consistency(financial_df['revenue'])

        # 净利润增长
        if 'net_profit' in financial_df.columns:
            factors['profit_qoq'] = self.calc_qoq_growth(financial_df['net_profit'])
            factors['profit_yoy'] = self.calc_yoy_growth(financial_df['net_profit'])
            factors['profit_cagr'] = self.calc_cagr(financial_df['net_profit'])
            factors['profit_acceleration'] = self.calc_acceleration(financial_df['net_profit'])
            factors['profit_consistency'] = self.calc_consistency(financial_df['net_profit'])

        # EPS增长
        if 'eps' in financial_df.columns:
            factors['eps_qoq'] = self.calc_qoq_growth(financial_df['eps'])
            factors['eps_yoy'] = self.calc_yoy_growth(financial_df['eps'])
            factors['eps_cagr'] = self.calc_cagr(financial_df['eps'])

        # 经营现金流增长
        if 'operating_cashflow' in financial_df.columns:
            factors['ocf_yoy'] = self.calc_yoy_growth(financial_df['operating_cashflow'])

        logger.info(f"成长因子计算完成，共 {len(factors.columns)} 个因子")

        return factors


if __name__ == '__main__':
    # 测试代码
    np.random.seed(42)

    # 生成模拟财务数据
    n = 100
    financial_data = pd.DataFrame({
        'code': [f'00000{i%10}' for i in range(n)],
        'date': pd.date_range('2023-01-01', periods=n, freq='Q'),
        'net_profit': np.random.uniform(1e8, 1e9, n),
        'revenue': np.random.uniform(1e9, 1e10, n),
        'equity': np.random.uniform(5e8, 5e9, n),
        'total_assets': np.random.uniform(1e9, 1e10, n),
        'cost': np.random.uniform(5e8, 5e9, n),
        'gross_margin': np.random.uniform(0.2, 0.5, n),
        'net_margin': np.random.uniform(0.05, 0.2, n),
        'roe': np.random.uniform(0.05, 0.25, n),
        'revenue_yoy': np.random.uniform(-0.1, 0.3, n),
        'net_profit_yoy': np.random.uniform(-0.2, 0.5, n),
        'debt_ratio': np.random.uniform(0.2, 0.7, n),
        'current_ratio': np.random.uniform(0.8, 2.5, n)
    })

    # 计算质量因子
    quality_calc = QualityFactorCalculator()
    quality_factors = quality_calc.calculate_all_quality_factors(financial_data)

    print("\n质量因子:")
    print(quality_factors.head())

    # 计算成长因子
    growth_calc = GrowthFactorCalculator()
    growth_factors = growth_calc.calculate_all_growth_factors(financial_data)

    print("\n成长因子:")
    print(growth_factors.head())
