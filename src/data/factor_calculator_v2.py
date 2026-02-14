"""
增强版因子计算模块 v2.0
新增因子：
1. 小市值因子 - 偏好中小盘(100-500亿)
2. 质量因子 - ROE稳定性、毛利率、现金流质量
3. 反转因子 - 绩优股回撤买入
4. 低波动因子 - 低波动溢价
5. PEG因子 - 成长性价比
6. 行业动量因子 - 行业轮动
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedFactorCalculator:
    """增强版因子计算器"""

    # 默认因子权重配置 - 可针对不同策略调整
    DEFAULT_WEIGHTS = {
        # === 价值因子 ===
        'dividend_yield': 0.10,      # 股息率
        'pe_value': 0.05,            # PE估值(低PE高分)

        # === 质量因子 ===
        'roe': 0.12,                 # ROE
        'roe_stability': 0.08,       # ROE稳定性
        'cash_flow_quality': 0.05,   # 现金流质量

        # === 成长因子 ===
        'profit_growth': 0.12,       # 净利润增长
        'revenue_growth': 0.05,      # 营收增长
        'peg': 0.08,                 # PEG(成长性价比)

        # === 规模因子 ===
        'small_cap': 0.10,           # 小市值因子(偏好中小盘)

        # === 动量/反转因子 ===
        'momentum': 0.10,            # 动量
        'reversal': 0.10,            # 反转(绩优股回撤买入)

        # === 风险因子 ===
        'low_volatility': 0.05,      # 低波动
    }

    # 稳健型策略权重 - 偏价值、高股息、低波动
    STABLE_WEIGHTS = {
        'dividend_yield': 0.18,
        'pe_value': 0.08,
        'roe': 0.15,
        'roe_stability': 0.10,
        'cash_flow_quality': 0.08,
        'profit_growth': 0.08,
        'revenue_growth': 0.03,
        'peg': 0.05,
        'small_cap': 0.05,
        'momentum': 0.05,
        'reversal': 0.10,
        'low_volatility': 0.05,
    }

    # 进取型策略权重 - 偏成长、动量、小市值
    AGGRESSIVE_WEIGHTS = {
        'dividend_yield': 0.05,
        'pe_value': 0.03,
        'roe': 0.10,
        'roe_stability': 0.05,
        'cash_flow_quality': 0.03,
        'profit_growth': 0.18,
        'revenue_growth': 0.08,
        'peg': 0.12,
        'small_cap': 0.15,
        'momentum': 0.12,
        'reversal': 0.06,
        'low_volatility': 0.03,
    }

    def __init__(self,
                 factor_weights: Dict[str, float] = None,
                 strategy_type: str = 'balanced'):
        """
        初始化因子计算器

        Args:
            factor_weights: 自定义因子权重
            strategy_type: 策略类型
                - 'balanced': 均衡型(默认)
                - 'stable': 稳健型(R4)
                - 'aggressive': 进取型(R5)
        """
        if factor_weights is not None:
            self.factor_weights = factor_weights
        elif strategy_type == 'stable':
            self.factor_weights = self.STABLE_WEIGHTS.copy()
        elif strategy_type == 'aggressive':
            self.factor_weights = self.AGGRESSIVE_WEIGHTS.copy()
        else:
            self.factor_weights = self.DEFAULT_WEIGHTS.copy()

        self.strategy_type = strategy_type

        # 验证权重和
        total_weight = sum(self.factor_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"因子权重和为{total_weight:.2f}，将自动归一化")
            for k in self.factor_weights:
                self.factor_weights[k] /= total_weight

        logger.info(f"策略类型: {strategy_type}")
        logger.info("因子权重配置:")
        for factor, weight in sorted(self.factor_weights.items(), key=lambda x: -x[1]):
            if weight > 0:
                logger.info(f"  {factor}: {weight:.1%}")

    # ==================== 价值因子 ====================

    def calculate_dividend_yield_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        股息率因子
        高股息 = 高分
        """
        logger.info("计算股息率因子...")
        result = df[['code']].copy()

        # 使用百分位排名
        result['dividend_score'] = df['dividend_yield'].rank(pct=True) * 100

        # 对超高股息额外加分（>5%）
        high_div_mask = df['dividend_yield'] > 0.05
        result.loc[high_div_mask, 'dividend_score'] = result.loc[high_div_mask, 'dividend_score'].clip(lower=85)

        logger.info(f"  中位数股息率: {df['dividend_yield'].median():.2%}")
        return result

    def calculate_pe_value_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        PE估值因子
        低PE = 高分（排除负PE）
        """
        logger.info("计算PE估值因子...")
        result = df[['code']].copy()

        # 只考虑正PE
        valid_pe = df['pe'].copy()
        valid_pe[valid_pe <= 0] = np.nan
        valid_pe[valid_pe > 200] = np.nan  # 排除极端值

        # 低PE高分（反向排名）
        result['pe_score'] = (1 - valid_pe.rank(pct=True)) * 100
        result['pe_score'] = result['pe_score'].fillna(30)  # 无效PE给低分

        logger.info(f"  PE中位数: {valid_pe.median():.1f}")
        return result

    # ==================== 质量因子 ====================

    def calculate_roe_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ROE因子
        高ROE = 高分
        """
        logger.info("计算ROE因子...")
        result = df[['code']].copy()

        roe = df['roe'].copy()
        roe[roe < 0] = 0  # 负ROE归零
        roe[roe > 50] = 50  # 上限50%

        result['roe_score'] = roe.rank(pct=True) * 100

        logger.info(f"  ROE中位数: {df['roe'].median():.1f}%")
        return result

    def calculate_roe_stability_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ROE稳定性因子
        需要历史ROE数据: roe_y1, roe_y2, roe_y3 (近3年)
        稳定性 = 1 / (1 + 标准差/均值)
        """
        logger.info("计算ROE稳定性因子...")
        result = df[['code']].copy()

        if 'roe_y1' in df.columns and 'roe_y2' in df.columns and 'roe_y3' in df.columns:
            roe_cols = ['roe_y1', 'roe_y2', 'roe_y3']
            roe_mean = df[roe_cols].mean(axis=1)
            roe_std = df[roe_cols].std(axis=1)

            # 变异系数的倒数（稳定性）
            cv = roe_std / (roe_mean.abs() + 1e-6)
            stability = 1 / (1 + cv)

            result['roe_stability_score'] = stability.rank(pct=True) * 100
        else:
            # 没有历史数据，用当期ROE的高低代替
            result['roe_stability_score'] = df['roe'].clip(lower=0).rank(pct=True) * 80

        return result

    def calculate_cash_flow_quality_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        现金流质量因子
        经营现金流/净利润 > 1 说明盈利质量高
        """
        logger.info("计算现金流质量因子...")
        result = df[['code']].copy()

        if 'ocf' in df.columns and 'net_profit' in df.columns:
            # 现金流/净利润比率
            cf_ratio = df['ocf'] / (df['net_profit'].abs() + 1e-6)
            cf_ratio = cf_ratio.clip(-2, 3)  # 限制极端值

            result['cash_flow_score'] = cf_ratio.rank(pct=True) * 100
        else:
            result['cash_flow_score'] = 50  # 无数据默认中等

        return result

    # ==================== 成长因子 ====================

    def calculate_profit_growth_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        净利润增长因子
        """
        logger.info("计算净利润增长因子...")
        result = df[['code']].copy()

        growth = df['net_profit_yoy'].copy()

        # 分段打分
        def growth_to_score(g):
            if pd.isna(g):
                return 50
            if g >= 100:
                return 100
            elif g >= 50:
                return 85 + (g - 50) / 50 * 15
            elif g >= 30:
                return 75 + (g - 30) / 20 * 10
            elif g >= 15:
                return 65 + (g - 15) / 15 * 10
            elif g >= 0:
                return 50 + g / 15 * 15
            elif g >= -20:
                return 30 + (g + 20) / 20 * 20
            else:
                return max(0, 30 + g / 50 * 30)

        result['profit_growth_score'] = growth.apply(growth_to_score)

        logger.info(f"  净利润增速中位数: {growth.median():.1f}%")
        return result

    def calculate_revenue_growth_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        营收增长因子
        """
        logger.info("计算营收增长因子...")
        result = df[['code']].copy()

        if 'revenue_yoy' in df.columns:
            growth = df['revenue_yoy'].clip(-50, 100)
            result['revenue_growth_score'] = growth.rank(pct=True) * 100
        else:
            result['revenue_growth_score'] = 50

        return result

    def calculate_peg_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        PEG因子 (成长性价比)
        PEG = PE / 净利润增速
        低PEG = 高分（成长被低估）
        """
        logger.info("计算PEG因子...")
        result = df[['code']].copy()

        pe = df['pe'].copy()
        growth = df['net_profit_yoy'].copy()

        # 只计算有效PEG (PE>0, growth>5%)
        valid_mask = (pe > 0) & (pe < 100) & (growth > 5)

        peg = pd.Series(index=df.index, dtype=float)
        peg[valid_mask] = pe[valid_mask] / growth[valid_mask]
        peg = peg.clip(0, 5)  # 限制范围

        # 低PEG高分
        result['peg_score'] = (1 - peg.rank(pct=True)) * 100
        result['peg_score'] = result['peg_score'].fillna(40)  # 无效PEG给较低分

        # 高成长低PE额外加分
        bonus_mask = (growth > 30) & (pe < 30) & (pe > 0)
        result.loc[bonus_mask, 'peg_score'] = result.loc[bonus_mask, 'peg_score'].clip(lower=80)

        return result

    # ==================== 规模因子 ====================

    def calculate_small_cap_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        小市值因子 (偏好中小盘)
        最优区间: 100-500亿
        """
        logger.info("计算小市值因子...")
        result = df[['code']].copy()

        cap = df['market_cap'].copy()  # 单位：亿

        def cap_to_score(c):
            if pd.isna(c):
                return 50
            # 最优区间 100-500亿，得分最高
            if 100 <= c <= 500:
                return 100
            elif 50 <= c < 100:
                return 90
            elif 500 < c <= 1000:
                return 85
            elif 30 <= c < 50:
                return 75
            elif 1000 < c <= 2000:
                return 70
            elif c < 30:
                return 50  # 微盘股流动性风险
            else:  # > 2000亿
                return 60  # 大盘股增长空间有限

        result['small_cap_score'] = cap.apply(cap_to_score)

        # 统计分布
        logger.info(f"  市值中位数: {cap.median():.0f}亿")
        logger.info(f"  100-500亿占比: {((cap >= 100) & (cap <= 500)).mean():.1%}")

        return result

    # ==================== 动量/反转因子 ====================

    def calculate_momentum_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        动量因子
        近1个月涨幅排名
        """
        logger.info("计算动量因子...")
        result = df[['code']].copy()

        momentum = df['momentum_1m'].copy() if 'momentum_1m' in df.columns else df['momentum'].copy()

        result['momentum_score'] = momentum.rank(pct=True) * 100

        logger.info(f"  动量中位数: {momentum.median():.1f}%")
        return result

    def calculate_reversal_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        反转因子 (绩优股回撤买入)

        核心逻辑：
        1. 筛选绩优股（ROE>10%, 净利润增速>0%）
        2. 找出近期回撤大的（近3个月跌幅>15%）
        3. 这类股票给高分（预期反弹）
        """
        logger.info("计算反转因子（绩优股回撤买入）...")
        result = df[['code']].copy()

        # 绩优股条件
        is_quality = (df['roe'] > 10) & (df['net_profit_yoy'] > 0)

        # 回撤幅度（用3个月动量的负值，跌得多=回撤大）
        if 'momentum_3m' in df.columns:
            drawdown = -df['momentum_3m']
        elif 'drawdown' in df.columns:
            drawdown = df['drawdown']
        else:
            drawdown = -df['momentum'] if 'momentum' in df.columns else pd.Series(0, index=df.index)

        # 基础分数：回撤大的高分
        base_score = drawdown.rank(pct=True) * 60 + 20

        # 绩优股回撤加分
        reversal_bonus = pd.Series(0, index=df.index)

        # 绩优股且回撤>15%，大幅加分
        quality_drawdown_mask = is_quality & (drawdown > 15)
        reversal_bonus[quality_drawdown_mask] = 40

        # 绩优股且回撤>10%，中等加分
        quality_drawdown_mask2 = is_quality & (drawdown > 10) & (drawdown <= 15)
        reversal_bonus[quality_drawdown_mask2] = 25

        # 绩优股且小幅回撤，小加分
        quality_drawdown_mask3 = is_quality & (drawdown > 5) & (drawdown <= 10)
        reversal_bonus[quality_drawdown_mask3] = 10

        result['reversal_score'] = (base_score + reversal_bonus).clip(0, 100)

        n_quality_drawdown = quality_drawdown_mask.sum()
        logger.info(f"  绩优股回撤>15%: {n_quality_drawdown}只（重点关注）")

        return result

    # ==================== 风险因子 ====================

    def calculate_low_volatility_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        低波动因子
        波动率低 = 高分
        """
        logger.info("计算低波动因子...")
        result = df[['code']].copy()

        if 'volatility' in df.columns:
            vol = df['volatility'].copy()
            vol = vol.clip(0, 100)
            # 低波动高分
            result['low_vol_score'] = (1 - vol.rank(pct=True)) * 100
        else:
            result['low_vol_score'] = 50

        return result

    # ==================== 综合得分 ====================

    def calculate_composite_score(self, scores_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        计算综合得分
        """
        logger.info("="*60)
        logger.info("计算综合得分")
        logger.info("="*60)

        # 合并所有得分
        result = None
        score_columns = []

        factor_to_column = {
            'dividend_yield': 'dividend_score',
            'pe_value': 'pe_score',
            'roe': 'roe_score',
            'roe_stability': 'roe_stability_score',
            'cash_flow_quality': 'cash_flow_score',
            'profit_growth': 'profit_growth_score',
            'revenue_growth': 'revenue_growth_score',
            'peg': 'peg_score',
            'small_cap': 'small_cap_score',
            'momentum': 'momentum_score',
            'reversal': 'reversal_score',
            'low_volatility': 'low_vol_score',
        }

        for factor_name, score_df in scores_dict.items():
            if result is None:
                result = score_df.copy()
            else:
                score_col = [c for c in score_df.columns if c != 'code'][0]
                result = result.merge(score_df, on='code', how='outer')

        # 填充缺失值
        for col in result.columns:
            if col != 'code' and '_score' in col:
                result[col] = result[col].fillna(50)

        # 计算加权综合得分
        result['composite_score'] = 0

        for factor, weight in self.factor_weights.items():
            if weight > 0:
                col = factor_to_column.get(factor)
                if col and col in result.columns:
                    result['composite_score'] += result[col] * weight

        # 统计
        logger.info(f"综合得分分布:")
        logger.info(f"  最小: {result['composite_score'].min():.1f}")
        logger.info(f"  25%:  {result['composite_score'].quantile(0.25):.1f}")
        logger.info(f"  50%:  {result['composite_score'].median():.1f}")
        logger.info(f"  75%:  {result['composite_score'].quantile(0.75):.1f}")
        logger.info(f"  最大: {result['composite_score'].max():.1f}")

        return result

    def calculate_all_scores(self, factor_data: pd.DataFrame) -> pd.DataFrame:
        """
        一站式计算所有因子得分

        Args:
            factor_data: 包含所有原始因子数据的DataFrame
                必需列: code, dividend_yield, pe, roe, net_profit_yoy, market_cap, momentum
                可选列: roe_y1/y2/y3, ocf, net_profit, revenue_yoy, momentum_3m, volatility
        """
        logger.info("="*60)
        logger.info(f"开始计算所有因子得分 (策略类型: {self.strategy_type})")
        logger.info("="*60)

        scores = {}

        # 价值因子
        if self.factor_weights.get('dividend_yield', 0) > 0:
            scores['dividend'] = self.calculate_dividend_yield_score(factor_data)

        if self.factor_weights.get('pe_value', 0) > 0:
            scores['pe'] = self.calculate_pe_value_score(factor_data)

        # 质量因子
        if self.factor_weights.get('roe', 0) > 0:
            scores['roe'] = self.calculate_roe_score(factor_data)

        if self.factor_weights.get('roe_stability', 0) > 0:
            scores['roe_stability'] = self.calculate_roe_stability_score(factor_data)

        if self.factor_weights.get('cash_flow_quality', 0) > 0:
            scores['cash_flow'] = self.calculate_cash_flow_quality_score(factor_data)

        # 成长因子
        if self.factor_weights.get('profit_growth', 0) > 0:
            scores['profit_growth'] = self.calculate_profit_growth_score(factor_data)

        if self.factor_weights.get('revenue_growth', 0) > 0:
            scores['revenue_growth'] = self.calculate_revenue_growth_score(factor_data)

        if self.factor_weights.get('peg', 0) > 0:
            scores['peg'] = self.calculate_peg_score(factor_data)

        # 规模因子
        if self.factor_weights.get('small_cap', 0) > 0:
            scores['small_cap'] = self.calculate_small_cap_score(factor_data)

        # 动量/反转因子
        if self.factor_weights.get('momentum', 0) > 0:
            scores['momentum'] = self.calculate_momentum_score(factor_data)

        if self.factor_weights.get('reversal', 0) > 0:
            scores['reversal'] = self.calculate_reversal_score(factor_data)

        # 风险因子
        if self.factor_weights.get('low_volatility', 0) > 0:
            scores['low_vol'] = self.calculate_low_volatility_score(factor_data)

        # 综合得分
        result = self.calculate_composite_score(scores)

        logger.info("="*60)
        logger.info("所有因子得分计算完成")
        logger.info("="*60)

        return result


# ==================== 工具函数 ====================

def get_factor_weights_for_market_regime(regime: str) -> Dict[str, float]:
    """
    根据市场环境返回推荐的因子权重

    Args:
        regime: 市场环境
            - 'bull': 牛市 - 偏成长、动量
            - 'bear': 熊市 - 偏价值、低波动、反转
            - 'volatile': 震荡市 - 偏质量、股息
    """
    if regime == 'bull':
        return {
            'dividend_yield': 0.05,
            'pe_value': 0.03,
            'roe': 0.10,
            'roe_stability': 0.05,
            'cash_flow_quality': 0.02,
            'profit_growth': 0.20,
            'revenue_growth': 0.10,
            'peg': 0.12,
            'small_cap': 0.15,
            'momentum': 0.15,
            'reversal': 0.03,
            'low_volatility': 0.00,
        }
    elif regime == 'bear':
        return {
            'dividend_yield': 0.18,
            'pe_value': 0.10,
            'roe': 0.12,
            'roe_stability': 0.10,
            'cash_flow_quality': 0.08,
            'profit_growth': 0.05,
            'revenue_growth': 0.02,
            'peg': 0.05,
            'small_cap': 0.02,
            'momentum': 0.03,
            'reversal': 0.15,
            'low_volatility': 0.10,
        }
    else:  # volatile / neutral
        return EnhancedFactorCalculator.DEFAULT_WEIGHTS.copy()


if __name__ == '__main__':
    # 测试代码
    print("="*60)
    print("测试增强版因子计算器")
    print("="*60)

    # 创建测试数据
    np.random.seed(42)
    n = 100

    test_data = pd.DataFrame({
        'code': [f'{i:06d}' for i in range(n)],
        'dividend_yield': np.random.uniform(0, 0.08, n),
        'pe': np.random.uniform(5, 80, n),
        'roe': np.random.uniform(-5, 35, n),
        'net_profit_yoy': np.random.uniform(-30, 100, n),
        'market_cap': np.random.uniform(20, 3000, n),
        'momentum': np.random.uniform(-20, 30, n),
        'momentum_3m': np.random.uniform(-30, 40, n),
    })

    # 稳健型策略
    print("\n【稳健型策略】")
    calc_stable = EnhancedFactorCalculator(strategy_type='stable')
    scores_stable = calc_stable.calculate_all_scores(test_data)

    print("\nTop 10 稳健型得分:")
    print(scores_stable.nlargest(10, 'composite_score')[['code', 'composite_score']])

    # 进取型策略
    print("\n【进取型策略】")
    calc_aggressive = EnhancedFactorCalculator(strategy_type='aggressive')
    scores_aggressive = calc_aggressive.calculate_all_scores(test_data)

    print("\nTop 10 进取型得分:")
    print(scores_aggressive.nlargest(10, 'composite_score')[['code', 'composite_score']])
