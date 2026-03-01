"""
Alpha158 风格的技术因子库
参考 Microsoft Qlib 的 Alpha158 因子设计

因子类别：
1. K线形态因子 (K-Level): KMID, KCLOSE, KSUMP, KSUMN, KMIN, KMAX
2. 动量因子 (Momentum): ROC, MA, MACD, RSI, BOLL
3. 波动率因子 (Volatility): STD, BETA
4. 成交量因子 (Volume): VMA, VSTD, VWAP
5. 时间衰减因子 (Decay): 多周期加权平均
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Alpha158FactorCalculator:
    """Alpha158 风格因子计算器"""

    def __init__(self, windows: List[int] = None):
        """
        初始化因子计算器

        Args:
            windows: 计算窗口列表，默认 [5, 10, 20, 30, 60]
        """
        self.windows = windows or [5, 10, 20, 30, 60]
        logger.info(f"Alpha158因子计算器初始化，窗口: {self.windows}")

    # ==================== K线形态因子 ====================

    def calc_KMID(self, df: pd.DataFrame, window: int) -> pd.Series:
        """
        KMID = (close - low) / (high - low)
        表示收盘价在当日区间中的位置
        """
        return (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-12)

    def calc_KCLOSE(self, df: pd.DataFrame, window: int) -> pd.Series:
        """
        KCLOSE = (close - N日最低) / (N日最高 - N日最低)
        表示收盘价在N日区间中的位置
        """
        roll_high = df['high'].rolling(window).max()
        roll_low = df['low'].rolling(window).min()
        return (df['close'] - roll_low) / (roll_high - roll_low + 1e-12)

    def calc_KSUMP(self, df: pd.DataFrame, window: int) -> pd.Series:
        """
        KSUMP = sum(max(0, close - pre_close), N) / sum(abs(close - pre_close), N)
        N日上涨幅度占比
        """
        change = df['close'].diff()
        pos_sum = change.clip(lower=0).rolling(window).sum()
        abs_sum = change.abs().rolling(window).sum()
        return pos_sum / (abs_sum + 1e-12)

    def calc_KSUMN(self, df: pd.DataFrame, window: int) -> pd.Series:
        """
        KSUMN = sum(min(0, close - pre_close), N) / sum(abs(close - pre_close), N)
        N日下跌幅度占比
        """
        change = df['close'].diff()
        neg_sum = change.clip(upper=0).rolling(window).sum().abs()
        abs_sum = change.abs().rolling(window).sum()
        return neg_sum / (abs_sum + 1e-12)

    def calc_KMIN(self, df: pd.DataFrame, window: int) -> pd.Series:
        """
        KMIN = min(close, N) / max(close, N)
        N日收盘价最小/最大比
        """
        roll_max = df['close'].rolling(window).max()
        roll_min = df['close'].rolling(window).min()
        return roll_min / (roll_max + 1e-12)

    def calc_KMAX(self, df: pd.DataFrame, window: int) -> pd.Series:
        """
        KMAX = max(close - min(close, N), max(close, N) - close) / max(close, N) - min(close, N)
        """
        roll_max = df['close'].rolling(window).max()
        roll_min = df['close'].rolling(window).min()
        up = roll_max - df['close']
        down = df['close'] - roll_min
        return np.maximum(up, down) / (roll_max - roll_min + 1e-12)

    # ==================== 动量因子 ====================

    def calc_ROC(self, df: pd.DataFrame, window: int) -> pd.Series:
        """
        ROC = (close - N日前close) / N日前close
        N日收益率
        """
        return df['close'].pct_change(periods=window)

    def calc_MA(self, df: pd.DataFrame, window: int) -> pd.Series:
        """
        MA = close / N日均价
        当前价格相对均价的位置
        """
        ma = df['close'].rolling(window).mean()
        return df['close'] / (ma + 1e-12)

    def calc_MACD(self, df: pd.DataFrame, window: int) -> pd.Series:
        """
        MACD相关因子（简化版）
        """
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        return (dif - dea) / (df['close'] + 1e-12)

    def calc_RSI(self, df: pd.DataFrame, window: int) -> pd.Series:
        """
        RSI = 100 - 100 / (1 + RS)
        RS = 平均上涨幅度 / 平均下跌幅度
        """
        change = df['close'].diff()
        gain = change.clip(lower=0)
        loss = (-change).clip(lower=0)

        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()

        rs = avg_gain / (avg_loss + 1e-12)
        return 100 - (100 / (1 + rs))

    def calc_BOLL(self, df: pd.DataFrame, window: int) -> pd.Series:
        """
        BOLL = (close - 中轨) / 标准差
        当前价格在布林带中的位置
        """
        ma = df['close'].rolling(window).mean()
        std = df['close'].rolling(window).std()
        return (df['close'] - ma) / (std + 1e-12)

    # ==================== 波动率因子 ====================

    def calc_STD(self, df: pd.DataFrame, window: int) -> pd.Series:
        """
        STD = N日收益率标准差
        """
        ret = df['close'].pct_change()
        return ret.rolling(window).std()

    def calc_BETA(self, df: pd.DataFrame, window: int, index_close: pd.Series = None) -> pd.Series:
        """
        BETA = Cov(股票收益, 指数收益) / Var(指数收益)
        相对市场的贝塔值
        """
        ret_stock = df['close'].pct_change()

        if index_close is not None:
            ret_index = index_close.pct_change()
        else:
            # 如果没有指数数据，使用自身均值作为替代
            ret_index = ret_stock.rolling(window).mean()

        cov = ret_stock.rolling(window).cov(ret_index)
        var = ret_index.rolling(window).var()
        return cov / (var + 1e-12)

    # ==================== 成交量因子 ====================

    def calc_VMA(self, df: pd.DataFrame, window: int) -> pd.Series:
        """
        VMA = volume / N日平均成交量
        当前成交量相对均量的位置
        """
        vma = df['volume'].rolling(window).mean()
        return df['volume'] / (vma + 1e-12)

    def calc_VSTD(self, df: pd.DataFrame, window: int) -> pd.Series:
        """
        VSTD = N日成交量标准差 / N日成交量均值
        成交量波动率
        """
        vstd = df['volume'].rolling(window).std()
        vma = df['volume'].rolling(window).mean()
        return vstd / (vma + 1e-12)

    def calc_VWAP(self, df: pd.DataFrame, window: int) -> pd.Series:
        """
        VWAP = 成交量加权均价
        当前价格相对VWAP的位置
        """
        vwap = (df['close'] * df['volume']).rolling(window).sum() / \
               (df['volume'].rolling(window).sum() + 1e-12)
        return df['close'] / (vwap + 1e-12)

    # ==================== 高级因子 ====================

    def calc_TURNOVER(self, df: pd.DataFrame, window: int) -> pd.Series:
        """
        换手率因子（如果有换手率数据）
        """
        if 'turnover' in df.columns:
            return df['turnover'].rolling(window).mean()
        elif 'volume' in df.columns and 'shares' in df.columns:
            return (df['volume'] / df['shares']).rolling(window).mean()
        else:
            # 使用成交量相对值作为替代
            return df['volume'] / (df['volume'].rolling(60).mean() + 1e-12)

    def calc_QTLM(self, df: pd.DataFrame, window: int) -> pd.Series:
        """
        QTLM = (high - low) / volume
        价格波动 / 成交量，反映单位成交量带来的价格变化
        """
        return (df['high'] - df['low']) / (df['volume'] + 1e-12)

    def calc_IMAX(self, df: pd.DataFrame, window: int) -> pd.Series:
        """
        IMAX = high / close
        日内最高价相对收盘价的位置
        """
        return df['high'] / (df['close'] + 1e-12)

    def calc_IMIN(self, df: pd.DataFrame, window: int) -> pd.Series:
        """
        IMIN = low / close
        日内最低价相对收盘价的位置
        """
        return df['low'] / (df['close'] + 1e-12)

    # ==================== 时间衰减因子 ====================

    def calc_CORR(self, df: pd.DataFrame, window: int) -> pd.Series:
        """
        CORR = close与volume的N日相关性
        量价相关性
        """
        return df['close'].rolling(window).corr(df['volume'])

    def calc_CMP(self, df: pd.DataFrame, window: int) -> pd.Series:
        """
        CMP = 累计收益率衰减因子
        给近期收益更高权重
        """
        ret = df['close'].pct_change()
        weights = np.exp(-np.arange(window) / window)
        weights = weights / weights.sum()

        def weighted_sum(x):
            if len(x) < window:
                return np.nan
            return (x.values * weights[-len(x):]).sum()

        return ret.rolling(window).apply(weighted_sum, raw=False)

    # ==================== 批量计算 ====================

    def calculate_all_factors(self,
                             price_df: pd.DataFrame,
                             include_volume_factors: bool = True) -> pd.DataFrame:
        """
        计算所有Alpha158风格因子

        Args:
            price_df: DataFrame with columns [date, code, open, high, low, close, volume]
            include_volume_factors: 是否包含成交量因子

        Returns:
            DataFrame with all factors
        """
        logger.info(f"开始计算Alpha158因子，数据量: {len(price_df)}")

        # 确保数据排序
        price_df = price_df.sort_values(['code', 'date'])

        # 按股票分组计算
        all_factors = []

        for code, group in price_df.groupby('code'):
            group = group.sort_values('date')
            factors = {'date': group['date'].values, 'code': code}

            # 对每个窗口计算因子
            for window in self.windows:
                suffix = f'_{window}d'

                # K线形态因子
                factors[f'KMID{suffix}'] = self.calc_KMID(group, window).values
                factors[f'KCLOSE{suffix}'] = self.calc_KCLOSE(group, window).values
                factors[f'KSUMP{suffix}'] = self.calc_KSUMP(group, window).values
                factors[f'KSUMN{suffix}'] = self.calc_KSUMN(group, window).values
                factors[f'KMIN{suffix}'] = self.calc_KMIN(group, window).values
                factors[f'KMAX{suffix}'] = self.calc_KMAX(group, window).values

                # 动量因子
                factors[f'ROC{suffix}'] = self.calc_ROC(group, window).values
                factors[f'MA{suffix}'] = self.calc_MA(group, window).values
                factors[f'RSI{suffix}'] = self.calc_RSI(group, window).values
                factors[f'BOLL{suffix}'] = self.calc_BOLL(group, window).values

                # 波动率因子
                factors[f'STD{suffix}'] = self.calc_STD(group, window).values

                # 成交量因子
                if include_volume_factors and 'volume' in group.columns:
                    factors[f'VMA{suffix}'] = self.calc_VMA(group, window).values
                    factors[f'VSTD{suffix}'] = self.calc_VSTD(group, window).values
                    factors[f'VWAP{suffix}'] = self.calc_VWAP(group, window).values

                # 高级因子
                factors[f'IMAX{suffix}'] = self.calc_IMAX(group, window).values
                factors[f'IMIN{suffix}'] = self.calc_IMIN(group, window).values

            # 计算MACD（不需要窗口参数）
            factors['MACD'] = self.calc_MACD(group, 0).values

            all_factors.append(pd.DataFrame(factors))

        result = pd.concat(all_factors, ignore_index=True)
        logger.info(f"计算完成，共 {len(result.columns) - 2} 个因子")

        return result

    def get_factor_list(self, include_volume_factors: bool = True) -> List[str]:
        """
        获取所有因子名称列表

        Args:
            include_volume_factors: 是否包含成交量因子

        Returns:
            因子名称列表
        """
        factors = []

        for window in self.windows:
            suffix = f'_{window}d'

            # K线形态因子
            factors.extend([
                f'KMID{suffix}', f'KCLOSE{suffix}', f'KSUMP{suffix}',
                f'KSUMN{suffix}', f'KMIN{suffix}', f'KMAX{suffix}'
            ])

            # 动量因子
            factors.extend([
                f'ROC{suffix}', f'MA{suffix}', f'RSI{suffix}', f'BOLL{suffix}'
            ])

            # 波动率因子
            factors.append(f'STD{suffix}')

            # 成交量因子
            if include_volume_factors:
                factors.extend([
                    f'VMA{suffix}', f'VSTD{suffix}', f'VWAP{suffix}'
                ])

            # 高级因子
            factors.extend([f'IMAX{suffix}', f'IMIN{suffix}'])

        # MACD
        factors.append('MACD')

        return factors


def quick_calc_factors(price_df: pd.DataFrame,
                       windows: List[int] = None) -> pd.DataFrame:
    """
    快速计算因子的便捷函数

    Args:
        price_df: DataFrame with [date, code, open, high, low, close, volume]
        windows: 计算窗口列表

    Returns:
        DataFrame with all factors
    """
    calculator = Alpha158FactorCalculator(windows=windows)
    return calculator.calculate_all_factors(price_df)


if __name__ == '__main__':
    # 测试代码
    np.random.seed(42)

    # 生成模拟数据
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    codes = ['000001', '000002', '600000']

    data = []
    for code in codes:
        base_price = np.random.uniform(10, 50)
        for i, date in enumerate(dates):
            open_p = base_price * (1 + np.random.randn() * 0.02)
            close_p = open_p * (1 + np.random.randn() * 0.02)
            high_p = max(open_p, close_p) * (1 + abs(np.random.randn()) * 0.01)
            low_p = min(open_p, close_p) * (1 - abs(np.random.randn()) * 0.01)
            volume = np.random.uniform(1e6, 1e7)

            data.append({
                'date': date,
                'code': code,
                'open': open_p,
                'high': high_p,
                'low': low_p,
                'close': close_p,
                'volume': volume
            })
            base_price = close_p

    df = pd.DataFrame(data)

    # 计算因子
    calculator = Alpha158FactorCalculator(windows=[5, 10, 20])
    factors_df = calculator.calculate_all_factors(df)

    print("\n计算的因子列表:")
    print(calculator.get_factor_list())

    print(f"\n因子数据形状: {factors_df.shape}")
    print("\n因子数据样例:")
    print(factors_df.head())
