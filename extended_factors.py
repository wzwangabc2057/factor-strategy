"""
扩展因子库 - 50+ 技术因子
目标：提升模型 IC 和预测能力
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class ExtendedFactorCalculator:
    """扩展因子计算器 - 50+ 因子"""

    def __init__(self):
        self.factor_names = []

    def calculate_all_factors(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有扩展因子

        Args:
            prices_df: DataFrame with columns [date, code, open, high, low, close, volume, amount]

        Returns:
            DataFrame with factors for each stock
        """
        logger.info("计算扩展因子...")

        grouped = prices_df.groupby('code')
        factor_list = []

        for code, group in grouped:
            group = group.sort_values('date')

            # 提取价格和成交量数据
            close = group['close'].values.astype(float)
            high = group['high'].values.astype(float)
            low = group['low'].values.astype(float)
            open_price = group['open'].values.astype(float)
            volume = group['volume'].values.astype(float)
            amount = group.get('amount', volume * close).values.astype(float)

            if len(close) < 60:
                continue

            factors = {'code': code, 'date': group['date'].iloc[-1]}

            try:
                # ========== 动量因子 (10个) ==========
                for period in [5, 10, 20, 60]:
                    if len(close) > period:
                        factors[f'momentum_{period}d'] = (close[-1] / close[-period] - 1) * 100

                # 价格动量加速度
                if len(close) > 20:
                    mom_5 = (close[-1] / close[-5] - 1)
                    mom_10 = (close[-1] / close[-10] - 1)
                    mom_20 = (close[-1] / close[-20] - 1)
                    factors['momentum_accel'] = (mom_5 - mom_10/2) * 100
                    factors['momentum_diff'] = (mom_5 - mom_20/4) * 100

                # 收益偏度
                if len(close) > 20:
                    returns = np.diff(np.log(close[-21:]))
                    factors['return_skew'] = float(pd.Series(returns).skew())

                # ========== RSI 因子 (3个) ==========
                for period in [6, 14, 28]:
                    if len(close) > period + 1:
                        factors[f'rsi_{period}'] = self._calc_rsi(close, period)

                # ========== 波动率因子 (6个) ==========
                for period in [10, 20, 60]:
                    if len(close) > period:
                        returns = np.diff(np.log(close[-period:]))
                        factors[f'volatility_{period}d'] = np.std(returns) * np.sqrt(252) * 100
                        factors[f'vol_of_vol_{period}d'] = np.std(np.abs(returns)) * 100

                # Parkinson 波动率
                if len(high) > 10 and len(low) > 10:
                    factors['parkinson_vol'] = self._calc_parkinson_vol(high[-10:], low[-10:])

                # ========== 成交量因子 (6个) ==========
                for period in [5, 10, 20]:
                    if len(volume) > period and np.mean(volume[-period:]) > 0:
                        factors[f'volume_ratio_{period}d'] = volume[-1] / np.mean(volume[-period:])
                        factors[f'volume_trend_{period}d'] = np.corrcoef(np.arange(period), volume[-period:])[0,1]

                # OBV 动量
                if len(close) > 10 and len(volume) > 10:
                    factors['obv_momentum'] = self._calc_obv_momentum(close[-11:], volume[-11:])

                # ========== 价格位置因子 (6个) ==========
                for period in [10, 20, 60]:
                    if len(close) > period:
                        period_high = np.max(high[-period:])
                        period_low = np.min(low[-period:])
                        if period_high > period_low:
                            factors[f'price_position_{period}d'] = (close[-1] - period_low) / (period_high - period_low) * 100
                            factors[f'distance_high_{period}d'] = (period_high - close[-1]) / close[-1] * 100

                # ========== 均线因子 (8个) ==========
                for period in [5, 10, 20, 60]:
                    if len(close) > period:
                        ma = np.mean(close[-period:])
                        factors[f'ma_{period}d'] = ma / close[-1] - 1  # MA偏离
                        factors[f'ma_deviation_{period}d'] = (close[-1] - ma) / ma * 100

                # 均线多头排列
                if len(close) > 60:
                    ma5 = np.mean(close[-5:])
                    ma10 = np.mean(close[-10:])
                    ma20 = np.mean(close[-20:])
                    ma60 = np.mean(close[-60:])
                    factors['ma_bullish'] = 1 if (ma5 > ma10 > ma20 > ma60) else 0

                # ========== MACD 因子 (3个) ==========
                if len(close) > 35:
                    macd, signal, hist = self._calc_macd(close)
                    factors['macd'] = macd
                    factors['macd_signal'] = signal
                    factors['macd_hist'] = hist

                # ========== 布林带因子 (3个) ==========
                if len(close) > 20:
                    bb_upper, bb_lower, bb_width = self._calc_bollinger(close)
                    factors['bb_position'] = (close[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper > bb_lower else 0.5
                    factors['bb_width'] = bb_width
                    factors['bb_squeeze'] = 1 if bb_width < 0.05 else 0

                # ========== KDJ 因子 (3个) ==========
                if len(close) > 20:
                    k, d, j = self._calc_kdj(high[-20:], low[-20:], close[-20:])
                    factors['kdj_k'] = k
                    factors['kdj_d'] = d
                    factors['kdj_j'] = j

                # ========== 形态因子 (4个) ==========
                if len(close) > 5:
                    # 上影线
                    factors['upper_shadow'] = (high[-1] - max(open_price[-1], close[-1])) / (high[-1] - low[-1] + 1e-8) * 100
                    # 下影线
                    factors['lower_shadow'] = (min(open_price[-1], close[-1]) - low[-1]) / (high[-1] - low[-1] + 1e-8) * 100
                    # 实体大小
                    factors['body_size'] = abs(close[-1] - open_price[-1]) / (high[-1] - low[-1] + 1e-8) * 100
                    # 涨跌
                    factors['is_bullish'] = 1 if close[-1] > open_price[-1] else 0

                # ========== 趋势因子 (3个) ==========
                if len(close) > 20:
                    # 价格斜率
                    x = np.arange(20)
                    slope, _ = np.polyfit(x, close[-20:], 1)
                    factors['price_slope'] = slope / close[-1] * 100

                    # 趋势强度 (ADXR 风格)
                    factors['trend_strength'] = abs(factors.get('price_slope', 0)) * factors.get('rsi_14', 50) / 50

                # 价格效率
                if len(close) > 20:
                    path_length = np.sum(np.abs(np.diff(close[-20:])))
                    straight_distance = abs(close[-1] - close[-20])
                    factors['price_efficiency'] = straight_distance / (path_length + 1e-8)

                factor_list.append(factors)

            except Exception as e:
                logger.debug(f"计算 {code} 因子失败: {e}")
                continue

        result = pd.DataFrame(factor_list)
        self.factor_names = [c for c in result.columns if c not in ['code', 'date']]
        logger.info(f"计算完成: {len(result)} 只股票, {len(self.factor_names)} 个因子")

        return result

    def _calc_rsi(self, close: np.ndarray, period: int) -> float:
        """计算 RSI"""
        delta = np.diff(close[-period-1:])
        gain = np.mean(delta[delta > 0]) if np.any(delta > 0) else 0
        loss = np.mean(-delta[delta < 0]) if np.any(delta < 0) else 0.001
        return 100 - 100 / (1 + gain / loss)

    def _calc_parkinson_vol(self, high: np.ndarray, low: np.ndarray) -> float:
        """计算 Parkinson 波动率"""
        return np.sqrt(np.mean(np.log(high / low) ** 2) / (4 * np.log(2))) * np.sqrt(252) * 100

    def _calc_obv_momentum(self, close: np.ndarray, volume: np.ndarray) -> float:
        """计算 OBV 动量"""
        obv = np.zeros(len(close))
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
        return (obv[-1] - obv[-6]) / (abs(obv[-6]) + 1) * 100

    def _calc_macd(self, close: np.ndarray) -> Tuple[float, float, float]:
        """计算 MACD"""
        # EMA
        ema12 = pd.Series(close).ewm(span=12).mean().values
        ema26 = pd.Series(close).ewm(span=26).mean().values
        macd = ema12[-1] - ema26[-1]
        macd_series = ema12 - ema26
        signal = pd.Series(macd_series).ewm(span=9).mean().values[-1]
        hist = macd - signal
        return float(macd), float(signal), float(hist)

    def _calc_bollinger(self, close: np.ndarray) -> Tuple[float, float, float]:
        """计算布林带"""
        ma = np.mean(close[-20:])
        std = np.std(close[-20:])
        bb_upper = ma + 2 * std
        bb_lower = ma - 2 * std
        bb_width = (bb_upper - bb_lower) / ma
        return bb_upper, bb_lower, bb_width

    def _calc_kdj(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> Tuple[float, float, float]:
        """计算 KDJ"""
        period = len(close)
        lowest = np.min(low)
        highest = np.max(high)

        if highest == lowest:
            rsv = 50
        else:
            rsv = (close[-1] - lowest) / (highest - lowest) * 100

        # 简化计算
        k = rsv  # 实际应该用平滑
        d = rsv
        j = 3 * k - 2 * d

        return float(k), float(d), float(j)


if __name__ == '__main__':
    # 测试
    calculator = ExtendedFactorCalculator()
    print(f"因子计算器已就绪")
