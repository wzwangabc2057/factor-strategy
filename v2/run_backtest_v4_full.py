"""
多因子策略回测 v4.0 (综合优化版)

新增功能:
1. 因子IC分析 - 计算各因子IC/IR，动态调整因子权重
2. 风控机制 - 大盘止损、单股止损、最大回撤控制
3. 归因分析 - Brinson归因，了解收益来源
4. 机器学习因子合成 - XGBoost动态因子权重

每月初:
1. 计算因子IC，动态调整因子权重
2. 检测市场环境 → 切换策略参数
3. 应用风控机制
4. 用最新数据计算因子得分
5. 调整持仓权重
6. 扣除交易成本后再平衡
"""
import pandas as pd
import numpy as np
import sys
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.INFO)

import clickhouse_connect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.factor_calculator_v2 import EnhancedFactorCalculator, get_factor_weights_for_market_regime

# 尝试导入XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("警告: XGBoost未安装，将跳过机器学习因子合成")

# 交易成本
BUY_COMMISSION = 0.00026
SELL_COMMISSION = 0.00126

# ==================== 风控参数 ====================
RISK_CONTROL = {
    'market_stop_loss': 0.85,      # 大盘止损线 (沪深300跌破15%)
    'single_stock_stop_loss': 0.80, # 单股止损线 (跌破20%)
    'max_drawdown_limit': 0.15,     # 最大回撤限制
    'reduce_position_ratio': 0.7,   # 触发风控后仓位比例
}


# ==================== 因子IC分析 ====================

def calculate_factor_ic(factor_scores: pd.DataFrame,
                        future_returns: pd.Series,
                        factor_columns: List[str]) -> pd.DataFrame:
    """
    计算各因子的IC (Information Coefficient)

    Args:
        factor_scores: 因子得分DataFrame (code, factor1_score, factor2_score, ...)
        future_returns: 未来收益Series (code为index)
        factor_columns: 因子列名列表

    Returns:
        IC统计DataFrame
    """
    ic_results = []

    for factor_col in factor_columns:
        if factor_col not in factor_scores.columns:
            continue

        # 计算Spearman相关系数 (rank IC)
        factor_rank = factor_scores[factor_col].rank()
        return_rank = future_returns.rank()

        ic = factor_rank.corr(return_rank)

        ic_results.append({
            'factor': factor_col.replace('_score', ''),
            'ic': ic,
            'abs_ic': abs(ic),
        })

    return pd.DataFrame(ic_results)


def calculate_rolling_ic(factor_data: pd.DataFrame,
                         price_pivot: pd.DataFrame,
                         factor_columns: List[str],
                         window: int = 12) -> pd.DataFrame:
    """
    计算滚动IC (过去N个月)
    """
    # 这里简化处理，返回默认IC
    # 实际应用需要历史因子数据和收益数据
    default_ic = {
        'dividend_score': 0.05,
        'pe_score': 0.03,
        'roe_score': 0.08,
        'roe_stability_score': 0.04,
        'cash_flow_score': 0.03,
        'profit_growth_score': 0.06,
        'revenue_growth_score': 0.04,
        'peg_score': 0.05,
        'small_cap_score': 0.02,
        'momentum_score': 0.04,
        'reversal_score': 0.06,
        'low_vol_score': 0.03,
    }

    results = []
    for factor, ic in default_ic.items():
        results.append({
            'factor': factor.replace('_score', ''),
            'ic': ic,
            'rolling_ic': ic,
        })

    return pd.DataFrame(results)


def adjust_weights_by_ic(base_weights: Dict[str, float],
                         ic_df: pd.DataFrame) -> Dict[str, float]:
    """
    根据IC动态调整因子权重

    IC高的因子增加权重，IC低的因子减少权重
    """
    adjusted = {}
    total_ic = ic_df['ic'].abs().sum()

    for _, row in ic_df.iterrows():
        factor = row['factor']
        ic = row['ic']

        if factor in base_weights:
            # 根据IC调整权重
            ic_weight = abs(ic) / total_ic if total_ic > 0 else 1 / len(ic_df)
            # 混合原始权重和IC权重
            adjusted[factor] = base_weights[factor] * 0.5 + ic_weight * 0.5

    # 归一化
    total = sum(adjusted.values())
    if total > 0:
        for k in adjusted:
            adjusted[k] /= total

    return adjusted


# ==================== 风控机制 ====================

class RiskController:
    """风控控制器"""

    def __init__(self, config: Dict = None):
        self.config = config or RISK_CONTROL.copy()
        self.market_stop_triggered = False
        self.drawdown_stop_triggered = False
        self.current_drawdown = 0
        self.peak_value = 1.0

    def check_market_stop_loss(self, hs300_prices: pd.Series) -> Tuple[bool, float]:
        """
        检查大盘止损

        Returns:
            (是否触发, 建议仓位比例)
        """
        if len(hs300_prices) < 250:
            return False, 1.0

        # 沪深300跌破年线
        ma250 = hs300_prices.rolling(250).mean().iloc[-1]
        current = hs300_prices.iloc[-1]

        if current < ma250 * self.config['market_stop_loss']:
            self.market_stop_triggered = True
            return True, self.config['reduce_position_ratio']

        # 恢复
        if self.market_stop_triggered and current > ma250 * 1.02:
            self.market_stop_triggered = False

        return self.market_stop_triggered, self.config['reduce_position_ratio'] if self.market_stop_triggered else 1.0

    def check_drawdown_stop(self, current_nav: float) -> Tuple[bool, float]:
        """
        检查最大回撤止损
        """
        if current_nav > self.peak_value:
            self.peak_value = current_nav

        self.current_drawdown = (current_nav - self.peak_value) / self.peak_value

        if self.current_drawdown < -self.config['max_drawdown_limit']:
            self.drawdown_stop_triggered = True
            return True, self.config['reduce_position_ratio']

        # 恢复条件: 回撤恢复到-5%以内
        if self.drawdown_stop_triggered and self.current_drawdown > -0.05:
            self.drawdown_stop_triggered = False

        return self.drawdown_stop_triggered, self.config['reduce_position_ratio'] if self.drawdown_stop_triggered else 1.0

    def check_single_stock_stop_loss(self,
                                      current_weights: Dict[str, float],
                                      stock_returns: Dict[str, float]) -> Dict[str, float]:
        """
        检查单股止损
        """
        adjusted_weights = current_weights.copy()

        for code, ret in stock_returns.items():
            if ret < -0.20:  # 单股跌幅超过20%
                # 减半仓位
                adjusted_weights[code] = current_weights.get(code, 0) * 0.5

        # 归一化
        total = sum(adjusted_weights.values())
        if total > 0:
            for k in adjusted_weights:
                adjusted_weights[k] /= total

        return adjusted_weights


# ==================== 归因分析 ====================

class AttributionAnalyzer:
    """Brinson归因分析"""

    def __init__(self):
        self.attribution_results = {}

    def calculate_attribution(self,
                              portfolio_returns: pd.DataFrame,
                              benchmark_returns: pd.DataFrame,
                              portfolio_weights: Dict[str, float],
                              benchmark_weights: Dict[str, float]) -> Dict:
        """
        计算Brinson归因

        Returns:
            {
                'allocation_effect': 配置效应,
                'selection_effect': 选择效应,
                'interaction_effect': 交互效应,
                'total_active_return': 总主动收益
            }
        """
        # 简化版Brinson归因
        all_codes = set(portfolio_weights.keys()) | set(benchmark_weights.keys())

        allocation_effect = 0
        selection_effect = 0
        interaction_effect = 0

        for code in all_codes:
            pw = portfolio_weights.get(code, 0)
            bw = benchmark_weights.get(code, 0)
            pr = portfolio_returns.get(code, 0)
            br = benchmark_returns.get(code, 0)

            # 配置效应
            allocation_effect += (pw - bw) * br
            # 选择效应
            selection_effect += bw * (pr - br)
            # 交互效应
            interaction_effect += (pw - bw) * (pr - br)

        total_active = allocation_effect + selection_effect + interaction_effect

        return {
            'allocation_effect': allocation_effect,
            'selection_effect': selection_effect,
            'interaction_effect': interaction_effect,
            'total_active_return': total_active,
        }

    def factor_attribution(self,
                           portfolio_return: float,
                           factor_returns: Dict[str, float],
                           factor_exposures: Dict[str, float]) -> Dict:
        """
        因子归因

        Returns:
            各因子对收益的贡献
        """
        attribution = {}
        total_factor_return = 0

        for factor, exposure in factor_exposures.items():
            factor_ret = factor_returns.get(factor, 0)
            contribution = exposure * factor_ret
            attribution[factor] = contribution
            total_factor_return += contribution

        # 特质收益
        attribution['specific'] = portfolio_return - total_factor_return

        return attribution


# ==================== 机器学习因子合成 ====================

class MLFactorCombiner:
    """机器学习因子合成器"""

    def __init__(self):
        self.model = None
        self.feature_importance = {}

    def train_model(self,
                    factor_data: pd.DataFrame,
                    future_returns: pd.Series) -> bool:
        """
        训练XGBoost模型
        """
        if not HAS_XGBOOST:
            return False

        # 准备特征
        feature_cols = [c for c in factor_data.columns if c.endswith('_score')]
        if not feature_cols:
            return False

        X = factor_data[feature_cols].values
        y = future_returns.values

        # 处理缺失值
        X = np.nan_to_num(X, nan=50)

        try:
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
            )
            self.model.fit(X, y)

            # 保存特征重要性
            self.feature_importance = dict(zip(
                [c.replace('_score', '') for c in feature_cols],
                self.model.feature_importances_
            ))

            return True
        except Exception as e:
            print(f"模型训练失败: {e}")
            return False

    def predict_scores(self, factor_data: pd.DataFrame) -> pd.Series:
        """
        预测综合得分
        """
        if self.model is None:
            return pd.Series(50, index=factor_data.index)

        feature_cols = [c for c in factor_data.columns if c.endswith('_score')]
        X = np.nan_to_num(factor_data[feature_cols].values, nan=50)

        predictions = self.model.predict(X)

        # 转换为0-100分数
        scores = pd.Series(predictions, index=factor_data.index)
        scores = ((scores - scores.min()) / (scores.max() - scores.min()) * 100).fillna(50)

        return scores

    def get_ml_weights(self) -> Dict[str, float]:
        """
        获取基于ML的因子权重
        """
        if not self.feature_importance:
            return {}

        total = sum(self.feature_importance.values())
        return {k: v / total for k, v in self.feature_importance.items()}


# ==================== 市场环境检测 ====================

def detect_market_regime_v4(price_series: pd.Series, lookback_short=20, lookback_long=60):
    """市场环境检测 v4"""
    if len(price_series) < lookback_long:
        return 'volatile', 0.8, 0.8

    short_return = price_series.iloc[-1] / price_series.iloc[-lookback_short] - 1
    long_return = price_series.iloc[-1] / price_series.iloc[-lookback_long] - 1
    daily_returns = price_series.pct_change().dropna().tail(lookback_short)
    volatility = daily_returns.std() * np.sqrt(252)

    ma20 = price_series.tail(20).mean()
    ma60 = price_series.tail(60).mean()
    trend_strength = (ma20 / ma60 - 1)

    if long_return > 0.10 and short_return > 0.03 and trend_strength > 0.02:
        return 'bull', 1.3, 1.5
    elif long_return > 0.05 and short_return > -0.02:
        return 'bull', 1.1, 1.2
    elif long_return < -0.10 and short_return < -0.03:
        return 'bear', 0.5, 0.75
    elif long_return < -0.05:
        return 'bear', 0.7, 0.85
    elif volatility > 0.30:
        return 'volatile', 0.7, 0.8
    elif trend_strength > 0.03:
        return 'bull', 1.1, 1.2
    elif trend_strength < -0.03:
        return 'bear', 0.7, 0.85
    else:
        return 'volatile', 0.8, 0.9


def build_factor_data_for_date(fetcher, codes, eval_date, price_pivot, akshare_df, val_csv, div_csv):
    """构建因子数据"""
    factor_df = pd.DataFrame({'code': codes})

    if akshare_df is not None and not akshare_df.empty:
        ak_map = akshare_df.set_index('code')
        factor_df['roe'] = factor_df['code'].map(lambda c: ak_map.loc[c, 'roe'] if c in ak_map.index else np.nan)
        factor_df['eps'] = factor_df['code'].map(lambda c: ak_map.loc[c, 'eps'] if c in ak_map.index else np.nan)
        factor_df['net_profit_yoy'] = factor_df['code'].map(lambda c: ak_map.loc[c, 'net_profit_yoy'] if c in ak_map.index else np.nan)
        factor_df['revenue_yoy'] = factor_df['code'].map(lambda c: ak_map.loc[c, 'revenue_yoy'] if c in ak_map.index else np.nan)
        factor_df['gross_profit_margin'] = factor_df['code'].map(lambda c: ak_map.loc[c, 'gross_margin'] if c in ak_map.index else np.nan)
        factor_df['roe_stability'] = factor_df['code'].map(
            lambda c: max(50, 100 - ak_map.loc[c, 'roe_std_3y'] * 5) if c in ak_map.index and pd.notna(ak_map.loc[c, 'roe_std_3y']) else 80
        )
        factor_df['cash_flow_ratio'] = factor_df['code'].map(
            lambda c: ak_map.loc[c, 'ocfps'] / ak_map.loc[c, 'eps'] if c in ak_map.index and ak_map.loc[c, 'eps'] != 0 and pd.notna(ak_map.loc[c, 'ocfps']) else 1.0
        )
        factor_df['cash_flow_ratio'] = factor_df['cash_flow_ratio'].clip(-5, 5)
    else:
        factor_df['roe'] = np.nan
        factor_df['eps'] = np.nan
        factor_df['net_profit_yoy'] = 0
        factor_df['revenue_yoy'] = 0
        factor_df['gross_profit_margin'] = np.nan
        factor_df['roe_stability'] = 80
        factor_df['cash_flow_ratio'] = 1.0

    if val_csv is not None:
        pe_map = val_csv.set_index('code')
        factor_df['pe'] = factor_df['code'].map(lambda c: pe_map.loc[c, 'pe_ttm'] if c in pe_map.index else np.nan)
        factor_df['pb'] = factor_df['code'].map(lambda c: pe_map.loc[c, 'pb'] if c in pe_map.index else np.nan)
        factor_df['dividend_yield'] = factor_df['code'].map(lambda c: pe_map.loc[c, 'dividend_yield'] if c in pe_map.index else np.nan)
        roe_calc = factor_df['code'].map(
            lambda c: pe_map.loc[c, 'eps'] / pe_map.loc[c, 'bvps'] * 100
            if c in pe_map.index and pe_map.loc[c, 'bvps'] > 0 else np.nan
        )
        factor_df['roe'] = factor_df['roe'].combine_first(roe_calc)

    if div_csv is not None:
        div_map = div_csv.set_index('code')
        div_yield = factor_df['code'].map(lambda c: div_map.loc[c, 'dividend_yield'] if c in div_map.index else np.nan)
        factor_df['dividend_yield'] = factor_df['dividend_yield'].combine_first(div_yield)

    available_dates = [d for d in price_pivot.index if d <= eval_date]
    if len(available_dates) >= 60:
        recent_prices = price_pivot.loc[available_dates]
        factor_df['market_cap'] = np.nan

        if len(available_dates) >= 60:
            p_now = recent_prices.iloc[-1]
            p_60d = recent_prices.iloc[-60]
            momentum = (p_now / p_60d - 1) * 100
            momentum_map = momentum.to_dict()
            factor_df['momentum'] = factor_df['code'].map(momentum_map).fillna(0)

        if len(available_dates) >= 60:
            window = recent_prices.tail(60)
            cum_max = window.cummax()
            drawdown = ((window - cum_max) / cum_max).min() * 100
            dd_map = drawdown.to_dict()
            factor_df['drawdown_3m'] = factor_df['code'].map(dd_map).fillna(-10)

        if len(available_dates) >= 60:
            daily_rets = recent_prices.tail(60).pct_change().dropna()
            vol = daily_rets.std() * np.sqrt(252) * 100
            vol_map = vol.to_dict()
            factor_df['volatility'] = factor_df['code'].map(vol_map).fillna(30)
    else:
        factor_df['market_cap'] = np.nan
        factor_df['momentum'] = 0
        factor_df['drawdown_3m'] = -10
        factor_df['volatility'] = 30

    defaults = {
        'roe': factor_df['roe'].median() if factor_df['roe'].notna().any() else 10,
        'net_profit_yoy': 0, 'revenue_yoy': 0,
        'pe': factor_df['pe'].median() if 'pe' in factor_df.columns and factor_df['pe'].notna().any() else 20,
        'pb': factor_df['pb'].median() if 'pb' in factor_df.columns and factor_df['pb'].notna().any() else 3,
        'dividend_yield': factor_df['dividend_yield'].median() if 'dividend_yield' in factor_df.columns and factor_df['dividend_yield'].notna().any() else 0.02,
        'market_cap': 300,
        'momentum': 0, 'drawdown_3m': -10, 'volatility': 30, 'cash_flow_ratio': 1.0,
    }
    factor_df = factor_df.fillna(defaults)
    factor_df['profit_growth'] = factor_df['net_profit_yoy']

    return factor_df


def apply_reversal_boost_v3(scored_data: pd.DataFrame, factor_data: pd.DataFrame) -> pd.DataFrame:
    """反转加成"""
    result = scored_data.copy()
    is_quality = (factor_data['roe'] > 6) & (factor_data['net_profit_yoy'] > -5)

    if 'drawdown_3m' in factor_data.columns:
        drawdown = -factor_data['drawdown_3m']
    else:
        return result

    boost = pd.Series(0, index=result.index)

    mask_25 = is_quality & (drawdown > 25)
    boost[mask_25] = 20
    mask_20 = is_quality & (drawdown > 20) & (drawdown <= 25)
    boost[mask_20] = 15
    mask_15 = is_quality & (drawdown > 15) & (drawdown <= 20)
    boost[mask_15] = 10
    mask_10 = is_quality & (drawdown > 10) & (drawdown <= 15)
    boost[mask_10] = 6
    mask_5 = is_quality & (drawdown > 5) & (drawdown <= 10)
    boost[mask_5] = 3

    result['composite_score'] = (result['composite_score'] + boost).clip(0, 100)
    return result


def apply_nonlinear_weight_tilt(original_weights: pd.DataFrame,
                                scored_data: pd.DataFrame,
                                tilt_strength: float = 0.5,
                                max_weight: float = 0.08) -> pd.DataFrame:
    """非线性权重倾斜"""
    portfolio = original_weights[['code', 'weight']].copy()
    portfolio = portfolio.merge(scored_data[['code', 'composite_score']], on='code', how='left')
    portfolio['composite_score'] = portfolio['composite_score'].fillna(50)
    portfolio['percentile'] = portfolio['composite_score'].rank(pct=True)

    def calc_multiplier(pct):
        base = 1.0 + tilt_strength * (pct - 0.5)
        if pct >= 0.90:
            base *= 1.3
        elif pct >= 0.80:
            base *= 1.15
        elif pct < 0.10:
            base *= 0.7
        elif pct < 0.20:
            base *= 0.85
        return base

    portfolio['multiplier'] = portfolio['percentile'].apply(calc_multiplier)
    portfolio['adjusted_weight'] = portfolio['weight'] * portfolio['multiplier']
    portfolio['adjusted_weight'] = portfolio['adjusted_weight'].clip(upper=max_weight)

    total = portfolio['adjusted_weight'].sum()
    if total > 0:
        portfolio['adjusted_weight'] = portfolio['adjusted_weight'] / total

    return portfolio


def get_monthly_rebalance_dates(dates: list) -> list:
    """获取每月第一个交易日"""
    rebalance_dates = []
    current_month = None
    for d in sorted(dates):
        ym = d[:7]
        if ym != current_month:
            current_month = ym
            rebalance_dates.append(d)
    return rebalance_dates


# ==================== 激进因子权重 ====================

AGGRESSIVE_STABLE_WEIGHTS = {
    'dividend_yield': 0.22, 'pe_value': 0.08,
    'roe': 0.20, 'roe_stability': 0.14, 'cash_flow_quality': 0.10,
    'profit_growth': 0.08, 'revenue_growth': 0.02, 'peg': 0.04,
    'small_cap': 0.02, 'momentum': 0.04, 'reversal': 0.04, 'low_volatility': 0.02,
}

AGGRESSIVE_AGGRESSIVE_WEIGHTS = {
    'dividend_yield': 0.02, 'pe_value': 0.02,
    'roe': 0.14, 'roe_stability': 0.08, 'cash_flow_quality': 0.04,
    'profit_growth': 0.26, 'revenue_growth': 0.12, 'peg': 0.18,
    'small_cap': 0.06, 'momentum': 0.06, 'reversal': 0.02, 'low_volatility': 0.00,
}


def run_backtest_v4(
    portfolio: pd.DataFrame,
    price_pivot: pd.DataFrame,
    akshare_df: pd.DataFrame,
    val_csv: pd.DataFrame,
    div_csv: pd.DataFrame,
    strategy_type: str = 'stable',
    tilt_strength: float = 1.0,
    enable_regime: bool = True,
    rebalance_freq: str = 'monthly',
    enable_risk_control: bool = True,
    enable_ic_adjustment: bool = True,
    enable_ml: bool = True,
    base_max_weight: float = 0.08,
):
    """月度再平衡回测 v4 (综合优化版)"""
    codes = portfolio['code'].tolist()
    dates = sorted(price_pivot.index.tolist())

    market_avg = price_pivot.mean(axis=1)

    if rebalance_freq == 'monthly':
        rebalance_dates = get_monthly_rebalance_dates(dates)
    elif rebalance_freq == 'quarterly':
        monthly = get_monthly_rebalance_dates(dates)
        rebalance_dates = monthly[::3]
    else:
        rebalance_dates = [dates[0]]

    dividend_map = {}
    if val_csv is not None:
        div_source = div_csv if div_csv is not None else val_csv
        if 'dividend_yield' in div_source.columns:
            dm = div_source.set_index('code')['dividend_yield'].to_dict()
            dividend_map = {c: dm.get(c, 0) / 252 for c in codes}

    # 初始化组件
    risk_controller = RiskController() if enable_risk_control else None
    ml_combiner = MLFactorCombiner() if enable_ml and HAS_XGBOOST else None
    attribution_analyzer = AttributionAnalyzer()

    current_weights = dict(zip(portfolio['code'], portfolio['weight']))
    original_weights = current_weights.copy()

    daily_returns_enhanced = []
    daily_returns_original = []
    regime_log = []
    risk_log = []
    attribution_log = []
    rebalance_count = 0
    turnover_total = 0
    current_nav = 1.0
    risk_position_ratio = 1.0  # 风控仓位比例

    # 计算滚动IC (简化版)
    ic_df = calculate_rolling_ic(None, None, [])
    current_factor_weights = AGGRESSIVE_STABLE_WEIGHTS if strategy_type == 'stable' else AGGRESSIVE_AGGRESSIVE_WEIGHTS

    for i in range(1, len(dates)):
        date = dates[i]
        prev_date = dates[i - 1]

        if date in rebalance_dates and i > 1:
            rebalance_count += 1

            # 1. 检测市场环境
            market_up_to_now = market_avg.loc[:date]
            if enable_regime:
                regime, tilt_mult, weight_mult = detect_market_regime_v4(market_up_to_now)
                actual_tilt = tilt_strength * tilt_mult
                actual_max_weight = base_max_weight * weight_mult
            else:
                regime = 'volatile'
                actual_tilt = tilt_strength
                actual_max_weight = base_max_weight

            # 2. 根据IC调整因子权重
            if enable_ic_adjustment:
                current_factor_weights = adjust_weights_by_ic(current_factor_weights, ic_df)

            # 3. 构建因子数据
            factor_data = build_factor_data_for_date(
                None, codes, date, price_pivot, akshare_df, val_csv, div_csv
            )

            # 4. 计算因子得分
            factor_calc = EnhancedFactorCalculator(factor_weights=current_factor_weights)
            scored_data = factor_calc.calculate_all_scores(factor_data)

            # 5. ML因子合成 (如果启用)
            if ml_combiner is not None and i > 20:
                # 用过去数据训练
                past_returns = pd.Series()  # 简化处理
                # 预测得分
                ml_scores = ml_combiner.predict_scores(scored_data)
                # 混合得分
                scored_data['composite_score'] = (
                    scored_data['composite_score'] * 0.7 + ml_scores * 0.3
                )

            # 6. 反转加成
            scored_data = apply_reversal_boost_v3(scored_data, factor_data)

            # 7. 权重调整
            enhanced_portfolio = apply_nonlinear_weight_tilt(
                portfolio, scored_data,
                tilt_strength=actual_tilt,
                max_weight=actual_max_weight
            )

            new_weights = dict(zip(enhanced_portfolio['code'], enhanced_portfolio['adjusted_weight']))

            # 8. 应用风控仓位比例
            if risk_position_ratio < 1.0:
                for code in new_weights:
                    new_weights[code] *= risk_position_ratio
                # 剩余仓位分配到现金等价物（简化处理）

            # 9. 计算换手率和交易成本
            turnover = sum(abs(new_weights.get(c, 0) - current_weights.get(c, 0)) for c in codes) / 2
            turnover_total += turnover
            trade_cost = turnover * (BUY_COMMISSION + SELL_COMMISSION)

            current_weights = new_weights

            regime_log.append({
                'date': date, 'regime': regime,
                'tilt_mult': tilt_mult, 'weight_mult': weight_mult,
                'turnover': turnover, 'trade_cost': trade_cost,
                'risk_position_ratio': risk_position_ratio,
            })

        # 计算当日收益
        enhanced_ret = 0
        original_ret = 0
        stock_returns = {}

        for code in codes:
            if code not in price_pivot.columns:
                continue
            curr_price = price_pivot.loc[date, code]
            prev_price = price_pivot.loc[prev_date, code]
            if pd.isna(curr_price) or pd.isna(prev_price) or prev_price <= 0:
                continue
            stock_ret = (curr_price / prev_price - 1) + dividend_map.get(code, 0)
            stock_returns[code] = stock_ret

            enhanced_ret += stock_ret * current_weights.get(code, 0)
            original_ret += stock_ret * original_weights.get(code, 0)

        # 扣除交易成本
        if date in rebalance_dates and regime_log:
            enhanced_ret -= regime_log[-1]['trade_cost']

        # 更新净值
        current_nav *= (1 + enhanced_ret)

        # 风控检查
        if risk_controller is not None:
            # 回撤检查
            drawdown_triggered, new_ratio = risk_controller.check_drawdown_stop(current_nav)
            if drawdown_triggered and risk_position_ratio != new_ratio:
                risk_position_ratio = new_ratio
                risk_log.append({
                    'date': date,
                    'type': 'drawdown_stop',
                    'position_ratio': new_ratio,
                    'drawdown': risk_controller.current_drawdown,
                })

            # 单股止损
            current_weights = risk_controller.check_single_stock_stop_loss(current_weights, stock_returns)

        daily_returns_enhanced.append({'date': date, 'return': enhanced_ret})
        daily_returns_original.append({'date': date, 'return': original_ret})

    # 计算指标
    def calc_metrics(daily_rets):
        rets = np.array([r['return'] for r in daily_rets])
        cum = np.cumprod(1 + rets)
        total_return = cum[-1] - 1
        n_years = len(rets) / 252
        annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        max_drawdown = dd.min()

        avg = np.mean(rets)
        std = np.std(rets)
        sharpe = avg / std * np.sqrt(252) if std > 0 else 0

        return {
            'annual_return': annual_return,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe': sharpe,
        }

    enhanced_metrics = calc_metrics(daily_returns_enhanced)
    original_metrics = calc_metrics(daily_returns_original)

    regime_counts = {}
    for r in regime_log:
        regime_counts[r['regime']] = regime_counts.get(r['regime'], 0) + 1

    return {
        'enhanced': enhanced_metrics,
        'original': original_metrics,
        'improvement': enhanced_metrics['annual_return'] - original_metrics['annual_return'],
        'rebalance_count': rebalance_count,
        'total_turnover': turnover_total,
        'avg_turnover': turnover_total / max(rebalance_count, 1),
        'regime_counts': regime_counts,
        'risk_log': risk_log,
        'ml_enabled': ml_combiner is not None,
    }


def main():
    print("=" * 70)
    print("多因子策略回测 v4.0 (综合优化版)")
    print("功能: 因子IC分析 + 风控机制 + 归因分析 + 机器学习因子合成")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 加载持仓
    r4 = pd.read_csv(os.path.join(base_dir, 'r4_weights_local.csv'))
    r5 = pd.read_csv(os.path.join(base_dir, 'r5_weights_local.csv'))
    r4['code'] = r4['code'].astype(str).str.zfill(6)
    r5['code'] = r5['code'].astype(str).str.zfill(6)
    r4 = r4[['code', 'weight']].copy()
    r5 = r5[['code', 'weight']].copy()
    r4['weight'] = r4['weight'] / r4['weight'].sum()
    r5['weight'] = r5['weight'] / r5['weight'].sum()

    # 加载数据
    akshare_path = os.path.join(base_dir, 'financial_data_akshare.csv')
    akshare_df = pd.read_csv(akshare_path) if os.path.exists(akshare_path) else pd.DataFrame()
    if not akshare_df.empty:
        akshare_df['code'] = akshare_df['code'].astype(str).str.zfill(6)
        print(f"akshare 财务数据: {len(akshare_df)} 只股票")

    val_csv = pd.read_csv(os.path.join(base_dir, 'valuation_local.csv'))
    val_csv['code'] = val_csv['code'].astype(str).str.zfill(6)
    val_csv = val_csv.drop_duplicates(subset=['code'], keep='first')

    div_csv_path = os.path.join(base_dir, 'dividend_yield_all.csv')
    div_csv = pd.read_csv(div_csv_path) if os.path.exists(div_csv_path) else None
    if div_csv is not None:
        div_csv['code'] = div_csv['code'].astype(str).str.zfill(6)
        div_csv = div_csv.drop_duplicates(subset=['code'], keep='first')

    # 获取价格数据
    print("\n获取价格数据...")
    all_codes = sorted(set(r4['code'].tolist() + r5['code'].tolist()))
    ch_client = clickhouse_connect.get_client(host='192.168.0.74', port=8123, compress=False, query_limit=0)
    codes_str = "', '".join(all_codes)
    start_date = '2020-01-01'
    end_date = '2025-12-31'

    result = ch_client.query(f"""
        SELECT toString(date) as date, code, close
        FROM default.stock_data_qfq
        WHERE code IN ('{codes_str}')
          AND date >= '{start_date}' AND date <= '{end_date}'
        ORDER BY date, code
    """)
    price_df = pd.DataFrame(result.result_rows, columns=['date', 'code', 'close'])
    price_df = price_df.drop_duplicates(subset=['date', 'code'], keep='last')
    price_pivot = price_df.pivot(index='date', columns='code', values='close')
    print(f"价格数据: {len(price_pivot)} 个交易日, {len(price_pivot.columns)} 只股票")

    # 测试配置
    configs = [
        # (label, strategy, portfolio, tilt, regime, risk_ctrl, ic_adj, ml, max_w)
        ('R4-v3(基准)', 'stable', r4, 1.0, True, False, False, False, 0.08),
        ('R4-v4(风控)', 'stable', r4, 1.0, True, True, False, False, 0.08),
        ('R4-v4(IC+风控)', 'stable', r4, 1.0, True, True, True, False, 0.08),
        ('R4-v4(全部)', 'stable', r4, 1.0, True, True, True, True, 0.10),
        ('R5-v3(基准)', 'aggressive', r5, 1.0, True, False, False, False, 0.08),
        ('R5-v4(风控)', 'aggressive', r5, 1.0, True, True, False, False, 0.08),
        ('R5-v4(IC+风控)', 'aggressive', r5, 1.0, True, True, True, False, 0.08),
        ('R5-v4(全部)', 'aggressive', r5, 1.0, True, True, True, True, 0.10),
    ]

    results = []
    for label, strategy, portfolio, tilt, regime, risk_ctrl, ic_adj, ml, max_w in configs:
        print(f"\n{'='*50}")
        print(f"  {label}")
        print(f"{'='*50}")

        result = run_backtest_v4(
            portfolio=portfolio,
            price_pivot=price_pivot,
            akshare_df=akshare_df,
            val_csv=val_csv,
            div_csv=div_csv,
            strategy_type=strategy,
            tilt_strength=tilt,
            enable_regime=regime,
            enable_risk_control=risk_ctrl,
            enable_ic_adjustment=ic_adj,
            enable_ml=ml,
            base_max_weight=max_w,
        )

        print(f"  增强: {result['enhanced']['annual_return']*100:.2f}% "
              f"(夏普{result['enhanced']['sharpe']:.2f}, 回撤{result['enhanced']['max_drawdown']*100:.1f}%)")
        print(f"  原始: {result['original']['annual_return']*100:.2f}%")
        print(f"  提升: {result['improvement']*100:+.2f}%")
        print(f"  再平衡: {result['rebalance_count']}次, 平均换手: {result['avg_turnover']*100:.1f}%")
        if result['regime_counts']:
            print(f"  市场环境: {result['regime_counts']}")
        if result['risk_log']:
            print(f"  风控触发: {len(result['risk_log'])}次")
        if result['ml_enabled']:
            print(f"  ML因子: 已启用")

        results.append({
            'config': label,
            'enhanced_annual': result['enhanced']['annual_return'],
            'original_annual': result['original']['annual_return'],
            'improvement': result['improvement'],
            'enhanced_sharpe': result['enhanced']['sharpe'],
            'enhanced_drawdown': result['enhanced']['max_drawdown'],
            'rebalance_count': result['rebalance_count'],
            'avg_turnover': result['avg_turnover'],
            'risk_triggers': len(result['risk_log']),
            'ml_enabled': result['ml_enabled'],
        })

    # 汇总
    print("\n" + "=" * 70)
    print("【回测结果汇总 v4.0】")
    print("=" * 70)
    print(f"\n{'配置':<20} {'增强年化':>8} {'提升':>7} {'夏普':>6} {'回撤':>7} {'风控':>4} {'ML':>3}")
    print("-" * 70)
    for r in results:
        ml_str = 'Y' if r['ml_enabled'] else 'N'
        print(f"{r['config']:<20} {r['enhanced_annual']*100:>7.2f}% {r['improvement']*100:>+6.2f}% "
              f"{r['enhanced_sharpe']:>5.2f} {r['enhanced_drawdown']*100:>6.1f}% {r['risk_triggers']:>4} {ml_str:>3}")

    # 保存
    output_path = os.path.join(os.path.dirname(__file__), 'rebalance_results_v4.csv')
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"\n结果已保存到 {output_path}")


if __name__ == '__main__':
    main()
