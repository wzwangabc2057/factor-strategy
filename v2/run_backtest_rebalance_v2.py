"""
月度再平衡 + 动态因子择时回测 v2.0 (优化版)

优化内容:
1. 放宽反转条件 - ROE>8%(原12%), 回撤>10%(原15%), 分级加成
2. 动态tilt_strength - 牛市1.2, 熊市0.6, 震荡0.8
3. 优化市场环境检测 - 增加趋势强度判断
4. 单股最大权重约束 - 防止过度集中
5. 优化因子权重配置

每月初:
1. 检测市场环境 (牛/熊/震荡) → 切换因子权重 + 调整倾斜力度
2. 用最新动量/波动率/回撤重新计算因子得分
3. 按因子得分调整持仓权重
4. 扣除交易成本后再平衡
"""
import pandas as pd
import numpy as np
import sys
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.INFO)

import clickhouse_connect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.factor_calculator_v2 import EnhancedFactorCalculator, get_factor_weights_for_market_regime


# 交易成本
BUY_COMMISSION = 0.00026   # 买入佣金 0.026%
SELL_COMMISSION = 0.00126  # 卖出佣金+印花税 0.126%


# ==================== 优化的因子权重配置 ====================

# 稳健型策略 - 优化后的因子权重
OPTIMIZED_STABLE_WEIGHTS = {
    # 价值因子
    'dividend_yield': 0.20,      # 股息率（+2%）
    'pe_value': 0.08,
    # 质量因子
    'roe': 0.18,                 # ROE（+3%）
    'roe_stability': 0.12,       # ROE稳定性（+2%）
    'cash_flow_quality': 0.08,
    # 成长因子
    'profit_growth': 0.08,
    'revenue_growth': 0.03,
    'peg': 0.05,
    # 规模因子
    'small_cap': 0.03,           # （-2%）
    # 动量/反转因子
    'momentum': 0.05,
    'reversal': 0.07,            # （-3%）
    # 风险因子
    'low_volatility': 0.03,
}

# 进取型策略 - 优化后的因子权重
OPTIMIZED_AGGRESSIVE_WEIGHTS = {
    # 价值因子
    'dividend_yield': 0.03,
    'pe_value': 0.02,
    # 质量因子
    'roe': 0.12,
    'roe_stability': 0.06,
    'cash_flow_quality': 0.03,
    # 成长因子
    'profit_growth': 0.22,       # （+4%）
    'revenue_growth': 0.10,      # （+2%）
    'peg': 0.15,                 # （+3%）
    # 规模因子
    'small_cap': 0.12,
    # 动量/反转因子
    'momentum': 0.10,
    'reversal': 0.03,
    # 风险因子
    'low_volatility': 0.02,
}


def detect_market_regime(price_series: pd.Series, lookback_short=20, lookback_long=60):
    """
    检测市场环境 (优化版)
    Args:
        price_series: 市场指数/均价序列 (日期为index)
        lookback_short: 短期回看天数
        lookback_long: 长期回看天数
    Returns:
        (regime, tilt_multiplier) - 环境类型和倾斜力度乘数
    """
    if len(price_series) < lookback_long:
        return 'volatile', 0.8

    # 短期动量 (20日)
    short_return = price_series.iloc[-1] / price_series.iloc[-lookback_short] - 1
    # 长期动量 (60日)
    long_return = price_series.iloc[-1] / price_series.iloc[-lookback_long] - 1
    # 波动率 (20日)
    daily_returns = price_series.pct_change().dropna().tail(lookback_short)
    volatility = daily_returns.std() * np.sqrt(252)

    # 趋势强度 (MA20/MA60)
    ma20 = price_series.tail(20).mean()
    ma60 = price_series.tail(60).mean()
    trend_strength = (ma20 / ma60 - 1)

    # 优化判断逻辑 - 增加趋势强度
    if long_return > 0.08 and short_return > 0 and trend_strength > 0:
        return 'bull', 1.2  # 强牛市，更激进
    elif long_return > 0.05 and short_return > -0.02:
        return 'bull', 1.0  # 温和牛市
    elif long_return < -0.08 and short_return < 0:
        return 'bear', 0.6  # 强熊市，保守
    elif long_return < -0.05:
        return 'bear', 0.8  # 温和熊市
    elif volatility > 0.28:
        return 'volatile', 0.8  # 高波动震荡
    elif trend_strength > 0.02:
        return 'bull', 1.0  # 趋势向上
    elif trend_strength < -0.02:
        return 'bear', 0.8  # 趋势向下
    else:
        return 'volatile', 0.8  # 默认震荡


def build_factor_data_for_date(fetcher, codes, eval_date, price_pivot, akshare_df, val_csv, div_csv):
    """
    为特定日期构建因子数据
    """
    factor_df = pd.DataFrame({'code': codes})

    # 1. 静态财务因子 (from akshare)
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

    # 2. PE/PB/股息率 (from CSV)
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

    # 3. 动态行情因子
    available_dates = [d for d in price_pivot.index if d <= eval_date]
    if len(available_dates) >= 60:
        recent_prices = price_pivot.loc[available_dates]
        factor_df['market_cap'] = np.nan

        # 动量 (60日)
        if len(available_dates) >= 60:
            p_now = recent_prices.iloc[-1]
            p_60d = recent_prices.iloc[-60]
            momentum = (p_now / p_60d - 1) * 100
            momentum_map = momentum.to_dict()
            factor_df['momentum'] = factor_df['code'].map(momentum_map).fillna(0)

        # 3个月最大回撤
        if len(available_dates) >= 60:
            window = recent_prices.tail(60)
            cum_max = window.cummax()
            drawdown = ((window - cum_max) / cum_max).min() * 100
            dd_map = drawdown.to_dict()
            factor_df['drawdown_3m'] = factor_df['code'].map(dd_map).fillna(-10)

        # 波动率 (60日)
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

    # 4. 填充缺失值
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


def apply_reversal_boost(scored_data: pd.DataFrame, factor_data: pd.DataFrame) -> pd.DataFrame:
    """
    优化版反转加成 - 放宽条件

    放宽条件:
    - ROE > 8% (原12%)
    - 净利润增速 > 0% (原5%)
    - 回撤 > 10% (原15%) - 分级加成
    """
    result = scored_data.copy()

    # 绩优股条件 (放宽)
    is_quality = (factor_data['roe'] > 8) & (factor_data['net_profit_yoy'] > 0)

    # 回撤幅度
    if 'drawdown_3m' in factor_data.columns:
        drawdown = -factor_data['drawdown_3m']
    else:
        return result

    # 分级加成
    boost = pd.Series(0, index=result.index)

    # 大幅回撤 (>20%)
    mask_20 = is_quality & (drawdown > 20)
    boost[mask_20] = 15

    # 中等回撤 (15-20%)
    mask_15 = is_quality & (drawdown > 15) & (drawdown <= 20)
    boost[mask_15] = 10

    # 小幅回撤 (10-15%)
    mask_10 = is_quality & (drawdown > 10) & (drawdown <= 15)
    boost[mask_10] = 5

    # 轻微回撤 (5-10%)
    mask_5 = is_quality & (drawdown > 5) & (drawdown <= 10)
    boost[mask_5] = 2

    # 应用加成
    result['composite_score'] = (result['composite_score'] + boost).clip(0, 100)

    return result


def apply_mild_weight_tilt(original_weights: pd.DataFrame,
                           scored_data: pd.DataFrame,
                           tilt_strength: float = 0.5,
                           max_weight: float = 0.08) -> pd.DataFrame:
    """温和的权重调整"""
    portfolio = original_weights[['code', 'weight']].copy()

    # 合并得分
    portfolio = portfolio.merge(scored_data[['code', 'composite_score']], on='code', how='left')
    portfolio['composite_score'] = portfolio['composite_score'].fillna(50)

    # 计算百分位
    portfolio['percentile'] = portfolio['composite_score'].rank(pct=True)

    # 线性倾斜
    portfolio['multiplier'] = 1.0 + tilt_strength * (portfolio['percentile'] - 0.5)

    # 计算调整后权重
    portfolio['adjusted_weight'] = portfolio['weight'] * portfolio['multiplier']

    # 约束
    portfolio['adjusted_weight'] = portfolio['adjusted_weight'].clip(upper=max_weight)

    # 归一化
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


def run_rebalance_backtest(
    portfolio: pd.DataFrame,
    price_pivot: pd.DataFrame,
    akshare_df: pd.DataFrame,
    val_csv: pd.DataFrame,
    div_csv: pd.DataFrame,
    strategy_type: str = 'stable',
    tilt_strength: float = 1.0,
    enable_regime: bool = True,
    rebalance_freq: str = 'monthly',
    enable_reversal_boost: bool = True,
    max_single_weight: float = 0.08,
    use_optimized_weights: bool = True,
):
    """月度再平衡回测 (优化版)"""
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

    current_weights = dict(zip(portfolio['code'], portfolio['weight']))
    original_weights = current_weights.copy()

    daily_returns_enhanced = []
    daily_returns_original = []
    regime_log = []
    rebalance_count = 0
    turnover_total = 0

    for i in range(1, len(dates)):
        date = dates[i]
        prev_date = dates[i - 1]

        if date in rebalance_dates and i > 1:
            rebalance_count += 1

            # 1. 检测市场环境和倾斜力度
            market_up_to_now = market_avg.loc[:date]
            if enable_regime:
                regime, tilt_mult = detect_market_regime(market_up_to_now)
                actual_tilt = tilt_strength * tilt_mult
            else:
                regime = 'volatile'
                actual_tilt = tilt_strength

            # 2. 获取因子权重
            if enable_regime and regime != 'volatile':
                factor_weights = get_factor_weights_for_market_regime(regime)
            elif use_optimized_weights:
                factor_weights = OPTIMIZED_STABLE_WEIGHTS if strategy_type == 'stable' else OPTIMIZED_AGGRESSIVE_WEIGHTS
            else:
                factor_weights = None

            # 3. 构建因子数据
            factor_data = build_factor_data_for_date(
                None, codes, date, price_pivot, akshare_df, val_csv, div_csv
            )

            # 4. 计算因子得分
            if factor_weights is not None:
                factor_calc = EnhancedFactorCalculator(factor_weights=factor_weights)
            else:
                factor_calc = EnhancedFactorCalculator(strategy_type=strategy_type)
            scored_data = factor_calc.calculate_all_scores(factor_data)

            # 5. 反转加成
            if enable_reversal_boost:
                scored_data = apply_reversal_boost(scored_data, factor_data)

            # 6. 权重调整
            enhanced_portfolio = apply_mild_weight_tilt(
                portfolio, scored_data,
                tilt_strength=actual_tilt,
                max_weight=max_single_weight
            )

            new_weights = dict(zip(enhanced_portfolio['code'], enhanced_portfolio['adjusted_weight']))

            # 7. 计算换手率和交易成本
            turnover = sum(abs(new_weights.get(c, 0) - current_weights.get(c, 0)) for c in codes) / 2
            turnover_total += turnover
            trade_cost = turnover * (BUY_COMMISSION + SELL_COMMISSION)

            current_weights = new_weights

            regime_log.append({
                'date': date, 'regime': regime,
                'tilt_mult': tilt_mult if enable_regime else 1.0,
                'turnover': turnover, 'trade_cost': trade_cost
            })

        # 计算当日收益
        enhanced_ret = 0
        original_ret = 0
        for code in codes:
            if code not in price_pivot.columns:
                continue
            curr_price = price_pivot.loc[date, code]
            prev_price = price_pivot.loc[prev_date, code]
            if pd.isna(curr_price) or pd.isna(prev_price) or prev_price <= 0:
                continue
            stock_ret = (curr_price / prev_price - 1) + dividend_map.get(code, 0)

            enhanced_ret += stock_ret * current_weights.get(code, 0)
            original_ret += stock_ret * original_weights.get(code, 0)

        if date in rebalance_dates and regime_log:
            enhanced_ret -= regime_log[-1]['trade_cost']

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
    }


def main():
    print("=" * 70)
    print("月度再平衡 + 动态因子择时回测 v2.0 (优化版)")
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
        # (label, strategy, portfolio, tilt, regime, freq, reversal, max_w, optimized_weights)
        ('R4-v1(基准)', 'stable', r4, 1.0, True, 'monthly', False, 0.08, False),
        ('R4-v2(优化)', 'stable', r4, 1.0, True, 'monthly', True, 0.08, True),
        ('R4-v2(激进)', 'stable', r4, 1.0, True, 'monthly', True, 0.10, True),
        ('R5-v1(基准)', 'aggressive', r5, 1.0, True, 'monthly', False, 0.08, False),
        ('R5-v2(优化)', 'aggressive', r5, 1.0, True, 'monthly', True, 0.08, True),
        ('R5-v2(激进)', 'aggressive', r5, 1.0, True, 'monthly', True, 0.10, True),
    ]

    results = []
    for label, strategy, portfolio, tilt, regime, freq, reversal, max_w, opt_weights in configs:
        print(f"\n{'='*50}")
        print(f"  {label}")
        print(f"{'='*50}")

        result = run_rebalance_backtest(
            portfolio=portfolio,
            price_pivot=price_pivot,
            akshare_df=akshare_df,
            val_csv=val_csv,
            div_csv=div_csv,
            strategy_type=strategy,
            tilt_strength=tilt,
            enable_regime=regime,
            rebalance_freq=freq,
            enable_reversal_boost=reversal,
            max_single_weight=max_w,
            use_optimized_weights=opt_weights,
        )

        print(f"  增强: {result['enhanced']['annual_return']*100:.2f}% "
              f"(夏普{result['enhanced']['sharpe']:.2f}, 回撤{result['enhanced']['max_drawdown']*100:.1f}%)")
        print(f"  原始: {result['original']['annual_return']*100:.2f}%")
        print(f"  提升: {result['improvement']*100:+.2f}%")
        print(f"  再平衡: {result['rebalance_count']}次, "
              f"平均换手: {result['avg_turnover']*100:.1f}%")
        if result['regime_counts']:
            print(f"  市场环境: {result['regime_counts']}")

        results.append({
            'config': label,
            'enhanced_annual': result['enhanced']['annual_return'],
            'original_annual': result['original']['annual_return'],
            'improvement': result['improvement'],
            'enhanced_sharpe': result['enhanced']['sharpe'],
            'enhanced_drawdown': result['enhanced']['max_drawdown'],
            'rebalance_count': result['rebalance_count'],
            'avg_turnover': result['avg_turnover'],
            'regime_counts': str(result['regime_counts']),
        })

    # 汇总
    print("\n" + "=" * 70)
    print("【回测结果汇总 v2.0】")
    print("=" * 70)
    print(f"\n{'配置':<20} {'增强年化':>8} {'提升':>7} {'夏普':>6} {'回撤':>7} {'调仓次数':>6}")
    print("-" * 65)
    for r in results:
        print(f"{r['config']:<20} {r['enhanced_annual']*100:>7.2f}% {r['improvement']*100:>+6.2f}% "
              f"{r['enhanced_sharpe']:>5.2f} {r['enhanced_drawdown']*100:>6.1f}% {r['rebalance_count']:>5}")

    # 保存
    output_path = os.path.join(os.path.dirname(__file__), 'rebalance_results_v2.csv')
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"\n结果已保存到 {output_path}")


if __name__ == '__main__':
    main()
