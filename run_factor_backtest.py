#!/usr/bin/env python
"""
多因子权重策略回测
- 自动从 ClickHouse 获取股票池（stock_financial 100只）
- 12因子打分 + 权重分档调整
- 年度滚动调仓，大盘止损
- 支持 R4 稳健型 / R5 进取型
"""

import pandas as pd
import numpy as np
import clickhouse_connect
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 配置
# ============================================================================

CLICKHOUSE_HOST = '192.168.0.74'
CLICKHOUSE_PORT = 8123

# 市值过滤（亿）
MIN_MARKET_CAP = 50

# 行业约束
MAX_INDUSTRY_WEIGHT = 0.20

# 大盘止损
MARKET_STOP_LOSS = {
    'enabled': True,
    'ma_period': 250,
    'position_reduce': 0.70,
    'recovery_buffer': 1.02,
}

# 权重调整档位
WEIGHT_TIERS = {
    0.95: 1.50,
    0.90: 1.30,
    0.80: 1.15,
    0.60: 1.00,
    0.40: 0.90,
    0.20: 0.80,
    0.10: 0.70,
    0.00: 0.50,
}

# R4 稳健型因子权重
WEIGHTS_STABLE = {
    'dividend_yield': 0.18,
    'roe': 0.15,
    'roe_stability': 0.10,
    'reversal': 0.10,
    'pe_value': 0.10,
    'profit_growth': 0.10,
    'small_cap': 0.08,
    'momentum': 0.05,
    'low_volatility': 0.07,
    'financial_health': 0.07,
}

# R5 进取型因子权重
WEIGHTS_AGGRESSIVE = {
    'profit_growth': 0.18,
    'small_cap': 0.15,
    'momentum': 0.12,
    'roe': 0.12,
    'revenue_growth': 0.10,
    'reversal': 0.08,
    'dividend_yield': 0.08,
    'pe_value': 0.05,
    'low_volatility': 0.05,
    'financial_health': 0.07,
}


# ============================================================================
# 数据获取
# ============================================================================

class DataFetcher:
    def __init__(self):
        self.client = clickhouse_connect.get_client(host=CLICKHOUSE_HOST, port=CLICKHOUSE_PORT)

    def get_stock_pool(self, strategy_type='stable'):
        """从 CSV 加载 R4/R5 持仓权重"""
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if strategy_type == 'stable':
            csv_path = os.path.join(base_dir, 'r4_weights_local.csv')
        else:
            csv_path = os.path.join(base_dir, 'r5_weights_local.csv')

        df = pd.read_csv(csv_path)
        # 使用 original_weight 列作为原始权重
        if 'original_weight' in df.columns:
            portfolio = df[['code', 'original_weight']].copy()
            portfolio.columns = ['code', 'weight']
        else:
            portfolio = df[['code', 'weight']].copy()
        # 代码转为6位字符串
        portfolio['code'] = portfolio['code'].astype(str).str.zfill(6)
        # 归一化
        portfolio['weight'] = portfolio['weight'] / portfolio['weight'].sum()
        return portfolio

    def get_latest_report_date(self, as_of_date):
        dt = datetime.strptime(as_of_date, '%Y-%m-%d')
        year, month = dt.year, dt.month
        if month >= 5:
            return f'{year-1}-12-31'
        else:
            return f'{year-2}-12-31'

    def load_valuation_csv(self):
        """从 valuation_local.csv 加载估值数据（PE/PB/ROE/EPS/市值）"""
        if not hasattr(self, '_valuation_df'):
            import os
            base_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(base_dir, 'valuation_local.csv')
            df = pd.read_csv(csv_path)
            df['code'] = df['code'].astype(str).str.zfill(6)
            # total_mv 单位是万，转亿
            df['market_cap'] = df['total_mv'] / 10000
            self._valuation_df = df
        return self._valuation_df

    def load_dividend_csv(self):
        """从 dividend_yield_all.csv 加载股息率"""
        if not hasattr(self, '_dividend_df'):
            import os
            base_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(base_dir, 'dividend_yield_all.csv')
            df = pd.read_csv(csv_path)
            df['code'] = df['code'].astype(str).str.zfill(6)
            self._dividend_df = df
        return self._dividend_df

    def get_financial_data(self, codes, as_of_date):
        """从 CSV + ClickHouse 获取财务数据"""
        # 先从 valuation CSV 获取基础数据
        val_df = self.load_valuation_csv()
        val_df = val_df[val_df['code'].isin(codes)].copy()

        # 基础字段
        fin_df = val_df[['code', 'roe', 'eps', 'bvps', 'pe_ttm', 'pb',
                         'market_cap', 'dividend_yield']].copy()
        fin_df = fin_df.rename(columns={'pe_ttm': 'pe'})

        # 补充 asset_liability_ratio（从 ClickHouse stock_financial 获取，能匹配多少算多少）
        codes_str = "','".join(codes)
        report_date = self.get_latest_report_date(as_of_date)
        try:
            query = f"""
                SELECT code, asset_liability_ratio
                FROM default.stock_financial
                WHERE code IN ('{codes_str}') AND report_date <= '{report_date}'
                ORDER BY report_date DESC LIMIT 1 BY code
            """
            result = self.client.query(query)
            health_df = pd.DataFrame(result.result_rows,
                columns=['code', 'asset_liability_ratio'])
            fin_df = fin_df.merge(health_df, on='code', how='left')
        except:
            pass
        if 'asset_liability_ratio' not in fin_df.columns:
            fin_df['asset_liability_ratio'] = 50.0
        fin_df['asset_liability_ratio'] = fin_df['asset_liability_ratio'].fillna(50.0)

        # 补充增长率（从 ClickHouse 获取）
        yoy_date = f'{int(report_date[:4])-1}{report_date[4:]}'
        try:
            growth_query = f"""
                WITH latest AS (
                    SELECT code, eps, operating_revenue
                    FROM default.stock_financial
                    WHERE code IN ('{codes_str}') AND report_date <= '{report_date}'
                    ORDER BY report_date DESC LIMIT 1 BY code
                ),
                yoy AS (
                    SELECT code, eps as yoy_eps, operating_revenue as yoy_revenue
                    FROM default.stock_financial
                    WHERE code IN ('{codes_str}') AND report_date <= '{yoy_date}'
                    ORDER BY report_date DESC LIMIT 1 BY code
                )
                SELECT latest.code,
                       CASE WHEN yoy.yoy_eps > 0
                            THEN (latest.eps / yoy.yoy_eps - 1) * 100 ELSE 0 END as profit_growth,
                       CASE WHEN yoy.yoy_revenue > 0
                            THEN (latest.operating_revenue / yoy.yoy_revenue - 1) * 100 ELSE 0 END as revenue_growth
                FROM latest LEFT JOIN yoy ON latest.code = yoy.code
            """
            growth_result = self.client.query(growth_query)
            growth_df = pd.DataFrame(growth_result.result_rows,
                columns=['code', 'profit_growth', 'revenue_growth'])
            fin_df = fin_df.merge(growth_df, on='code', how='left')
        except:
            pass
        if 'profit_growth' not in fin_df.columns:
            fin_df['profit_growth'] = 0.0
        if 'revenue_growth' not in fin_df.columns:
            fin_df['revenue_growth'] = 0.0
        fin_df['profit_growth'] = fin_df['profit_growth'].fillna(0.0)
        fin_df['revenue_growth'] = fin_df['revenue_growth'].fillna(0.0)

        return fin_df

    def get_market_data(self, codes, as_of_date):
        """获取市值、价格"""
        codes_str = "','".join(codes)
        query = f"""
            SELECT code, close as price,
                   liutongshizhi / 100000000 as market_cap
            FROM default.stock_data_qfq
            WHERE code IN ('{codes_str}') AND date <= '{as_of_date}'
            ORDER BY date DESC LIMIT 1 BY code
        """
        result = self.client.query(query)
        return pd.DataFrame(result.result_rows, columns=['code', 'price', 'market_cap'])

    def get_momentum(self, codes, as_of_date, days=60):
        codes_str = "','".join(codes)
        query = f"""
            WITH latest AS (
                SELECT code, close as latest_close
                FROM default.stock_data_qfq
                WHERE code IN ('{codes_str}') AND date <= '{as_of_date}'
                ORDER BY date DESC LIMIT 1 BY code
            ),
            past AS (
                SELECT code, close as past_close
                FROM default.stock_data_qfq
                WHERE code IN ('{codes_str}')
                  AND date <= date_sub(day, {days}, toDate('{as_of_date}'))
                ORDER BY date DESC LIMIT 1 BY code
            )
            SELECT latest.code,
                   (latest.latest_close / past.past_close - 1) * 100 as momentum
            FROM latest JOIN past ON latest.code = past.code
        """
        result = self.client.query(query)
        return pd.DataFrame(result.result_rows, columns=['code', 'momentum'])

    def get_drawdown(self, codes, as_of_date, months=3):
        codes_str = "','".join(codes)
        query = f"""
            WITH price_data AS (
                SELECT code, date, close
                FROM default.stock_data_qfq
                WHERE code IN ('{codes_str}')
                  AND date >= date_sub(month, {months}, toDate('{as_of_date}'))
                  AND date <= '{as_of_date}'
            ),
            max_price AS (
                SELECT code, max(close) as high FROM price_data GROUP BY code
            ),
            latest_price AS (
                SELECT code, close as latest FROM price_data
                ORDER BY date DESC LIMIT 1 BY code
            )
            SELECT max_price.code,
                   (latest_price.latest / max_price.high - 1) * 100 as drawdown
            FROM max_price JOIN latest_price ON max_price.code = latest_price.code
        """
        result = self.client.query(query)
        return pd.DataFrame(result.result_rows, columns=['code', 'drawdown'])

    def get_volatility(self, codes, as_of_date, days=60):
        codes_str = "','".join(codes)
        query = f"""
            WITH daily_returns AS (
                SELECT code,
                       close / lagInFrame(close, 1) OVER (PARTITION BY code ORDER BY date) - 1 as ret
                FROM default.stock_data_qfq
                WHERE code IN ('{codes_str}')
                  AND date >= date_sub(day, {days}, toDate('{as_of_date}'))
                  AND date <= '{as_of_date}'
            )
            SELECT code, stddevPop(ret) * sqrt(252) * 100 as volatility
            FROM daily_returns WHERE ret IS NOT NULL
            GROUP BY code
        """
        result = self.client.query(query)
        return pd.DataFrame(result.result_rows, columns=['code', 'volatility'])

    def get_prices(self, codes, start_date, end_date):
        codes_str = "','".join(codes)
        query = f"""
            SELECT toString(date) as date, code, close
            FROM default.stock_data_qfq
            WHERE code IN ('{codes_str}')
              AND date >= '{start_date}' AND date <= '{end_date}'
            ORDER BY date, code
        """
        result = self.client.query(query)
        df = pd.DataFrame(result.result_rows, columns=['date', 'code', 'close'])
        return df.drop_duplicates(subset=['date', 'code'], keep='last')

    def get_index_prices(self, start_date, end_date):
        """获取沪深300指数"""
        try:
            query = f"""
                SELECT toString(date) as date, close
                FROM default.stock_index
                WHERE code = '000300'
                  AND date >= '{start_date}' AND date <= '{end_date}'
                ORDER BY date
            """
            result = self.client.query(query)
            if result.result_rows:
                return pd.DataFrame(result.result_rows, columns=['date', 'close'])
        except:
            pass
        return pd.DataFrame()


# ============================================================================
# 因子计算 & 权重调整
# ============================================================================

def percentile_score(series, ascending=True):
    if ascending:
        return series.rank(pct=True) * 100
    return (1 - series.rank(pct=True)) * 100


def calculate_factor_scores(df, weights):
    result = df.copy()

    if 'roe' in df.columns:
        result['roe_score'] = percentile_score(df['roe'])
    if 'pe' in df.columns:
        valid_pe = df['pe'].where(df['pe'] > 0, np.nan)
        result['pe_value_score'] = percentile_score(valid_pe, ascending=False)
    if 'dividend_yield' in df.columns:
        result['dividend_yield_score'] = percentile_score(df['dividend_yield'])
    if 'asset_liability_ratio' in df.columns:
        result['financial_health_score'] = percentile_score(df['asset_liability_ratio'], ascending=False)
    if 'profit_growth' in df.columns:
        result['profit_growth_score'] = percentile_score(df['profit_growth'])
    if 'revenue_growth' in df.columns:
        result['revenue_growth_score'] = percentile_score(df['revenue_growth'])
    if 'market_cap' in df.columns:
        result['small_cap_score'] = percentile_score(df['market_cap'], ascending=False)
    if 'momentum' in df.columns:
        result['momentum_score'] = percentile_score(df['momentum'])
    if 'drawdown' in df.columns and 'roe' in df.columns:
        is_quality = df['roe'] > 15
        dd_score = percentile_score(df['drawdown'])
        result['reversal_score'] = np.where(is_quality, dd_score, 50)
    if 'volatility' in df.columns:
        result['low_volatility_score'] = percentile_score(df['volatility'], ascending=False)

    for factor in ['roe_stability']:
        if f'{factor}_score' not in result.columns:
            result[f'{factor}_score'] = 50

    # 综合得分
    composite = pd.Series(0.0, index=df.index)
    for factor, weight in weights.items():
        score_col = f'{factor}_score'
        if score_col in result.columns:
            composite += result[score_col].fillna(50) * weight
    result['composite_score'] = composite

    return result


def adjust_weights(portfolio, scored_df):
    merged = portfolio.merge(scored_df[['code', 'composite_score']], on='code', how='left')
    merged['percentile'] = merged['composite_score'].rank(pct=True)

    def get_multiplier(pct):
        for threshold, mult in sorted(WEIGHT_TIERS.items(), reverse=True):
            if pct >= threshold:
                return mult
        return 0.5

    merged['multiplier'] = merged['percentile'].apply(get_multiplier)
    merged['adjusted_weight'] = merged['weight'] * merged['multiplier']
    merged['adjusted_weight'] /= merged['adjusted_weight'].sum()
    return merged


# ============================================================================
# 回测引擎
# ============================================================================

def run_backtest(strategy_type='stable', start_date='2020-01-01', end_date='2025-12-31'):
    weights = WEIGHTS_STABLE if strategy_type == 'stable' else WEIGHTS_AGGRESSIVE
    label = 'R4 稳健型' if strategy_type == 'stable' else 'R5 进取型'

    print("=" * 60)
    print(f"多因子权重策略回测 [{label}]")
    print(f"回测区间: {start_date} ~ {end_date}")
    print("=" * 60)

    fetcher = DataFetcher()

    # 获取股票池
    portfolio = fetcher.get_stock_pool(strategy_type)
    codes = portfolio['code'].tolist()
    print(f"\n股票池: {len(codes)} 只")

    # 获取调仓日期（每年初）
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    rebalance_dates = []
    y = start_dt.year
    while y <= end_dt.year:
        rd = f'{y}-01-01'
        if rd >= start_date:
            rebalance_dates.append(rd)
        y += 1
    print(f"调仓日期: {rebalance_dates}")

    # 获取价格数据
    extended_start = (start_dt - timedelta(days=400)).strftime('%Y-%m-%d')
    prices = fetcher.get_prices(codes, extended_start, end_date)
    price_pivot = prices.pivot(index='date', columns='code', values='close')

    # 指数数据
    index_prices = fetcher.get_index_prices(extended_start, end_date)

    dates = sorted([d for d in price_pivot.index if d >= start_date])
    print(f"交易日: {len(dates)} 天\n")

    # 初始化
    original_weights = dict(zip(portfolio['code'], portfolio['weight']))
    current_weights = original_weights.copy()
    current_rebalance_idx = 0

    # 大盘止损
    market_position = 1.0
    stop_loss_count = 0

    enhanced_returns = []
    original_returns = []

    for i in range(1, len(dates)):
        curr_date = dates[i]
        prev_date = dates[i - 1]

        # 调仓
        while (current_rebalance_idx < len(rebalance_dates) and
               curr_date >= rebalance_dates[current_rebalance_idx]):
            rb_date = rebalance_dates[current_rebalance_idx]
            print(f"  {rb_date}: 重新计算因子...")

            # 构建因子数据（CSV 提供基础估值，ClickHouse 补充动量等）
            factor_df = fetcher.get_financial_data(codes, rb_date)
            mom_df = fetcher.get_momentum(codes, rb_date)
            dd_df = fetcher.get_drawdown(codes, rb_date)
            vol_df = fetcher.get_volatility(codes, rb_date)

            factor_df = factor_df.merge(mom_df, on='code', how='left')
            factor_df = factor_df.merge(dd_df, on='code', how='left')
            factor_df = factor_df.merge(vol_df, on='code', how='left')

            # 市值过滤
            factor_df = factor_df[factor_df['market_cap'] >= MIN_MARKET_CAP]
            factor_df = factor_df.fillna(0)

            if len(factor_df) > 0:
                scored = calculate_factor_scores(factor_df, weights)
                adjusted = adjust_weights(portfolio[portfolio['code'].isin(factor_df['code'])], scored)
                current_weights = dict(zip(adjusted['code'], adjusted['adjusted_weight']))
                print(f"    通过筛选: {len(factor_df)} 只")

            current_rebalance_idx += 1

        # 计算日收益
        curr = price_pivot.loc[curr_date]
        prev = price_pivot.loc[prev_date]

        # 大盘止损检查
        if MARKET_STOP_LOSS['enabled'] and not index_prices.empty:
            idx = index_prices[index_prices['date'] <= curr_date]
            if len(idx) >= MARKET_STOP_LOSS['ma_period']:
                ma = idx['close'].rolling(MARKET_STOP_LOSS['ma_period']).mean().iloc[-1]
                price_now = idx['close'].iloc[-1]
                if price_now < ma:
                    if market_position == 1.0:
                        stop_loss_count += 1
                    market_position = MARKET_STOP_LOSS['position_reduce']
                elif price_now > ma * MARKET_STOP_LOSS['recovery_buffer']:
                    market_position = 1.0

        enh_ret = orig_ret = 0
        for code in codes:
            if code in curr.index and pd.notna(curr[code]) and pd.notna(prev[code]) and prev[code] > 0:
                ret = curr[code] / prev[code] - 1
                enh_ret += ret * current_weights.get(code, 0) * market_position
                orig_ret += ret * original_weights.get(code, 0)

        enhanced_returns.append(enh_ret)
        original_returns.append(orig_ret)

    # 绩效计算
    def calc_metrics(returns):
        cum = np.cumprod([1 + r for r in returns])
        total = cum[-1] - 1
        n_years = len(returns) / 252
        annual = (1 + total) ** (1 / n_years) - 1 if n_years > 0 else 0
        max_dd = np.min(cum / np.maximum.accumulate(cum) - 1)
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        return {
            'annual_return': annual,
            'total_return': total,
            'max_drawdown': max_dd,
            'sharpe': sharpe,
            'final_nav': cum[-1],
        }

    enh = calc_metrics(enhanced_returns)
    orig = calc_metrics(original_returns)

    print(f"\n{'='*60}")
    print(f"回测结果 [{label}]")
    print(f"{'='*60}")
    print(f"  {'指标':<20} {'等权原始':>12} {'因子增强':>12}")
    print(f"  {'-'*44}")
    print(f"  {'年化收益':<20} {orig['annual_return']*100:>11.2f}% {enh['annual_return']*100:>11.2f}%")
    print(f"  {'总收益':<20} {orig['total_return']*100:>11.2f}% {enh['total_return']*100:>11.2f}%")
    print(f"  {'最大回撤':<20} {orig['max_drawdown']*100:>11.2f}% {enh['max_drawdown']*100:>11.2f}%")
    print(f"  {'夏普比率':<20} {orig['sharpe']:>12.2f} {enh['sharpe']:>12.2f}")
    print(f"  {'最终净值':<20} {orig['final_nav']:>12.4f} {enh['final_nav']:>12.4f}")
    if MARKET_STOP_LOSS['enabled']:
        print(f"\n  大盘止损触发: {stop_loss_count} 次")

    return {'enhanced': enh, 'original': orig, 'strategy_type': strategy_type}


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 60)
    print("多因子权重策略回测系统")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    start_date = '2020-01-01'
    end_date = '2025-12-31'

    # R4 稳健型
    r4 = run_backtest('stable', start_date, end_date)

    print("\n")

    # R5 进取型
    r5 = run_backtest('aggressive', start_date, end_date)

    # 总结
    print(f"\n{'='*60}")
    print("回测总结")
    print(f"{'='*60}")
    print(f"""
┌──────────────┬───────────┬───────────┬──────────┬────────┐
│   策略       │ 等权年化  │ 增强年化  │ 最大回撤 │  夏普  │
├──────────────┼───────────┼───────────┼──────────┼────────┤
│ R4 稳健型    │  {r4['original']['annual_return']*100:>6.2f}%  │  {r4['enhanced']['annual_return']*100:>6.2f}%  │ {r4['enhanced']['max_drawdown']*100:>6.2f}% │  {r4['enhanced']['sharpe']:.2f}  │
│ R5 进取型    │  {r5['original']['annual_return']*100:>6.2f}%  │  {r5['enhanced']['annual_return']*100:>6.2f}%  │ {r5['enhanced']['max_drawdown']*100:>6.2f}% │  {r5['enhanced']['sharpe']:.2f}  │
└──────────────┴───────────┴───────────┴──────────┴────────┘
""")

    # 保存结果
    result = {
        'run_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'backtest_period': f'{start_date} ~ {end_date}',
        'R4_stable': r4,
        'R5_aggressive': r5,
    }
    with open('factor_backtest_result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print("结果已保存到 factor_backtest_result.json")


if __name__ == '__main__':
    main()
