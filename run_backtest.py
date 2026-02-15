"""
增强版多因子策略回测 v2 - 修复版
修复:
1. 使用原始weight作为baseline (而非adjusted_weight)
2. 消除双重调整bug
3. 温和化权重调整系数 (0.75x~1.35x)
4. R4/R5 使用不同因子权重产生不同增强结果
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

import clickhouse_connect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.factor_calculator_v2 import EnhancedFactorCalculator


class LocalDataFetcher:
    """本地数据获取器 - 从ClickHouse获取所有数据"""

    def __init__(self):
        self.ch_client = clickhouse_connect.get_client(host='192.168.0.74', port=8123, compress=False, query_limit=0)

    def get_stock_prices(self, codes: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """从ClickHouse获取复权价格"""
        codes_str = "', '".join(codes)
        query = f"""
        SELECT toString(date) as date, code, close
        FROM default.stock_data_qfq
        WHERE code IN ('{codes_str}')
          AND date >= '{start_date}'
          AND date <= '{end_date}'
        ORDER BY date, code
        """
        result = self.ch_client.query(query)
        df = pd.DataFrame(result.result_rows, columns=['date', 'code', 'close'])
        return df

    def get_financial_data(self, codes: List[str], report_date: str = None) -> pd.DataFrame:
        """从本地ClickHouse获取财务数据"""
        codes_str = "', '".join(codes)

        if report_date:
            query = f"""
            SELECT code,
                   toString(report_date) as report_date,
                   roe, eps, bvps,
                   net_profit, operating_revenue, operating_profit,
                   gross_profit_margin, net_profit_margin,
                   total_assets, shareholders_equity,
                   asset_liability_ratio
            FROM default.stock_financial
            WHERE code IN ('{codes_str}')
              AND report_date = '{report_date}'
            """
        else:
            query = f"""
            SELECT code,
                   report_date,
                   roe, eps, bvps,
                   net_profit, operating_revenue, operating_profit,
                   gross_profit_margin, net_profit_margin,
                   total_assets, shareholders_equity,
                   asset_liability_ratio
            FROM default.stock_financial
            WHERE code IN ('{codes_str}')
              AND report_date = '2024-12-31'
            """

        result = self.ch_client.query(query)
        df = pd.DataFrame(result.result_rows, columns=[
            'code', 'report_date', 'roe', 'eps', 'bvps',
            'net_profit', 'operating_revenue', 'operating_profit',
            'gross_profit_margin', 'net_profit_margin',
            'total_assets', 'shareholders_equity', 'asset_liability_ratio'
        ])
        return df

    def get_financial_growth(self, codes: List[str]) -> pd.DataFrame:
        """计算财务增长率 (同比)"""
        codes_str = "', '".join(codes)

        query = f"""
        WITH latest AS (
            SELECT code, eps, operating_revenue, report_date
            FROM default.stock_financial
            WHERE code IN ('{codes_str}')
              AND report_date = '2024-12-31'
        ),
        yoy AS (
            SELECT code, eps as yoy_eps, operating_revenue as yoy_revenue
            FROM default.stock_financial
            WHERE code IN ('{codes_str}')
              AND report_date = '2023-12-31'
        )
        SELECT latest.code,
               CASE WHEN yoy.yoy_eps != 0 AND yoy.yoy_eps IS NOT NULL AND yoy.yoy_eps > 0
                    THEN (latest.eps / yoy.yoy_eps - 1) * 100
                    ELSE 0 END as profit_growth,
               CASE WHEN yoy.yoy_revenue != 0 AND yoy.yoy_revenue IS NOT NULL
                    THEN (latest.operating_revenue / yoy.yoy_revenue - 1) * 100
                    ELSE 0 END as revenue_growth
        FROM latest
        LEFT JOIN yoy ON latest.code = yoy.code
        """

        try:
            result = self.ch_client.query(query)
            df = pd.DataFrame(result.result_rows, columns=['code', 'profit_growth', 'revenue_growth'])
            df = df.drop_duplicates(subset=['code'], keep='first')
            return df
        except Exception as e:
            print(f"计算增长率失败: {e}")
            return pd.DataFrame()

    def get_market_data(self, codes: List[str], date: str = None) -> pd.DataFrame:
        """获取市值数据"""
        codes_str = "', '".join(codes)

        if date:
            query = f"""
            SELECT code,
                   liutongshizhi as market_cap,
                   huanshoulv as turnover_rate
            FROM default.stock_data_qfq
            WHERE code IN ('{codes_str}')
              AND date = '{date}'
            """
        else:
            query = f"""
            SELECT code,
                   liutongshizhi as market_cap,
                   huanshoulv as turnover_rate
            FROM default.stock_data_qfq
            WHERE code IN ('{codes_str}')
              AND (code, date) IN (
                  SELECT code, max(date)
                  FROM default.stock_data_qfq
                  WHERE code IN ('{codes_str}')
                  GROUP BY code
              )
            """

        try:
            result = self.ch_client.query(query)
            df = pd.DataFrame(result.result_rows, columns=['code', 'market_cap', 'turnover_rate'])
            df['market_cap'] = df['market_cap'] / 1e8
            df = df.drop_duplicates(subset=['code'], keep='first')
            return df
        except Exception as e:
            print(f"获取市值数据失败: {e}")
            return pd.DataFrame()

    def get_price_momentum(self, codes: List[str], end_date: str, lookback_days: int = 60) -> pd.DataFrame:
        """计算价格动量"""
        codes_str = "', '".join(codes)

        query = f"""
        WITH latest AS (
            SELECT code, close as latest_close
            FROM default.stock_data_qfq
            WHERE code IN ('{codes_str}')
              AND date <= '{end_date}'
            ORDER BY date DESC
            LIMIT 1 BY code
        ),
        past AS (
            SELECT code, close as past_close
            FROM default.stock_data_qfq
            WHERE code IN ('{codes_str}')
              AND date <= date_sub(day, {lookback_days}, toDate('{end_date}'))
            ORDER BY date DESC
            LIMIT 1 BY code
        )
        SELECT latest.code,
               (latest.latest_close / past.past_close - 1) * 100 as momentum
        FROM latest
        JOIN past ON latest.code = past.code
        """

        try:
            result = self.ch_client.query(query)
            df = pd.DataFrame(result.result_rows, columns=['code', 'momentum'])
            return df
        except Exception as e:
            print(f"计算动量失败: {e}")
            return pd.DataFrame()

    def get_drawdown_3m(self, codes: List[str], end_date: str) -> pd.DataFrame:
        """计算3个月回撤"""
        codes_str = "', '".join(codes)

        query = f"""
        WITH price_data AS (
            SELECT code, date, close
            FROM default.stock_data_qfq
            WHERE code IN ('{codes_str}')
              AND date >= date_sub(month, 3, toDate('{end_date}'))
              AND date <= '{end_date}'
        ),
        max_price AS (
            SELECT code, max(close) as high_3m
            FROM price_data
            GROUP BY code
        ),
        latest_price AS (
            SELECT code, close as latest_close
            FROM price_data
            ORDER BY date DESC
            LIMIT 1 BY code
        )
        SELECT max_price.code,
               (latest_price.latest_close / max_price.high_3m - 1) * 100 as drawdown_3m
        FROM max_price
        JOIN latest_price ON max_price.code = latest_price.code
        """

        try:
            result = self.ch_client.query(query)
            df = pd.DataFrame(result.result_rows, columns=['code', 'drawdown_3m'])
            return df
        except Exception as e:
            print(f"计算回撤失败: {e}")
            return pd.DataFrame()

    def get_volatility(self, codes: List[str], end_date: str, lookback_days: int = 60) -> pd.DataFrame:
        """计算波动率"""
        codes_str = "', '".join(codes)

        query = f"""
        WITH daily_returns AS (
            SELECT code,
                   date,
                   close / lagInFrame(close, 1) OVER (PARTITION BY code ORDER BY date) - 1 as daily_return
            FROM default.stock_data_qfq
            WHERE code IN ('{codes_str}')
              AND date >= date_sub(day, {lookback_days}, toDate('{end_date}'))
              AND date <= '{end_date}'
        )
        SELECT code, stddevPop(daily_return) * sqrt(252) * 100 as volatility
        FROM daily_returns
        WHERE daily_return IS NOT NULL
        GROUP BY code
        """

        try:
            result = self.ch_client.query(query)
            df = pd.DataFrame(result.result_rows, columns=['code', 'volatility'])
            return df
        except Exception as e:
            print(f"计算波动率失败: {e}")
            return pd.DataFrame()


def build_factor_data(fetcher: LocalDataFetcher, codes: List[str], eval_date: str, min_market_cap: float = 0) -> pd.DataFrame:
    """从本地数据构建因子DataFrame"""
    print(f"\n构建因子数据 (评估日期: {eval_date})...")

    # 1. 获取财务数据
    print("  获取财务数据...")
    financial_df = fetcher.get_financial_data(codes)
    print(f"    获取到 {len(financial_df)} 只股票")

    # 2. 获取财务增长率
    print("  计算财务增长率...")
    growth_df = fetcher.get_financial_growth(codes)
    print(f"    计算完成 {len(growth_df)} 只股票")

    # 3. 获取市值数据
    print("  获取市值数据...")
    market_df = fetcher.get_market_data(codes)
    print(f"    获取到 {len(market_df)} 只股票")

    # 4. 获取动量
    print("  计算动量...")
    momentum_df = fetcher.get_price_momentum(codes, eval_date, lookback_days=60)
    print(f"    计算完成 {len(momentum_df)} 只股票")

    # 5. 获取3个月回撤
    print("  计算3个月回撤...")
    drawdown_df = fetcher.get_drawdown_3m(codes, eval_date)
    print(f"    计算完成 {len(drawdown_df)} 只股票")

    # 6. 获取波动率
    print("  计算波动率...")
    volatility_df = fetcher.get_volatility(codes, eval_date)
    print(f"    计算完成 {len(volatility_df)} 只股票")

    # 合并所有数据
    factor_df = pd.DataFrame({'code': codes})

    if not financial_df.empty:
        financial_df_unique = financial_df.drop_duplicates(subset=['code'], keep='first')
        factor_df = factor_df.merge(
            financial_df_unique[['code', 'roe', 'eps', 'net_profit_margin', 'gross_profit_margin']],
            on='code', how='left'
        )
    else:
        for col in ['roe', 'eps', 'net_profit_margin', 'gross_profit_margin']:
            factor_df[col] = np.nan

    if not growth_df.empty:
        growth_df_unique = growth_df.drop_duplicates(subset=['code'], keep='first')
        growth_df_unique = growth_df_unique.rename(columns={
            'profit_growth': 'net_profit_yoy',
            'revenue_growth': 'revenue_yoy'
        })
        factor_df = factor_df.merge(growth_df_unique, on='code', how='left')
    else:
        factor_df['net_profit_yoy'] = np.nan
        factor_df['revenue_yoy'] = np.nan

    if not market_df.empty:
        factor_df = factor_df.merge(market_df[['code', 'market_cap']], on='code', how='left')
    else:
        factor_df['market_cap'] = np.nan

    if not momentum_df.empty:
        factor_df = factor_df.merge(momentum_df, on='code', how='left')
    else:
        factor_df['momentum'] = np.nan

    if not drawdown_df.empty:
        factor_df = factor_df.merge(drawdown_df, on='code', how='left')
    else:
        factor_df['drawdown_3m'] = np.nan

    if not volatility_df.empty:
        factor_df = factor_df.merge(volatility_df, on='code', how='left')
    else:
        factor_df['volatility'] = np.nan

    # 从valuation_local.csv获取PE、PB、ROE、股息率
    print("  从valuation_local.csv获取PE/PB/ROE数据...")
    try:
        csv_path = os.path.join(os.path.dirname(__file__), 'valuation_local.csv')
        val_csv = pd.read_csv(csv_path)
        val_csv['code'] = val_csv['code'].astype(str).str.zfill(6)

        val_csv['roe_calc'] = np.where(
            val_csv['bvps'] > 0,
            val_csv['eps'] / val_csv['bvps'] * 100,
            np.nan
        )

        pe_df = val_csv[['code', 'pe_ttm', 'pb', 'dividend_yield', 'roe_calc']].copy()
        pe_df = pe_df.rename(columns={'pe_ttm': 'pe'})
        pe_df = pe_df.drop_duplicates(subset=['code'], keep='first')
        pe_df = pe_df[pe_df['code'].isin(codes)]

        factor_df = factor_df.merge(pe_df, on='code', how='left')

        if 'roe_calc' in factor_df.columns:
            factor_df['roe'] = factor_df['roe_calc'].combine_first(factor_df['roe'])
            factor_df = factor_df.drop(columns=['roe_calc'])

        div_csv_path = os.path.join(os.path.dirname(__file__), 'dividend_yield_all.csv')
        if os.path.exists(div_csv_path):
            div_csv = pd.read_csv(div_csv_path)
            div_csv['code'] = div_csv['code'].astype(str).str.zfill(6)
            div_csv = div_csv.drop_duplicates(subset=['code'], keep='first')
            div_dict = dict(zip(div_csv['code'], div_csv['dividend_yield']))
            factor_df['dividend_yield'] = factor_df['code'].map(div_dict).combine_first(factor_df['dividend_yield'])
    except Exception as e:
        print(f"    获取PE数据失败: {e}")
        factor_df['pe'] = np.nan
        factor_df['pb'] = np.nan
        factor_df['dividend_yield'] = np.nan

    factor_df = factor_df.drop_duplicates(subset=['code'], keep='first')

    # 市值过滤
    if min_market_cap > 0:
        factor_df = factor_df[factor_df['market_cap'] >= min_market_cap]

    # 填充缺失值
    defaults = {
        'roe': factor_df['roe'].median() if factor_df['roe'].notna().any() else 10,
        'net_profit_yoy': 0, 'revenue_yoy': 0,
        'pe': factor_df['pe'].median() if factor_df['pe'].notna().any() else 20,
        'pb': factor_df['pb'].median() if factor_df['pb'].notna().any() else 3,
        'dividend_yield': factor_df['dividend_yield'].median() if factor_df['dividend_yield'].notna().any() else 0.02,
        'market_cap': factor_df['market_cap'].median() if factor_df['market_cap'].notna().any() and factor_df['market_cap'].median() > 0 else 300,
        'momentum': 0, 'drawdown_3m': -10, 'volatility': 30, 'cash_flow_ratio': 1.0
    }
    factor_df = factor_df.fillna(defaults)
    factor_df['roe_stability'] = 80
    factor_df['profit_growth'] = factor_df['net_profit_yoy']

    print(f"  因子数据构建完成: {len(factor_df)} 只股票")
    return factor_df


def apply_mild_weight_tilt(original_weights: pd.DataFrame,
                           scored_data: pd.DataFrame,
                           tilt_strength: float = 0.3,
                           max_weight: float = 0.08) -> pd.DataFrame:
    """
    温和的权重调整 - 核心修复

    原来的问题: 10档 multiplier (0.3x~2.5x) 过于激进，
    配合 max_weight=5% clip + 归一化后，所有策略趋同。

    新方法: 用 composite_score 的百分位做线性倾斜，
    tilt_strength 控制调整幅度。

    multiplier = 1 + tilt_strength * (percentile - 0.5)
    即: 中位数股票不变, top 股票超配, bottom 股票低配
    tilt_strength=0.3 时: top → 1.15x, bottom → 0.85x
    tilt_strength=0.5 时: top → 1.25x, bottom → 0.75x
    """
    portfolio = original_weights.copy()

    # 合并得分
    portfolio = portfolio.merge(
        scored_data[['code', 'composite_score']],
        on='code', how='left'
    )
    portfolio['composite_score'] = portfolio['composite_score'].fillna(50)

    # 计算百分位
    portfolio['percentile'] = portfolio['composite_score'].rank(pct=True)

    # 线性倾斜: 中位数不变
    portfolio['multiplier'] = 1.0 + tilt_strength * (portfolio['percentile'] - 0.5)

    # 计算调整后权重
    portfolio['adjusted_weight'] = portfolio['weight'] * portfolio['multiplier']

    # 约束: 不让单股过大
    portfolio['adjusted_weight'] = portfolio['adjusted_weight'].clip(upper=max_weight)

    # 归一化
    total = portfolio['adjusted_weight'].sum()
    if total > 0:
        portfolio['adjusted_weight'] = portfolio['adjusted_weight'] / total

    return portfolio


class BacktestEngine:
    """回测引擎"""

    def __init__(self):
        self.ch_client = clickhouse_connect.get_client(host='192.168.0.74', port=8123, compress=False, query_limit=0)
        self.data_fetcher = LocalDataFetcher()

    def get_daily_prices(self, codes: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """获取复权日线数据"""
        codes_str = "', '".join(codes)
        query = f"""
        SELECT toString(date) as date, code, close
        FROM default.stock_data_qfq
        WHERE code IN ('{codes_str}')
          AND date >= '{start_date}'
          AND date <= '{end_date}'
        ORDER BY date, code
        """
        result = self.ch_client.query(query)
        df = pd.DataFrame(result.result_rows, columns=['date', 'code', 'close'])
        return df

    def run_backtest(self,
                     portfolio: pd.DataFrame,
                     start_date: str,
                     end_date: str,
                     initial_capital: float = 1000000,
                     strategy_type: str = 'stable',
                     tilt_strength: float = 0.3) -> Dict:
        """运行回测"""
        print("=" * 60)
        print(f"运行回测 ({strategy_type}, tilt={tilt_strength})")
        print(f"回测区间: {start_date} ~ {end_date}")
        print("=" * 60)

        codes = portfolio['code'].tolist()

        # 构建因子数据
        factor_data = build_factor_data(self.data_fetcher, codes, end_date, min_market_cap=0)

        # 只保留有因子数据的股票
        filtered_codes = factor_data['code'].tolist()
        if len(filtered_codes) < len(codes):
            print(f"\n注意: {len(codes) - len(filtered_codes)} 只股票无因子数据")
            portfolio = portfolio[portfolio['code'].isin(filtered_codes)].copy()
            portfolio['weight'] = portfolio['weight'] / portfolio['weight'].sum()
            codes = filtered_codes

        # 计算因子得分 (R4/R5 用不同因子权重)
        print(f"\n计算因子得分 (策略: {strategy_type})...")
        factor_calc = EnhancedFactorCalculator(strategy_type=strategy_type)
        scored_data = factor_calc.calculate_all_scores(factor_data)

        # 温和权重调整 (核心修复)
        print(f"\n温和权重调整 (tilt_strength={tilt_strength})...")
        enhanced_portfolio = apply_mild_weight_tilt(
            portfolio, scored_data,
            tilt_strength=tilt_strength,
            max_weight=0.08
        )

        # 统计调整幅度
        enhanced_portfolio['weight_change'] = enhanced_portfolio['adjusted_weight'] / enhanced_portfolio['weight'] - 1
        increased = (enhanced_portfolio['weight_change'] > 0.05).sum()
        decreased = (enhanced_portfolio['weight_change'] < -0.05).sum()
        print(f"  增加>5%: {increased}只, 减少>5%: {decreased}只")

        # Top 10 调整
        top10 = enhanced_portfolio.nlargest(10, 'adjusted_weight')
        print(f"\n  Top 10 权重:")
        for _, row in top10.iterrows():
            print(f"    {row['code']}: {row['weight']*100:.2f}% -> {row['adjusted_weight']*100:.2f}% "
                  f"(得分{row['composite_score']:.0f}, pct{row['percentile']:.0%})")

        # 获取价格数据
        print(f"\n获取 {len(codes)} 只股票价格数据...")
        price_df = self.get_daily_prices(codes, start_date, end_date)
        print(f"获取到 {len(price_df)} 条价格数据")

        price_df = price_df.drop_duplicates(subset=['date', 'code'], keep='last')
        price_pivot = price_df.pivot(index='date', columns='code', values='close')
        dates = sorted(price_pivot.index.tolist())
        print(f"共 {len(dates)} 个交易日")

        # 股息率
        dividend_dict = dict(zip(factor_data['code'], factor_data['dividend_yield']))
        daily_dividend = {code: div / 252 for code, div in dividend_dict.items()}

        # 计算每日收益
        def calc_daily_returns(weight_dict):
            daily_returns = []
            for i in range(1, len(dates)):
                date = dates[i]
                prev_date = dates[i-1]
                daily_return = 0
                for code in codes:
                    if code in price_pivot.columns:
                        curr_price = price_pivot.loc[date, code]
                        prev_price = price_pivot.loc[prev_date, code]
                        if pd.notna(curr_price) and pd.notna(prev_price) and prev_price > 0:
                            price_return = (curr_price / prev_price - 1)
                            div_return = daily_dividend.get(code, 0)
                            stock_return = price_return + div_return
                            weight = weight_dict.get(code, 0)
                            daily_return += stock_return * weight
                daily_returns.append({'date': date, 'return': daily_return})
            returns_df = pd.DataFrame(daily_returns)
            returns_df['cumulative'] = (1 + returns_df['return']).cumprod()
            return returns_df

        # 增强版收益
        enhanced_weight_dict = dict(zip(enhanced_portfolio['code'], enhanced_portfolio['adjusted_weight']))
        enhanced_returns_df = calc_daily_returns(enhanced_weight_dict)

        # 原始版收益
        orig_weight_dict = dict(zip(portfolio['code'], portfolio['weight']))
        original_returns_df = calc_daily_returns(orig_weight_dict)

        # 计算绩效指标
        def calc_metrics(returns_df):
            total_return = returns_df['cumulative'].iloc[-1] - 1
            n_years = len(returns_df) / 252
            annual_return = (1 + total_return) ** (1/n_years) - 1

            cummax = returns_df['cumulative'].cummax()
            drawdown = (returns_df['cumulative'] - cummax) / cummax
            max_drawdown = drawdown.min()

            daily_returns = returns_df['return']
            sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0

            return {
                'annual_return': annual_return,
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'sharpe': sharpe,
                'final_nav': returns_df['cumulative'].iloc[-1]
            }

        enhanced_metrics = calc_metrics(enhanced_returns_df)
        original_metrics = calc_metrics(original_returns_df)

        # 打印结果
        print(f"\n{'='*50}")
        print(f"增强版: 年化{enhanced_metrics['annual_return']*100:.2f}%, "
              f"回撤{enhanced_metrics['max_drawdown']*100:.2f}%, "
              f"夏普{enhanced_metrics['sharpe']:.2f}")
        print(f"原始版: 年化{original_metrics['annual_return']*100:.2f}%, "
              f"回撤{original_metrics['max_drawdown']*100:.2f}%, "
              f"夏普{original_metrics['sharpe']:.2f}")
        diff = (enhanced_metrics['annual_return'] - original_metrics['annual_return']) * 100
        print(f"提升: {diff:+.2f}%")
        print(f"{'='*50}")

        return {
            'enhanced': enhanced_metrics,
            'original': original_metrics,
            'enhanced_returns': enhanced_returns_df,
            'original_returns': original_returns_df,
            'enhanced_weights': enhanced_portfolio,
            'factor_data': scored_data,
            'strategy_type': strategy_type
        }


def main():
    print("=" * 70)
    print("增强版多因子策略回测 v2 (修复版)")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 加载持仓 - FIX: 使用原始 weight 列 (不是 adjusted_weight)
    base_dir = os.path.dirname(__file__)
    print(f"\n加载持仓文件...")

    r4 = pd.read_csv(os.path.join(base_dir, 'r4_weights_local.csv'))
    r5 = pd.read_csv(os.path.join(base_dir, 'r5_weights_local.csv'))

    r4['code'] = r4['code'].astype(str).str.zfill(6)
    r5['code'] = r5['code'].astype(str).str.zfill(6)

    # FIX: 始终使用原始 weight 列, 不使用 adjusted_weight
    r4 = r4[['code', 'weight']].copy()
    r5 = r5[['code', 'weight']].copy()

    # 归一化
    r4['weight'] = r4['weight'] / r4['weight'].sum()
    r5['weight'] = r5['weight'] / r5['weight'].sum()

    print(f"R4 稳健型: {len(r4)} 只股票, 最大权重: {r4['weight'].max()*100:.2f}%")
    print(f"R5 进取型: {len(r5)} 只股票, 最大权重: {r5['weight'].max()*100:.2f}%")

    engine = BacktestEngine()

    start_date = '2020-01-01'
    end_date = '2025-12-31'

    # R4 回测 (稳健型: 温和倾斜)
    print("\n" + "=" * 70)
    print("【R4 稳健型策略回测】")
    print("=" * 70)
    r4_result = engine.run_backtest(
        portfolio=r4,
        start_date=start_date,
        end_date=end_date,
        strategy_type='stable',
        tilt_strength=0.3
    )

    # R5 回测 (进取型: 稍强倾斜)
    print("\n" + "=" * 70)
    print("【R5 进取型策略回测】")
    print("=" * 70)
    r5_result = engine.run_backtest(
        portfolio=r5,
        start_date=start_date,
        end_date=end_date,
        strategy_type='aggressive',
        tilt_strength=0.4
    )

    # 结果对比
    print("\n" + "=" * 70)
    print("【回测结果总结】")
    print("=" * 70)

    print(f"""
┌───────────┬───────────┬───────────┬──────────┬───────────┐
│  策略     │ 原始年化  │ 增强年化  │   提升   │ 夏普比率  │
├───────────┼───────────┼───────────┼──────────┼───────────┤
│ R4 稳健   │  {r4_result['original']['annual_return']*100:>6.2f}%  │  {r4_result['enhanced']['annual_return']*100:>6.2f}%  │ {(r4_result['enhanced']['annual_return']-r4_result['original']['annual_return'])*100:>+5.2f}% │ {r4_result['original']['sharpe']:.2f}→{r4_result['enhanced']['sharpe']:.2f} │
│ R5 进取   │  {r5_result['original']['annual_return']*100:>6.2f}%  │  {r5_result['enhanced']['annual_return']*100:>6.2f}%  │ {(r5_result['enhanced']['annual_return']-r5_result['original']['annual_return'])*100:>+5.2f}% │ {r5_result['original']['sharpe']:.2f}→{r5_result['enhanced']['sharpe']:.2f} │
└───────────┴───────────┴───────────┴──────────┴───────────┘
""")

    # 保存结果
    output_dir = os.path.dirname(os.path.abspath(__file__))
    r4_result['enhanced_weights'].to_csv(os.path.join(output_dir, 'r4_weights_result.csv'), index=False)
    r5_result['enhanced_weights'].to_csv(os.path.join(output_dir, 'r5_weights_result.csv'), index=False)

    # 保存JSON结果
    result_json = {
        'run_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'backtest_period': f'{start_date} ~ {end_date}',
        'R4_stable': {
            'enhanced': {k: v for k, v in r4_result['enhanced'].items() if isinstance(v, (int, float))},
            'original': {k: v for k, v in r4_result['original'].items() if isinstance(v, (int, float))},
            'strategy_type': 'stable'
        },
        'R5_aggressive': {
            'enhanced': {k: v for k, v in r5_result['enhanced'].items() if isinstance(v, (int, float))},
            'original': {k: v for k, v in r5_result['original'].items() if isinstance(v, (int, float))},
            'strategy_type': 'aggressive'
        }
    }
    json_path = os.path.join(output_dir, 'factor_backtest_result.json')
    with open(json_path, 'w') as f:
        json.dump(result_json, f, indent=2)

    print(f"\n结果已保存")
    print("\n回测完成!")


if __name__ == '__main__':
    main()
