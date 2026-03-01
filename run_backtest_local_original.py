"""
增强版多因子策略回测 - 使用本地ClickHouse财务数据
直接从本地stock_financial表获取财务数据
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import clickhouse_connect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.factor_calculator_v2 import EnhancedFactorCalculator
from strategy.enhanced_strategy_v2 import EnhancedStrategyV2


class LocalDataFetcher:
    """本地数据获取器 - 从ClickHouse获取所有数据"""

    def __init__(self):
        self.ch_client = clickhouse_connect.get_client(host='127.0.0.1', port=8123)

    def get_stock_prices(self, codes: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """从ClickHouse获取复权价格"""
        codes_str = "', '".join(codes)
        query = f"""
        SELECT toString(date) as date, code, close
        FROM default.stock_data_qfq_new
        WHERE code IN ('{codes_str}')
          AND date >= '{start_date}'
          AND date <= '{end_date}'
        ORDER BY date, code
        """
        result = self.ch_client.query(query)
        df = pd.DataFrame(result.result_rows, columns=['date', 'code', 'close'])
        return df

    def get_financial_data(self, codes: List[str], report_date: str = None) -> pd.DataFrame:
        """
        从本地ClickHouse获取财务数据
        如果不指定report_date，获取最新一期数据
        """
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
            # 获取每只股票最新年报数据 (使用2024年年报，因为季报ROE未年化)
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

    def get_financial_growth(self, codes: List[str], report_date: str = None) -> pd.DataFrame:
        """计算财务增长率 (同比) - 使用EPS计算，因为net_profit数据可能为0"""
        codes_str = "', '".join(codes)

        # 使用EPS计算同比增长率 (EPS = 每股收益，与净利润成正比)
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
            # 去重
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
                   toFloat64OrNull(liutongshizhi) as market_cap,
                   toFloat64OrNull(huanshoulv) as turnover_rate
            FROM default.stock_data
            WHERE code IN ('{codes_str}')
              AND date = '{date}'
            """
        else:
            # 获取最新数据
            query = f"""
            SELECT code,
                   toFloat64OrNull(liutongshizhi) as market_cap,
                   toFloat64OrNull(huanshoulv) as turnover_rate
            FROM default.stock_data
            WHERE code IN ('{codes_str}')
              AND (code, date) IN (
                  SELECT code, max(date)
                  FROM default.stock_data
                  WHERE code IN ('{codes_str}')
                  GROUP BY code
              )
            """

        try:
            result = self.ch_client.query(query)
            df = pd.DataFrame(result.result_rows, columns=['code', 'market_cap', 'turnover_rate'])
            # 市值单位转为亿
            df['market_cap'] = df['market_cap'] / 1e8
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
            FROM default.stock_data_qfq_new
            WHERE code IN ('{codes_str}')
              AND date <= '{end_date}'
            ORDER BY date DESC
            LIMIT 1 BY code
        ),
        past AS (
            SELECT code, close as past_close
            FROM default.stock_data_qfq_new
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
            FROM default.stock_data_qfq_new
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
            FROM default.stock_data_qfq_new
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


def build_factor_data(fetcher: LocalDataFetcher, codes: List[str], eval_date: str, min_market_cap: float = 100) -> pd.DataFrame:
    """
    从本地数据构建因子DataFrame

    Args:
        min_market_cap: 最小市值(亿)，默认100亿
    """
    print(f"\n构建因子数据 (评估日期: {eval_date})...")
    print(f"  市值过滤: >= {min_market_cap}亿")

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
    print("  合并因子数据...")
    factor_df = pd.DataFrame({'code': codes})

    # 合并财务数据 (确保每个code只有一行)
    if not financial_df.empty:
        # 去重，保留第一条（最新的）
        financial_df_unique = financial_df.drop_duplicates(subset=['code'], keep='first')
        factor_df = factor_df.merge(
            financial_df_unique[['code', 'roe', 'eps', 'net_profit_margin', 'gross_profit_margin']],
            on='code', how='left'
        )
    else:
        for col in ['roe', 'eps', 'net_profit_margin', 'gross_profit_margin']:
            factor_df[col] = np.nan

    # 合并增长数据 (确保每个code只有一行)
    if not growth_df.empty:
        growth_df_unique = growth_df.drop_duplicates(subset=['code'], keep='first')
        # 重命名列以匹配因子计算器期望的名称
        growth_df_unique = growth_df_unique.rename(columns={
            'profit_growth': 'net_profit_yoy',
            'revenue_growth': 'revenue_yoy'
        })
        factor_df = factor_df.merge(growth_df_unique, on='code', how='left')
    else:
        factor_df['net_profit_yoy'] = np.nan
        factor_df['revenue_yoy'] = np.nan

    # 合并市值数据
    if not market_df.empty:
        factor_df = factor_df.merge(market_df[['code', 'market_cap']], on='code', how='left')
    else:
        factor_df['market_cap'] = np.nan

    # 合并动量
    if not momentum_df.empty:
        factor_df = factor_df.merge(momentum_df, on='code', how='left')
    else:
        factor_df['momentum'] = np.nan

    # 合并回撤
    if not drawdown_df.empty:
        factor_df = factor_df.merge(drawdown_df, on='code', how='left')
    else:
        factor_df['drawdown_3m'] = np.nan

    # 合并波动率
    if not volatility_df.empty:
        factor_df = factor_df.merge(volatility_df, on='code', how='left')
    else:
        factor_df['volatility'] = np.nan

    # 从本地valuation_local表获取PE、PB、和正确计算的ROE
    print("  获取本地PE/PB/ROE数据...")
    try:
        codes_str = "', '".join(codes)
        pe_query = f"""
        SELECT code, pe_ttm as pe, pb, dividend_yield, total_mv,
               CASE WHEN bvps > 0 THEN eps / bvps * 100 ELSE NULL END as roe_calc
        FROM default.valuation_local
        WHERE code IN ('{codes_str}')
        """
        pe_result = fetcher.ch_client.query(pe_query)
        pe_df = pd.DataFrame(pe_result.result_rows, columns=['code', 'pe', 'pb', 'dividend_yield', 'total_mv', 'roe_calc'])
        pe_df = pe_df.drop_duplicates(subset=['code'], keep='first')

        # 合并PE/PB/ROE数据
        factor_df = factor_df.merge(pe_df, on='code', how='left')
        print(f"    获取到 {len(pe_df[pe_df['pe'].notna()])} 只股票PE数据")

        # 使用正确计算的ROE替换原来的ROE
        factor_df['roe'] = factor_df['roe_calc'].combine_first(factor_df['roe'])
        print(f"    ROE中位数: {factor_df['roe'].median():.2f}%")

        # 更新市值数据 (使用更准确的total_mv)
        factor_df['market_cap'] = factor_df['total_mv'].combine_first(factor_df['market_cap'])
    except Exception as e:
        print(f"    获取PE数据失败: {e}")
        factor_df['pe'] = np.nan
        factor_df['pb'] = np.nan
        factor_df['dividend_yield'] = np.nan

    # 市值过滤
    before_filter = len(factor_df)
    factor_df = factor_df[factor_df['market_cap'] >= min_market_cap]
    print(f"  市值过滤: {before_filter} -> {len(factor_df)} 只股票 (过滤掉 {before_filter - len(factor_df)} 只小市值股票)")

    # 填充缺失值
    defaults = {
        'roe': factor_df['roe'].median() if factor_df['roe'].notna().any() else 10,
        'net_profit_yoy': 0,
        'revenue_yoy': 0,
        'pe': factor_df['pe'].median() if factor_df['pe'].notna().any() else 20,
        'pb': factor_df['pb'].median() if factor_df['pb'].notna().any() else 3,
        'dividend_yield': factor_df['dividend_yield'].median() if factor_df['dividend_yield'].notna().any() else 0.02,
        'market_cap': factor_df['market_cap'].median() if factor_df['market_cap'].notna().any() else 200,
        'momentum': 0,
        'drawdown_3m': -10,
        'volatility': 30,
        'cash_flow_ratio': 1.0
    }
    factor_df = factor_df.fillna(defaults)

    # 添加ROE稳定性 (基于净利润率的稳定性估计)
    factor_df['roe_stability'] = 80  # 默认值

    # 为兼容性添加profit_growth别名
    factor_df['profit_growth'] = factor_df['net_profit_yoy']

    print(f"  因子数据构建完成: {len(factor_df)} 只股票")

    # 显示因子统计
    print(f"\n  因子统计:")
    print(f"    ROE中位数: {factor_df['roe'].median():.2f}%")
    print(f"    PE中位数: {factor_df['pe'].median():.1f}")
    print(f"    PB中位数: {factor_df['pb'].median():.2f}")
    print(f"    股息率中位数: {factor_df['dividend_yield'].median()*100:.2f}%")
    print(f"    净利润增速中位数: {factor_df['net_profit_yoy'].median():.2f}%")
    print(f"    市值中位数: {factor_df['market_cap'].median():.0f}亿")
    print(f"    动量中位数: {factor_df['momentum'].median():.2f}%")
    print(f"    3个月回撤中位数: {factor_df['drawdown_3m'].median():.2f}%")

    return factor_df


class BacktestEngine:
    """回测引擎"""

    def __init__(self):
        self.ch_client = clickhouse_connect.get_client(host='127.0.0.1', port=8123)
        self.data_fetcher = LocalDataFetcher()

    def get_daily_prices(self, codes: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """获取复权日线数据"""
        codes_str = "', '".join(codes)
        query = f"""
        SELECT toString(date) as date, code, close
        FROM default.stock_data_qfq_new
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
                     strategy_type: str = 'stable') -> Dict:
        """
        运行回测
        """
        print("=" * 60)
        print(f"运行回测 ({strategy_type})")
        print(f"回测区间: {start_date} ~ {end_date}")
        print("=" * 60)

        codes = portfolio['code'].tolist()

        # 构建因子数据 (市值过滤 >= 100亿)
        factor_data = build_factor_data(self.data_fetcher, codes, end_date, min_market_cap=100)

        # 更新codes列表 (只保留通过市值过滤的股票)
        filtered_codes = factor_data['code'].tolist()
        if len(filtered_codes) < len(codes):
            print(f"\n注意: {len(codes) - len(filtered_codes)} 只股票因市值<100亿被过滤")
            portfolio = portfolio[portfolio['code'].isin(filtered_codes)].copy()
            # 重新归一化权重
            portfolio['weight'] = portfolio['weight'] / portfolio['weight'].sum()
            codes = filtered_codes

        # 计算因子得分
        print("\n计算因子得分...")
        factor_calc = EnhancedFactorCalculator(strategy_type=strategy_type)
        scored_data = factor_calc.calculate_all_scores(factor_data)

        # 调整权重
        print("\n调整权重...")
        strategy = EnhancedStrategyV2()

        # 构建factor_data用于绩优股回撤判断，需要匹配strategy期望的列名
        factor_data_for_strategy = factor_data[['code', 'roe', 'net_profit_yoy', 'drawdown_3m']].copy()
        factor_data_for_strategy = factor_data_for_strategy.rename(columns={'drawdown_3m': 'momentum_3m'})
        factor_data_for_strategy = factor_data_for_strategy.fillna({'roe': 10, 'net_profit_yoy': 0, 'momentum_3m': -10})

        enhanced_weights = strategy.adjust_weights(
            original_portfolio=portfolio,
            composite_scores=scored_data,
            factor_data=factor_data_for_strategy
        )

        # 获取价格数据
        print(f"\n获取 {len(codes)} 只股票价格数据...")
        price_df = self.get_daily_prices(codes, start_date, end_date)
        print(f"获取到 {len(price_df)} 条价格数据")

        price_pivot = price_df.pivot(index='date', columns='code', values='close')
        dates = sorted(price_pivot.index.tolist())
        print(f"共 {len(dates)} 个交易日")

        # 获取股息率数据
        dividend_dict = dict(zip(factor_data['code'], factor_data['dividend_yield']))
        # 计算每日股息收益 (年化股息率 / 252个交易日)
        daily_dividend = {code: div / 252 for code, div in dividend_dict.items()}

        # 计算每日收益 - 增强版 (包含股息收入)
        enhanced_weight_dict = dict(zip(enhanced_weights['code'], enhanced_weights['adjusted_weight']))
        enhanced_daily_returns = []

        for i in range(1, len(dates)):
            date = dates[i]
            prev_date = dates[i-1]

            daily_return = 0
            for code in codes:
                if code in price_pivot.columns:
                    curr_price = price_pivot.loc[date, code]
                    prev_price = price_pivot.loc[prev_date, code]

                    if pd.notna(curr_price) and pd.notna(prev_price) and prev_price > 0:
                        # 价格收益
                        price_return = (curr_price / prev_price - 1)
                        # 股息收益
                        div_return = daily_dividend.get(code, 0)
                        # 总收益 = 价格收益 + 股息收益
                        stock_return = price_return + div_return
                        weight = enhanced_weight_dict.get(code, 0)
                        daily_return += stock_return * weight

            enhanced_daily_returns.append({'date': date, 'return': daily_return})

        enhanced_returns_df = pd.DataFrame(enhanced_daily_returns)
        enhanced_returns_df['cumulative'] = (1 + enhanced_returns_df['return']).cumprod()

        # 计算每日收益 - 原始版 (包含股息收入)
        orig_weight_dict = dict(zip(portfolio['code'], portfolio['weight']))
        original_daily_returns = []

        for i in range(1, len(dates)):
            date = dates[i]
            prev_date = dates[i-1]

            daily_return = 0
            for code in codes:
                if code in price_pivot.columns:
                    curr_price = price_pivot.loc[date, code]
                    prev_price = price_pivot.loc[prev_date, code]

                    if pd.notna(curr_price) and pd.notna(prev_price) and prev_price > 0:
                        # 价格收益
                        price_return = (curr_price / prev_price - 1)
                        # 股息收益
                        div_return = daily_dividend.get(code, 0)
                        # 总收益 = 价格收益 + 股息收益
                        stock_return = price_return + div_return
                        weight = orig_weight_dict.get(code, 0)
                        daily_return += stock_return * weight

            original_daily_returns.append({'date': date, 'return': daily_return})

        original_returns_df = pd.DataFrame(original_daily_returns)
        original_returns_df['cumulative'] = (1 + original_returns_df['return']).cumprod()

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

            calmar = abs(annual_return / max_drawdown) if max_drawdown != 0 else 0

            return {
                'annual_return': annual_return,
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'sharpe': sharpe,
                'calmar': calmar,
                'final_nav': returns_df['cumulative'].iloc[-1]
            }

        enhanced_metrics = calc_metrics(enhanced_returns_df)
        original_metrics = calc_metrics(original_returns_df)

        # 打印结果
        print(f"\n增强版 绩效指标:")
        print(f"  年化收益: {enhanced_metrics['annual_return']*100:.2f}%")
        print(f"  最大回撤: {enhanced_metrics['max_drawdown']*100:.2f}%")
        print(f"  夏普比率: {enhanced_metrics['sharpe']:.2f}")
        print(f"  最终净值: {enhanced_metrics['final_nav']:.4f}")

        print(f"\n原始版 绩效指标:")
        print(f"  年化收益: {original_metrics['annual_return']*100:.2f}%")
        print(f"  最大回撤: {original_metrics['max_drawdown']*100:.2f}%")
        print(f"  夏普比率: {original_metrics['sharpe']:.2f}")
        print(f"  最终净值: {original_metrics['final_nav']:.4f}")

        return {
            'enhanced': enhanced_metrics,
            'original': original_metrics,
            'enhanced_returns': enhanced_returns_df,
            'original_returns': original_returns_df,
            'enhanced_weights': enhanced_weights,
            'factor_data': scored_data
        }


def main():
    print("=" * 70)
    print("增强版多因子策略回测 - 本地数据版")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 加载持仓
    portfolio_file = "/Users/kangbing/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/wxid_gusa1p3piit022_6ca9/msg/file/2026-01/四大类策略持仓20251231.xlsx"

    print(f"\n加载持仓文件...")
    r4 = pd.read_excel(portfolio_file, sheet_name='稳健型股票组合(风险等级R4)')
    r5 = pd.read_excel(portfolio_file, sheet_name='进取型股票组合(风险等级R5)')

    r4.columns = ['code', 'weight']
    r5.columns = ['code', 'weight']
    r4['code'] = r4['code'].apply(lambda x: x.split('.')[0])
    r5['code'] = r5['code'].apply(lambda x: x.split('.')[0])

    print(f"R4 稳健型: {len(r4)} 只股票")
    print(f"R5 进取型: {len(r5)} 只股票")

    # 初始化回测引擎
    engine = BacktestEngine()

    # 回测参数
    start_date = '2019-01-01'
    end_date = '2025-12-31'

    # R4 回测
    print("\n" + "=" * 70)
    print("【R4 稳健型策略回测】")
    print("=" * 70)
    r4_result = engine.run_backtest(
        portfolio=r4,
        start_date=start_date,
        end_date=end_date,
        strategy_type='stable'
    )

    # R5 回测
    print("\n" + "=" * 70)
    print("【R5 进取型策略回测】")
    print("=" * 70)
    r5_result = engine.run_backtest(
        portfolio=r5,
        start_date=start_date,
        end_date=end_date,
        strategy_type='aggressive'
    )

    # 结果对比
    print("\n" + "=" * 70)
    print("【回测结果总结】")
    print("=" * 70)

    print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│                         回测结果对比                                │
├─────────────┬───────────┬───────────┬──────────┬───────────────────┤
│   策略      │ 原始年化  │ 增强年化  │   提升   │     夏普比率      │
├─────────────┼───────────┼───────────┼──────────┼───────────────────┤
│ R4 稳健型   │  {r4_result['original']['annual_return']*100:>6.2f}%  │  {r4_result['enhanced']['annual_return']*100:>6.2f}%  │ {(r4_result['enhanced']['annual_return']-r4_result['original']['annual_return'])*100:>+6.2f}% │ {r4_result['original']['sharpe']:.2f} → {r4_result['enhanced']['sharpe']:.2f}       │
│ R5 进取型   │  {r5_result['original']['annual_return']*100:>6.2f}%  │  {r5_result['enhanced']['annual_return']*100:>6.2f}%  │ {(r5_result['enhanced']['annual_return']-r5_result['original']['annual_return'])*100:>+6.2f}% │ {r5_result['original']['sharpe']:.2f} → {r5_result['enhanced']['sharpe']:.2f}       │
└─────────────┴───────────┴───────────┴──────────┴───────────────────┘
""")

    # 显示Top权重调整
    print("\n【权重调整 Top 10 - R4稳健型】")
    print(f"{'代码':<10} {'原权重':>10} {'新权重':>10} {'得分':>8}")
    print("-" * 45)
    top10_r4 = r4_result['enhanced_weights'].nlargest(10, 'adjusted_weight')
    for _, row in top10_r4.iterrows():
        orig_w = r4[r4['code'] == row['code']]['weight'].values[0] * 100 if len(r4[r4['code'] == row['code']]) > 0 else 0
        score = row.get('composite_score', 0)
        print(f"{row['code']:<10} {orig_w:>9.2f}% {row['adjusted_weight']*100:>9.2f}% {score:>8.1f}")

    # 保存结果
    output_dir = os.path.dirname(__file__)
    r4_result['enhanced_weights'].to_csv(f"{output_dir}/r4_weights_local.csv", index=False)
    r5_result['enhanced_weights'].to_csv(f"{output_dir}/r5_weights_local.csv", index=False)

    print(f"\n结果已保存到: {output_dir}")
    print("\n回测完成!")


if __name__ == '__main__':
    main()
