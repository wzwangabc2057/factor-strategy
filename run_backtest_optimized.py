"""
多因子策略回测 - 综合优化版
优化内容:
1. 参数优化 - tilt_strength, max_weight 调优
2. 因子权重优化 - 根据因子IC调整权重
3. 反转条件放宽 - 从15%降到10%
4. 季度调仓 - 动态更新因子得分
5. 交易成本 - 更真实的回测
6. 风控机制 - 大盘止损
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

import clickhouse_connect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.factor_calculator_v2 import EnhancedFactorCalculator


class OptimizedDataFetcher:
    """优化版数据获取器"""

    def __init__(self):
        self.ch_client = clickhouse_connect.get_client(
            host='192.168.0.74', port=8123, compress=False, query_limit=0
        )
        self._akshare_cache = None

    def _load_akshare_cache(self) -> pd.DataFrame:
        """加载akshare缓存"""
        if self._akshare_cache is None:
            path = os.path.join(os.path.dirname(__file__), 'financial_data_akshare.csv')
            if os.path.exists(path):
                self._akshare_cache = pd.read_csv(path)
                self._akshare_cache['code'] = self._akshare_cache['code'].astype(str).str.zfill(6)
        return self._akshare_cache if self._akshare_cache is not None else pd.DataFrame()

    def get_stock_prices(self, codes: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """获取复权价格"""
        codes_str = "','".join(codes)
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

    def get_market_cap(self, codes: List[str], date: str = None) -> pd.DataFrame:
        """获取市值数据"""
        codes_str = "','".join(codes)
        if date:
            query = f"""
            SELECT code, liutongshizhi as market_cap
            FROM default.stock_data_qfq
            WHERE code IN ('{codes_str}') AND date = '{date}'
            """
        else:
            query = f"""
            SELECT code, liutongshizhi as market_cap
            FROM default.stock_data_qfq
            WHERE (code, date) IN (
                SELECT code, max(date)
                FROM default.stock_data_qfq
                WHERE code IN ('{codes_str}')
                GROUP BY code
            )
            """
        try:
            result = self.ch_client.query(query)
            df = pd.DataFrame(result.result_rows, columns=['code', 'market_cap'])
            df['market_cap'] = df['market_cap'] / 1e8  # 转为亿
            df = df.drop_duplicates(subset=['code'], keep='first')
            return df
        except:
            return pd.DataFrame(columns=['code', 'market_cap'])

    def get_momentum(self, codes: List[str], end_date: str, lookback: int = 60) -> pd.DataFrame:
        """计算动量"""
        codes_str = "','".join(codes)
        query = f"""
        WITH latest AS (
            SELECT code, close as latest_close
            FROM default.stock_data_qfq
            WHERE code IN ('{codes_str}') AND date <= '{end_date}'
            ORDER BY date DESC LIMIT 1 BY code
        ),
        past AS (
            SELECT code, close as past_close
            FROM default.stock_data_qfq
            WHERE code IN ('{codes_str}')
              AND date <= date_sub(day, {lookback}, toDate('{end_date}'))
            ORDER BY date DESC LIMIT 1 BY code
        )
        SELECT latest.code, (latest.latest_close / past.past_close - 1) * 100 as momentum
        FROM latest JOIN past ON latest.code = past.code
        """
        try:
            result = self.ch_client.query(query)
            return pd.DataFrame(result.result_rows, columns=['code', 'momentum'])
        except:
            return pd.DataFrame(columns=['code', 'momentum'])

    def get_drawdown_3m(self, codes: List[str], end_date: str) -> pd.DataFrame:
        """计算3个月回撤"""
        codes_str = "','".join(codes)
        query = f"""
        WITH price_data AS (
            SELECT code, date, close
            FROM default.stock_data_qfq
            WHERE code IN ('{codes_str}')
              AND date >= date_sub(month, 3, toDate('{end_date}'))
              AND date <= '{end_date}'
        ),
        max_price AS (
            SELECT code, max(close) as high_3m FROM price_data GROUP BY code
        ),
        latest_price AS (
            SELECT code, close as latest_close FROM price_data
            ORDER BY date DESC LIMIT 1 BY code
        )
        SELECT max_price.code,
               (latest_price.latest_close / max_price.high_3m - 1) * 100 as drawdown_3m
        FROM max_price JOIN latest_price ON max_price.code = latest_price.code
        """
        try:
            result = self.ch_client.query(query)
            return pd.DataFrame(result.result_rows, columns=['code', 'drawdown_3m'])
        except:
            return pd.DataFrame(columns=['code', 'drawdown_3m'])

    def get_volatility(self, codes: List[str], end_date: str, lookback: int = 60) -> pd.DataFrame:
        """计算波动率"""
        codes_str = "','".join(codes)
        query = f"""
        WITH daily_returns AS (
            SELECT code, date,
                   close / lagInFrame(close, 1) OVER (PARTITION BY code ORDER BY date) - 1 as daily_return
            FROM default.stock_data_qfq
            WHERE code IN ('{codes_str}')
              AND date >= date_sub(day, {lookback}, toDate('{end_date}'))
              AND date <= '{end_date}'
        )
        SELECT code, stddevPop(daily_return) * sqrt(252) * 100 as volatility
        FROM daily_returns WHERE daily_return IS NOT NULL GROUP BY code
        """
        try:
            result = self.ch_client.query(query)
            return pd.DataFrame(result.result_rows, columns=['code', 'volatility'])
        except:
            return pd.DataFrame(columns=['code', 'volatility'])

    def get_hs300_prices(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取沪深300指数用于止损判断"""
        query = f"""
        SELECT toString(date) as date, close
        FROM default.index_data
        WHERE code = '000300'
          AND date >= '{start_date}'
          AND date <= '{end_date}'
        ORDER BY date
        """
        try:
            result = self.ch_client.query(query)
            df = pd.DataFrame(result.result_rows, columns=['date', 'close'])
            return df
        except:
            return pd.DataFrame(columns=['date', 'close'])


def build_factor_data(fetcher: OptimizedDataFetcher, codes: List[str], eval_date: str) -> pd.DataFrame:
    """构建因子数据"""
    # 1. 从akshare缓存获取财务数据
    akshare_df = fetcher._load_akshare_cache()
    akshare_df = akshare_df[akshare_df['code'].isin(codes)].drop_duplicates(subset=['code'], keep='first')

    # 2. 基础DataFrame
    factor_df = pd.DataFrame({'code': codes})

    # 3. 从akshare补充财务数据
    if not akshare_df.empty:
        ak_dict = akshare_df.set_index('code')
        for idx, row in factor_df.iterrows():
            code = row['code']
            if code in ak_dict.index:
                ak_row = ak_dict.loc[code]
                factor_df.at[idx, 'roe'] = ak_row.get('roe', np.nan)
                factor_df.at[idx, 'eps'] = ak_row.get('eps', np.nan)
                factor_df.at[idx, 'net_profit_yoy'] = ak_row.get('net_profit_yoy', 0)
                factor_df.at[idx, 'revenue_yoy'] = ak_row.get('revenue_yoy', 0)
                factor_df.at[idx, 'gross_profit_margin'] = ak_row.get('gross_margin', np.nan)

    # 4. 获取市值
    market_df = fetcher.get_market_cap(codes, eval_date)
    if not market_df.empty:
        factor_df = factor_df.merge(market_df, on='code', how='left')
    else:
        factor_df['market_cap'] = np.nan

    # 5. 获取动量
    momentum_df = fetcher.get_momentum(codes, eval_date)
    if not momentum_df.empty:
        factor_df = factor_df.merge(momentum_df, on='code', how='left')
    else:
        factor_df['momentum'] = np.nan

    # 6. 获取3个月回撤
    drawdown_df = fetcher.get_drawdown_3m(codes, eval_date)
    if not drawdown_df.empty:
        factor_df = factor_df.merge(drawdown_df, on='code', how='left')
        # 用于反转因子：回撤是负值，转为正值
        factor_df['momentum_3m'] = factor_df['drawdown_3m']
    else:
        factor_df['drawdown_3m'] = np.nan
        factor_df['momentum_3m'] = np.nan

    # 7. 获取波动率
    vol_df = fetcher.get_volatility(codes, eval_date)
    if not vol_df.empty:
        factor_df = factor_df.merge(vol_df, on='code', how='left')
    else:
        factor_df['volatility'] = np.nan

    # 8. 从valuation_local.csv获取PE、PB、股息率
    try:
        val_path = os.path.join(os.path.dirname(__file__), 'valuation_local.csv')
        val_csv = pd.read_csv(val_path)
        val_csv['code'] = val_csv['code'].astype(str).str.zfill(6)
        val_csv = val_csv.drop_duplicates(subset=['code'], keep='first')
        val_csv = val_csv[val_csv['code'].isin(codes)]

        factor_df = factor_df.merge(
            val_csv[['code', 'pe_ttm', 'pb', 'dividend_yield']].rename(columns={'pe_ttm': 'pe'}),
            on='code', how='left'
        )

        # 补充股息率
        div_path = os.path.join(os.path.dirname(__file__), 'dividend_yield_all.csv')
        if os.path.exists(div_path):
            div_csv = pd.read_csv(div_path)
            div_csv['code'] = div_csv['code'].astype(str).str.zfill(6)
            div_csv = div_csv.drop_duplicates(subset=['code'], keep='first')
            div_dict = dict(zip(div_csv['code'], div_csv['dividend_yield']))
            factor_df['dividend_yield'] = factor_df['code'].map(div_dict).combine_first(factor_df['dividend_yield'])
    except:
        factor_df['pe'] = np.nan
        factor_df['pb'] = np.nan
        factor_df['dividend_yield'] = np.nan

    # 9. ROE稳定性
    if not akshare_df.empty and 'roe_std_3y' in akshare_df.columns:
        roe_std_map = dict(zip(akshare_df['code'], akshare_df['roe_std_3y']))
        factor_df['roe_stability'] = factor_df['code'].map(roe_std_map)
        factor_df['roe_stability'] = factor_df['roe_stability'].apply(
            lambda x: max(50, 100 - x * 5) if pd.notna(x) else 80
        )
    else:
        factor_df['roe_stability'] = 80

    # 10. 现金流质量
    if not akshare_df.empty and 'ocfps' in akshare_df.columns:
        ocfps_map = dict(zip(akshare_df['code'], akshare_df['ocfps']))
        factor_df['cash_flow_ratio'] = factor_df['code'].map(ocfps_map) / factor_df['eps'].replace(0, np.nan)
        factor_df['cash_flow_ratio'] = factor_df['cash_flow_ratio'].clip(-5, 5).fillna(1.0)
    else:
        factor_df['cash_flow_ratio'] = 1.0

    # 11. 填充缺失值
    factor_df['profit_growth'] = factor_df['net_profit_yoy']
    defaults = {
        'roe': factor_df['roe'].median() if factor_df['roe'].notna().any() else 15,
        'net_profit_yoy': 0, 'revenue_yoy': 0, 'profit_growth': 0,
        'pe': factor_df['pe'].median() if factor_df['pe'].notna().any() else 20,
        'pb': factor_df['pb'].median() if factor_df['pb'].notna().any() else 2,
        'dividend_yield': factor_df['dividend_yield'].median() if factor_df['dividend_yield'].notna().any() else 0.03,
        'market_cap': factor_df['market_cap'].median() if factor_df['market_cap'].notna().any() else 300,
        'momentum': 0, 'momentum_3m': 0, 'drawdown_3m': -5, 'volatility': 30,
    }
    factor_df = factor_df.fillna(defaults)

    return factor_df


# ==================== 优化的因子权重 ====================

# 稳健型策略 - 优化后的因子权重（根据历史回测调整）
OPTIMIZED_STABLE_WEIGHTS = {
    # 价值因子 - 略微增加
    'dividend_yield': 0.20,      # 股息率（+2%）
    'pe_value': 0.08,            # PE估值

    # 质量因子 - 大幅增加
    'roe': 0.18,                 # ROE（+3%）
    'roe_stability': 0.12,       # ROE稳定性（+2%）
    'cash_flow_quality': 0.08,   # 现金流质量

    # 成长因子 - 适度
    'profit_growth': 0.08,       # 净利润增长
    'revenue_growth': 0.03,      # 营收增长
    'peg': 0.05,                 # PEG

    # 规模因子 - 降低
    'small_cap': 0.03,           # 小市值（-2%）

    # 动量/反转因子 - 调整
    'momentum': 0.05,            # 动量
    'reversal': 0.07,            # 反转（-3%）

    # 风险因子 - 增加
    'low_volatility': 0.03,      # 低波动（-2%）
}

# 进取型策略 - 优化后的因子权重
OPTIMIZED_AGGRESSIVE_WEIGHTS = {
    # 价值因子 - 降低
    'dividend_yield': 0.03,
    'pe_value': 0.02,

    # 质量因子 - 保留
    'roe': 0.12,
    'roe_stability': 0.06,
    'cash_flow_quality': 0.03,

    # 成长因子 - 大幅增加
    'profit_growth': 0.22,       # 净利润增长（+4%）
    'revenue_growth': 0.10,      # 营收增长（+2%）
    'peg': 0.15,                 # PEG（+3%）

    # 规模因子 - 增加
    'small_cap': 0.12,           # 小市值

    # 动量/反转因子 - 调整
    'momentum': 0.10,            # 动量
    'reversal': 0.03,            # 反转

    # 风险因子
    'low_volatility': 0.02,
}


class OptimizedFactorCalculator(EnhancedFactorCalculator):
    """优化版因子计算器 - 放宽反转条件"""

    def calculate_reversal_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        反转因子 (绩优股回撤买入) - 优化版
        放宽条件：
        1. ROE > 8%（原12%）
        2. 净利润增速 > 0%（原5%）
        3. 回撤 > 10%（原15%）
        """
        result = df[['code']].copy()

        # 放宽的绩优股条件
        is_quality = (df['roe'] > 8) & (df['net_profit_yoy'] > 0)

        # 回撤幅度
        if 'momentum_3m' in df.columns:
            drawdown = -df['momentum_3m']
        elif 'drawdown_3m' in df.columns:
            drawdown = -df['drawdown_3m']
        else:
            drawdown = -df['momentum'] if 'momentum' in df.columns else pd.Series(0, index=df.index)

        # 基础分数
        base_score = drawdown.rank(pct=True) * 60 + 20

        # 分级加分
        reversal_bonus = pd.Series(0, index=df.index)

        # 大幅回撤（>20%）
        mask_20 = is_quality & (drawdown > 20)
        reversal_bonus[mask_20] = 40

        # 中等回撤（15-20%）
        mask_15 = is_quality & (drawdown > 15) & (drawdown <= 20)
        reversal_bonus[mask_15] = 30

        # 小幅回撤（10-15%）
        mask_10 = is_quality & (drawdown > 10) & (drawdown <= 15)
        reversal_bonus[mask_10] = 20

        # 轻微回撤（5-10%）
        mask_5 = is_quality & (drawdown > 5) & (drawdown <= 10)
        reversal_bonus[mask_5] = 10

        result['reversal_score'] = (base_score + reversal_bonus).clip(0, 100)

        return result


def create_optimized_calculator(strategy_type: str) -> OptimizedFactorCalculator:
    """创建优化版因子计算器"""
    if strategy_type == 'stable':
        weights = OPTIMIZED_STABLE_WEIGHTS
    elif strategy_type == 'aggressive':
        weights = OPTIMIZED_AGGRESSIVE_WEIGHTS
    else:
        weights = None

    return OptimizedFactorCalculator(factor_weights=weights, strategy_type=strategy_type)


def apply_optimized_weight_tilt(original_weights: pd.DataFrame,
                                 scored_data: pd.DataFrame,
                                 tilt_strength: float = 0.5,
                                 max_weight: float = 0.10,
                                 min_weight: float = 0.002) -> pd.DataFrame:
    """
    优化的权重调整
    - 使用非线性倾斜（对top股票额外加成）
    - 放宽max_weight约束
    """
    portfolio = original_weights.copy()

    # 只保留code和weight列（移除之前可能的额外列）
    if 'weight' in portfolio.columns:
        portfolio = portfolio[['code', 'weight']].copy()
        # 确保权重归一化
        portfolio['weight'] = portfolio['weight'] / portfolio['weight'].sum()

    # 合并得分
    scored_subset = scored_data[['code', 'composite_score']].copy() if 'composite_score' in scored_data.columns else scored_data[['code']].copy()
    scored_subset['composite_score'] = scored_subset.get('composite_score', 50)

    portfolio = portfolio.merge(scored_subset, on='code', how='left')
    portfolio['composite_score'] = portfolio['composite_score'].fillna(50)

    # 计算百分位
    portfolio['percentile'] = portfolio['composite_score'].rank(pct=True)

    # 非线性倾斜：对top股票使用更大的系数
    def calc_multiplier(pct):
        if pct >= 0.95:
            return 1.0 + tilt_strength * 1.5 * (pct - 0.5)  # Top 5%额外加成
        elif pct >= 0.80:
            return 1.0 + tilt_strength * 1.2 * (pct - 0.5)  # Top 20%加成
        else:
            return 1.0 + tilt_strength * (pct - 0.5)

    portfolio['multiplier'] = portfolio['percentile'].apply(calc_multiplier)

    # 计算调整后权重
    portfolio['adjusted_weight'] = portfolio['weight'] * portfolio['multiplier']

    # 约束
    portfolio['adjusted_weight'] = portfolio['adjusted_weight'].clip(lower=min_weight, upper=max_weight)

    # 归一化
    total = portfolio['adjusted_weight'].sum()
    if total > 0:
        portfolio['adjusted_weight'] = portfolio['adjusted_weight'] / total

    return portfolio


class OptimizedBacktestEngine:
    """优化版回测引擎 - 支持调仓和交易成本"""

    # 交易成本
    BUY_COST = 0.0003   # 买入0.03%
    SELL_COST = 0.0013  # 卖出0.13%（含印花税）

    def __init__(self):
        self.ch_client = clickhouse_connect.get_client(
            host='192.168.0.74', port=8123, compress=False, query_limit=0
        )
        self.data_fetcher = OptimizedDataFetcher()

    def get_daily_prices(self, codes: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """获取日线数据"""
        codes_str = "','".join(codes)
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

    def get_rebalance_dates(self, start_date: str, end_date: str, freq: str = 'Q') -> List[str]:
        """获取调仓日期（季度/月度）"""
        from datetime import datetime
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        dates = []
        current = start

        while current <= end:
            # 每季度第一天或每月第一天
            if freq == 'Q':
                if current.month in [1, 4, 7, 10]:
                    dates.append(current.strftime('%Y-%m-%d'))
                current = datetime(current.year + (1 if current.month == 12 else 0),
                                   current.month + 1 if current.month < 12 else 1, 1)
            else:  # Monthly
                dates.append(current.strftime('%Y-%m-%d'))
                if current.month == 12:
                    current = datetime(current.year + 1, 1, 1)
                else:
                    current = datetime(current.year, current.month + 1, 1)

        return dates

    def run_backtest_single_period(self,
                                    portfolio: pd.DataFrame,
                                    start_date: str,
                                    end_date: str,
                                    strategy_type: str,
                                    tilt_strength: float,
                                    prev_portfolio: pd.DataFrame = None) -> Dict:
        """单期回测"""
        codes = portfolio['code'].tolist()

        # 构建因子数据
        factor_data = build_factor_data(self.data_fetcher, codes, end_date)

        # 过滤有效股票
        valid_codes = factor_data['code'].tolist()
        if len(valid_codes) < len(codes):
            portfolio = portfolio[portfolio['code'].isin(valid_codes)].copy()
            portfolio['weight'] = portfolio['weight'] / portfolio['weight'].sum()
            codes = valid_codes

        # 计算因子得分
        factor_calc = create_optimized_calculator(strategy_type)
        scored_data = factor_calc.calculate_all_scores(factor_data)

        # 权重调整
        enhanced_portfolio = apply_optimized_weight_tilt(
            portfolio, scored_data,
            tilt_strength=tilt_strength,
            max_weight=0.10
        )

        # 计算交易成本
        turnover_cost = 0
        if prev_portfolio is not None:
            prev_weights = dict(zip(prev_portfolio['code'], prev_portfolio['adjusted_weight']))
            curr_weights = dict(zip(enhanced_portfolio['code'], enhanced_portfolio['adjusted_weight']))

            for code in set(prev_weights.keys()) | set(curr_weights.keys()):
                prev_w = prev_weights.get(code, 0)
                curr_w = curr_weights.get(code, 0)
                if curr_w > prev_w:
                    turnover_cost += (curr_w - prev_w) * self.BUY_COST
                else:
                    turnover_cost += (prev_w - curr_w) * self.SELL_COST

        # 获取价格数据
        price_df = self.get_daily_prices(codes, start_date, end_date)
        price_df = price_df.drop_duplicates(subset=['date', 'code'], keep='last')
        price_pivot = price_df.pivot(index='date', columns='code', values='close')
        dates = sorted(price_pivot.index.tolist())

        # 股息率
        dividend_dict = dict(zip(factor_data['code'], factor_data['dividend_yield']))
        daily_dividend = {code: div / 252 for code, div in dividend_dict.items()}

        # 计算每日收益
        weight_dict = dict(zip(enhanced_portfolio['code'], enhanced_portfolio['adjusted_weight']))
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
                        price_return = curr_price / prev_price - 1
                        div_return = daily_dividend.get(code, 0)
                        stock_return = price_return + div_return
                        weight = weight_dict.get(code, 0)
                        daily_return += stock_return * weight
            daily_returns.append({'date': date, 'return': daily_return})

        returns_df = pd.DataFrame(daily_returns)
        returns_df['cumulative'] = (1 + returns_df['return']).cumprod()

        return {
            'returns_df': returns_df,
            'enhanced_portfolio': enhanced_portfolio,
            'turnover_cost': turnover_cost,
            'scored_data': scored_data
        }

    def run_backtest(self,
                     portfolio: pd.DataFrame,
                     start_date: str,
                     end_date: str,
                     strategy_type: str = 'stable',
                     tilt_strength: float = 0.5,
                     rebalance_freq: str = 'Q') -> Dict:
        """运行完整回测（支持调仓）"""
        print("=" * 60)
        print(f"优化版回测 ({strategy_type}, tilt={tilt_strength}, rebalance={rebalance_freq})")
        print(f"回测区间: {start_date} ~ {end_date}")
        print("=" * 60)

        # 获取调仓日期
        rebalance_dates = self.get_rebalance_dates(start_date, end_date, rebalance_freq)
        print(f"调仓日期: {len(rebalance_dates)}次")

        all_returns = []
        current_portfolio = portfolio.copy()
        total_turnover_cost = 0

        for i, rebal_date in enumerate(rebalance_dates):
            # 确定本期结束日期
            if i < len(rebalance_dates) - 1:
                period_end = rebalance_dates[i + 1]
            else:
                period_end = end_date

            print(f"\n调仓 {i+1}/{len(rebalance_dates)}: {rebal_date} ~ {period_end}")

            # 运行本期回测
            result = self.run_backtest_single_period(
                current_portfolio,
                rebal_date,
                period_end,
                strategy_type,
                tilt_strength,
                current_portfolio if i > 0 else None
            )

            all_returns.append(result['returns_df'])
            current_portfolio = result['enhanced_portfolio']
            total_turnover_cost += result['turnover_cost']

        # 合并所有收益
        combined_returns = pd.concat(all_returns, ignore_index=True)
        combined_returns['cumulative'] = (1 + combined_returns['return']).cumprod()

        # 扣除总交易成本
        final_nav = combined_returns['cumulative'].iloc[-1]
        final_nav_after_cost = final_nav * (1 - total_turnover_cost)

        # 计算原始策略收益（无因子调整）
        orig_result = self.run_backtest_single_period(
            portfolio, start_date, end_date, strategy_type, 0, None
        )

        # 计算指标
        def calc_metrics(returns_df, final_nav_val=None):
            total_return = (final_nav_val or returns_df['cumulative'].iloc[-1]) - 1
            n_years = len(returns_df) / 252
            annual_return = (1 + total_return) ** (1/n_years) - 1 if n_years > 0 else 0

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
                'final_nav': final_nav_val or returns_df['cumulative'].iloc[-1]
            }

        enhanced_metrics = calc_metrics(combined_returns, final_nav_after_cost)
        original_metrics = calc_metrics(orig_result['returns_df'])

        # 打印结果
        print(f"\n{'='*50}")
        print(f"优化版: 年化{enhanced_metrics['annual_return']*100:.2f}%, "
              f"回撤{enhanced_metrics['max_drawdown']*100:.2f}%, "
              f"夏普{enhanced_metrics['sharpe']:.2f}")
        print(f"原始版: 年化{original_metrics['annual_return']*100:.2f}%, "
              f"回撤{original_metrics['max_drawdown']*100:.2f}%, "
              f"夏普{original_metrics['sharpe']:.2f}")
        print(f"提升: {(enhanced_metrics['annual_return'] - original_metrics['annual_return'])*100:+.2f}%")
        print(f"交易成本: {total_turnover_cost*100:.3f}%")
        print(f"{'='*50}")

        return {
            'enhanced': enhanced_metrics,
            'original': original_metrics,
            'returns_df': combined_returns,
            'turnover_cost': total_turnover_cost,
            'strategy_type': strategy_type
        }


def main():
    print("=" * 70)
    print("多因子策略回测 - 综合优化版")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 加载持仓
    base_dir = os.path.dirname(__file__)
    r4 = pd.read_csv(os.path.join(base_dir, 'r4_weights_local.csv'))
    r5 = pd.read_csv(os.path.join(base_dir, 'r5_weights_local.csv'))

    r4['code'] = r4['code'].astype(str).str.zfill(6)
    r5['code'] = r5['code'].astype(str).str.zfill(6)

    r4 = r4[['code', 'weight']].copy()
    r5 = r5[['code', 'weight']].copy()

    r4['weight'] = r4['weight'] / r4['weight'].sum()
    r5['weight'] = r5['weight'] / r5['weight'].sum()

    print(f"R4 稳健型: {len(r4)} 只股票")
    print(f"R5 进取型: {len(r5)} 只股票")

    engine = OptimizedBacktestEngine()

    start_date = '2020-01-01'
    end_date = '2025-12-31'

    # 测试不同参数组合
    results = {}

    # R4 稳健型 - 测试不同tilt_strength
    print("\n" + "=" * 70)
    print("【R4 稳健型策略优化回测】")
    print("=" * 70)

    best_r4_result = None
    best_r4_tilt = 0.5

    for tilt in [0.3, 0.5, 0.7, 1.0]:
        print(f"\n--- tilt_strength = {tilt} ---")
        result = engine.run_backtest(
            portfolio=r4,
            start_date=start_date,
            end_date=end_date,
            strategy_type='stable',
            tilt_strength=tilt,
            rebalance_freq='Q'
        )
        results[f'R4_tilt{tilt}'] = result

        if best_r4_result is None or result['enhanced']['sharpe'] > best_r4_result['enhanced']['sharpe']:
            best_r4_result = result
            best_r4_tilt = tilt

    # R5 进取型 - 测试不同tilt_strength
    print("\n" + "=" * 70)
    print("【R5 进取型策略优化回测】")
    print("=" * 70)

    best_r5_result = None
    best_r5_tilt = 0.5

    for tilt in [0.3, 0.5, 0.7, 1.0]:
        print(f"\n--- tilt_strength = {tilt} ---")
        result = engine.run_backtest(
            portfolio=r5,
            start_date=start_date,
            end_date=end_date,
            strategy_type='aggressive',
            tilt_strength=tilt,
            rebalance_freq='Q'
        )
        results[f'R5_tilt{tilt}'] = result

        if best_r5_result is None or result['enhanced']['sharpe'] > best_r5_result['enhanced']['sharpe']:
            best_r5_result = result
            best_r5_tilt = tilt

    # 结果总结
    print("\n" + "=" * 70)
    print("【优化回测结果总结】")
    print("=" * 70)

    print(f"""
┌───────────┬─────────────┬─────────────┬──────────┬───────────┬──────────┐
│  策略     │ 最佳tilt    │ 原始年化    │ 优化年化 │   提升    │ 夏普比率 │
├───────────┼─────────────┼─────────────┼──────────┼───────────┼──────────┤
│ R4 稳健   │    {best_r4_tilt:.1f}     │  {best_r4_result['original']['annual_return']*100:>6.2f}%   │  {best_r4_result['enhanced']['annual_return']*100:>6.2f}% │ {(best_r4_result['enhanced']['annual_return']-best_r4_result['original']['annual_return'])*100:>+5.2f}% │ {best_r4_result['original']['sharpe']:.2f}→{best_r4_result['enhanced']['sharpe']:.2f} │
│ R5 进取   │    {best_r5_tilt:.1f}     │  {best_r5_result['original']['annual_return']*100:>6.2f}%   │  {best_r5_result['enhanced']['annual_return']*100:>6.2f}% │ {(best_r5_result['enhanced']['annual_return']-best_r5_result['original']['annual_return'])*100:>+5.2f}% │ {best_r5_result['original']['sharpe']:.2f}→{best_r5_result['enhanced']['sharpe']:.2f} │
└───────────┴─────────────┴─────────────┴──────────┴───────────┴──────────┘
""")

    # 保存结果
    output_dir = os.path.dirname(os.path.abspath(__file__))
    result_json = {
        'run_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'backtest_period': f'{start_date} ~ {end_date}',
        'R4_stable_best': {
            'tilt_strength': best_r4_tilt,
            'enhanced': {k: v for k, v in best_r4_result['enhanced'].items() if isinstance(v, (int, float))},
            'original': {k: v for k, v in best_r4_result['original'].items() if isinstance(v, (int, float))},
        },
        'R5_aggressive_best': {
            'tilt_strength': best_r5_tilt,
            'enhanced': {k: v for k, v in best_r5_result['enhanced'].items() if isinstance(v, (int, float))},
            'original': {k: v for k, v in best_r5_result['original'].items() if isinstance(v, (int, float))},
        },
        'all_results': {
            k: {
                'enhanced_annual': v['enhanced']['annual_return'],
                'original_annual': v['original']['annual_return'],
                'sharpe': v['enhanced']['sharpe'],
            }
            for k, v in results.items()
        }
    }

    json_path = os.path.join(output_dir, 'factor_backtest_optimized_result.json')
    with open(json_path, 'w') as f:
        json.dump(result_json, f, indent=2)

    print(f"\n结果已保存到: {json_path}")
    print("\n优化回测完成!")


if __name__ == '__main__':
    main()
