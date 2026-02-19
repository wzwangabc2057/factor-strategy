#!/usr/bin/env python
"""
多因子权重策略回测 v2.1
核心策略: 保住v1.7赢家因子权重 + 新因子小比例增强

v2.0教训: 新因子稀释了原始赢家(dividend_yield/roe/reversal)权重,
          导致收益下降1-2%. 正确做法是赢家因子不动, 新因子只占10-15%.

v2.1改进:
1. R4/R5核心因子权重恢复到v1.7水平
2. 新因子总权重控制在12% (每个2-3%)
3. 保留季度调仓+交易成本+10档权重
4. 同时跑 年度/季度 两种调仓频率做对比
"""

import pandas as pd
import numpy as np
import clickhouse_connect
from datetime import datetime, timedelta
import json
import warnings
import os

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

# 交易成本
TRANSACTION_COSTS = {
    'buy_commission': 0.00026,    # 买入佣金 0.026%
    'sell_commission': 0.00126,   # 卖出佣金+印花税 0.126%
}

# 调仓频率配置 (同时对比)
REBALANCE_QUARTERLY = [1, 4, 7, 10]   # 季度调仓
REBALANCE_ANNUAL = [1]                  # 年度调仓 (同v1.7)

# 10档权重调整系数 (v2精细版)
WEIGHT_TIERS_10 = [
    (0.95, 2.50),   # Top 5%:    大幅超配
    (0.90, 2.00),   # 5-10%:     超配
    (0.80, 1.60),   # 10-20%:    中等超配
    (0.70, 1.30),   # 20-30%:    轻度超配
    (0.60, 1.10),   # 30-40%:    微超配
    (0.40, 1.00),   # 40-60%:    保持原权重
    (0.30, 0.90),   # 60-70%:    微低配
    (0.20, 0.75),   # 70-80%:    低配
    (0.10, 0.50),   # 80-90%:    大幅低配
    (0.00, 0.30),   # Bottom 10%: 极低配
]

# ============================================================================
# R4 稳健型 v2.1: 核心因子恢复v1.7 + 新因子小比例增强 (12%)
# ============================================================================
WEIGHTS_STABLE_V2 = {
    # === v1.7 原始赢家因子 (88%) ===
    'dividend_yield': 0.16,      # 股息率 (v1.7=18%, 微降2%给新因子)
    'roe': 0.14,                 # ROE (v1.7=15%)
    'roe_stability': 0.09,       # ROE稳定性 (v1.7=10%)
    'reversal': 0.10,            # 反转 (v1.7=10%, 保持)
    'pe_value': 0.09,            # PE估值 (v1.7=10%)
    'profit_growth': 0.09,       # 净利润增速 (v1.7=10%)
    'small_cap': 0.06,           # 小市值 (v1.7=8%, 微降)
    'momentum': 0.04,            # 动量 (v1.7=5%)
    'low_volatility': 0.06,      # 低波动 (v1.7=7%)
    'financial_health': 0.05,    # 财务健康 (v1.7=7%, 微降)

    # === 新增因子 - 小比例增强 (12%) ===
    'pb_value': 0.03,            # PB估值 (NEW)
    'gross_margin': 0.03,        # 毛利率 (NEW)
    'turnover': 0.02,            # 换手率 (NEW)
    'operating_quality': 0.02,   # 营业利润质量 (NEW)
    'net_margin': 0.01,          # 净利率 (NEW)
    'asset_turnover': 0.01,      # 总资产周转率 (NEW)
}

# ============================================================================
# R5 进取型 v2.1: 核心因子恢复v1.7 + 新因子小比例增强 (12%)
# ============================================================================
WEIGHTS_AGGRESSIVE_V2 = {
    # === v1.7 原始赢家因子 (88%) ===
    'profit_growth': 0.16,       # 净利润增速 (v1.7=18%)
    'small_cap': 0.13,           # 小市值 (v1.7=15%)
    'momentum': 0.11,            # 动量 (v1.7=12%)
    'roe': 0.11,                 # ROE (v1.7=12%)
    'revenue_growth': 0.09,      # 营收增速 (v1.7=10%)
    'reversal': 0.07,            # 反转 (v1.7=8%)
    'dividend_yield': 0.07,      # 股息率 (v1.7=8%)
    'pe_value': 0.04,            # PE估值 (v1.7=5%)
    'low_volatility': 0.04,      # 低波动 (v1.7=5%)
    'financial_health': 0.06,    # 财务健康 (v1.7=7%)

    # === 新增因子 - 小比例增强 (12%) ===
    'turnover': 0.03,            # 换手率 (NEW) - A股强因子
    'gross_margin': 0.03,        # 毛利率 (NEW)
    'pb_value': 0.02,            # PB估值 (NEW)
    'operating_quality': 0.02,   # 营业利润质量 (NEW)
    'asset_turnover': 0.01,      # 总资产周转率 (NEW)
    'net_margin': 0.01,          # 净利率 (NEW)
}

# 验证权重和
for name, w in [('R4_v2', WEIGHTS_STABLE_V2), ('R5_v2', WEIGHTS_AGGRESSIVE_V2)]:
    total = sum(w.values())
    assert abs(total - 1.0) < 0.01, f"{name} 权重和={total:.2f}, 不等于1"


# ============================================================================
# 数据获取 (扩展版)
# ============================================================================

class DataFetcherV2:
    """扩展数据获取器, 支持18因子所需的全部数据"""

    def __init__(self):
        self.client = clickhouse_connect.get_client(host=CLICKHOUSE_HOST, port=CLICKHOUSE_PORT)
        self._valuation_df = None
        self._dividend_df = None

    def get_stock_pool(self, strategy_type='stable'):
        """从 CSV 加载 R4/R5 持仓权重"""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if strategy_type == 'stable':
            csv_path = os.path.join(base_dir, 'r4_weights_local.csv')
        else:
            csv_path = os.path.join(base_dir, 'r5_weights_local.csv')

        df = pd.read_csv(csv_path)
        if 'original_weight' in df.columns:
            portfolio = df[['code', 'original_weight']].copy()
            portfolio.columns = ['code', 'weight']
        else:
            portfolio = df[['code', 'weight']].copy()
        portfolio['code'] = portfolio['code'].astype(str).str.zfill(6)
        portfolio['weight'] = portfolio['weight'] / portfolio['weight'].sum()
        return portfolio

    def get_latest_report_date(self, as_of_date):
        """根据调仓日期确定可用的最新报告期"""
        dt = datetime.strptime(as_of_date, '%Y-%m-%d')
        year, month = dt.year, dt.month
        # 季度调仓 — 更精细的报告期选择
        if month <= 4:
            return f'{year - 2}-12-31'     # 1-4月: 前年年报
        elif month <= 8:
            return f'{year - 1}-12-31'     # 5-8月: 上年年报
        elif month <= 10:
            return f'{year}-06-30'         # 9-10月: 当年中报
        else:
            return f'{year}-09-30'         # 11-12月: 当年三季报

    def load_valuation_csv(self):
        """加载本地估值数据"""
        if self._valuation_df is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(base_dir, 'valuation_local.csv')
            df = pd.read_csv(csv_path)
            df['code'] = df['code'].astype(str).str.zfill(6)
            df['market_cap'] = df['total_mv'] / 10000  # 万→亿
            self._valuation_df = df
        return self._valuation_df

    def load_dividend_csv(self):
        """加载股息率数据"""
        if self._dividend_df is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(base_dir, 'dividend_yield_all.csv')
            df = pd.read_csv(csv_path)
            df['code'] = df['code'].astype(str).str.zfill(6)
            self._dividend_df = df
        return self._dividend_df

    def get_financial_data(self, codes, as_of_date):
        """获取财务数据 (基础 + 新增因子字段)"""
        val_df = self.load_valuation_csv()
        val_df = val_df[val_df['code'].isin(codes)].copy()

        # 基础字段 (含PB — 新增因子)
        fin_df = val_df[['code', 'roe', 'eps', 'bvps', 'pe_ttm', 'pb',
                         'market_cap', 'dividend_yield']].copy()
        fin_df = fin_df.rename(columns={'pe_ttm': 'pe'})

        codes_str = "','".join(codes)
        report_date = self.get_latest_report_date(as_of_date)
        yoy_date = f'{int(report_date[:4]) - 1}{report_date[4:]}'

        # ------ 从 ClickHouse 获取扩展财务数据 ------
        try:
            query = f"""
                SELECT code,
                       asset_liability_ratio,
                       gross_profit_margin,
                       net_profit_margin,
                       operating_profit,
                       net_profit,
                       operating_revenue,
                       total_assets
                FROM default.stock_financial
                WHERE code IN ('{codes_str}') AND report_date <= '{report_date}'
                ORDER BY report_date DESC LIMIT 1 BY code
            """
            result = self.client.query(query)
            ext_df = pd.DataFrame(result.result_rows,
                                  columns=['code', 'asset_liability_ratio',
                                           'gross_profit_margin', 'net_profit_margin',
                                           'operating_profit', 'net_profit_ch',
                                           'operating_revenue', 'total_assets'])
            fin_df = fin_df.merge(ext_df, on='code', how='left')
        except Exception as e:
            print(f"  [WARN] 获取扩展财务数据失败: {e}")

        # 填充默认值
        defaults = {
            'asset_liability_ratio': 50.0,
            'gross_profit_margin': 30.0,
            'net_profit_margin': 10.0,
            'operating_profit': 0,
            'net_profit_ch': 0,
            'operating_revenue': 0,
            'total_assets': 1,
        }
        for col, val in defaults.items():
            if col not in fin_df.columns:
                fin_df[col] = val
            fin_df[col] = fin_df[col].fillna(val)

        # ------ 计算新增因子: 营业利润质量 ------
        # operating_quality = 营业利润 / 净利润 (越接近1越好，排除非经常性损益)
        fin_df['operating_quality'] = np.where(
            fin_df['net_profit_ch'].abs() > 1e-6,
            fin_df['operating_profit'] / fin_df['net_profit_ch'].abs(),
            1.0
        )
        fin_df['operating_quality'] = fin_df['operating_quality'].clip(-2, 3)

        # ------ 计算新增因子: 总资产周转率 ------
        fin_df['asset_turnover'] = np.where(
            fin_df['total_assets'] > 1e-6,
            fin_df['operating_revenue'] / fin_df['total_assets'],
            0.5
        )
        fin_df['asset_turnover'] = fin_df['asset_turnover'].clip(0, 5)

        # ------ 增长率 (利润 + 营收) ------
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
        except Exception as e:
            print(f"  [WARN] 获取增长率数据失败: {e}")

        for col in ['profit_growth', 'revenue_growth']:
            if col not in fin_df.columns:
                fin_df[col] = 0.0
            fin_df[col] = fin_df[col].fillna(0.0)

        # ------ ROE稳定性 (真实3年历史) ------
        try:
            roe_query = f"""
                SELECT code, roe, report_date
                FROM default.stock_financial
                WHERE code IN ('{codes_str}')
                  AND report_date >= date_sub(year, 3, toDate('{report_date}'))
                  AND report_date <= '{report_date}'
                  AND toMonth(report_date) = 12
                ORDER BY code, report_date
            """
            roe_result = self.client.query(roe_query)
            if roe_result.result_rows:
                roe_hist = pd.DataFrame(roe_result.result_rows,
                                        columns=['code', 'roe_hist', 'report_date_hist'])
                roe_stability = roe_hist.groupby('code')['roe_hist'].agg(['mean', 'std']).reset_index()
                roe_stability['roe_cv'] = roe_stability['std'] / (roe_stability['mean'].abs() + 1e-6)
                roe_stability['roe_stability'] = 1 / (1 + roe_stability['roe_cv'])
                fin_df = fin_df.merge(roe_stability[['code', 'roe_stability']], on='code', how='left')
        except Exception as e:
            print(f"  [WARN] 获取ROE稳定性数据失败: {e}")

        if 'roe_stability' not in fin_df.columns:
            fin_df['roe_stability'] = 0.5
        fin_df['roe_stability'] = fin_df['roe_stability'].fillna(0.5)

        return fin_df

    def get_momentum(self, codes, as_of_date, days=60):
        """获取动量因子"""
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
        """获取回撤数据"""
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
        """获取波动率"""
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

    # ==================== 新增因子数据获取 ====================

    def get_turnover(self, codes, as_of_date, days=20):
        """
        获取换手率因子 (NEW)
        返回: 近N日平均换手率 + 换手率标准差(稳定性)
        """
        codes_str = "','".join(codes)
        query = f"""
            SELECT code,
                   avg(huanshoulv) as avg_turnover,
                   stddevPop(huanshoulv) as std_turnover
            FROM default.stock_data_qfq
            WHERE code IN ('{codes_str}')
              AND date >= date_sub(day, {days}, toDate('{as_of_date}'))
              AND date <= '{as_of_date}'
            GROUP BY code
        """
        result = self.client.query(query)
        df = pd.DataFrame(result.result_rows, columns=['code', 'avg_turnover', 'std_turnover'])
        # 换手率稳定性 = 1 / (1 + CV)
        df['turnover_stability'] = 1 / (1 + df['std_turnover'] / (df['avg_turnover'].abs() + 1e-6))
        return df

    def get_prices(self, codes, start_date, end_date):
        """获取价格数据"""
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
# 因子计算 (18因子)
# ============================================================================

def percentile_score(series, ascending=True):
    """百分位打分 (0-100)"""
    if ascending:
        return series.rank(pct=True) * 100
    return (1 - series.rank(pct=True)) * 100


def calculate_factor_scores_v2(df, weights):
    """
    18因子综合打分

    原始12因子:
    - roe, pe_value, dividend_yield, financial_health
    - profit_growth, revenue_growth, small_cap
    - momentum, reversal, low_volatility
    - roe_stability, peg

    新增6因子:
    - pb_value, gross_margin, net_margin
    - operating_quality, asset_turnover, turnover
    """
    result = df.copy()

    # === 原有因子 ===
    if 'roe' in df.columns:
        roe = df['roe'].clip(lower=0, upper=50)
        result['roe_score'] = percentile_score(roe)

    if 'pe' in df.columns:
        valid_pe = df['pe'].where((df['pe'] > 0) & (df['pe'] < 200), np.nan)
        result['pe_value_score'] = percentile_score(valid_pe, ascending=False)

    if 'dividend_yield' in df.columns:
        result['dividend_yield_score'] = percentile_score(df['dividend_yield'])
        # 超高股息加分 (>5%)
        high_div = df['dividend_yield'] > 5
        result.loc[high_div, 'dividend_yield_score'] = result.loc[high_div, 'dividend_yield_score'].clip(lower=85)

    if 'asset_liability_ratio' in df.columns:
        result['financial_health_score'] = percentile_score(df['asset_liability_ratio'], ascending=False)

    if 'profit_growth' in df.columns:
        result['profit_growth_score'] = percentile_score(df['profit_growth'].clip(-50, 200))

    if 'revenue_growth' in df.columns:
        result['revenue_growth_score'] = percentile_score(df['revenue_growth'].clip(-50, 200))

    if 'market_cap' in df.columns:
        result['small_cap_score'] = percentile_score(df['market_cap'], ascending=False)

    if 'momentum' in df.columns:
        result['momentum_score'] = percentile_score(df['momentum'])

    if 'drawdown' in df.columns and 'roe' in df.columns:
        is_quality = df['roe'] > 15
        dd_score = percentile_score(df['drawdown'])
        result['reversal_score'] = np.where(is_quality, dd_score * 1.3, dd_score * 0.7)
        result['reversal_score'] = np.clip(result['reversal_score'], 0, 100)

    if 'volatility' in df.columns:
        result['low_volatility_score'] = percentile_score(df['volatility'], ascending=False)

    if 'roe_stability' in df.columns:
        result['roe_stability_score'] = percentile_score(df['roe_stability'])
    else:
        result['roe_stability_score'] = 50

    # PEG因子
    if 'pe' in df.columns and 'profit_growth' in df.columns:
        valid_mask = (df['pe'] > 0) & (df['pe'] < 100) & (df['profit_growth'] > 5)
        peg = pd.Series(np.nan, index=df.index)
        peg[valid_mask] = df.loc[valid_mask, 'pe'] / df.loc[valid_mask, 'profit_growth']
        result['peg_score'] = percentile_score(peg.clip(0, 5), ascending=False)
        result['peg_score'] = result['peg_score'].fillna(40)
        # 高成长低PE加分
        bonus = (df['profit_growth'] > 30) & (df['pe'] > 0) & (df['pe'] < 30)
        result.loc[bonus, 'peg_score'] = result.loc[bonus, 'peg_score'].clip(lower=80)

    # === 新增因子 ===

    # 1. PB估值因子 - 低PB高分
    if 'pb' in df.columns:
        valid_pb = df['pb'].where((df['pb'] > 0) & (df['pb'] < 20), np.nan)
        result['pb_value_score'] = percentile_score(valid_pb, ascending=False)
        result['pb_value_score'] = result['pb_value_score'].fillna(30)

    # 2. 毛利率因子 - 高毛利高分
    if 'gross_profit_margin' in df.columns:
        gpm = df['gross_profit_margin'].clip(0, 80)
        result['gross_margin_score'] = percentile_score(gpm)

    # 3. 净利率因子 - 高净利率高分
    if 'net_profit_margin' in df.columns:
        npm = df['net_profit_margin'].clip(-20, 50)
        result['net_margin_score'] = percentile_score(npm)

    # 4. 营业利润质量因子 - 越接近1越好
    if 'operating_quality' in df.columns:
        # 最优: 0.8~1.2, 偏离越大越差
        oq = df['operating_quality']
        # 转换为得分: 接近1给高分
        oq_score = 100 - (oq - 1.0).abs() * 50
        result['operating_quality_score'] = oq_score.clip(0, 100)

    # 5. 总资产周转率因子 - 高周转高分
    if 'asset_turnover' in df.columns:
        result['asset_turnover_score'] = percentile_score(df['asset_turnover'].clip(0, 3))

    # 6. 换手率因子 - 低换手率高分 (低换手率溢价)
    if 'avg_turnover' in df.columns:
        result['turnover_score'] = percentile_score(df['avg_turnover'], ascending=False)

    # 换手率稳定性 - 稳定高分
    if 'turnover_stability' in df.columns:
        result['turnover_stability_score'] = percentile_score(df['turnover_stability'])

    # === 综合得分 ===
    factor_to_score = {
        'dividend_yield': 'dividend_yield_score',
        'pe_value': 'pe_value_score',
        'pb_value': 'pb_value_score',
        'peg': 'peg_score',
        'roe': 'roe_score',
        'roe_stability': 'roe_stability_score',
        'gross_margin': 'gross_margin_score',
        'net_margin': 'net_margin_score',
        'operating_quality': 'operating_quality_score',
        'profit_growth': 'profit_growth_score',
        'revenue_growth': 'revenue_growth_score',
        'asset_turnover': 'asset_turnover_score',
        'small_cap': 'small_cap_score',
        'momentum': 'momentum_score',
        'reversal': 'reversal_score',
        'turnover': 'turnover_score',
        'turnover_stability': 'turnover_stability_score',
        'low_volatility': 'low_volatility_score',
        'financial_health': 'financial_health_score',
    }

    composite = pd.Series(0.0, index=df.index)
    active_factors = 0
    for factor, weight in weights.items():
        score_col = factor_to_score.get(factor)
        if score_col and score_col in result.columns:
            composite += result[score_col].fillna(50) * weight
            active_factors += 1

    result['composite_score'] = composite

    return result, active_factors


# ============================================================================
# 权重调整 (10档精细版)
# ============================================================================

def get_multiplier_10tier(percentile):
    """10档精细权重系数"""
    for threshold, mult in WEIGHT_TIERS_10:
        if percentile >= threshold:
            return mult
    return 0.3


def adjust_weights_v2(portfolio, scored_df):
    """根据综合得分调整权重 (10档)"""
    merged = portfolio.merge(scored_df[['code', 'composite_score']], on='code', how='left')
    merged['composite_score'] = merged['composite_score'].fillna(50)
    merged['percentile'] = merged['composite_score'].rank(pct=True)

    merged['multiplier'] = merged['percentile'].apply(get_multiplier_10tier)
    merged['original_weight'] = merged['weight']
    merged['adjusted_weight'] = merged['weight'] * merged['multiplier']
    merged['adjusted_weight'] /= merged['adjusted_weight'].sum()

    return merged


# ============================================================================
# 回测引擎 v2
# ============================================================================

def run_backtest_v2(strategy_type='stable', start_date='2020-01-01', end_date='2025-12-31',
                    rebalance_months=None):
    """
    v2.1 回测: 核心因子保持 + 新因子增强 + 可选调仓频率
    """
    if rebalance_months is None:
        rebalance_months = REBALANCE_QUARTERLY
    weights = WEIGHTS_STABLE_V2 if strategy_type == 'stable' else WEIGHTS_AGGRESSIVE_V2
    freq_label = '季度' if len(rebalance_months) > 1 else '年度'
    label = f"R4 稳健型" if strategy_type == 'stable' else f"R5 进取型"
    label += f" ({freq_label}调仓)"

    print("=" * 70)
    print(f"多因子权重策略回测 v2.1 [{label}]")
    print(f"回测区间: {start_date} ~ {end_date}")
    print(f"因子数量: {len(weights)} | 调仓频率: {freq_label} | 交易成本: ON")
    print("=" * 70)

    fetcher = DataFetcherV2()

    # 获取股票池
    portfolio = fetcher.get_stock_pool(strategy_type)
    codes = portfolio['code'].tolist()
    print(f"\n股票池: {len(codes)} 只")

    # 生成调仓日期
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    rebalance_dates = []
    for y in range(start_dt.year, end_dt.year + 1):
        for m in rebalance_months:
            rd = f'{y}-{m:02d}-01'
            if start_date <= rd <= end_date:
                rebalance_dates.append(rd)
    print(f"调仓日期: {len(rebalance_dates)} 次 ({freq_label})")

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
    prev_weights = {}  # 用于计算交易成本
    current_rebalance_idx = 0

    # 大盘止损
    market_position = 1.0
    stop_loss_count = 0

    enhanced_returns = []
    original_returns = []
    enhanced_returns_no_cost = []
    total_tx_cost = 0.0

    for i in range(1, len(dates)):
        curr_date = dates[i]
        prev_date = dates[i - 1]

        rebalance_cost = 0.0

        # 调仓
        while (current_rebalance_idx < len(rebalance_dates) and
               curr_date >= rebalance_dates[current_rebalance_idx]):
            rb_date = rebalance_dates[current_rebalance_idx]
            print(f"  [{rb_date}] 季度调仓, 重新计算因子...")

            # 构建因子数据
            factor_df = fetcher.get_financial_data(codes, rb_date)
            mom_df = fetcher.get_momentum(codes, rb_date)
            dd_df = fetcher.get_drawdown(codes, rb_date)
            vol_df = fetcher.get_volatility(codes, rb_date)
            turnover_df = fetcher.get_turnover(codes, rb_date)

            factor_df = factor_df.merge(mom_df, on='code', how='left')
            factor_df = factor_df.merge(dd_df, on='code', how='left')
            factor_df = factor_df.merge(vol_df, on='code', how='left')
            factor_df = factor_df.merge(turnover_df, on='code', how='left')

            # 股票池已经是精选的，不再做市值过滤（CSV的total_mv单位不可靠）
            factor_df = factor_df.fillna(0)

            if len(factor_df) > 0:
                scored, n_factors = calculate_factor_scores_v2(factor_df, weights)
                adjusted = adjust_weights_v2(
                    portfolio[portfolio['code'].isin(factor_df['code'])], scored)

                prev_weights = current_weights.copy()
                current_weights = dict(zip(adjusted['code'], adjusted['adjusted_weight']))

                # 计算调仓交易成本
                turnover_ratio = sum(
                    abs(current_weights.get(c, 0) - prev_weights.get(c, 0))
                    for c in set(list(current_weights.keys()) + list(prev_weights.keys()))
                ) / 2  # 单边换手率

                # 卖出部分的成本 + 买入部分的成本
                rebalance_cost = (turnover_ratio * TRANSACTION_COSTS['sell_commission'] +
                                  turnover_ratio * TRANSACTION_COSTS['buy_commission'])
                total_tx_cost += rebalance_cost

                # 统计
                top5 = sorted(current_weights.items(), key=lambda x: -x[1])[:5]
                top5_weight = sum(w for _, w in top5)
                print(f"    活跃因子: {n_factors} | 通过筛选: {len(factor_df)} 只")
                print(f"    换手率: {turnover_ratio:.1%} | 交易成本: {rebalance_cost:.4%}")
                print(f"    Top5权重: {top5_weight:.1%} ({', '.join(c for c, _ in top5)})")

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

        enh_ret = orig_ret = enh_ret_no_cost = 0
        for code in codes:
            if code in curr.index and pd.notna(curr[code]) and pd.notna(prev[code]) and prev[code] > 0:
                ret = curr[code] / prev[code] - 1
                enh_ret += ret * current_weights.get(code, 0) * market_position
                enh_ret_no_cost += ret * current_weights.get(code, 0) * market_position
                orig_ret += ret * original_weights.get(code, 0)

        # 扣减当日分摊的交易成本 (调仓日)
        if rebalance_cost > 0:
            enh_ret -= rebalance_cost

        enhanced_returns.append(enh_ret)
        enhanced_returns_no_cost.append(enh_ret_no_cost)
        original_returns.append(orig_ret)

    # ============================================================================
    # 绩效计算
    # ============================================================================
    def calc_metrics(returns, label=''):
        cum = np.cumprod([1 + r for r in returns])
        total = cum[-1] - 1
        n_years = len(returns) / 252
        annual = (1 + total) ** (1 / n_years) - 1 if n_years > 0 else 0
        max_dd = np.min(cum / np.maximum.accumulate(cum) - 1)
        daily_std = np.std(returns)
        sharpe = np.mean(returns) / daily_std * np.sqrt(252) if daily_std > 0 else 0
        calmar = annual / abs(max_dd) if max_dd != 0 else 0
        win_days = sum(1 for r in returns if r > 0)
        win_rate = win_days / len(returns) if returns else 0
        return {
            'annual_return': annual,
            'total_return': total,
            'max_drawdown': max_dd,
            'sharpe': sharpe,
            'calmar': calmar,
            'win_rate': win_rate,
            'final_nav': cum[-1],
            'volatility': daily_std * np.sqrt(252),
        }

    enh = calc_metrics(enhanced_returns, '因子增强(含成本)')
    enh_nc = calc_metrics(enhanced_returns_no_cost, '因子增强(无成本)')
    orig = calc_metrics(original_returns, '等权原始')

    print(f"\n{'=' * 70}")
    print(f"回测结果 [{label}]")
    print(f"{'=' * 70}")
    print(f"  {'指标':<16} {'等权原始':>14} {'增强(无成本)':>14} {'增强(含成本)':>14}")
    print(f"  {'-' * 58}")
    print(f"  {'年化收益':<16} {orig['annual_return']*100:>13.2f}% {enh_nc['annual_return']*100:>13.2f}% {enh['annual_return']*100:>13.2f}%")
    print(f"  {'总收益':<16} {orig['total_return']*100:>13.2f}% {enh_nc['total_return']*100:>13.2f}% {enh['total_return']*100:>13.2f}%")
    print(f"  {'最大回撤':<16} {orig['max_drawdown']*100:>13.2f}% {enh_nc['max_drawdown']*100:>13.2f}% {enh['max_drawdown']*100:>13.2f}%")
    print(f"  {'夏普比率':<16} {orig['sharpe']:>14.2f} {enh_nc['sharpe']:>14.2f} {enh['sharpe']:>14.2f}")
    print(f"  {'卡尔玛比率':<16} {orig['calmar']:>14.2f} {enh_nc['calmar']:>14.2f} {enh['calmar']:>14.2f}")
    print(f"  {'胜率':<16} {orig['win_rate']*100:>13.2f}% {enh_nc['win_rate']*100:>13.2f}% {enh['win_rate']*100:>13.2f}%")
    print(f"  {'年化波动率':<16} {orig['volatility']*100:>13.2f}% {enh_nc['volatility']*100:>13.2f}% {enh['volatility']*100:>13.2f}%")
    print(f"  {'最终净值':<16} {orig['final_nav']:>14.4f} {enh_nc['final_nav']:>14.4f} {enh['final_nav']:>14.4f}")
    print(f"\n  累计交易成本: {total_tx_cost*100:.4f}%")
    if MARKET_STOP_LOSS['enabled']:
        print(f"  大盘止损触发: {stop_loss_count} 次")

    return {
        'enhanced': enh,
        'enhanced_no_cost': enh_nc,
        'original': orig,
        'strategy_type': strategy_type,
        'total_tx_cost': total_tx_cost,
        'stop_loss_count': stop_loss_count,
    }


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 70)
    print("多因子权重策略回测系统 v2.1")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"策略: 核心因子保持v1.7权重 + 新因子12%增强")
    print(f"对比: 年度调仓 vs 季度调仓")
    print("=" * 70)

    start_date = '2020-01-01'
    end_date = '2025-12-31'

    # 4组对比: R4年度 / R4季度 / R5年度 / R5季度
    r4_annual = run_backtest_v2('stable', start_date, end_date, REBALANCE_ANNUAL)
    print("\n")
    r4_quarterly = run_backtest_v2('stable', start_date, end_date, REBALANCE_QUARTERLY)
    print("\n")
    r5_annual = run_backtest_v2('aggressive', start_date, end_date, REBALANCE_ANNUAL)
    print("\n")
    r5_quarterly = run_backtest_v2('aggressive', start_date, end_date, REBALANCE_QUARTERLY)

    # ============================================================================
    # 完整对比
    # ============================================================================
    print(f"\n{'=' * 70}")
    print("v2.1 完整对比 (所有组合含交易成本)")
    print(f"{'=' * 70}")

    def fmt(r):
        e = r['enhanced']
        return f"{e['annual_return']*100:>6.2f}%  {e['max_drawdown']*100:>6.2f}%  {e['sharpe']:>5.2f}  {e['calmar']:>5.2f}  {e['win_rate']*100:>5.1f}%  {r['total_tx_cost']*100:>5.2f}%"

    print(f"  {'策略':<22} {'年化':>8} {'回撤':>8} {'夏普':>6} {'卡尔玛':>6} {'胜率':>7} {'成本':>7}")
    print(f"  {'-' * 68}")
    print(f"  {'R4 稳健 (年度调仓)':<22} {fmt(r4_annual)}")
    print(f"  {'R4 稳健 (季度调仓)':<22} {fmt(r4_quarterly)}")
    print(f"  {'R5 进取 (年度调仓)':<22} {fmt(r5_annual)}")
    print(f"  {'R5 进取 (季度调仓)':<22} {fmt(r5_quarterly)}")
    print(f"  {'-' * 68}")
    print(f"  v1.7参考 R4: 14.66%年化  -15.68%回撤  0.98夏普  (12因子/年度/无成本)")
    print(f"  v1.7参考 R5: 16.61%年化  -19.77%回撤  1.01夏普  (12因子/年度/无成本)")

    # 判断最优组合
    all_results = {
        'R4_annual': r4_annual, 'R4_quarterly': r4_quarterly,
        'R5_annual': r5_annual, 'R5_quarterly': r5_quarterly,
    }
    best = max(all_results.items(), key=lambda x: x[1]['enhanced']['sharpe'])
    print(f"\n  最优夏普: {best[0]} (夏普={best[1]['enhanced']['sharpe']:.2f})")

    # 保存结果
    result = {
        'version': 'v2.1',
        'run_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'backtest_period': f'{start_date} ~ {end_date}',
        'strategy': '核心因子保持v1.7权重 + 新因子12%增强',
        'R4_annual': r4_annual,
        'R4_quarterly': r4_quarterly,
        'R5_annual': r5_annual,
        'R5_quarterly': r5_quarterly,
    }
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'factor_backtest_result_v2.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存到 {output_path}")


if __name__ == '__main__':
    main()
