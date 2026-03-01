#!/usr/bin/env python
"""
多因子权重策略回测 v3.0 — 多策略对比实验

基于华泰/广发金工研究和最新量化论文，设计5组对比实验:

策略A: v1.7复刻 — 原始12因子 + 原始8档权重 + 年度调仓 (含交易成本基线)
策略B: 新因子增强 — 16因子 + v1.7温和8档 + 年度调仓
策略C: 半年度调仓 — 16因子 + v1.7温和8档 + 半年调仓
策略D: IC动态加权 — 16因子 + IC加权动态权重 + 半年调仓 (核心创新)
策略E: 激进10档 — 16因子 + 激进10档 + 半年调仓 (对照组)

核心创新 — IC加权:
  不用固定的因子权重，而是根据每个因子在过去1-2个周期的
  实际选股表现(IC = 因子得分与后续收益的相关系数)动态调整权重。
  表现好的因子自动加权，表现差的自动降权。
  研究表明IC加权比等权/固定权重提升2-5%年化。
"""

import pandas as pd
import numpy as np
import clickhouse_connect
from datetime import datetime, timedelta
from scipy import stats
import json
import warnings
import os

warnings.filterwarnings('ignore')

# ============================================================================
# 基础配置
# ============================================================================

CLICKHOUSE_HOST = '192.168.0.74'
CLICKHOUSE_PORT = 8123

MARKET_STOP_LOSS = {
    'enabled': True,
    'ma_period': 250,
    'position_reduce': 0.70,
    'recovery_buffer': 1.02,
}

TRANSACTION_COSTS = {
    'buy_commission': 0.00026,
    'sell_commission': 0.00126,
}

# ============================================================================
# 权重档位配置 (多套)
# ============================================================================

# v1.7 原始温和8档 (range: 0.50~1.50, ratio=3x)
TIERS_MILD = [
    (0.95, 1.50), (0.90, 1.30), (0.80, 1.15), (0.60, 1.00),
    (0.40, 0.90), (0.20, 0.80), (0.10, 0.70), (0.00, 0.50),
]

# 激进10档 (range: 0.30~2.50, ratio=8.3x) — 对照组
TIERS_AGGRESSIVE = [
    (0.95, 2.50), (0.90, 2.00), (0.80, 1.60), (0.70, 1.30),
    (0.60, 1.10), (0.40, 1.00), (0.30, 0.90), (0.20, 0.75),
    (0.10, 0.50), (0.00, 0.30),
]

# ============================================================================
# 因子权重配置
# ============================================================================

# v1.7 原始12因子 (R4稳健型)
FACTORS_V17_R4 = {
    'dividend_yield': 0.18, 'roe': 0.15, 'roe_stability': 0.10,
    'reversal': 0.10, 'pe_value': 0.10, 'profit_growth': 0.10,
    'small_cap': 0.08, 'momentum': 0.05, 'low_volatility': 0.07,
    'financial_health': 0.07,
}

# v1.7 原始12因子 (R5进取型)
FACTORS_V17_R5 = {
    'profit_growth': 0.18, 'small_cap': 0.15, 'momentum': 0.12,
    'roe': 0.12, 'revenue_growth': 0.10, 'reversal': 0.08,
    'dividend_yield': 0.08, 'pe_value': 0.05, 'low_volatility': 0.05,
    'financial_health': 0.07,
}

# v3 16因子: v1.7核心 + 12%新因子 (R4)
FACTORS_V3_R4 = {
    'dividend_yield': 0.16, 'roe': 0.14, 'roe_stability': 0.09,
    'reversal': 0.10, 'pe_value': 0.09, 'profit_growth': 0.09,
    'small_cap': 0.06, 'momentum': 0.04, 'low_volatility': 0.06,
    'financial_health': 0.05,
    # 新因子 12%
    'pb_value': 0.03, 'gross_margin': 0.03, 'turnover': 0.02,
    'operating_quality': 0.02, 'net_margin': 0.01, 'asset_turnover': 0.01,
}

# v3 16因子 (R5)
FACTORS_V3_R5 = {
    'profit_growth': 0.16, 'small_cap': 0.13, 'momentum': 0.11,
    'roe': 0.11, 'revenue_growth': 0.09, 'reversal': 0.07,
    'dividend_yield': 0.07, 'pe_value': 0.04, 'low_volatility': 0.04,
    'financial_health': 0.06,
    # 新因子 12%
    'turnover': 0.03, 'gross_margin': 0.03, 'pb_value': 0.02,
    'operating_quality': 0.02, 'asset_turnover': 0.01, 'net_margin': 0.01,
}

# IC动态加权的初始权重 (等权起步, 后续根据IC动态调整)
FACTORS_IC_INIT_R4 = {k: 1.0 / 16 for k in FACTORS_V3_R4.keys()}
FACTORS_IC_INIT_R5 = {k: 1.0 / 16 for k in FACTORS_V3_R5.keys()}


# ============================================================================
# 数据获取 (复用v2的DataFetcher)
# ============================================================================

class DataFetcherV3:
    def __init__(self):
        self.client = clickhouse_connect.get_client(
            host=CLICKHOUSE_HOST,
            port=CLICKHOUSE_PORT,
            compress=False,
            query_limit=10000000  # 设置更大的查询限制
        )
        self._valuation_df = None
        self._dividend_df = None

    def get_stock_pool(self, strategy_type='stable'):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir,
            'r4_weights_local.csv' if strategy_type == 'stable' else 'r5_weights_local.csv')
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
        dt = datetime.strptime(as_of_date, '%Y-%m-%d')
        year, month = dt.year, dt.month
        if month <= 4:
            return f'{year - 2}-12-31'
        elif month <= 8:
            return f'{year - 1}-12-31'
        elif month <= 10:
            return f'{year}-06-30'
        else:
            return f'{year}-09-30'

    def load_valuation_csv(self):
        if self._valuation_df is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            df = pd.read_csv(os.path.join(base_dir, 'valuation_local.csv'))
            df['code'] = df['code'].astype(str).str.zfill(6)
            df['market_cap'] = df['total_mv'] / 10000
            self._valuation_df = df
        return self._valuation_df

    def get_financial_data(self, codes, as_of_date):
        val_df = self.load_valuation_csv()
        val_df = val_df[val_df['code'].isin(codes)].copy()
        fin_df = val_df[['code', 'roe', 'eps', 'bvps', 'pe_ttm', 'pb',
                         'market_cap', 'dividend_yield']].copy()
        fin_df = fin_df.rename(columns={'pe_ttm': 'pe'})

        codes_str = "','".join(codes)
        report_date = self.get_latest_report_date(as_of_date)
        yoy_date = f'{int(report_date[:4]) - 1}{report_date[4:]}'

        # 扩展财务数据
        try:
            query = f"""
                SELECT code, asset_liability_ratio, gross_profit_margin,
                       net_profit_margin, operating_profit, net_profit,
                       operating_revenue, total_assets
                FROM default.stock_financial
                WHERE code IN ('{codes_str}') AND report_date <= '{report_date}'
                ORDER BY report_date DESC LIMIT 1 BY code
            """
            result = self.client.query(query)
            ext_df = pd.DataFrame(result.result_rows,
                columns=['code', 'asset_liability_ratio', 'gross_profit_margin',
                         'net_profit_margin', 'operating_profit', 'net_profit_ch',
                         'operating_revenue', 'total_assets'])
            fin_df = fin_df.merge(ext_df, on='code', how='left')
        except:
            pass

        defaults = {'asset_liability_ratio': 50.0, 'gross_profit_margin': 30.0,
                    'net_profit_margin': 10.0, 'operating_profit': 0,
                    'net_profit_ch': 0, 'operating_revenue': 0, 'total_assets': 1}
        for col, val in defaults.items():
            if col not in fin_df.columns:
                fin_df[col] = val
            fin_df[col] = fin_df[col].fillna(val)

        fin_df['operating_quality'] = np.where(
            fin_df['net_profit_ch'].abs() > 1e-6,
            fin_df['operating_profit'] / fin_df['net_profit_ch'].abs(), 1.0
        ).clip(-2, 3)
        fin_df['asset_turnover'] = np.where(
            fin_df['total_assets'] > 1e-6,
            fin_df['operating_revenue'] / fin_df['total_assets'], 0.5
        ).clip(0, 5)

        # 增长率
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
                            THEN (latest.eps / yoy.yoy_eps - 1) * 100 ELSE 0 END,
                       CASE WHEN yoy.yoy_revenue > 0
                            THEN (latest.operating_revenue / yoy.yoy_revenue - 1) * 100 ELSE 0 END
                FROM latest LEFT JOIN yoy ON latest.code = yoy.code
            """
            growth_result = self.client.query(growth_query)
            growth_df = pd.DataFrame(growth_result.result_rows,
                columns=['code', 'profit_growth', 'revenue_growth'])
            fin_df = fin_df.merge(growth_df, on='code', how='left')
        except:
            pass

        for col in ['profit_growth', 'revenue_growth']:
            if col not in fin_df.columns:
                fin_df[col] = 0.0
            fin_df[col] = fin_df[col].fillna(0.0)

        # ROE稳定性
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
                roe_hist = pd.DataFrame(roe_result.result_rows, columns=['code', 'roe_hist', 'rd'])
                roe_stab = roe_hist.groupby('code')['roe_hist'].agg(['mean', 'std']).reset_index()
                roe_stab['roe_stability'] = 1 / (1 + roe_stab['std'] / (roe_stab['mean'].abs() + 1e-6))
                fin_df = fin_df.merge(roe_stab[['code', 'roe_stability']], on='code', how='left')
        except:
            pass

        if 'roe_stability' not in fin_df.columns:
            fin_df['roe_stability'] = 0.5
        fin_df['roe_stability'] = fin_df['roe_stability'].fillna(0.5)
        return fin_df

    def get_momentum(self, codes, as_of_date, days=60):
        codes_str = "','".join(codes)
        query = f"""
            WITH latest AS (
                SELECT code, close as lc FROM default.stock_data_qfq
                WHERE code IN ('{codes_str}') AND date <= '{as_of_date}'
                ORDER BY date DESC LIMIT 1 BY code
            ),
            past AS (
                SELECT code, close as pc FROM default.stock_data_qfq
                WHERE code IN ('{codes_str}')
                  AND date <= date_sub(day, {days}, toDate('{as_of_date}'))
                ORDER BY date DESC LIMIT 1 BY code
            )
            SELECT latest.code, (latest.lc / past.pc - 1) * 100
            FROM latest JOIN past ON latest.code = past.code
        """
        result = self.client.query(query)
        return pd.DataFrame(result.result_rows, columns=['code', 'momentum'])

    def get_drawdown(self, codes, as_of_date, months=3):
        codes_str = "','".join(codes)
        query = f"""
            WITH pd AS (
                SELECT code, date, close FROM default.stock_data_qfq
                WHERE code IN ('{codes_str}')
                  AND date >= date_sub(month, {months}, toDate('{as_of_date}'))
                  AND date <= '{as_of_date}'
            ),
            mx AS (SELECT code, max(close) as h FROM pd GROUP BY code),
            lt AS (SELECT code, close as l FROM pd ORDER BY date DESC LIMIT 1 BY code)
            SELECT mx.code, (lt.l / mx.h - 1) * 100 FROM mx JOIN lt ON mx.code = lt.code
        """
        result = self.client.query(query)
        return pd.DataFrame(result.result_rows, columns=['code', 'drawdown'])

    def get_volatility(self, codes, as_of_date, days=60):
        codes_str = "','".join(codes)
        query = f"""
            WITH dr AS (
                SELECT code, close / lagInFrame(close, 1)
                    OVER (PARTITION BY code ORDER BY date) - 1 as ret
                FROM default.stock_data_qfq
                WHERE code IN ('{codes_str}')
                  AND date >= date_sub(day, {days}, toDate('{as_of_date}'))
                  AND date <= '{as_of_date}'
            )
            SELECT code, stddevPop(ret) * sqrt(252) * 100
            FROM dr WHERE ret IS NOT NULL GROUP BY code
        """
        result = self.client.query(query)
        return pd.DataFrame(result.result_rows, columns=['code', 'volatility'])

    def get_turnover(self, codes, as_of_date, days=20):
        codes_str = "','".join(codes)
        query = f"""
            SELECT code, avg(huanshoulv) as avg_turnover,
                   stddevPop(huanshoulv) as std_turnover
            FROM default.stock_data_qfq
            WHERE code IN ('{codes_str}')
              AND date >= date_sub(day, {days}, toDate('{as_of_date}'))
              AND date <= '{as_of_date}'
            GROUP BY code
        """
        result = self.client.query(query)
        return pd.DataFrame(result.result_rows, columns=['code', 'avg_turnover', 'std_turnover'])

    def get_prices(self, codes, start_date, end_date):
        codes_str = "','".join(codes)
        query = f"""
            SELECT toString(date), code, close FROM default.stock_data_qfq
            WHERE code IN ('{codes_str}')
              AND date >= '{start_date}' AND date <= '{end_date}'
            ORDER BY date, code
        """
        result = self.client.query(query)
        df = pd.DataFrame(result.result_rows, columns=['date', 'code', 'close'])
        return df.drop_duplicates(subset=['date', 'code'], keep='last')

    def get_index_prices(self, start_date, end_date):
        try:
            query = f"""
                SELECT toString(date), close FROM default.stock_index
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
# 因子打分 (支持所有16个因子)
# ============================================================================

def pct_score(s, ascending=True):
    return s.rank(pct=True) * 100 if ascending else (1 - s.rank(pct=True)) * 100

def score_all_factors(df):
    """
    对所有可能的因子打分，返回 {factor_name: score_series}
    不做加权合成，让调用者自己决定权重
    """
    scores = {}

    if 'roe' in df.columns:
        scores['roe'] = pct_score(df['roe'].clip(0, 50))
    if 'pe' in df.columns:
        valid_pe = df['pe'].where((df['pe'] > 0) & (df['pe'] < 200), np.nan)
        scores['pe_value'] = pct_score(valid_pe, ascending=False).fillna(30)
    if 'dividend_yield' in df.columns:
        s = pct_score(df['dividend_yield'])
        s[df['dividend_yield'] > 5] = s[df['dividend_yield'] > 5].clip(lower=85)
        scores['dividend_yield'] = s
    if 'asset_liability_ratio' in df.columns:
        scores['financial_health'] = pct_score(df['asset_liability_ratio'], ascending=False)
    if 'profit_growth' in df.columns:
        scores['profit_growth'] = pct_score(df['profit_growth'].clip(-50, 200))
    if 'revenue_growth' in df.columns:
        scores['revenue_growth'] = pct_score(df['revenue_growth'].clip(-50, 200))
    if 'market_cap' in df.columns:
        scores['small_cap'] = pct_score(df['market_cap'], ascending=False)
    if 'momentum' in df.columns:
        scores['momentum'] = pct_score(df['momentum'])
    if 'drawdown' in df.columns:
        is_q = df['roe'] > 15 if 'roe' in df.columns else False
        dd_s = pct_score(df['drawdown'])
        scores['reversal'] = np.where(is_q, dd_s * 1.2, dd_s * 0.8).clip(0, 100)
        scores['reversal'] = pd.Series(scores['reversal'], index=df.index)
    if 'volatility' in df.columns:
        scores['low_volatility'] = pct_score(df['volatility'], ascending=False)
    if 'roe_stability' in df.columns:
        scores['roe_stability'] = pct_score(df['roe_stability'])
    # PEG
    if 'pe' in df.columns and 'profit_growth' in df.columns:
        valid = (df['pe'] > 0) & (df['pe'] < 100) & (df['profit_growth'] > 5)
        peg = pd.Series(np.nan, index=df.index)
        peg[valid] = df.loc[valid, 'pe'] / df.loc[valid, 'profit_growth']
        scores['peg'] = pct_score(peg.clip(0, 5), ascending=False).fillna(40)
    # 新因子
    if 'pb' in df.columns:
        valid_pb = df['pb'].where((df['pb'] > 0) & (df['pb'] < 20), np.nan)
        scores['pb_value'] = pct_score(valid_pb, ascending=False).fillna(30)
    if 'gross_profit_margin' in df.columns:
        scores['gross_margin'] = pct_score(df['gross_profit_margin'].clip(0, 80))
    if 'net_profit_margin' in df.columns:
        scores['net_margin'] = pct_score(df['net_profit_margin'].clip(-20, 50))
    if 'operating_quality' in df.columns:
        oq = 100 - (df['operating_quality'] - 1.0).abs() * 50
        scores['operating_quality'] = oq.clip(0, 100)
    if 'asset_turnover' in df.columns:
        scores['asset_turnover'] = pct_score(df['asset_turnover'].clip(0, 3))
    if 'avg_turnover' in df.columns:
        scores['turnover'] = pct_score(df['avg_turnover'], ascending=False)

    return scores


def composite_score(factor_scores, weights):
    """用给定权重合成综合得分"""
    keys = list(factor_scores.keys())
    if not keys:
        return pd.Series(50.0)
    idx = factor_scores[keys[0]].index
    composite = pd.Series(0.0, index=idx)
    total_w = 0
    for factor, weight in weights.items():
        if factor in factor_scores:
            composite += factor_scores[factor].fillna(50) * weight
            total_w += weight
    # 归一化到实际使用的权重
    if total_w > 0 and abs(total_w - 1.0) > 0.01:
        composite = composite / total_w
    return composite


# ============================================================================
# IC加权动态因子权重 (核心创新)
# ============================================================================

class ICWeightedManager:
    """
    IC加权管理器
    根据因子在上一期的实际选股表现(IC)动态调整因子权重

    IC = rank_corr(因子得分, 下一期收益)
    - IC > 0: 因子有效, 加权
    - IC < 0: 因子反向, 反转或降权
    - IC ≈ 0: 因子无效, 降权

    使用ICIR(IC均值/IC标准差)来进一步考虑稳定性
    """

    def __init__(self, base_weights, decay=0.7):
        """
        Args:
            base_weights: 初始因子权重 (dict)
            decay: IC衰减系数, 越小越重视近期表现
        """
        self.base_weights = base_weights.copy()
        self.decay = decay
        self.ic_history = {f: [] for f in base_weights}
        self.prev_scores = None  # 上一期的因子得分
        self.prev_codes = None

    def update_ic(self, returns_dict):
        """
        根据上一期因子得分和本期实际收益, 计算各因子IC

        Args:
            returns_dict: {code: return} 上一期到本期的实际收益率
        """
        if self.prev_scores is None:
            return

        # 对齐代码
        common_codes = [c for c in self.prev_codes if c in returns_dict]
        if len(common_codes) < 20:
            return  # 样本太少不计算

        actual_returns = [returns_dict[c] for c in common_codes]

        for factor, score_series in self.prev_scores.items():
            factor_vals = []
            for c in common_codes:
                if c in score_series.index:
                    factor_vals.append(score_series.loc[c])
                else:
                    factor_vals.append(50.0)

            # Spearman rank correlation (IC)
            try:
                ic, _ = stats.spearmanr(factor_vals, actual_returns)
                if np.isnan(ic):
                    ic = 0.0
            except:
                ic = 0.0

            self.ic_history[factor].append(ic)

    def get_dynamic_weights(self):
        """
        根据IC历史计算动态权重

        方法: ICIR加权 (华泰金工推荐)
        权重 ∝ max(ICIR, 0) + base_weight * 0.3
        """
        weights = {}

        has_ic = any(len(v) > 0 for v in self.ic_history.values())
        if not has_ic:
            return self.base_weights.copy()

        for factor in self.base_weights:
            ics = self.ic_history.get(factor, [])
            if len(ics) == 0:
                weights[factor] = self.base_weights[factor]
                continue

            # 指数衰减加权的IC均值
            w = np.array([self.decay ** i for i in range(len(ics) - 1, -1, -1)])
            w = w / w.sum()
            ic_mean = np.average(ics, weights=w)
            ic_std = np.std(ics) if len(ics) > 1 else 0.05

            # ICIR
            icir = ic_mean / (ic_std + 1e-6)

            # 权重 = max(ICIR, 0) + 基础权重的30%保底
            # 这样即使IC为负, 因子也不会完全被踢掉
            raw_w = max(icir, 0) + self.base_weights[factor] * 0.3
            weights[factor] = raw_w

        # 归一化
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def save_scores(self, factor_scores, codes):
        """保存本期因子得分, 用于下一期IC计算"""
        self.prev_scores = {f: s.copy() for f, s in factor_scores.items()}
        self.prev_codes = codes


# ============================================================================
# 权重调整
# ============================================================================

def get_multiplier(percentile, tiers):
    for threshold, mult in tiers:
        if percentile >= threshold:
            return mult
    return tiers[-1][1]

def adjust_weights(portfolio, composite_scores, tiers):
    merged = portfolio.copy()
    merged = merged.merge(
        pd.DataFrame({'code': composite_scores.index, 'composite_score': composite_scores.values}),
        on='code', how='left')
    merged['composite_score'] = merged['composite_score'].fillna(50)
    merged['percentile'] = merged['composite_score'].rank(pct=True)
    merged['multiplier'] = merged['percentile'].apply(lambda p: get_multiplier(p, tiers))
    merged['original_weight'] = merged['weight']
    merged['adjusted_weight'] = merged['weight'] * merged['multiplier']
    merged['adjusted_weight'] /= merged['adjusted_weight'].sum()
    return merged


# ============================================================================
# 统一回测引擎
# ============================================================================

def run_single_backtest(strategy_type, factor_weights, tiers, rebalance_months,
                        use_ic_weighting=False, start_date='2020-01-01',
                        end_date='2025-12-31', label=''):
    """通用回测函数"""
    fetcher = DataFetcherV3()
    portfolio = fetcher.get_stock_pool(strategy_type)
    codes = portfolio['code'].tolist()

    freq = {1: '年度', 2: '半年', 4: '季度'}.get(len(rebalance_months), f'{len(rebalance_months)}次/年')
    ic_label = '+IC动态' if use_ic_weighting else ''
    print(f"\n{'='*65}")
    print(f"[{label}] {len(factor_weights)}因子 | {freq}调仓 | "
          f"{'温和' if tiers == TIERS_MILD else '激进'}档位{ic_label}")
    print(f"{'='*65}")

    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    rebalance_dates = []
    for y in range(start_dt.year, end_dt.year + 1):
        for m in rebalance_months:
            rd = f'{y}-{m:02d}-01'
            if start_date <= rd <= end_date:
                rebalance_dates.append(rd)

    extended_start = (start_dt - timedelta(days=400)).strftime('%Y-%m-%d')
    prices = fetcher.get_prices(codes, extended_start, end_date)
    price_pivot = prices.pivot(index='date', columns='code', values='close')
    index_prices = fetcher.get_index_prices(extended_start, end_date)

    dates = sorted([d for d in price_pivot.index if d >= start_date])

    # 初始化
    original_weights = dict(zip(portfolio['code'], portfolio['weight']))
    current_weights = original_weights.copy()
    prev_weights = {}
    rb_idx = 0

    market_position = 1.0
    stop_loss_count = 0

    enhanced_returns = []
    original_returns = []
    total_tx_cost = 0.0

    # IC管理器
    ic_mgr = ICWeightedManager(factor_weights, decay=0.7) if use_ic_weighting else None
    prev_rb_date = None

    for i in range(1, len(dates)):
        curr_date = dates[i]
        prev_date = dates[i - 1]
        rebalance_cost = 0.0

        while rb_idx < len(rebalance_dates) and curr_date >= rebalance_dates[rb_idx]:
            rb_date = rebalance_dates[rb_idx]

            # IC更新: 计算上一期的因子表现
            if use_ic_weighting and ic_mgr and prev_rb_date:
                # 计算上一期各股票的实际收益
                returns_dict = {}
                for code in codes:
                    if (prev_rb_date in price_pivot.index and rb_date in price_pivot.index and
                            code in price_pivot.columns):
                        p1 = price_pivot.loc[prev_rb_date].get(code)
                        p2 = price_pivot.loc[rb_date].get(code)
                        if pd.notna(p1) and pd.notna(p2) and p1 > 0:
                            returns_dict[code] = p2 / p1 - 1
                ic_mgr.update_ic(returns_dict)

            # 获取因子数据
            factor_df = fetcher.get_financial_data(codes, rb_date)
            mom_df = fetcher.get_momentum(codes, rb_date)
            dd_df = fetcher.get_drawdown(codes, rb_date)
            vol_df = fetcher.get_volatility(codes, rb_date)
            turnover_df = fetcher.get_turnover(codes, rb_date)

            factor_df = factor_df.merge(mom_df, on='code', how='left')
            factor_df = factor_df.merge(dd_df, on='code', how='left')
            factor_df = factor_df.merge(vol_df, on='code', how='left')
            factor_df = factor_df.merge(turnover_df, on='code', how='left')
            factor_df = factor_df.fillna(0)
            factor_df = factor_df.set_index('code')

            if len(factor_df) > 0:
                # 计算所有因子得分
                all_scores = score_all_factors(factor_df)

                # IC动态权重 or 固定权重
                if use_ic_weighting and ic_mgr:
                    ic_mgr.save_scores(all_scores, factor_df.index.tolist())
                    active_weights = ic_mgr.get_dynamic_weights()
                else:
                    active_weights = factor_weights

                # 合成综合得分
                comp = composite_score(all_scores, active_weights)
                comp.index = factor_df.index

                # 权重调整
                factor_df_reset = factor_df.reset_index()
                adjusted = adjust_weights(
                    portfolio[portfolio['code'].isin(factor_df_reset['code'])],
                    comp, tiers)

                prev_weights = current_weights.copy()
                current_weights = dict(zip(adjusted['code'], adjusted['adjusted_weight']))

                # 交易成本
                turnover_ratio = sum(
                    abs(current_weights.get(c, 0) - prev_weights.get(c, 0))
                    for c in set(list(current_weights.keys()) + list(prev_weights.keys()))
                ) / 2
                rebalance_cost = (turnover_ratio * TRANSACTION_COSTS['sell_commission'] +
                                  turnover_ratio * TRANSACTION_COSTS['buy_commission'])
                total_tx_cost += rebalance_cost

                n_active = sum(1 for f in active_weights if f in all_scores and active_weights[f] > 0.001)
                print(f"  [{rb_date}] 因子:{n_active} | 换手:{turnover_ratio:.1%} | 成本:{rebalance_cost:.4%}")

                if use_ic_weighting and ic_mgr and len(ic_mgr.ic_history.get('roe', [])) > 0:
                    # 显示IC最高和最低的因子
                    dw = active_weights
                    top3 = sorted(dw.items(), key=lambda x: -x[1])[:3]
                    print(f"         IC权重Top3: {', '.join(f'{k}={v:.1%}' for k,v in top3)}")

            prev_rb_date = rb_date
            rb_idx += 1

        # 日收益
        curr = price_pivot.loc[curr_date]
        prev = price_pivot.loc[prev_date]

        # 大盘止损
        if MARKET_STOP_LOSS['enabled'] and not index_prices.empty:
            idx = index_prices[index_prices['date'] <= curr_date]
            if len(idx) >= MARKET_STOP_LOSS['ma_period']:
                ma = idx['close'].rolling(MARKET_STOP_LOSS['ma_period']).mean().iloc[-1]
                pn = idx['close'].iloc[-1]
                if pn < ma:
                    if market_position == 1.0:
                        stop_loss_count += 1
                    market_position = MARKET_STOP_LOSS['position_reduce']
                elif pn > ma * MARKET_STOP_LOSS['recovery_buffer']:
                    market_position = 1.0

        enh_ret = orig_ret = 0
        for code in codes:
            if code in curr.index and pd.notna(curr[code]) and pd.notna(prev[code]) and prev[code] > 0:
                ret = curr[code] / prev[code] - 1
                enh_ret += ret * current_weights.get(code, 0) * market_position
                orig_ret += ret * original_weights.get(code, 0)

        if rebalance_cost > 0:
            enh_ret -= rebalance_cost

        enhanced_returns.append(enh_ret)
        original_returns.append(orig_ret)

    # 绩效
    def calc(returns):
        cum = np.cumprod([1 + r for r in returns])
        total = cum[-1] - 1
        n_years = len(returns) / 252
        annual = (1 + total) ** (1 / n_years) - 1 if n_years > 0 else 0
        max_dd = np.min(cum / np.maximum.accumulate(cum) - 1)
        daily_std = np.std(returns)
        sharpe = np.mean(returns) / daily_std * np.sqrt(252) if daily_std > 0 else 0
        calmar = annual / abs(max_dd) if max_dd != 0 else 0
        win_rate = sum(1 for r in returns if r > 0) / len(returns) if returns else 0
        return dict(annual_return=annual, total_return=total, max_drawdown=max_dd,
                    sharpe=sharpe, calmar=calmar, win_rate=win_rate, final_nav=cum[-1])

    enh = calc(enhanced_returns)
    orig = calc(original_returns)

    print(f"\n  年化: {enh['annual_return']*100:.2f}% | 回撤: {enh['max_drawdown']*100:.2f}% | "
          f"夏普: {enh['sharpe']:.2f} | 卡尔玛: {enh['calmar']:.2f} | "
          f"成本: {total_tx_cost*100:.2f}%")

    return {
        'label': label, 'enhanced': enh, 'original': orig,
        'total_tx_cost': total_tx_cost, 'stop_loss_count': stop_loss_count,
    }


# ============================================================================
# 主函数: 5组对比实验
# ============================================================================

def main():
    print("=" * 65)
    print("多因子策略 v3.0 — 5组对比实验")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    sd, ed = '2020-01-01', '2025-12-31'
    results = {}

    for st, st_label in [('stable', 'R4'), ('aggressive', 'R5')]:
        fw_v17 = FACTORS_V17_R4 if st == 'stable' else FACTORS_V17_R5
        fw_v3 = FACTORS_V3_R4 if st == 'stable' else FACTORS_V3_R5
        fw_ic = FACTORS_IC_INIT_R4 if st == 'stable' else FACTORS_IC_INIT_R5

        # 策略A: v1.7复刻 (基线)
        results[f'{st_label}_A'] = run_single_backtest(
            st, fw_v17, TIERS_MILD, [1],
            label=f'{st_label}-A v1.7复刻', start_date=sd, end_date=ed)

        # 策略B: 新因子 + 温和档位 + 年度
        results[f'{st_label}_B'] = run_single_backtest(
            st, fw_v3, TIERS_MILD, [1],
            label=f'{st_label}-B 新因子+年度', start_date=sd, end_date=ed)

        # 策略C: 新因子 + 温和档位 + 半年度
        results[f'{st_label}_C'] = run_single_backtest(
            st, fw_v3, TIERS_MILD, [1, 7],
            label=f'{st_label}-C 新因子+半年', start_date=sd, end_date=ed)

        # 策略D: IC动态加权 + 温和档位 + 半年度 (核心创新)
        results[f'{st_label}_D'] = run_single_backtest(
            st, fw_v3, TIERS_MILD, [1, 7],
            use_ic_weighting=True,
            label=f'{st_label}-D IC动态+半年', start_date=sd, end_date=ed)

        # 策略E: 新因子 + 激进10档 + 半年度 (对照)
        results[f'{st_label}_E'] = run_single_backtest(
            st, fw_v3, TIERS_AGGRESSIVE, [1, 7],
            label=f'{st_label}-E 激进档位对照', start_date=sd, end_date=ed)

    # ============================================================================
    # 汇总对比
    # ============================================================================
    print(f"\n\n{'='*90}")
    print("完整对比表 (所有策略含交易成本)")
    print(f"{'='*90}")
    print(f"  {'策略':<28} {'年化':>8} {'回撤':>8} {'夏普':>6} {'卡尔玛':>7} {'成本':>7}")
    print(f"  {'-'*72}")

    for key in ['R4_A', 'R4_B', 'R4_C', 'R4_D', 'R4_E']:
        r = results[key]
        e = r['enhanced']
        print(f"  {r['label']:<28} {e['annual_return']*100:>7.2f}% {e['max_drawdown']*100:>7.2f}% "
              f"{e['sharpe']:>6.2f} {e['calmar']:>7.2f} {r['total_tx_cost']*100:>6.2f}%")

    print(f"  {'-'*72}")
    print(f"  v1.7参考(无成本)             14.66%  -15.68%   0.98      ---      0%")
    print()

    for key in ['R5_A', 'R5_B', 'R5_C', 'R5_D', 'R5_E']:
        r = results[key]
        e = r['enhanced']
        print(f"  {r['label']:<28} {e['annual_return']*100:>7.2f}% {e['max_drawdown']*100:>7.2f}% "
              f"{e['sharpe']:>6.2f} {e['calmar']:>7.2f} {r['total_tx_cost']*100:>6.2f}%")

    print(f"  {'-'*72}")
    print(f"  v1.7参考(无成本)             16.61%  -19.77%   1.01      ---      0%")

    # 找最优
    print(f"\n  {'最优策略':}")
    for prefix in ['R4', 'R5']:
        keys = [f'{prefix}_{x}' for x in 'ABCDE']
        best_sharpe = max(keys, key=lambda k: results[k]['enhanced']['sharpe'])
        best_calmar = max(keys, key=lambda k: results[k]['enhanced']['calmar'])
        best_return = max(keys, key=lambda k: results[k]['enhanced']['annual_return'])
        print(f"  {prefix} 最高夏普: {results[best_sharpe]['label']} ({results[best_sharpe]['enhanced']['sharpe']:.2f})")
        print(f"  {prefix} 最高卡尔玛: {results[best_calmar]['label']} ({results[best_calmar]['enhanced']['calmar']:.2f})")
        print(f"  {prefix} 最高收益: {results[best_return]['label']} ({results[best_return]['enhanced']['annual_return']*100:.2f}%)")

    # 保存
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'factor_backtest_result_v3.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'version': 'v3.0',
            'run_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'experiments': {k: v for k, v in results.items()},
        }, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存到 {output_path}")


if __name__ == '__main__':
    main()
