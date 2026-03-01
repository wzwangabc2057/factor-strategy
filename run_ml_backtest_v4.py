#!/usr/bin/env python
"""
ML增强因子策略回测 v4.3 — 恢复V1高收益基线 + 季度ML

架构:
  ClickHouse → DataFetcherML → FactorPipeline → IC筛选 → FeatureBuilder
  → MLSignalGenerator (LightGBM+XGBoost集成) → 季度/月度调仓回测

v4.3改进:
  - 恢复V1的10因子集中权重作为新基线 (V1曾达24.38%)
  - 添加MIN_MARKET_CAP=50过滤 (V1关键特征)
  - 季度ML调仓替代月度 (降低3倍交易成本)
  - 修复IC_THRESHOLDS NameError

策略对比:
  V1(基线): R5, 10集中因子, 0.5-1.5x, 年度调仓, 无ML, 市值过滤
  C (基线): R5, 16固定因子, 0.5-1.5x, 半年调仓, 无ML
  H (ML季度温和): R5, IC筛选因子, 0.5-1.5x, 季度调仓, LGB+XGB, 市值过滤
  I (ML季度激进): R5, IC筛选因子, 0.2-3.0x, 季度调仓, LGB+XGB, 市值过滤
"""

import pandas as pd
import numpy as np
import clickhouse_connect
from datetime import datetime, timedelta
from scipy import stats
import json
import warnings
import os
import sys
import logging
import traceback
import argparse

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

# ML imports
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.data.alpha158_factors import Alpha158FactorCalculator
from src.data.extended_factors import ExtendedFactorCalculator

# ============================================================================
# 配置
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

# 权重档位
TIERS_MILD = [
    (0.95, 1.50), (0.90, 1.30), (0.80, 1.15), (0.60, 1.00),
    (0.40, 0.90), (0.20, 0.80), (0.10, 0.70), (0.00, 0.50),
]

TIERS_ML_AGGRESSIVE = [
    (0.95, 3.00), (0.90, 2.20), (0.80, 1.60), (0.65, 1.20),
    (0.35, 1.00), (0.00, 0.20),
]

# V1 R5 10因子集中权重 (V1曾达24.38%年化)
FACTORS_V1_R5 = {
    'profit_growth': 0.18, 'small_cap': 0.15, 'momentum': 0.12,
    'roe': 0.12, 'revenue_growth': 0.10, 'reversal': 0.08,
    'dividend_yield': 0.08, 'pe_value': 0.05, 'low_volatility': 0.05,
    'financial_health': 0.07,
}

# R5 16因子权重 (基线策略C用)
FACTORS_V3_R5 = {
    'profit_growth': 0.16, 'small_cap': 0.13, 'momentum': 0.11,
    'roe': 0.11, 'revenue_growth': 0.09, 'reversal': 0.07,
    'dividend_yield': 0.07, 'pe_value': 0.04, 'low_volatility': 0.04,
    'financial_health': 0.06,
    'turnover': 0.03, 'gross_margin': 0.03, 'pb_value': 0.02,
    'operating_quality': 0.02, 'asset_turnover': 0.01, 'net_margin': 0.01,
}

# 市值过滤 (V1关键特征: 过滤流通市值<50亿的小票)
MIN_MARKET_CAP = 50

# LightGBM 超参数 (v4.1: 更多树+更低学习率+更强正则化)
LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 23,
    'learning_rate': 0.01,
    'feature_fraction': 0.6,
    'bagging_fraction': 0.6,
    'bagging_freq': 5,
    'reg_alpha': 0.5,
    'reg_lambda': 2.0,
    'min_child_samples': 30,
    'verbose': -1,
    'n_estimators': 1500,
    'n_jobs': 4,
}

# XGBoost 超参数 (v4.1: 更保守防过拟合)
XGB_PARAMS = {
    'objective': 'reg:squarederror',
    'max_depth': 4,
    'learning_rate': 0.01,
    'subsample': 0.6,
    'colsample_bytree': 0.6,
    'reg_alpha': 0.5,
    'reg_lambda': 2.0,
    'min_child_weight': 10,
    'n_estimators': 1500,
    'n_jobs': 4,
}

# IC 筛选配置 (v4.2: Top-K排名替代阈值筛选)
IC_CONFIG = {
    'top_k': 35,           # 始终选择IC排名前35的因子
    'min_observations': 2,  # 至少2次IC观测
    'warmup_months': 2,     # 预热期: 头2个月用全部因子
}


# ============================================================================
# DataFetcherML — 继承v3并增加OHLCV
# ============================================================================

class DataFetcherML:
    """ML增强数据获取器"""

    def __init__(self):
        self.client = clickhouse_connect.get_client(
            host=CLICKHOUSE_HOST, port=CLICKHOUSE_PORT,
            compress=False, query_limit=10000000
        )
        self._valuation_df = None

    def get_stock_pool(self):
        """获取R5股票池"""
        csv_path = os.path.join(BASE_DIR, 'r5_weights_local.csv')
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
            df = pd.read_csv(os.path.join(BASE_DIR, 'valuation_local.csv'))
            df['code'] = df['code'].astype(str).str.zfill(6)
            df['market_cap'] = df['total_mv']  # total_mv已经是亿元
            self._valuation_df = df
        return self._valuation_df

    def get_financial_data(self, codes, as_of_date):
        """获取基本面数据 (复用v3逻辑)"""
        val_df = self.load_valuation_csv()
        val_df = val_df[val_df['code'].isin(codes)].copy()
        fin_df = val_df[['code', 'roe', 'eps', 'bvps', 'pe_ttm', 'pb',
                         'market_cap', 'dividend_yield']].copy()
        fin_df = fin_df.rename(columns={'pe_ttm': 'pe'})
        codes_str = "','".join(codes)
        report_date = self.get_latest_report_date(as_of_date)
        yoy_date = f'{int(report_date[:4]) - 1}{report_date[4:]}'

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

    def get_ohlcv(self, codes, start_date, end_date):
        """获取OHLCV数据用于技术因子计算"""
        codes_str = "','".join(codes)
        query = f"""
            SELECT toString(date) as date, code,
                   open, high, low, close, vol, amount
            FROM default.stock_data_qfq
            WHERE code IN ('{codes_str}')
              AND date >= '{start_date}' AND date <= '{end_date}'
            ORDER BY code, date
            LIMIT 10000000
        """
        result = self.client.query(query)
        df = pd.DataFrame(result.result_rows,
            columns=['date', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount'])
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.drop_duplicates(subset=['date', 'code'], keep='last')
        logger.info(f"OHLCV数据: {len(df)}行, {df['code'].nunique()}只股票, "
                    f"{df['date'].min()} ~ {df['date'].max()}")
        return df

    def get_ohlcv_amount(self, codes, start_date, end_date):
        """获取OHLCV+amount数据用于ExtendedFactorCalculator"""
        return self.get_ohlcv(codes, start_date, end_date)


# ============================================================================
# 因子打分 (复用v3)
# ============================================================================

def pct_score(s, ascending=True):
    return s.rank(pct=True) * 100 if ascending else (1 - s.rank(pct=True)) * 100


def score_all_factors(df):
    """对16个基本面因子打分"""
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
    comp = pd.Series(0.0, index=idx)
    total_w = 0
    for factor, weight in weights.items():
        if factor in factor_scores:
            comp += factor_scores[factor].fillna(50) * weight
            total_w += weight
    if total_w > 0 and abs(total_w - 1.0) > 0.01:
        comp = comp / total_w
    return comp


# ============================================================================
# FactorPipeline — 编排所有因子计算
# ============================================================================

class FactorPipeline:
    """
    因子计算编排器
    整合 Alpha158 + Extended + 基本面因子
    输出: 每只股票在指定截面日期的因子值 DataFrame
    """

    def __init__(self):
        self.alpha158_calc = Alpha158FactorCalculator(windows=[5, 10, 20, 60])
        self.extended_calc = ExtendedFactorCalculator()

    def compute_technical_factors(self, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术因子 (Alpha158 + Extended)
        输入: OHLCV DataFrame [date, code, open, high, low, close, volume]
        输出: 每只股票最新日期的技术因子 DataFrame [code, factor1, factor2, ...]
        """
        results = {}

        # Alpha158: 返回时间序列, 取每只股票的最后一行
        try:
            alpha_df = self.alpha158_calc.calculate_all_factors(ohlcv_df)
            if len(alpha_df) > 0:
                # 取每只股票最后一行
                alpha_latest = alpha_df.sort_values('date').groupby('code').tail(1)
                alpha_cols = [c for c in alpha_latest.columns if c not in ['date', 'code']]
                for _, row in alpha_latest.iterrows():
                    code = row['code']
                    if code not in results:
                        results[code] = {}
                    for col in alpha_cols:
                        val = row[col]
                        if pd.notna(val) and np.isfinite(val):
                            results[code][f'a158_{col}'] = val
        except Exception as e:
            logger.warning(f"Alpha158因子计算失败: {e}")

        # Extended: 返回每只股票一行
        try:
            ext_df = self.extended_calc.calculate_all_factors(ohlcv_df)
            if len(ext_df) > 0:
                ext_cols = [c for c in ext_df.columns if c not in ['date', 'code']]
                for _, row in ext_df.iterrows():
                    code = row['code']
                    if code not in results:
                        results[code] = {}
                    for col in ext_cols:
                        val = row[col]
                        if pd.notna(val) and np.isfinite(val):
                            results[code][f'ext_{col}'] = val
        except Exception as e:
            logger.warning(f"Extended因子计算失败: {e}")

        if not results:
            return pd.DataFrame()

        # 转为DataFrame
        factor_df = pd.DataFrame.from_dict(results, orient='index')
        factor_df.index.name = 'code'
        factor_df = factor_df.reset_index()

        logger.info(f"技术因子: {len(factor_df)}只股票, {len(factor_df.columns)-1}个因子")
        return factor_df

    def compute_fundamental_scores(self, fetcher, codes, as_of_date):
        """
        计算基本面因子得分 (复用v3的score_all_factors)
        输出: DataFrame [code, fund_factor1, fund_factor2, ...]
        """
        factor_df = fetcher.get_financial_data(codes, as_of_date)
        mom_df = fetcher.get_momentum(codes, as_of_date)
        dd_df = fetcher.get_drawdown(codes, as_of_date)
        vol_df = fetcher.get_volatility(codes, as_of_date)
        turnover_df = fetcher.get_turnover(codes, as_of_date)

        factor_df = factor_df.merge(mom_df, on='code', how='left')
        factor_df = factor_df.merge(dd_df, on='code', how='left')
        factor_df = factor_df.merge(vol_df, on='code', how='left')
        factor_df = factor_df.merge(turnover_df, on='code', how='left')
        factor_df = factor_df.fillna(0)
        factor_df = factor_df.set_index('code')

        all_scores = score_all_factors(factor_df)

        # 将因子得分转为DataFrame
        fund_df = pd.DataFrame(index=factor_df.index)
        for name, score_series in all_scores.items():
            fund_df[f'fund_{name}'] = score_series
        fund_df = fund_df.reset_index()

        return fund_df, all_scores

    def compute_all_factors(self, fetcher, codes, as_of_date, ohlcv_df=None):
        """
        计算所有因子 (技术 + 基本面)
        返回: (合并的因子DataFrame, 基本面得分dict)
        """
        # 1. 基本面因子得分
        fund_df, fund_scores = self.compute_fundamental_scores(fetcher, codes, as_of_date)

        # 2. 技术因子
        tech_df = pd.DataFrame()
        if ohlcv_df is not None and len(ohlcv_df) > 0:
            tech_df = self.compute_technical_factors(ohlcv_df)

        # 3. 合并
        if len(tech_df) > 0:
            merged = fund_df.merge(tech_df, on='code', how='left')
        else:
            merged = fund_df

        # 填充NaN
        factor_cols = [c for c in merged.columns if c != 'code']
        for col in factor_cols:
            merged[col] = merged[col].fillna(0.0)

        return merged, fund_scores


# ============================================================================
# IC筛选器 — 滚动窗口IC因子筛选
# ============================================================================

class ICScreener:
    """基于IC的因子筛选器 (v4.2: Top-K排名替代阈值)"""

    def __init__(self, top_k=35, min_observations=2, warmup_months=2):
        self.top_k = top_k
        self.min_observations = min_observations
        self.warmup_months = warmup_months
        self.ic_history = {}  # {factor: [(date, ic_value), ...]}
        self.update_count = 0

    def update(self, factor_values_dict, forward_returns_dict, date_label):
        """
        更新IC历史
        factor_values_dict: {code: {factor_name: value}}
        forward_returns_dict: {code: forward_return}
        """
        common_codes = [c for c in forward_returns_dict if c in factor_values_dict]
        if len(common_codes) < 30:
            return

        returns_arr = np.array([forward_returns_dict[c] for c in common_codes])

        for factor_name in list(factor_values_dict[common_codes[0]].keys()):
            vals = []
            for c in common_codes:
                v = factor_values_dict[c].get(factor_name, np.nan)
                vals.append(v)
            vals = np.array(vals, dtype=float)

            # 移除NaN
            valid = ~(np.isnan(vals) | np.isnan(returns_arr))
            if valid.sum() < 20:
                continue

            try:
                ic, pval = stats.spearmanr(vals[valid], returns_arr[valid])
                if np.isnan(ic):
                    continue
            except:
                continue

            if factor_name not in self.ic_history:
                self.ic_history[factor_name] = []
            self.ic_history[factor_name].append((date_label, ic, pval))

        self.update_count += 1

    def get_selected_factors(self):
        """
        Top-K因子选择: 按|IC均值|排名选前K个
        预热期内返回空列表(由调用方决定fallback)
        """
        if self.update_count < self.warmup_months:
            return []

        # 计算每个因子的IC统计
        candidates = []
        for factor, history in self.ic_history.items():
            if len(history) < self.min_observations:
                continue

            ics = np.array([h[1] for h in history])
            ic_mean = np.mean(ics)
            ic_std = np.std(ics) if len(ics) > 1 else 1.0
            icir = ic_mean / (ic_std + 1e-8)

            candidates.append((factor, abs(ic_mean), ic_mean, icir))

        if not candidates:
            return []

        # 按 |IC均值| 降序排列，取前 top_k
        candidates.sort(key=lambda x: -x[1])
        selected = candidates[:self.top_k]

        factor_names = [s[0] for s in selected]
        logger.info(f"IC Top-K: 选中{len(factor_names)}/{len(self.ic_history)}因子 "
                    f"(|IC|范围: {selected[0][1]:.4f}~{selected[-1][1]:.4f})")
        for name, abs_ic, ic, icir in selected[:5]:
            logger.info(f"  Top: {name}: IC={ic:.4f}, |IC|={abs_ic:.4f}, ICIR={icir:.4f}")

        return factor_names

    def get_all_factor_stats(self):
        """获取所有因子的IC统计"""
        stats_list = []
        for factor, history in self.ic_history.items():
            if not history:
                continue
            ics = np.array([h[1] for h in history])
            stats_list.append({
                'factor': factor,
                'ic_mean': np.mean(ics),
                'ic_std': np.std(ics) if len(ics) > 1 else 0,
                'icir': np.mean(ics) / (np.std(ics) + 1e-8) if len(ics) > 1 else 0,
                'n_obs': len(ics),
            })
        return pd.DataFrame(stats_list)


# ============================================================================
# MLSignalGenerator — LightGBM + XGBoost 集成
# ============================================================================

class MLSignalGenerator:
    """ML信号生成器: LightGBM(60%) + XGBoost(40%) 集成"""

    def __init__(self, label_period=21):
        self.label_period = label_period
        self.lgb_model = None
        self.xgb_model = None
        self.imputer = None
        self.scaler = None
        self.feature_cols = None
        self.train_count = 0

    def prepare_training_data(self, factor_history, price_pivot, as_of_date,
                               feature_cols, gap_days=21):
        """
        准备训练数据 (expanding window, 无前视偏差)

        factor_history: list of (date_str, DataFrame[code, factor1, ...])
        price_pivot: DataFrame[date x code] close prices
        as_of_date: 当前日期 (训练数据截止到 as_of_date - gap_days)
        feature_cols: 使用的因子列
        gap_days: 预测周期 (同时用作防前视gap)

        返回: (X_train, y_train, X_val, y_val, feature_names)
        """
        cutoff = (datetime.strptime(as_of_date, '%Y-%m-%d') -
                  timedelta(days=gap_days)).strftime('%Y-%m-%d')

        all_X = []
        all_y = []

        for hist_date, factor_df in factor_history:
            if hist_date > cutoff:
                continue  # 防前视偏差

            # 计算label: 该日期后label_period天的收益率
            available_dates = sorted([d for d in price_pivot.index if d >= hist_date])
            if len(available_dates) < self.label_period + 1:
                continue

            target_date_idx = min(self.label_period, len(available_dates) - 1)
            target_date = available_dates[target_date_idx]
            base_date = available_dates[0]

            for _, row in factor_df.iterrows():
                code = row['code']
                if code not in price_pivot.columns:
                    continue

                p_base = price_pivot.loc[base_date].get(code)
                p_target = price_pivot.loc[target_date].get(code)

                if pd.isna(p_base) or pd.isna(p_target) or p_base <= 0:
                    continue

                ret = p_target / p_base - 1

                # 提取特征
                feat_vals = []
                for col in feature_cols:
                    v = row.get(col, 0.0)
                    feat_vals.append(float(v) if pd.notna(v) and np.isfinite(v) else 0.0)

                all_X.append(feat_vals)
                all_y.append(ret)

        if len(all_X) < 100:
            logger.warning(f"训练数据不足: {len(all_X)}条 (需要>=100)")
            return None, None, None, None, feature_cols

        X = np.array(all_X)
        y = np.array(all_y)

        # 划分训练/验证 (最近63天数据用于验证)
        n_val = max(int(len(X) * 0.15), 50)
        X_train, X_val = X[:-n_val], X[-n_val:]
        y_train, y_val = y[:-n_val], y[-n_val:]

        logger.info(f"训练数据: {len(X_train)}条训练, {len(X_val)}条验证, "
                    f"{len(feature_cols)}个特征, 标签范围[{y.min():.4f}, {y.max():.4f}]")

        return X_train, y_train, X_val, y_val, feature_cols

    def _adaptive_params(self, n_samples, n_features):
        """v4.2: 根据样本量自适应调整模型复杂度"""
        ratio = n_samples / max(n_features, 1)

        if ratio < 10:
            # 极少数据: 极简模型
            lgb_p = dict(LGB_PARAMS, num_leaves=8, learning_rate=0.05,
                         n_estimators=300, feature_fraction=0.4,
                         min_child_samples=50, reg_alpha=1.0, reg_lambda=5.0)
            xgb_p = dict(XGB_PARAMS, max_depth=2, learning_rate=0.05,
                         n_estimators=300, subsample=0.5, colsample_bytree=0.4,
                         min_child_weight=20, reg_alpha=1.0, reg_lambda=5.0)
            es = 30
        elif ratio < 25:
            # 中等数据: 温和模型
            lgb_p = dict(LGB_PARAMS, num_leaves=15, learning_rate=0.02,
                         n_estimators=800, feature_fraction=0.5,
                         min_child_samples=30, reg_alpha=0.5, reg_lambda=3.0)
            xgb_p = dict(XGB_PARAMS, max_depth=3, learning_rate=0.02,
                         n_estimators=800, subsample=0.6, colsample_bytree=0.5,
                         min_child_weight=10, reg_alpha=0.5, reg_lambda=3.0)
            es = 50
        else:
            # 充足数据: 用配置参数
            lgb_p = dict(LGB_PARAMS)
            xgb_p = dict(XGB_PARAMS)
            es = 100

        logger.info(f"  自适应参数: 样本={n_samples}, 特征={n_features}, "
                    f"比率={ratio:.1f}, 模式={'简' if ratio<10 else '中' if ratio<25 else '全'}")
        return lgb_p, xgb_p, es

    def train(self, X_train, y_train, X_val, y_val, feature_names):
        """训练LightGBM + XGBoost集成模型 (v4.2: 自适应复杂度)"""
        # 数据预处理
        self.imputer = SimpleImputer(strategy='median')
        X_train = self.imputer.fit_transform(X_train)
        X_val = self.imputer.transform(X_val)

        self.scaler = RobustScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)

        self.feature_cols = feature_names

        # v4.2: 自适应参数
        lgb_p, xgb_p, es_rounds = self._adaptive_params(len(X_train), X_train.shape[1])

        # LightGBM
        if HAS_LGB:
            lgb_params = {k: v for k, v in lgb_p.items()
                         if k not in ['n_estimators']}
            self.lgb_model = lgb.LGBMRegressor(
                n_estimators=lgb_p['n_estimators'], **lgb_params)

            self.lgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(es_rounds, verbose=False),
                           lgb.log_evaluation(period=200)]
            )
            lgb_pred = self.lgb_model.predict(X_val)
            lgb_ic = np.corrcoef(lgb_pred, y_val)[0, 1] if len(y_val) > 2 else 0
            logger.info(f"  LGB验证IC: {lgb_ic:.4f}, best_iter: {self.lgb_model.best_iteration_}")

        # XGBoost
        if HAS_XGB:
            xgb_params = {k: v for k, v in xgb_p.items()
                         if k not in ['n_estimators']}
            self.xgb_model = xgb.XGBRegressor(
                n_estimators=xgb_p['n_estimators'],
                early_stopping_rounds=es_rounds, **xgb_params)

            self.xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            xgb_pred = self.xgb_model.predict(X_val)
            xgb_ic = np.corrcoef(xgb_pred, y_val)[0, 1] if len(y_val) > 2 else 0
            logger.info(f"  XGB验证IC: {xgb_ic:.4f}, best_iter: {self.xgb_model.best_ntree_limit if hasattr(self.xgb_model, 'best_ntree_limit') else 'N/A'}")

        self.train_count += 1

    def predict(self, factor_df, feature_cols):
        """
        对当前截面生成ML信号
        返回: Series[code -> percentile_score (0-100)]
        """
        if self.lgb_model is None and self.xgb_model is None:
            logger.warning("模型未训练，使用等权")
            return pd.Series(50.0, index=factor_df['code'])

        # 提取特征
        X = np.zeros((len(factor_df), len(feature_cols)))
        for j, col in enumerate(feature_cols):
            if col in factor_df.columns:
                vals = factor_df[col].values.astype(float)
                X[:, j] = np.where(np.isfinite(vals), vals, 0.0)

        X = self.imputer.transform(X)
        X = self.scaler.transform(X)

        # 集成预测
        pred = np.zeros(len(X))
        n_models = 0

        if self.lgb_model is not None:
            pred += self.lgb_model.predict(X) * 0.6
            n_models += 0.6

        if self.xgb_model is not None:
            pred += self.xgb_model.predict(X) * 0.4
            n_models += 0.4

        if n_models > 0:
            pred /= n_models

        # 转为百分位排名 (0-100)
        ranks = pd.Series(pred, index=factor_df['code'].values).rank(pct=True) * 100

        return ranks

    def get_feature_importance(self, top_n=20):
        """获取特征重要性"""
        importance = {}
        if self.lgb_model is not None and hasattr(self.lgb_model, 'feature_importances_'):
            for i, col in enumerate(self.feature_cols):
                importance[col] = importance.get(col, 0) + self.lgb_model.feature_importances_[i] * 0.6
        if self.xgb_model is not None and hasattr(self.xgb_model, 'feature_importances_'):
            for i, col in enumerate(self.feature_cols):
                importance[col] = importance.get(col, 0) + self.xgb_model.feature_importances_[i] * 0.4

        sorted_imp = sorted(importance.items(), key=lambda x: -x[1])[:top_n]
        return sorted_imp


# ============================================================================
# 权重调整
# ============================================================================

def get_multiplier(percentile, tiers):
    for threshold, mult in tiers:
        if percentile >= threshold:
            return mult
    return tiers[-1][1]


def adjust_weights(portfolio, scores_series, tiers):
    """
    根据综合得分调整权重
    scores_series: Series[code -> score]
    """
    merged = portfolio.copy()
    score_df = pd.DataFrame({
        'code': scores_series.index, 'composite_score': scores_series.values})
    merged = merged.merge(score_df, on='code', how='left')
    merged['composite_score'] = merged['composite_score'].fillna(50)
    merged['percentile'] = merged['composite_score'].rank(pct=True)
    merged['multiplier'] = merged['percentile'].apply(lambda p: get_multiplier(p, tiers))
    merged['adjusted_weight'] = merged['weight'] * merged['multiplier']
    merged['adjusted_weight'] /= merged['adjusted_weight'].sum()
    return merged


def apply_turnover_damping(new_weights, prev_weights, max_turnover=0.30):
    """
    换手率阻尼: 限制单次调仓换手率不超过max_turnover
    """
    if not prev_weights:
        return new_weights

    # 计算目标换手率
    all_codes = set(list(new_weights.keys()) + list(prev_weights.keys()))
    raw_turnover = sum(
        abs(new_weights.get(c, 0) - prev_weights.get(c, 0))
        for c in all_codes
    ) / 2

    if raw_turnover <= max_turnover:
        return new_weights

    # 阻尼: 按比例缩小变化
    damping = max_turnover / raw_turnover
    damped = {}
    for c in all_codes:
        old_w = prev_weights.get(c, 0)
        new_w = new_weights.get(c, 0)
        damped[c] = old_w + (new_w - old_w) * damping

    # 归一化
    total = sum(damped.values())
    if total > 0:
        damped = {k: v / total for k, v in damped.items()}

    return damped


# ============================================================================
# 基线回测 (策略A/C, 无ML)
# ============================================================================

def run_baseline_backtest(factor_weights, tiers, rebalance_months,
                           start_date='2020-01-01', end_date='2025-12-31',
                           label='', min_market_cap=0):
    """基线回测 (复用v3逻辑), min_market_cap>0时过滤小市值"""
    fetcher = DataFetcherML()
    portfolio = fetcher.get_stock_pool()
    codes = portfolio['code'].tolist()

    freq = {1: '年度', 2: '半年', 4: '季度', 12: '月度'}.get(
        len(rebalance_months), f'{len(rebalance_months)}次/年')
    print(f"\n{'='*65}")
    print(f"[{label}] {len(factor_weights)}因子 | {freq}调仓 | "
          f"{'温和' if tiers == TIERS_MILD else '激进'}档位")
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

    original_weights = dict(zip(portfolio['code'], portfolio['weight']))
    current_weights = original_weights.copy()
    prev_weights = {}
    rb_idx = 0
    market_position = 1.0
    stop_loss_count = 0
    enhanced_returns = []
    original_returns = []
    total_tx_cost = 0.0

    for i in range(1, len(dates)):
        curr_date = dates[i]
        prev_date = dates[i - 1]
        rebalance_cost = 0.0

        while rb_idx < len(rebalance_dates) and curr_date >= rebalance_dates[rb_idx]:
            rb_date = rebalance_dates[rb_idx]

            factor_df = fetcher.get_financial_data(codes, rb_date)
            mom_df = fetcher.get_momentum(codes, rb_date)
            dd_df = fetcher.get_drawdown(codes, rb_date)
            vol_df = fetcher.get_volatility(codes, rb_date)
            turnover_df = fetcher.get_turnover(codes, rb_date)

            factor_df = factor_df.merge(mom_df, on='code', how='left')
            factor_df = factor_df.merge(dd_df, on='code', how='left')
            factor_df = factor_df.merge(vol_df, on='code', how='left')
            factor_df = factor_df.merge(turnover_df, on='code', how='left')
            factor_df = factor_df.fillna(0).set_index('code')

            # 市值过滤
            if min_market_cap > 0 and 'market_cap' in factor_df.columns:
                before = len(factor_df)
                factor_df = factor_df[factor_df['market_cap'] >= min_market_cap]
                if rb_idx == 0:
                    print(f"  市值过滤: {before} → {len(factor_df)} (>={min_market_cap}亿)")

            if len(factor_df) > 0:
                all_scores = score_all_factors(factor_df)
                comp = composite_score(all_scores, factor_weights)
                comp.index = factor_df.index

                adjusted = adjust_weights(
                    portfolio[portfolio['code'].isin(factor_df.index)],
                    comp, tiers)

                prev_weights = current_weights.copy()
                current_weights = dict(zip(adjusted['code'], adjusted['adjusted_weight']))

                turnover_ratio = sum(
                    abs(current_weights.get(c, 0) - prev_weights.get(c, 0))
                    for c in set(list(current_weights.keys()) + list(prev_weights.keys()))
                ) / 2
                rebalance_cost = (turnover_ratio * TRANSACTION_COSTS['sell_commission'] +
                                  turnover_ratio * TRANSACTION_COSTS['buy_commission'])
                total_tx_cost += rebalance_cost

                print(f"  [{rb_date}] 换手:{turnover_ratio:.1%} | 成本:{rebalance_cost:.4%}")

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
    enh = calc_performance(enhanced_returns)
    orig = calc_performance(original_returns)

    print(f"\n  年化: {enh['annual_return']*100:.2f}% | 回撤: {enh['max_drawdown']*100:.2f}% | "
          f"夏普: {enh['sharpe']:.2f} | 卡尔玛: {enh['calmar']:.2f} | "
          f"成本: {total_tx_cost*100:.2f}%")

    return {
        'label': label, 'enhanced': enh, 'original': orig,
        'total_tx_cost': total_tx_cost, 'stop_loss_count': stop_loss_count,
    }


# ============================================================================
# ML回测 (策略F/G)
# ============================================================================

def run_ml_backtest(tiers, max_turnover=0.30,
                     rebalance_months=None,
                     start_date='2020-01-01', end_date='2025-12-31',
                     label='', min_market_cap=0):
    """ML增强调仓回测, rebalance_months控制调仓频率(默认月度)"""
    if rebalance_months is None:
        rebalance_months = list(range(1, 13))  # 月度

    fetcher = DataFetcherML()
    portfolio = fetcher.get_stock_pool()
    codes = portfolio['code'].tolist()

    freq = {1: '年度', 2: '半年', 4: '季度', 12: '月度'}.get(
        len(rebalance_months), f'{len(rebalance_months)}次/年')
    tiers_name = '温和' if tiers == TIERS_MILD else '激进'
    mcap_str = f' | 市值>={min_market_cap}亿' if min_market_cap > 0 else ''
    print(f"\n{'='*65}")
    print(f"[{label}] ML增强 | {freq}调仓 | {tiers_name}档位 | 换手率≤{max_turnover:.0%}{mcap_str}")
    print(f"{'='*65}")

    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    # 调仓日期
    rebalance_dates = []
    for y in range(start_dt.year, end_dt.year + 1):
        for m in rebalance_months:
            rd = f'{y}-{m:02d}-01'
            if start_date <= rd <= end_date:
                rebalance_dates.append(rd)

    # 预加载价格数据
    # 需要额外2年历史数据用于: 技术因子(60天) + IC历史(2年) + ML训练
    train_start = (start_dt - timedelta(days=800)).strftime('%Y-%m-%d')
    extended_start = (start_dt - timedelta(days=400)).strftime('%Y-%m-%d')
    prices = fetcher.get_prices(codes, train_start, end_date)
    price_pivot = prices.pivot(index='date', columns='code', values='close')
    index_prices = fetcher.get_index_prices(extended_start, end_date)

    # 预加载OHLCV (用于技术因子)
    print(f"  加载OHLCV数据...")
    ohlcv_all = fetcher.get_ohlcv(codes, train_start, end_date)

    dates = sorted([d for d in price_pivot.index if d >= start_date])
    trading_days = sorted(price_pivot.index.tolist())

    def nearest_trading_day(date_str, direction='backward'):
        """找到最近的交易日 (backward=往前找, forward=往后找)"""
        if date_str in trading_days:
            return date_str
        if direction == 'backward':
            candidates = [d for d in trading_days if d <= date_str]
            return candidates[-1] if candidates else None
        else:
            candidates = [d for d in trading_days if d >= date_str]
            return candidates[0] if candidates else None

    # 初始化组件
    pipeline = FactorPipeline()
    ic_screener = ICScreener(**IC_CONFIG)
    ml_signal = MLSignalGenerator(label_period=21)

    original_weights = dict(zip(portfolio['code'], portfolio['weight']))
    current_weights = original_weights.copy()
    prev_weights = {}
    rb_idx = 0
    market_position = 1.0
    stop_loss_count = 0
    enhanced_returns = []
    original_returns = []
    total_tx_cost = 0.0

    # 因子历史 (用于ML训练)
    factor_history = []
    prev_factor_df = None
    prev_rb_date = None

    for i in range(1, len(dates)):
        curr_date = dates[i]
        prev_date = dates[i - 1]
        rebalance_cost = 0.0

        while rb_idx < len(rebalance_dates) and curr_date >= rebalance_dates[rb_idx]:
            rb_date = rebalance_dates[rb_idx]
            rb_dt = datetime.strptime(rb_date, '%Y-%m-%d')

            try:
                # === Step 1: 计算技术因子 ===
                # 取调仓日之前120天的OHLCV用于技术因子
                tech_start = (rb_dt - timedelta(days=120)).strftime('%Y-%m-%d')
                ohlcv_slice = ohlcv_all[
                    (ohlcv_all['date'] >= tech_start) & (ohlcv_all['date'] <= rb_date)
                ].copy()

                # === Step 2: 市值过滤 + 计算所有因子 ===
                active_codes = codes
                if min_market_cap > 0:
                    val_df = fetcher.load_valuation_csv()
                    cap_df = val_df[val_df['code'].isin(codes)][['code', 'market_cap']]
                    active_codes = cap_df[cap_df['market_cap'] >= min_market_cap]['code'].tolist()
                    if rb_idx == 0:
                        print(f"  市值过滤: {len(codes)} → {len(active_codes)} (>={min_market_cap}亿)")

                factor_df, fund_scores = pipeline.compute_all_factors(
                    fetcher, active_codes, rb_date, ohlcv_slice)

                if len(factor_df) == 0:
                    rb_idx += 1
                    continue

                factor_cols = [c for c in factor_df.columns if c != 'code']

                # === Step 3: 更新IC历史 ===
                if prev_factor_df is not None and prev_rb_date is not None:
                    # 计算上期到本期的实际收益
                    # v4.2 fix: 用最近交易日替代精确日期匹配
                    td_prev = nearest_trading_day(prev_rb_date, 'forward')
                    td_curr = nearest_trading_day(rb_date, 'forward')
                    returns_dict = {}
                    for code in codes:
                        if (td_prev and td_curr and
                            td_prev in price_pivot.index and
                            td_curr in price_pivot.index and
                            code in price_pivot.columns):
                            p1 = price_pivot.loc[td_prev].get(code)
                            p2 = price_pivot.loc[td_curr].get(code)
                            if pd.notna(p1) and pd.notna(p2) and p1 > 0:
                                returns_dict[code] = p2 / p1 - 1

                    if returns_dict:
                        # 构建因子值字典
                        fv_dict = {}
                        for _, row in prev_factor_df.iterrows():
                            c = row['code']
                            fv_dict[c] = {col: row[col] for col in
                                         [cc for cc in prev_factor_df.columns if cc != 'code']}
                        ic_screener.update(fv_dict, returns_dict, prev_rb_date)

                # 保存当前因子到历史
                factor_history.append((rb_date, factor_df.copy()))
                # 控制历史长度 (保留最近36个月)
                if len(factor_history) > 36:
                    factor_history = factor_history[-36:]

                prev_factor_df = factor_df.copy()
                prev_rb_date = rb_date

                # === Step 4: IC Top-K因子选择 ===
                selected_factors = ic_screener.get_selected_factors()

                # 预热期或选择不足时: 用全部因子
                if len(selected_factors) < 5:
                    selected_factors = factor_cols
                    logger.info(f"IC预热中, 使用全部{len(factor_cols)}个因子")

                # === Step 5: ML训练与预测 ===
                ml_trained = False
                if len(factor_history) >= 3:  # 至少3个月历史
                    try:
                        X_train, y_train, X_val, y_val, feat_names = \
                            ml_signal.prepare_training_data(
                                factor_history, price_pivot, rb_date,
                                selected_factors, gap_days=21)

                        if X_train is not None and len(X_train) >= 100:
                            ml_signal.train(X_train, y_train, X_val, y_val, feat_names)
                            ml_trained = True
                    except Exception as e:
                        logger.warning(f"ML训练失败: {e}")

                # === Step 6: 生成融合信号 ===
                # v4.2: 始终计算基本面得分，ML信号与基本面融合
                fund_comp = composite_score(fund_scores, FACTORS_V3_R5)
                # 转为百分位排名
                fund_rank = fund_comp.rank(pct=True) * 100

                if ml_trained:
                    # ML信号: 百分位得分
                    ml_rank = ml_signal.predict(factor_df, selected_factors)
                    # 融合: 60% ML + 40% 基本面 (保底基本面alpha)
                    ml_scores = pd.Series(index=fund_rank.index, dtype=float)
                    common = fund_rank.index.intersection(ml_rank.index)
                    ml_scores[common] = ml_rank[common] * 0.6 + fund_rank[common] * 0.4
                    # ML无覆盖的用纯基本面
                    ml_only = fund_rank.index.difference(ml_rank.index)
                    ml_scores[ml_only] = fund_rank[ml_only]
                else:
                    ml_scores = fund_rank

                # 权重调整
                adjusted = adjust_weights(
                    portfolio[portfolio['code'].isin(factor_df['code'])],
                    ml_scores, tiers)

                new_weights = dict(zip(adjusted['code'], adjusted['adjusted_weight']))

                # 换手率阻尼
                damped_weights = apply_turnover_damping(
                    new_weights, current_weights, max_turnover)

                prev_weights = current_weights.copy()
                current_weights = damped_weights

                # 交易成本
                turnover_ratio = sum(
                    abs(current_weights.get(c, 0) - prev_weights.get(c, 0))
                    for c in set(list(current_weights.keys()) + list(prev_weights.keys()))
                ) / 2
                rebalance_cost = (turnover_ratio * TRANSACTION_COSTS['sell_commission'] +
                                  turnover_ratio * TRANSACTION_COSTS['buy_commission'])
                total_tx_cost += rebalance_cost

                status = 'ML' if ml_trained else 'FUND'
                n_factors = len(selected_factors)
                print(f"  [{rb_date}] {status} | 因子:{n_factors} | "
                      f"换手:{turnover_ratio:.1%} | 成本:{rebalance_cost:.4%}")

                # 显示特征重要性 (每6个月一次)
                if ml_trained and ml_signal.train_count % 6 == 1:
                    imp = ml_signal.get_feature_importance(5)
                    if imp:
                        top_str = ', '.join(f'{name}={val:.0f}' for name, val in imp)
                        print(f"         Top特征: {top_str}")

            except Exception as e:
                logger.error(f"调仓{rb_date}失败: {e}")
                traceback.print_exc()

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
    enh = calc_performance(enhanced_returns)
    orig = calc_performance(original_returns)

    print(f"\n  年化: {enh['annual_return']*100:.2f}% | 回撤: {enh['max_drawdown']*100:.2f}% | "
          f"夏普: {enh['sharpe']:.2f} | 卡尔玛: {enh['calmar']:.2f} | "
          f"成本: {total_tx_cost*100:.2f}%")

    # IC统计
    ic_stats = ic_screener.get_all_factor_stats()
    if len(ic_stats) > 0:
        n_positive = (ic_stats['ic_mean'] > 0).sum()
        print(f"  IC统计: {len(ic_stats)}因子, {n_positive}正向, "
              f"平均|IC|={ic_stats['ic_mean'].abs().mean():.4f}")

    return {
        'label': label, 'enhanced': enh, 'original': orig,
        'total_tx_cost': total_tx_cost, 'stop_loss_count': stop_loss_count,
        'ml_train_count': ml_signal.train_count,
        'ic_stats': ic_stats.to_dict('records') if len(ic_stats) > 0 else [],
    }


# ============================================================================
# 绩效计算
# ============================================================================

def calc_performance(returns):
    if not returns:
        return dict(annual_return=0, total_return=0, max_drawdown=0,
                    sharpe=0, calmar=0, win_rate=0, final_nav=1)
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


# ============================================================================
# v4.4: Bayesian优化因子权重 (Optuna + Walk-Forward)
# ============================================================================

# V1因子名称列表 (与FACTORS_V1_R5顺序一致)
FACTOR_NAMES_V1 = list(FACTORS_V1_R5.keys())

# Walk-Forward 4折定义
WALK_FORWARD_FOLDS = [
    {'train_start': '2018-01-01', 'train_end': '2020-12-31',
     'test_start': '2021-01-01', 'test_end': '2021-12-31', 'name': 'Fold1(2021)'},
    {'train_start': '2018-01-01', 'train_end': '2021-12-31',
     'test_start': '2022-01-01', 'test_end': '2022-12-31', 'name': 'Fold2(2022)'},
    {'train_start': '2018-01-01', 'train_end': '2022-12-31',
     'test_start': '2023-01-01', 'test_end': '2023-12-31', 'name': 'Fold3(2023)'},
    {'train_start': '2018-01-01', 'train_end': '2023-12-31',
     'test_start': '2024-01-01', 'test_end': '2024-12-31', 'name': 'Fold4(2024)'},
]


def _build_tiers(top_mult, bottom_mult):
    """从顶层和底层乘数线性插值生成8档分层乘数"""
    n = 8
    thresholds = [0.95, 0.90, 0.80, 0.60, 0.40, 0.20, 0.10, 0.00]
    mults = [top_mult + (bottom_mult - top_mult) * i / (n - 1) for i in range(n)]
    return list(zip(thresholds, mults))


def _rebalance_months_from_freq(freq):
    """调仓频率字符串转月份列表"""
    if freq == 'semi':
        return [1, 7]
    elif freq == 'quarterly':
        return [1, 4, 7, 10]
    elif freq == 'annual':
        return [1]
    return [1, 7]


class OptunaTuner:
    """Optuna贝叶斯优化器: 优化因子权重 + 分层乘数 + 超参数"""

    def __init__(self, n_trials=200, timeout=7200, seed=42):
        if not HAS_OPTUNA:
            raise ImportError("请安装optuna: pip install optuna")
        self.n_trials = n_trials
        self.timeout = timeout
        self.seed = seed
        self.folds = WALK_FORWARD_FOLDS
        self.factor_names = FACTOR_NAMES_V1

    def objective(self, trial):
        """Optuna目标函数: Walk-Forward 4折平均得分"""
        # 1. 采样因子权重 (10维, 归一化到simplex)
        raw_weights = []
        for i, name in enumerate(self.factor_names):
            w = trial.suggest_float(f'w_{name}', 0.01, 1.0)
            raw_weights.append(w)
        w_sum = sum(raw_weights)
        norm_weights = [w / w_sum for w in raw_weights]
        factor_weights = dict(zip(self.factor_names, norm_weights))

        # 2. 采样分层乘数 (2维 → 插值8档, 单调递减)
        top_mult = trial.suggest_float('top_mult', 1.2, 3.0)
        bottom_mult = trial.suggest_float('bottom_mult', 0.1, 0.8)
        tiers = _build_tiers(top_mult, bottom_mult)

        # 3. 采样超参数
        rebal_freq = trial.suggest_categorical('rebalance', ['semi', 'quarterly'])
        rebalance_months = _rebalance_months_from_freq(rebal_freq)
        min_cap = trial.suggest_int('min_cap', 30, 100, step=10)
        ma_period = trial.suggest_categorical('ma_period', [120, 200, 250])

        # 暂时修改全局MA止损周期
        original_ma = MARKET_STOP_LOSS['ma_period']
        MARKET_STOP_LOSS['ma_period'] = ma_period

        # 4. Walk-Forward 4折验证
        fold_scores = []
        for fold_idx, fold in enumerate(self.folds):
            try:
                result = run_baseline_backtest(
                    factor_weights=factor_weights,
                    tiers=tiers,
                    rebalance_months=rebalance_months,
                    start_date=fold['test_start'],
                    end_date=fold['test_end'],
                    label=f'Trial{trial.number}-{fold["name"]}',
                    min_market_cap=min_cap,
                )
                sharpe = result['enhanced']['sharpe']
                annual_ret = result['enhanced']['annual_return']
                score = 0.7 * sharpe + 0.3 * annual_ret
                fold_scores.append(score)
            except Exception as e:
                logger.warning(f"Trial {trial.number} {fold['name']} 失败: {e}")
                fold_scores.append(-1.0)

            # Pruning: 报告中间结果，允许提前终止差的试验
            trial.report(np.mean(fold_scores), step=fold_idx)
            if trial.should_prune():
                MARKET_STOP_LOSS['ma_period'] = original_ma
                raise optuna.TrialPruned()

        MARKET_STOP_LOSS['ma_period'] = original_ma
        return np.mean(fold_scores)

    def optimize(self):
        """运行优化，返回最优参数"""
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
            study_name='factor_weight_optimization_v4.4',
        )

        # 添加V1C当前参数作为初始试验 (确保不会比基线差)
        v1c_params = {}
        for name in self.factor_names:
            v1c_params[f'w_{name}'] = FACTORS_V1_R5[name]
        v1c_params['top_mult'] = 1.50
        v1c_params['bottom_mult'] = 0.50
        v1c_params['rebalance'] = 'semi'
        v1c_params['min_cap'] = 50
        v1c_params['ma_period'] = 250
        study.enqueue_trial(v1c_params)

        logger.info(f"开始Optuna优化: {self.n_trials}次试验, 超时{self.timeout}秒")
        logger.info(f"参数空间: {len(self.factor_names)}因子权重 + 2分层 + 3超参数 = "
                     f"{len(self.factor_names) + 5}维")

        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True,
        )

        logger.info(f"优化完成: {len(study.trials)}次试验, "
                     f"最优得分={study.best_value:.4f}")
        return study

    def extract_best_params(self, study):
        """从Study中提取最优参数为可用格式"""
        bp = study.best_params

        # 因子权重 (归一化)
        raw = [bp[f'w_{name}'] for name in self.factor_names]
        w_sum = sum(raw)
        factor_weights = {name: raw[i] / w_sum
                          for i, name in enumerate(self.factor_names)}

        # 分层乘数
        tiers = _build_tiers(bp['top_mult'], bp['bottom_mult'])

        # 超参数
        rebalance_months = _rebalance_months_from_freq(bp['rebalance'])

        return {
            'factor_weights': factor_weights,
            'tiers': tiers,
            'rebalance_months': rebalance_months,
            'min_market_cap': bp['min_cap'],
            'ma_period': bp['ma_period'],
            'top_mult': bp['top_mult'],
            'bottom_mult': bp['bottom_mult'],
            'rebalance_freq': bp['rebalance'],
        }

    def validate_params(self, params):
        """验证最优参数的合理性"""
        issues = []
        # 检查因子权重无极端值
        for name, w in params['factor_weights'].items():
            if w > 0.4:
                issues.append(f"因子{name}权重过高: {w:.3f} > 0.4")
            if w < 0.01:
                issues.append(f"因子{name}权重过低: {w:.3f} < 0.01")
        # 检查分层乘数合理
        if params['top_mult'] > 2.5:
            issues.append(f"顶层乘数偏高: {params['top_mult']:.2f}")
        if params['bottom_mult'] < 0.15:
            issues.append(f"底层乘数偏低: {params['bottom_mult']:.2f}")
        return issues

    def analyze_top_trials(self, study, top_n=10):
        """分析top-N试验参数分布，检查稳定性"""
        trials = sorted(study.trials,
                        key=lambda t: t.value if t.value is not None else -999,
                        reverse=True)[:top_n]
        param_stats = {}
        for key in trials[0].params:
            vals = [t.params[key] for t in trials
                    if t.value is not None and key in t.params]
            if vals and isinstance(vals[0], (int, float)):
                param_stats[key] = {
                    'mean': np.mean(vals),
                    'std': np.std(vals),
                    'cv': np.std(vals) / np.mean(vals) if np.mean(vals) != 0 else 0,
                }
        return param_stats


def run_optuna_optimization(n_trials=200, timeout=7200):
    """v4.4入口: Optuna贝叶斯优化因子权重"""
    if not HAS_OPTUNA:
        print("错误: 未安装optuna，请运行: pip install optuna")
        return

    print("=" * 65)
    print("多因子策略 v4.4 — Bayesian优化因子权重 + Walk-Forward验证")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"试验次数: {n_trials} | 超时: {timeout}秒")
    print("=" * 65)

    tuner = OptunaTuner(n_trials=n_trials, timeout=timeout)

    # === Phase 1: 运行V1C基线对比 ===
    print("\n>>> Phase 1: V1C基线回测...")
    baseline = run_baseline_backtest(
        FACTORS_V1_R5, TIERS_MILD, [1, 7],
        label='V1C 基线', start_date='2020-01-01', end_date='2025-12-31',
        min_market_cap=MIN_MARKET_CAP)
    baseline_sharpe = baseline['enhanced']['sharpe']
    baseline_return = baseline['enhanced']['annual_return']
    print(f"  V1C基线: 年化={baseline_return*100:.2f}%, 夏普={baseline_sharpe:.2f}")

    # === Phase 2: Optuna优化 ===
    print(f"\n>>> Phase 2: Optuna优化 ({n_trials}次试验)...")
    study = tuner.optimize()

    # === Phase 3: 提取最优参数 ===
    best_params = tuner.extract_best_params(study)

    print(f"\n>>> Phase 3: 最优参数")
    print(f"  因子权重:")
    for name, w in sorted(best_params['factor_weights'].items(),
                           key=lambda x: -x[1]):
        orig = FACTORS_V1_R5.get(name, 0)
        delta = w - orig
        print(f"    {name:<20} {w:.4f}  (原{orig:.2f}, {'+'if delta>=0 else ''}{delta:.4f})")
    print(f"  分层: top={best_params['top_mult']:.2f}, "
          f"bottom={best_params['bottom_mult']:.2f}")
    print(f"  调仓: {best_params['rebalance_freq']}")
    print(f"  市值过滤: >={best_params['min_market_cap']}亿")
    print(f"  MA止损: {best_params['ma_period']}日")

    # 参数合理性检查
    issues = tuner.validate_params(best_params)
    if issues:
        print(f"\n  参数警告:")
        for issue in issues:
            print(f"    ! {issue}")

    # Top-10参数稳定性
    param_stats = tuner.analyze_top_trials(study)
    high_cv_params = {k: v for k, v in param_stats.items() if v['cv'] > 0.5}
    if high_cv_params:
        print(f"\n  参数稳定性警告 (CV>0.5):")
        for k, v in high_cv_params.items():
            print(f"    {k}: CV={v['cv']:.2f} (mean={v['mean']:.3f}, std={v['std']:.3f})")

    # === Phase 4: 全量回测 ===
    print(f"\n>>> Phase 4: 最优参数全量回测 (2020-2025)...")
    original_ma = MARKET_STOP_LOSS['ma_period']
    MARKET_STOP_LOSS['ma_period'] = best_params['ma_period']

    optimized = run_baseline_backtest(
        factor_weights=best_params['factor_weights'],
        tiers=best_params['tiers'],
        rebalance_months=best_params['rebalance_months'],
        start_date='2020-01-01', end_date='2025-12-31',
        label='OPT 优化后',
        min_market_cap=best_params['min_market_cap'],
    )

    MARKET_STOP_LOSS['ma_period'] = original_ma

    # === Phase 5: Walk-Forward各折详情 ===
    print(f"\n>>> Phase 5: Walk-Forward各折验证...")
    fold_results = []
    MARKET_STOP_LOSS['ma_period'] = best_params['ma_period']
    for fold in WALK_FORWARD_FOLDS:
        fr = run_baseline_backtest(
            factor_weights=best_params['factor_weights'],
            tiers=best_params['tiers'],
            rebalance_months=best_params['rebalance_months'],
            start_date=fold['test_start'], end_date=fold['test_end'],
            label=f'OPT-{fold["name"]}',
            min_market_cap=best_params['min_market_cap'],
        )
        fold_results.append({
            'fold': fold['name'],
            'sharpe': fr['enhanced']['sharpe'],
            'annual_return': fr['enhanced']['annual_return'],
            'max_drawdown': fr['enhanced']['max_drawdown'],
        })
    MARKET_STOP_LOSS['ma_period'] = original_ma

    print(f"\n  Walk-Forward各折结果:")
    print(f"  {'折':>12} {'年化':>8} {'回撤':>8} {'夏普':>6}")
    all_sharpes_ok = True
    for fr in fold_results:
        flag = '✓' if fr['sharpe'] > 0.9 else '✗'
        if fr['sharpe'] <= 0.9:
            all_sharpes_ok = False
        print(f"  {fr['fold']:>12} {fr['annual_return']*100:>7.2f}% "
              f"{fr['max_drawdown']*100:>7.2f}% {fr['sharpe']:>5.2f} {flag}")

    # === 汇总对比 ===
    opt_e = optimized['enhanced']
    base_e = baseline['enhanced']

    print(f"\n{'='*65}")
    print("v4.4 优化结果对比")
    print(f"{'='*65}")
    print(f"  {'指标':<16} {'V1C基线':>12} {'优化后':>12} {'变化':>10}")
    print(f"  {'-'*52}")
    print(f"  {'年化收益':<16} {base_e['annual_return']*100:>11.2f}% "
          f"{opt_e['annual_return']*100:>11.2f}% "
          f"{(opt_e['annual_return']-base_e['annual_return'])*100:>+9.2f}%")
    print(f"  {'夏普比率':<16} {base_e['sharpe']:>12.2f} {opt_e['sharpe']:>12.2f} "
          f"{opt_e['sharpe']-base_e['sharpe']:>+10.2f}")
    print(f"  {'最大回撤':<16} {base_e['max_drawdown']*100:>11.2f}% "
          f"{opt_e['max_drawdown']*100:>11.2f}% "
          f"{(opt_e['max_drawdown']-base_e['max_drawdown'])*100:>+9.2f}%")
    print(f"  {'卡尔玛比率':<16} {base_e['calmar']:>12.2f} {opt_e['calmar']:>12.2f} "
          f"{opt_e['calmar']-base_e['calmar']:>+10.2f}")
    print(f"  {'交易成本':<16} {baseline['total_tx_cost']*100:>11.2f}% "
          f"{optimized['total_tx_cost']*100:>11.2f}%")

    # 验证标准
    print(f"\n  验证标准:")
    print(f"  各折Sharpe > 0.9: {'✓ 通过' if all_sharpes_ok else '✗ 未通过'}")
    print(f"  优于V1C基线:      {'✓ 通过' if opt_e['annual_return'] > base_e['annual_return'] else '✗ 未通过'}")
    max_w = max(best_params['factor_weights'].values())
    print(f"  无极端权重(<0.4): {'✓ 通过' if max_w < 0.4 else '✗ 未通过'} (最大={max_w:.3f})")

    # === 保存结果 ===
    output = {
        'version': 'v4.4',
        'run_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'optimization': {
            'method': 'Optuna TPE Bayesian',
            'n_trials': len(study.trials),
            'best_score': study.best_value,
            'n_pruned': len([t for t in study.trials
                             if t.state == optuna.trial.TrialState.PRUNED]),
        },
        'best_params': {
            'factor_weights': best_params['factor_weights'],
            'top_mult': best_params['top_mult'],
            'bottom_mult': best_params['bottom_mult'],
            'rebalance_freq': best_params['rebalance_freq'],
            'min_market_cap': best_params['min_market_cap'],
            'ma_period': best_params['ma_period'],
        },
        'baseline_v1c': {
            'annual_return': base_e['annual_return'],
            'sharpe': base_e['sharpe'],
            'max_drawdown': base_e['max_drawdown'],
            'calmar': base_e['calmar'],
        },
        'optimized': {
            'annual_return': opt_e['annual_return'],
            'sharpe': opt_e['sharpe'],
            'max_drawdown': opt_e['max_drawdown'],
            'calmar': opt_e['calmar'],
            'total_tx_cost': optimized['total_tx_cost'],
        },
        'walk_forward_folds': fold_results,
        'param_stability': {k: {kk: round(vv, 4) for kk, vv in v.items()}
                            for k, v in param_stats.items()},
        'validation_issues': issues,
    }

    output_path = os.path.join(BASE_DIR, 'factor_backtest_result_v4.4.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存到 {output_path}")

    return output


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 65)
    print("多因子策略 v4.3 — V1基线恢复 + 季度ML + 市值过滤")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"LightGBM: {'可用' if HAS_LGB else '不可用'}")
    print(f"XGBoost: {'可用' if HAS_XGB else '不可用'}")
    print(f"MIN_MARKET_CAP: {MIN_MARKET_CAP}亿")
    print("=" * 65)

    if not HAS_LGB and not HAS_XGB:
        print("错误: LightGBM和XGBoost都未安装，无法运行ML策略")
        print("请运行: pip install lightgbm xgboost")
        return

    sd, ed = '2020-01-01', '2025-12-31'
    results = {}

    # 策略V1: V1基线 — 10集中因子 + 温和档位 + 年度调仓 + 市值过滤
    print("\n\n>>> 运行策略V1 (V1基线-年度-市值过滤)...")
    results['V1'] = run_baseline_backtest(
        FACTORS_V1_R5, TIERS_MILD, [1],
        label='V1 基线(10因子+年度)', start_date=sd, end_date=ed,
        min_market_cap=MIN_MARKET_CAP)

    # 策略C: 基线 — 16因子 + 温和档位 + 半年调仓 (参考)
    print("\n\n>>> 运行策略C (V3基线-半年)...")
    results['C'] = run_baseline_backtest(
        FACTORS_V3_R5, TIERS_MILD, [1, 7],
        label='C V3基线(16因子+半年)', start_date=sd, end_date=ed)

    # 策略V1C: V1因子 + 半年调仓 + 市值过滤
    print("\n\n>>> 运行策略V1C (V1因子+半年+市值过滤)...")
    results['V1C'] = run_baseline_backtest(
        FACTORS_V1_R5, TIERS_MILD, [1, 7],
        label='V1C 基线(10因子+半年)', start_date=sd, end_date=ed,
        min_market_cap=MIN_MARKET_CAP)

    # 策略H: ML季度温和 — IC筛选因子 + 温和档位 + 季度调仓 + 市值过滤
    print("\n\n>>> 运行策略H (ML季度温和)...")
    results['H'] = run_ml_backtest(
        TIERS_MILD, max_turnover=0.30,
        rebalance_months=[1, 4, 7, 10],
        label='H ML季度温和', start_date=sd, end_date=ed,
        min_market_cap=MIN_MARKET_CAP)

    # 策略I: ML季度激进 — IC筛选因子 + 激进档位 + 季度调仓 + 市值过滤
    print("\n\n>>> 运行策略I (ML季度激进)...")
    results['I'] = run_ml_backtest(
        TIERS_ML_AGGRESSIVE, max_turnover=0.25,
        rebalance_months=[1, 4, 7, 10],
        label='I ML季度激进', start_date=sd, end_date=ed,
        min_market_cap=MIN_MARKET_CAP)

    # ============================================================================
    # 汇总对比
    # ============================================================================
    print(f"\n\n{'='*90}")
    print("完整对比表 (所有策略含交易成本)")
    print(f"{'='*90}")
    print(f"  {'策略':<32} {'年化':>8} {'回撤':>8} {'夏普':>6} {'卡尔玛':>7} {'成本':>7}")
    print(f"  {'-'*76}")

    for key in ['V1', 'C', 'V1C', 'H', 'I']:
        r = results[key]
        e = r['enhanced']
        extra = ''
        if 'ml_train_count' in r:
            extra = f" [ML训练{r['ml_train_count']}次]"
        print(f"  {r['label']:<32} {e['annual_return']*100:>7.2f}% {e['max_drawdown']*100:>7.2f}% "
              f"{e['sharpe']:>6.2f} {e['calmar']:>7.2f} {r['total_tx_cost']*100:>6.2f}%{extra}")

    print(f"  {'-'*76}")
    print(f"  v3最优R5-C参考                  16.69%  -22.07%   1.07     ---      0.08%")
    print(f"  v1最优R5-enhanced参考            24.38%  -22.92%   1.24     ---      ---")

    # 找最优
    print(f"\n  最优策略:")
    best_sharpe = max(results.keys(), key=lambda k: results[k]['enhanced']['sharpe'])
    best_return = max(results.keys(), key=lambda k: results[k]['enhanced']['annual_return'])
    best_calmar = max(results.keys(), key=lambda k: results[k]['enhanced']['calmar'])
    print(f"  最高夏普: {results[best_sharpe]['label']} ({results[best_sharpe]['enhanced']['sharpe']:.2f})")
    print(f"  最高收益: {results[best_return]['label']} ({results[best_return]['enhanced']['annual_return']*100:.2f}%)")
    print(f"  最高卡尔玛: {results[best_calmar]['label']} ({results[best_calmar]['enhanced']['calmar']:.2f})")

    # 验收检查 (对最高收益策略)
    best_r = results[best_return]['enhanced']
    print(f"\n  验收标准检查 (策略{best_return}):")
    print(f"  年化收益 ≥ 30%: {best_r['annual_return']*100:.2f}% {'✓' if best_r['annual_return'] >= 0.30 else '✗'}")
    print(f"  夏普比率 ≥ 1.2: {best_r['sharpe']:.2f} {'✓' if best_r['sharpe'] >= 1.2 else '✗'}")
    print(f"  最大回撤 ≥ -25%: {best_r['max_drawdown']*100:.2f}% {'✓' if best_r['max_drawdown'] >= -0.25 else '✗'}")

    # 保存结果
    output_path = os.path.join(BASE_DIR, 'factor_backtest_result_v4.json')

    # 清理不可序列化的数据
    serializable_results = {}
    for k, v in results.items():
        sv = {key: val for key, val in v.items() if key != 'ic_stats'}
        if 'ic_stats' in v and v['ic_stats']:
            sv['ic_stats_count'] = len(v['ic_stats'])
        serializable_results[k] = sv

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'version': 'v4.3',
            'run_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'config': {
                'lgb_params': LGB_PARAMS,
                'xgb_params': XGB_PARAMS,
                'ic_config': IC_CONFIG,
                'min_market_cap': MIN_MARKET_CAP,
                'label_period': 21,
            },
            'experiments': serializable_results,
        }, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存到 {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='多因子策略回测 v4.3/v4.4')
    parser.add_argument('--optimize', action='store_true',
                        help='运行v4.4 Optuna贝叶斯优化')
    parser.add_argument('--n-trials', type=int, default=200,
                        help='Optuna试验次数 (默认200)')
    parser.add_argument('--timeout', type=int, default=7200,
                        help='Optuna超时秒数 (默认7200)')
    args = parser.parse_args()

    if args.optimize:
        run_optuna_optimization(n_trials=args.n_trials, timeout=args.timeout)
    else:
        main()
