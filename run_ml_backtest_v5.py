#!/usr/bin/env python
"""
ML驱动选股回测 v5.0 — 目标30%+年化

核心改进 vs v4.3:
  1. 截面排名标签 (消除市场beta噪声)
  2. 全量150+因子 (不做IC预筛选, 让树模型自动选择)
  3. Top-30集中持仓 + 波动率倒数加权
  4. 月度调仓 (捕捉短期动量alpha)
  5. 纯ML驱动 (不融合线性基本面分数)

架构:
  ClickHouse → DataFetcherML → FactorEngine(150+因子) → 截面排名标准化
  → MLScorer(LGB+XGB, 排名标签) → Top-30选股 → 波动率倒数加权 → 月度调仓
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

TRANSACTION_COSTS = {
    'buy_commission': 0.00026,
    'sell_commission': 0.00126,
}

MARKET_STOP_LOSS = {
    'enabled': True,
    'ma_period': 250,
    'position_reduce': 0.70,  # Run8验证: 0.70比0.50多1.5%收益
    'recovery_buffer': 1.02,
}

# v5 ML参数: 更强表达力, 适配150+因子
LGB_PARAMS_V5 = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.02,
    'feature_fraction': 0.6,
    'bagging_fraction': 0.6,
    'bagging_freq': 5,
    'reg_alpha': 0.3,
    'reg_lambda': 3.0,
    'min_child_samples': 80,
    'verbose': -1,
    'n_estimators': 2000,
    'n_jobs': 4,
}

XGB_PARAMS_V5 = {
    'objective': 'reg:squarederror',
    'max_depth': 4,
    'learning_rate': 0.02,
    'subsample': 0.6,
    'colsample_bytree': 0.6,
    'reg_alpha': 0.3,
    'reg_lambda': 3.0,
    'min_child_weight': 40,
    'n_estimators': 2000,
    'n_jobs': 4,
}

# 组合参数
N_STOCKS = 30           # 持股数量(30只集中alpha)
MAX_WEIGHT = 0.10       # 单股上限10%(更集中配置高信心股票)
MIN_WEIGHT = 0.01       # 单股下限1%
SECTOR_CAP = 0.25       # 行业上限25%
MAX_TURNOVER = 0.35     # 月度最大换手35%(控制成本)
HOLDING_BONUS = 0.20    # 已持仓股票的ML分数加成20%(平衡换手与灵活性)
MIN_MARKET_CAP = 50     # 最小市值50亿
LABEL_PERIOD = 21       # 标签预测期21天(匹配月度调仓节奏)
MIN_TRAIN_MONTHS = 24   # 最少训练期24个月
MIN_TURNOVER_RATE = 0.5 # 20日均换手率下限0.5%

# CSI300/中盘 配比参数
CSI300_RATIO = 0.40     # CSI300成分股目标配比40% (按持股数量)
                        # 即Top-50中: 20只CSI300 + 30只非CSI300

# 动态配比参数 (根据CSI300 vs CSI1000相对强弱调整)
DYNAMIC_RATIO = True          # 是否启用动态配比
RATIO_LOOKBACK_MONTHS = 3     # 回看月数 (滚动3个月相对强弱)
RATIO_LARGE_CAP_STRONG = 0.55 # 大盘跑赢时CSI300配比 (55%)
RATIO_SMALL_CAP_STRONG = 0.25 # 小盘跑赢时CSI300配比 (25%)
RATIO_NEUTRAL = 0.40          # 中性时CSI300配比 (40%)
RATIO_THRESHOLD = 3.0         # 相对强弱信号阈值 (3%)


# ============================================================================
# DataFetcherML — 复用v4的ClickHouse数据获取
# ============================================================================

class DataFetcherML:
    """数据获取器 (与v4共用ClickHouse接口)"""

    def __init__(self):
        self.client = clickhouse_connect.get_client(
            host=CLICKHOUSE_HOST, port=CLICKHOUSE_PORT,
            compress=False, query_limit=10000000
        )
        self._valuation_df = None
        self._csi300_codes = None

    def get_csi300_constituents(self):
        """获取CSI300成分股列表 (从stock_block表)"""
        if self._csi300_codes is not None:
            return self._csi300_codes
        try:
            result = self.client.query(
                "SELECT DISTINCT stock_code FROM stock_block "
                "WHERE block_name = '沪深300' AND block_type = 'index'"
            )
            self._csi300_codes = set(
                str(r[0]).zfill(6) for r in result.result_rows
            )
            logger.info(f"CSI300成分股: {len(self._csi300_codes)}只")
        except Exception as e:
            logger.warning(f"获取CSI300成分失败: {e}, 使用空集合")
            self._csi300_codes = set()
        return self._csi300_codes

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
            df['market_cap'] = df['total_mv']
            self._valuation_df = df
        return self._valuation_df

    def get_financial_data(self, codes, as_of_date):
        """获取基本面数据"""
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

    def get_index_prices_multi(self, start_date, end_date, codes=('000300', '000852')):
        """获取多个指数价格 (用于动态配比计算)"""
        try:
            codes_str = "','".join(codes)
            query = f"""
                SELECT toString(date), code, close FROM default.stock_index
                WHERE code IN ('{codes_str}')
                  AND date >= '{start_date}' AND date <= '{end_date}'
                ORDER BY code, date
            """
            result = self.client.query(query)
            if result.result_rows:
                return pd.DataFrame(result.result_rows, columns=['date', 'code', 'close'])
        except Exception as e:
            logger.warning(f"获取多指数价格失败: {e}")
        return pd.DataFrame()

    def get_tushare_factors(self, codes, as_of_date):
        """从tushare表计算财务增长/质量因子"""
        # 将6位code转为tushare格式
        code_map = {}
        for c in codes:
            c6 = str(c).zfill(6)
            suffix = 'SH' if c6.startswith(('6', '5')) else 'SZ'
            code_map[c6] = f'{c6}.{suffix}'
        ts_codes = list(code_map.values())
        ts_str = "','".join(ts_codes)
        reverse_map = {v: k for k, v in code_map.items()}

        # 确定最新可用报告期 (同get_latest_report_date逻辑)
        report_date = self.get_latest_report_date(as_of_date).replace('-', '')
        dt = datetime.strptime(as_of_date, '%Y-%m-%d')
        yoy_year = str(int(report_date[:4]) - 1)
        yoy_report_date = yoy_year + report_date[4:]

        result_rows = []

        try:
            # --- 1. 利润表: 最新期 + 同比期 ---
            inc_query = f"""
                SELECT ts_code, end_date, revenue, n_income_attr_p, operate_profit,
                       total_revenue, basic_eps
                FROM tushare_income
                WHERE ts_code IN ('{ts_str}')
                  AND end_date <= '{report_date}' AND report_type = '1'
                ORDER BY end_date DESC LIMIT 1 BY ts_code
            """
            inc_res = self.client.query(inc_query)
            latest_inc = {}
            for row in inc_res.result_rows:
                latest_inc[row[0]] = {
                    'end_date': row[1], 'revenue': row[2] or 0,
                    'net_profit': row[3] or 0, 'oper_profit': row[4] or 0,
                    'total_revenue': row[5] or 0, 'eps': row[6] or 0
                }

            # 同比期利润表
            inc_yoy_query = f"""
                SELECT ts_code, revenue, n_income_attr_p, operate_profit
                FROM tushare_income
                WHERE ts_code IN ('{ts_str}')
                  AND end_date <= '{yoy_report_date}' AND report_type = '1'
                ORDER BY end_date DESC LIMIT 1 BY ts_code
            """
            inc_yoy_res = self.client.query(inc_yoy_query)
            yoy_inc = {}
            for row in inc_yoy_res.result_rows:
                yoy_inc[row[0]] = {
                    'revenue': row[1] or 0, 'net_profit': row[2] or 0,
                    'oper_profit': row[3] or 0
                }

            # --- 2. 资产负债表 ---
            bs_query = f"""
                SELECT ts_code, total_assets, total_liab,
                       total_hldr_eqy_exc_min_int, total_cur_assets,
                       total_cur_liab, accounts_receiv, inventories, goodwill
                FROM tushare_balancesheet
                WHERE ts_code IN ('{ts_str}')
                  AND end_date <= '{report_date}' AND report_type = '1'
                ORDER BY end_date DESC LIMIT 1 BY ts_code
            """
            bs_res = self.client.query(bs_query)
            bs_data = {}
            for row in bs_res.result_rows:
                bs_data[row[0]] = {
                    'total_assets': row[1] or 0, 'total_liab': row[2] or 0,
                    'equity': row[3] or 0, 'cur_assets': row[4] or 0,
                    'cur_liab': row[5] or 0, 'receivables': row[6] or 0,
                    'inventories': row[7] or 0, 'goodwill': row[8] or 0
                }

            # --- 3. 现金流量表 ---
            cf_query = f"""
                SELECT ts_code, n_cashflow_act, free_cashflow, c_fr_sale_sg
                FROM tushare_cashflow
                WHERE ts_code IN ('{ts_str}')
                  AND end_date <= '{report_date}' AND report_type = '1'
                ORDER BY end_date DESC LIMIT 1 BY ts_code
            """
            cf_res = self.client.query(cf_query)
            cf_data = {}
            for row in cf_res.result_rows:
                cf_data[row[0]] = {
                    'oper_cf': row[1] or 0, 'fcf': row[2] or 0,
                    'cash_from_sales': row[3] or 0
                }

            # --- 4. 每日指标 (最近交易日) ---
            as_of_str = as_of_date.replace('-', '')
            db_query = f"""
                SELECT ts_code, pe_ttm, pb, ps_ttm, turnover_rate_f,
                       dv_ttm, total_mv, circ_mv, free_share, float_share
                FROM tushare_daily_basic
                WHERE ts_code IN ('{ts_str}')
                  AND trade_date <= '{as_of_str}'
                ORDER BY trade_date DESC LIMIT 1 BY ts_code
            """
            db_res = self.client.query(db_query)
            daily_data = {}
            for row in db_res.result_rows:
                daily_data[row[0]] = {
                    'pe_ttm': row[1] or 0, 'pb': row[2] or 0,
                    'ps_ttm': row[3] or 0, 'turnover_rate_f': row[4] or 0,
                    'dv_ttm': row[5] or 0, 'total_mv': row[6] or 0,
                    'circ_mv': row[7] or 0, 'free_share': row[8] or 0,
                    'float_share': row[9] or 0
                }

            # --- 组装因子 ---
            for ts_code, code in reverse_map.items():
                factors = {}

                # 增长因子
                if ts_code in latest_inc and ts_code in yoy_inc:
                    li = latest_inc[ts_code]
                    yi = yoy_inc[ts_code]
                    if abs(yi['revenue']) > 1e-6:
                        factors['ts_revenue_yoy'] = (li['revenue'] / yi['revenue'] - 1) * 100
                    if abs(yi['net_profit']) > 1e-6:
                        factors['ts_profit_yoy'] = (li['net_profit'] / yi['net_profit'] - 1) * 100
                    if abs(yi['oper_profit']) > 1e-6:
                        factors['ts_oper_profit_yoy'] = (li['oper_profit'] / yi['oper_profit'] - 1) * 100

                # 盈利质量因子
                if ts_code in latest_inc:
                    li = latest_inc[ts_code]
                    if abs(li['total_revenue']) > 1e-6:
                        factors['ts_net_margin'] = li['net_profit'] / li['total_revenue'] * 100
                        factors['ts_oper_margin'] = li['oper_profit'] / li['total_revenue'] * 100

                # 现金流质量
                if ts_code in cf_data and ts_code in latest_inc:
                    cf = cf_data[ts_code]
                    li = latest_inc[ts_code]
                    if abs(li['net_profit']) > 1e-6:
                        factors['ts_cf_to_profit'] = cf['oper_cf'] / abs(li['net_profit'])
                    if abs(li['total_revenue']) > 1e-6:
                        factors['ts_cash_revenue_ratio'] = cf['cash_from_sales'] / li['total_revenue']

                # 自由现金流收益率
                if ts_code in cf_data and ts_code in daily_data:
                    if daily_data[ts_code]['total_mv'] > 1e-6:
                        factors['ts_fcf_yield'] = (cf_data[ts_code]['fcf'] /
                            (daily_data[ts_code]['total_mv'] * 10000)) * 100

                # 资产负债表因子
                if ts_code in bs_data:
                    bs = bs_data[ts_code]
                    if bs['total_assets'] > 1e-6:
                        factors['ts_debt_ratio'] = bs['total_liab'] / bs['total_assets'] * 100
                        factors['ts_goodwill_ratio'] = bs['goodwill'] / bs['total_assets'] * 100
                    if bs['cur_liab'] > 1e-6:
                        factors['ts_current_ratio'] = bs['cur_assets'] / bs['cur_liab']
                    if bs['equity'] > 1e-6 and ts_code in latest_inc:
                        factors['ts_roe'] = latest_inc[ts_code]['net_profit'] / bs['equity'] * 100
                    if bs['total_assets'] > 1e-6 and ts_code in latest_inc:
                        factors['ts_roa'] = latest_inc[ts_code]['net_profit'] / bs['total_assets'] * 100

                # 每日估值因子
                if ts_code in daily_data:
                    dd = daily_data[ts_code]
                    factors['ts_pe_ttm'] = dd['pe_ttm']
                    factors['ts_pb'] = dd['pb']
                    factors['ts_ps_ttm'] = dd['ps_ttm']
                    factors['ts_dv_ttm'] = dd['dv_ttm']
                    factors['ts_turnover_f'] = dd['turnover_rate_f']
                    # 筹码集中度: 自由流通股占比
                    if dd['float_share'] > 1e-6:
                        factors['ts_free_float_ratio'] = dd['free_share'] / dd['float_share']

                if factors:
                    factors['code'] = code
                    result_rows.append(factors)

        except Exception as e:
            logger.warning(f"Tushare因子计算失败: {e}")

        if result_rows:
            df = pd.DataFrame(result_rows)
            logger.info(f"Tushare财务因子: {len(df)}只股票, "
                        f"{len(df.columns)-1}个因子")
            return df
        return pd.DataFrame()

    def get_tushare_daily_factors(self, codes, as_of_date):
        """仅从tushare_daily_basic表获取估值/换手因子 (覆盖全市场)"""
        code_map = {}
        for c in codes:
            c6 = str(c).zfill(6)
            suffix = 'SH' if c6.startswith(('6', '5')) else 'SZ'
            code_map[c6] = f'{c6}.{suffix}'
        ts_codes = list(code_map.values())
        ts_str = "','".join(ts_codes)
        reverse_map = {v: k for k, v in code_map.items()}

        result_rows = []
        try:
            as_of_str = as_of_date.replace('-', '')
            db_query = f"""
                SELECT ts_code, pe_ttm, pb, ps_ttm, turnover_rate_f,
                       dv_ttm, total_mv, circ_mv, free_share, float_share,
                       volume_ratio
                FROM tushare_daily_basic
                WHERE ts_code IN ('{ts_str}')
                  AND trade_date <= '{as_of_str}'
                ORDER BY trade_date DESC LIMIT 1 BY ts_code
            """
            db_res = self.client.query(db_query)
            for row in db_res.result_rows:
                ts_code = row[0]
                code = reverse_map.get(ts_code)
                if not code:
                    continue
                dd = {
                    'pe_ttm': row[1] or 0, 'pb': row[2] or 0,
                    'ps_ttm': row[3] or 0, 'turnover_rate_f': row[4] or 0,
                    'dv_ttm': row[5] or 0, 'total_mv': row[6] or 0,
                    'circ_mv': row[7] or 0, 'free_share': row[8] or 0,
                    'float_share': row[9] or 0, 'volume_ratio': row[10] or 0,
                }
                factors = {'code': code}
                factors['ts_pe_ttm'] = dd['pe_ttm']
                factors['ts_pb'] = dd['pb']
                factors['ts_ps_ttm'] = dd['ps_ttm']
                factors['ts_dv_ttm'] = dd['dv_ttm']
                factors['ts_turnover_f'] = dd['turnover_rate_f']
                factors['ts_volume_ratio'] = dd['volume_ratio']
                # 筹码集中度
                if dd['float_share'] > 1e-6:
                    factors['ts_free_float_ratio'] = dd['free_share'] / dd['float_share']
                # 市值比 (流通/总)
                if dd['total_mv'] > 1e-6:
                    factors['ts_circ_mv_ratio'] = dd['circ_mv'] / dd['total_mv']
                result_rows.append(factors)
        except Exception as e:
            logger.warning(f"Tushare daily因子失败: {e}")

        if result_rows:
            df = pd.DataFrame(result_rows)
            logger.info(f"Tushare每日因子: {len(df)}只股票, "
                        f"{len(df.columns)-1}个因子")
            return df
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


# ============================================================================
# FactorEngine — 整合所有因子计算
# ============================================================================

class FactorEngine:
    """整合Alpha158 + Extended + 基本面因子, 输出截面排名"""

    def __init__(self):
        self.alpha158_calc = Alpha158FactorCalculator(windows=[5, 10, 20, 60])
        self.extended_calc = ExtendedFactorCalculator()

    def compute_all(self, fetcher, codes, as_of_date, ohlcv_df):
        """
        计算所有因子并做截面百分位排名标准化

        Returns:
            DataFrame[code, factor1_rank, factor2_rank, ...] 值域[0,1]
        """
        results = {}

        # 1. Alpha158 技术因子 (时间序列→取最后一行)
        try:
            alpha_df = self.alpha158_calc.calculate_all_factors(ohlcv_df)
            if len(alpha_df) > 0:
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

        # 2. Extended 技术因子 (每只股票一行)
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

        # 3. 基本面因子 (16个打分因子)
        try:
            fin_df = fetcher.get_financial_data(codes, as_of_date)
            mom_df = fetcher.get_momentum(codes, as_of_date)
            dd_df = fetcher.get_drawdown(codes, as_of_date)
            vol_df = fetcher.get_volatility(codes, as_of_date)
            turnover_df = fetcher.get_turnover(codes, as_of_date)

            fin_df = fin_df.merge(mom_df, on='code', how='left')
            fin_df = fin_df.merge(dd_df, on='code', how='left')
            fin_df = fin_df.merge(vol_df, on='code', how='left')
            fin_df = fin_df.merge(turnover_df, on='code', how='left')
            fin_df = fin_df.fillna(0)

            # 将原始基本面数值作为因子
            fund_cols = ['roe', 'pe', 'pb', 'dividend_yield', 'market_cap',
                         'profit_growth', 'revenue_growth', 'momentum', 'drawdown',
                         'volatility', 'avg_turnover', 'asset_liability_ratio',
                         'gross_profit_margin', 'net_profit_margin',
                         'operating_quality', 'asset_turnover', 'roe_stability']
            for _, row in fin_df.iterrows():
                code = row['code']
                if code not in results:
                    results[code] = {}
                for col in fund_cols:
                    if col in row.index:
                        val = row[col]
                        if pd.notna(val) and np.isfinite(float(val)):
                            results[code][f'fund_{col}'] = float(val)
        except Exception as e:
            logger.warning(f"基本面因子计算失败: {e}")

        # 4. Tushare每日指标因子 (仅daily_basic, 覆盖全市场)
        try:
            ts_df = fetcher.get_tushare_daily_factors(codes, as_of_date)
            if len(ts_df) > 0:
                ts_cols = [c for c in ts_df.columns if c != 'code']
                for _, row in ts_df.iterrows():
                    code = row['code']
                    if code not in results:
                        results[code] = {}
                    for col in ts_cols:
                        val = row[col]
                        if pd.notna(val) and np.isfinite(float(val)):
                            results[code][col] = float(val)
        except Exception as e:
            logger.warning(f"Tushare每日因子计算失败: {e}")

        if not results:
            return pd.DataFrame()

        # 转为DataFrame
        factor_df = pd.DataFrame.from_dict(results, orient='index')
        factor_df.index.name = 'code'
        factor_df = factor_df.reset_index()

        # 截面百分位排名标准化: 每个因子 → rank percentile [0, 1]
        factor_cols = [c for c in factor_df.columns if c != 'code']
        for col in factor_cols:
            factor_df[col] = factor_df[col].rank(pct=True)
            factor_df[col] = factor_df[col].fillna(0.5)

        logger.info(f"因子引擎: {len(factor_df)}只股票, {len(factor_cols)}个因子")
        return factor_df


# ============================================================================
# MLScorer — LightGBM + XGBoost 集成 (截面排名标签)
# ============================================================================

class MLScorer:
    """ML评分器: 排名标签 + LGB/XGB集成"""

    def __init__(self, label_period=LABEL_PERIOD):
        self.label_period = label_period
        self.lgb_model = None
        self.xgb_model = None
        self.imputer = None
        self.scaler = None
        self.feature_cols = None
        self.train_count = 0
        self.selected_features = None  # 特征筛选: 累积重要性排名
        self.feature_importance_acc = {}  # 累积特征重要性
        self.last_val_ic = 0.0  # 最近一次验证IC

    def build_rank_labels(self, price_pivot, date_str):
        """
        计算截面收益排名标签

        对于给定日期, 计算未来label_period天的收益率, 然后做截面排名[0,1]
        Returns: dict {code: rank_label}
        """
        available_dates = sorted([d for d in price_pivot.index if d >= date_str])
        if len(available_dates) < self.label_period + 1:
            return {}

        base_date = available_dates[0]
        target_idx = min(self.label_period, len(available_dates) - 1)
        target_date = available_dates[target_idx]

        p_base = price_pivot.loc[base_date]
        p_target = price_pivot.loc[target_date]

        returns = (p_target / p_base - 1).dropna()
        returns = returns[returns.index.isin(p_base.dropna().index)]

        if len(returns) < 30:
            return {}

        # 截面排名: 0=最差, 1=最好
        ranks = returns.rank(pct=True)
        return ranks.to_dict()

    def prepare_training_data(self, factor_history, price_pivot, as_of_date, feature_cols):
        """
        准备训练数据 (expanding window + 排名标签)

        factor_history: list of (date_str, DataFrame[code, factor1_rank, ...])
        price_pivot: DataFrame[date x code] close prices
        as_of_date: 当前日期
        feature_cols: 特征列
        """
        # 截止日期: as_of_date - label_period天, 防止前视偏差
        cutoff_dt = datetime.strptime(as_of_date, '%Y-%m-%d') - timedelta(days=self.label_period + 5)
        cutoff = cutoff_dt.strftime('%Y-%m-%d')

        all_X = []
        all_y = []

        for hist_date, factor_df in factor_history:
            if hist_date > cutoff:
                continue

            # 计算截面排名标签
            rank_labels = self.build_rank_labels(price_pivot, hist_date)
            if not rank_labels:
                continue

            for _, row in factor_df.iterrows():
                code = row['code']
                if code not in rank_labels:
                    continue

                feat_vals = []
                for col in feature_cols:
                    v = row.get(col, 0.5)
                    feat_vals.append(float(v) if pd.notna(v) and np.isfinite(v) else 0.5)

                all_X.append(feat_vals)
                all_y.append(rank_labels[code])

        if len(all_X) < 200:
            logger.warning(f"训练数据不足: {len(all_X)}条 (需要>=200)")
            return None, None, None, None

        X = np.array(all_X)
        y = np.array(all_y)

        # 验证集: 最后15%
        n_val = max(int(len(X) * 0.15), 100)
        X_train, X_val = X[:-n_val], X[-n_val:]
        y_train, y_val = y[:-n_val], y[-n_val:]

        logger.info(f"训练数据: {len(X_train)}条训练, {len(X_val)}条验证, "
                    f"{len(feature_cols)}特征, 标签[{y.min():.3f}, {y.max():.3f}]")

        return X_train, y_train, X_val, y_val

    def train(self, X_train, y_train, X_val, y_val, feature_names):
        """训练LightGBM + XGBoost集成"""
        # 预处理
        self.imputer = SimpleImputer(strategy='median')
        X_train = self.imputer.fit_transform(X_train)
        X_val = self.imputer.transform(X_val)

        self.scaler = RobustScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)

        self.feature_cols = feature_names

        # 自适应参数
        ratio = len(X_train) / max(len(feature_names), 1)
        if ratio < 15:
            lgb_p = dict(LGB_PARAMS_V5, num_leaves=31, n_estimators=800,
                         feature_fraction=0.5, min_child_samples=80,
                         reg_alpha=0.5, reg_lambda=3.0)
            xgb_p = dict(XGB_PARAMS_V5, max_depth=4, n_estimators=800,
                         subsample=0.6, colsample_bytree=0.5,
                         min_child_weight=30, reg_alpha=0.5, reg_lambda=3.0)
            es_rounds = 50
        else:
            lgb_p = dict(LGB_PARAMS_V5)
            xgb_p = dict(XGB_PARAMS_V5)
            es_rounds = 100

        logger.info(f"  样本/特征比: {ratio:.1f}, 模式: {'保守' if ratio<15 else '标准'}")

        # LightGBM
        if HAS_LGB:
            lgb_params = {k: v for k, v in lgb_p.items() if k != 'n_estimators'}
            self.lgb_model = lgb.LGBMRegressor(n_estimators=lgb_p['n_estimators'], **lgb_params)
            self.lgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(es_rounds, verbose=False),
                           lgb.log_evaluation(period=500)]
            )
            lgb_pred = self.lgb_model.predict(X_val)
            lgb_ic = stats.spearmanr(lgb_pred, y_val)[0] if len(y_val) > 2 else 0
            logger.info(f"  LGB验证IC: {lgb_ic:.4f}, best_iter: {self.lgb_model.best_iteration_}")

        # XGBoost
        if HAS_XGB:
            xgb_params = {k: v for k, v in xgb_p.items() if k != 'n_estimators'}
            self.xgb_model = xgb.XGBRegressor(
                n_estimators=xgb_p['n_estimators'],
                early_stopping_rounds=es_rounds, **xgb_params)
            self.xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            xgb_pred = self.xgb_model.predict(X_val)
            xgb_ic = stats.spearmanr(xgb_pred, y_val)[0] if len(y_val) > 2 else 0
            logger.info(f"  XGB验证IC: {xgb_ic:.4f}")

        # 记录验证IC
        val_ics = []
        if HAS_LGB and self.lgb_model is not None:
            val_ics.append(lgb_ic)
        if HAS_XGB and self.xgb_model is not None:
            val_ics.append(xgb_ic)
        self.last_val_ic = np.mean(val_ics) if val_ics else 0.0

        # 累积特征重要性 (用于特征筛选)
        self._update_feature_importance(feature_names)
        self.train_count += 1

    def _update_feature_importance(self, feature_names):
        """累积特征重要性，训练3次后开始筛选"""
        imp = {}
        if self.lgb_model is not None and hasattr(self.lgb_model, 'feature_importances_'):
            for i, col in enumerate(feature_names):
                imp[col] = imp.get(col, 0) + self.lgb_model.feature_importances_[i] * 0.6
        if self.xgb_model is not None and hasattr(self.xgb_model, 'feature_importances_'):
            for i, col in enumerate(feature_names):
                imp[col] = imp.get(col, 0) + self.xgb_model.feature_importances_[i] * 0.4

        # 指数加权累积 (新训练权重更高)
        decay = 0.7
        for col, v in imp.items():
            self.feature_importance_acc[col] = self.feature_importance_acc.get(col, 0) * decay + v

        # 训练≥3次后启用特征筛选: 保留top 80%
        if self.train_count >= 3 and self.feature_importance_acc:
            sorted_feats = sorted(self.feature_importance_acc.items(), key=lambda x: -x[1])
            n_keep = max(int(len(sorted_feats) * 0.85), 30)
            self.selected_features = set(f[0] for f in sorted_feats[:n_keep])
            n_dropped = len(sorted_feats) - n_keep
            if n_dropped > 0:
                logger.info(f"  特征筛选: 保留{n_keep}/{len(sorted_feats)}, 丢弃{n_dropped}低效特征")

    def get_active_features(self, feature_cols):
        """返回筛选后的特征列表"""
        if self.selected_features is None:
            return feature_cols
        return [c for c in feature_cols if c in self.selected_features]

    def predict(self, factor_df, feature_cols):
        """
        对当前截面生成ML分数
        Returns: Series[code -> ml_score]
        """
        if self.lgb_model is None and self.xgb_model is None:
            return pd.Series(0.5, index=factor_df['code'])

        X = np.zeros((len(factor_df), len(feature_cols)))
        for j, col in enumerate(feature_cols):
            if col in factor_df.columns:
                vals = factor_df[col].values.astype(float)
                X[:, j] = np.where(np.isfinite(vals), vals, 0.5)

        X = self.imputer.transform(X)
        X = self.scaler.transform(X)

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

        return pd.Series(pred, index=factor_df['code'].values)

    def get_feature_importance(self, top_n=20):
        """获取特征重要性"""
        importance = {}
        if self.lgb_model is not None and hasattr(self.lgb_model, 'feature_importances_'):
            for i, col in enumerate(self.feature_cols):
                importance[col] = importance.get(col, 0) + self.lgb_model.feature_importances_[i] * 0.6
        if self.xgb_model is not None and hasattr(self.xgb_model, 'feature_importances_'):
            for i, col in enumerate(self.feature_cols):
                importance[col] = importance.get(col, 0) + self.xgb_model.feature_importances_[i] * 0.4
        return sorted(importance.items(), key=lambda x: -x[1])[:top_n]


# ============================================================================
# PortfolioBuilder — 集中组合 + 波动率倒数加权
# ============================================================================

class PortfolioBuilder:
    """Top-N选股 + 波动率倒数加权 + CSI300/中盘分层配比"""

    def build(self, ml_scores, vol_dict, n_stocks=N_STOCKS,
              max_weight=MAX_WEIGHT, min_weight=MIN_WEIGHT,
              csi300_codes=None, csi300_ratio=CSI300_RATIO):
        """
        构建集中组合 (分层选股)

        ml_scores: Series[code -> ml_score]
        vol_dict: dict {code: annualized_volatility}
        csi300_codes: set of CSI300 constituent codes (None=不分层)
        csi300_ratio: CSI300目标配比 (按持股数量)
        Returns: dict {code: weight}
        """
        # 分层选股: CSI300内选N只 + 非CSI300内选M只
        if csi300_codes and len(csi300_codes) > 0:
            n_csi300 = max(1, round(n_stocks * csi300_ratio))
            n_other = n_stocks - n_csi300

            scores_300 = ml_scores[ml_scores.index.isin(csi300_codes)]
            scores_other = ml_scores[~ml_scores.index.isin(csi300_codes)]

            top_300 = scores_300.nlargest(min(n_csi300, len(scores_300))).index.tolist()
            top_other = scores_other.nlargest(min(n_other, len(scores_other))).index.tolist()
            top_codes = top_300 + top_other

            logger.info(f"  分层选股: CSI300={len(top_300)}只, 非CSI300={len(top_other)}只")
        else:
            top_codes = ml_scores.nlargest(n_stocks).index.tolist()

        if not top_codes:
            return {}

        # 波动率倒数加权
        weights = {}
        for code in top_codes:
            vol = vol_dict.get(code, 30.0)  # 默认30%年化波动率
            vol = max(vol, 5.0)  # 下限5%防止极端权重
            weights[code] = 1.0 / vol

        # 归一化
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        # 应用上下限
        weights = self._apply_weight_limits(weights, max_weight, min_weight)

        return weights

    def _apply_weight_limits(self, weights, max_w, min_w):
        """应用权重上下限, 迭代直到满足约束"""
        for _ in range(10):  # 最多迭代10次
            excess = 0
            clipped = {}
            for code, w in weights.items():
                if w > max_w:
                    excess += w - max_w
                    clipped[code] = max_w
                elif w < min_w:
                    excess -= min_w - w
                    clipped[code] = min_w
                else:
                    clipped[code] = w

            # 重新分配超额
            total = sum(clipped.values())
            if total > 0:
                clipped = {k: v / total for k, v in clipped.items()}

            # 检查是否满足约束
            all_ok = all(min_w - 0.001 <= v <= max_w + 0.001 for v in clipped.values())
            weights = clipped
            if all_ok:
                break

        return weights


# ============================================================================
# 动态风格配比
# ============================================================================

def compute_dynamic_csi300_ratio(index_data, as_of_date,
                                  lookback_months=RATIO_LOOKBACK_MONTHS):
    """
    根据CSI300 vs CSI1000滚动相对强弱计算动态CSI300配比

    逻辑:
      1. 计算过去N个月CSI300和CSI1000的累计收益
      2. 大盘跑赢 → 提高CSI300配比 (55%)
      3. 小盘跑赢 → 降低CSI300配比 (25%)
      4. 中性 → 维持40%

    index_data: DataFrame with columns ['date', 'code', 'close']
    as_of_date: 当前日期字符串 'YYYY-MM-DD'
    Returns: (ratio, signal_label)
    """
    if index_data is None or index_data.empty:
        return CSI300_RATIO, "NO_DATA"

    # 计算回看起点 (约N个月前)
    as_of_dt = datetime.strptime(as_of_date, '%Y-%m-%d')
    lookback_start = (as_of_dt - timedelta(days=lookback_months * 30)).strftime('%Y-%m-%d')

    csi300 = index_data[(index_data['code'] == '000300') &
                        (index_data['date'] >= lookback_start) &
                        (index_data['date'] <= as_of_date)]
    csi1000 = index_data[(index_data['code'] == '000852') &
                         (index_data['date'] >= lookback_start) &
                         (index_data['date'] <= as_of_date)]

    if len(csi300) < 20 or len(csi1000) < 20:
        return CSI300_RATIO, "INSUF_DATA"

    # 计算区间收益率
    ret_300 = (csi300['close'].iloc[-1] / csi300['close'].iloc[0] - 1) * 100
    ret_1000 = (csi1000['close'].iloc[-1] / csi1000['close'].iloc[0] - 1) * 100
    diff = ret_300 - ret_1000  # 正=大盘强, 负=小盘强

    if diff > RATIO_THRESHOLD:
        ratio = RATIO_LARGE_CAP_STRONG
        signal = "LARGE_CAP"
    elif diff < -RATIO_THRESHOLD:
        ratio = RATIO_SMALL_CAP_STRONG
        signal = "SMALL_CAP"
    else:
        ratio = RATIO_NEUTRAL
        signal = "NEUTRAL"

    logger.info(f"  动态配比: CSI300={ret_300:+.1f}% CSI1000={ret_1000:+.1f}% "
                f"diff={diff:+.1f}% → {signal} ratio={ratio:.0%}")
    return ratio, signal


# ============================================================================
# 换手率控制
# ============================================================================

def apply_turnover_damping(new_weights, prev_weights, max_turnover=MAX_TURNOVER):
    """限制单次调仓换手率"""
    if not prev_weights:
        return new_weights

    all_codes = set(list(new_weights.keys()) + list(prev_weights.keys()))
    raw_turnover = sum(
        abs(new_weights.get(c, 0) - prev_weights.get(c, 0))
        for c in all_codes
    ) / 2

    if raw_turnover <= max_turnover:
        return new_weights

    damping = max_turnover / raw_turnover
    damped = {}
    for c in all_codes:
        old_w = prev_weights.get(c, 0)
        new_w = new_weights.get(c, 0)
        damped[c] = old_w + (new_w - old_w) * damping

    # 移除极小权重 + 归一化
    damped = {k: v for k, v in damped.items() if v > 0.001}
    total = sum(damped.values())
    if total > 0:
        damped = {k: v / total for k, v in damped.items()}

    return damped


# ============================================================================
# 绩效计算
# ============================================================================

def calc_performance(returns_list):
    """计算策略绩效"""
    if not returns_list:
        return {'annual_return': 0, 'sharpe': 0, 'max_drawdown': 0, 'calmar': 0}

    r = np.array(returns_list)
    n_days = len(r)
    total_ret = np.prod(1 + r)
    annual_ret = total_ret ** (252 / max(n_days, 1)) - 1

    daily_std = np.std(r)
    sharpe = (np.mean(r) / (daily_std + 1e-10)) * np.sqrt(252)

    cumulative = np.cumprod(1 + r)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative / running_max - 1
    max_dd = np.min(drawdowns)

    calmar = annual_ret / abs(max_dd) if abs(max_dd) > 1e-6 else 0

    # 年度分解
    return {
        'annual_return': annual_ret,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'calmar': calmar,
        'total_return': total_ret - 1,
        'n_days': n_days,
    }


def calc_yearly_returns(dates, returns_list):
    """按年度分解收益"""
    if not returns_list or not dates:
        return {}

    yearly = {}
    for date_str, ret in zip(dates, returns_list):
        year = date_str[:4]
        if year not in yearly:
            yearly[year] = []
        yearly[year].append(ret)

    result = {}
    for year, rets in yearly.items():
        r = np.array(rets)
        total = np.prod(1 + r) - 1
        result[year] = total

    return result


# ============================================================================
# 主回测
# ============================================================================

def run_v5_backtest(start_date='2022-01-01', end_date='2025-12-31'):
    """
    v5.0 ML驱动选股回测

    - 前24个月 (2020-01 ~ 2021-12) 作为初始训练期
    - 2022-01 开始交易, 月度调仓
    """
    print("=" * 70)
    print("v5.0 ML驱动选股回测")
    print(f"回测期: {start_date} ~ {end_date}")
    print(f"配置: Top-{N_STOCKS} | 月度调仓 | 排名标签 | 150+因子 | 波动率倒数加权")
    print(f"风控: 单股≤{MAX_WEIGHT:.0%} | 换手≤{MAX_TURNOVER:.0%}/月 | 市值≥{MIN_MARKET_CAP}亿")
    print("=" * 70)

    fetcher = DataFetcherML()
    portfolio = fetcher.get_stock_pool()
    codes = portfolio['code'].tolist()

    # 训练期需要额外2年历史
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    train_start = (start_dt - timedelta(days=800)).strftime('%Y-%m-%d')
    extended_start = (start_dt - timedelta(days=400)).strftime('%Y-%m-%d')

    print(f"\n[1/5] 加载价格数据 ({train_start} ~ {end_date})...")
    prices = fetcher.get_prices(codes, train_start, end_date)
    price_pivot = prices.pivot(index='date', columns='code', values='close')
    index_prices = fetcher.get_index_prices(extended_start, end_date)

    # 加载多指数数据 (CSI300 + CSI1000, 用于动态配比)
    index_multi = pd.DataFrame()
    if DYNAMIC_RATIO:
        index_multi = fetcher.get_index_prices_multi(extended_start, end_date)
        if not index_multi.empty:
            logger.info(f"  多指数数据: {len(index_multi)}行 (CSI300+CSI1000)")
        else:
            logger.warning("  多指数数据加载失败, 将使用静态配比")
    print(f"  价格矩阵: {price_pivot.shape[0]}天 x {price_pivot.shape[1]}只股票")

    print(f"\n[2/5] 加载OHLCV数据...")
    ohlcv_all = fetcher.get_ohlcv(codes, train_start, end_date)

    # 市值过滤
    val_df = fetcher.load_valuation_csv()
    cap_df = val_df[val_df['code'].isin(codes)][['code', 'market_cap']]
    active_codes = cap_df[cap_df['market_cap'] >= MIN_MARKET_CAP]['code'].tolist()
    print(f"  市值过滤: {len(codes)} → {len(active_codes)} (≥{MIN_MARKET_CAP}亿)")

    # 流动性过滤: 在每次调仓时做
    codes_filtered = active_codes

    # 月度调仓日期 (从训练期开始, 为了积累因子历史)
    train_start_dt = start_dt - timedelta(days=MIN_TRAIN_MONTHS * 30 + 30)
    all_rebalance_dates = []
    for y in range(train_start_dt.year, end_dt.year + 1):
        for m in range(1, 13):
            rd = f'{y}-{m:02d}-01'
            if train_start_dt.strftime('%Y-%m-%d') <= rd <= end_date:
                all_rebalance_dates.append(rd)

    # 区分: 训练期调仓 vs 交易期调仓
    trade_rebalance_dates = [d for d in all_rebalance_dates if d >= start_date]
    train_rebalance_dates = [d for d in all_rebalance_dates if d < start_date]

    print(f"\n[3/5] 计算因子历史 ({len(train_rebalance_dates)}个训练月 + "
          f"{len(trade_rebalance_dates)}个交易月)...")

    # 初始化组件
    factor_engine = FactorEngine()
    ml_scorer = MLScorer(label_period=LABEL_PERIOD)
    portfolio_builder = PortfolioBuilder()

    # 预计算训练期因子
    factor_history = []  # [(date_str, factor_df), ...]

    for i, rb_date in enumerate(train_rebalance_dates):
        rb_dt = datetime.strptime(rb_date, '%Y-%m-%d')
        tech_start = (rb_dt - timedelta(days=120)).strftime('%Y-%m-%d')
        ohlcv_slice = ohlcv_all[
            (ohlcv_all['date'] >= tech_start) & (ohlcv_all['date'] <= rb_date)
            & (ohlcv_all['code'].isin(codes_filtered))
        ].copy()

        if len(ohlcv_slice) == 0:
            continue

        factor_df = factor_engine.compute_all(fetcher, codes_filtered, rb_date, ohlcv_slice)
        if len(factor_df) > 0:
            factor_history.append((rb_date, factor_df))
            if (i + 1) % 6 == 0 or i == len(train_rebalance_dates) - 1:
                logger.info(f"  训练期因子 {i+1}/{len(train_rebalance_dates)}: "
                           f"{rb_date}, {len(factor_df)}只股票, "
                           f"{len(factor_df.columns)-1}个因子")

    print(f"  训练期因子: {len(factor_history)}个截面")

    # ================================================================
    # 交易期回测
    # ================================================================
    print(f"\n[4/5] 开始交易回测...")

    trading_days = sorted(price_pivot.index.tolist())
    dates = sorted([d for d in price_pivot.index if d >= start_date])

    original_weights = dict(zip(portfolio['code'], portfolio['weight']))
    current_weights = {}  # v5: 初始空仓, 等ML信号
    prev_weights = {}
    rb_idx = 0
    market_position = 1.0
    stop_loss_count = 0
    enhanced_returns = []
    original_returns = []
    return_dates = []
    total_tx_cost = 0.0
    monthly_turnovers = []
    monthly_ics = []

    for i in range(1, len(dates)):
        curr_date = dates[i]
        prev_date = dates[i - 1]
        rebalance_cost = 0.0

        # 月度调仓
        while rb_idx < len(trade_rebalance_dates) and curr_date >= trade_rebalance_dates[rb_idx]:
            rb_date = trade_rebalance_dates[rb_idx]
            rb_dt = datetime.strptime(rb_date, '%Y-%m-%d')

            try:
                # Step 1: 计算当月因子
                tech_start = (rb_dt - timedelta(days=120)).strftime('%Y-%m-%d')
                ohlcv_slice = ohlcv_all[
                    (ohlcv_all['date'] >= tech_start) & (ohlcv_all['date'] <= rb_date)
                    & (ohlcv_all['code'].isin(codes_filtered))
                ].copy()

                if len(ohlcv_slice) == 0:
                    rb_idx += 1
                    continue

                factor_df = factor_engine.compute_all(
                    fetcher, codes_filtered, rb_date, ohlcv_slice)

                if len(factor_df) == 0:
                    rb_idx += 1
                    continue

                factor_history.append((rb_date, factor_df))
                feature_cols = [c for c in factor_df.columns if c != 'code']

                # Step 2: 流动性过滤
                turnover_df = fetcher.get_turnover(codes_filtered, rb_date)
                liquid_codes = turnover_df[
                    turnover_df['avg_turnover'] >= MIN_TURNOVER_RATE
                ]['code'].tolist() if len(turnover_df) > 0 else codes_filtered
                factor_df_liquid = factor_df[factor_df['code'].isin(liquid_codes)]

                if len(factor_df_liquid) < N_STOCKS:
                    factor_df_liquid = factor_df  # fallback: 不过滤

                # Step 3: 特征筛选 (训练≥3次后只用高效特征)
                active_features = ml_scorer.get_active_features(feature_cols)

                # Step 4: 训练ML (需要≥MIN_TRAIN_MONTHS个月历史)
                if len(factor_history) >= MIN_TRAIN_MONTHS:
                    train_result = ml_scorer.prepare_training_data(
                        factor_history, price_pivot, rb_date, active_features)

                    X_train, y_train, X_val, y_val = train_result

                    if X_train is not None and len(X_train) >= 200:
                        logger.info(f"\n  [{rb_date}] 训练ML模型... ({len(active_features)}特征)")
                        ml_scorer.train(X_train, y_train, X_val, y_val, active_features)

                        # 验证IC
                        val_pred = ml_scorer.predict(
                            factor_df_liquid[['code'] + active_features], active_features)
                        rank_labels = ml_scorer.build_rank_labels(price_pivot, rb_date)
                        if rank_labels:
                            common = [c for c in val_pred.index if c in rank_labels]
                            if len(common) > 20:
                                ic = stats.spearmanr(
                                    [val_pred[c] for c in common],
                                    [rank_labels[c] for c in common]
                                )[0]
                                monthly_ics.append(ic)
                                logger.info(f"  月度预测IC: {ic:.4f}")
                    else:
                        logger.info(f"  [{rb_date}] 训练数据不足, 跳过训练")
                else:
                    logger.info(f"  [{rb_date}] 训练期积累中 ({len(factor_history)}/{MIN_TRAIN_MONTHS})")

                # Step 5: ML预测 + 构建组合
                if ml_scorer.lgb_model is not None or ml_scorer.xgb_model is not None:
                    # IC门控: 验证IC太低时跳过调仓 (保持现有组合)
                    if ml_scorer.last_val_ic < 0.02 and current_weights:
                        logger.info(f"  [{rb_date}] IC门控: val_ic={ml_scorer.last_val_ic:.4f}<0.02, 跳过调仓")
                        rb_idx += 1
                        continue

                    ml_scores = ml_scorer.predict(
                        factor_df_liquid[['code'] + active_features], active_features)

                    # 持仓加成: 已持有的股票ML分数加成, 减少不必要换手
                    if current_weights:
                        held_codes = set(c for c, w in current_weights.items() if w > 0.001)
                        for code in ml_scores.index:
                            if code in held_codes:
                                ml_scores[code] *= (1 + HOLDING_BONUS)

                    # 获取波动率用于加权
                    vol_df = fetcher.get_volatility(codes_filtered, rb_date)
                    vol_dict = dict(zip(vol_df['code'], vol_df['volatility'])) if len(vol_df) > 0 else {}

                    # 计算动态配比 (基于CSI300 vs CSI1000相对强弱)
                    if DYNAMIC_RATIO and not index_multi.empty:
                        dynamic_ratio, ratio_signal = compute_dynamic_csi300_ratio(
                            index_multi, rb_date)
                    else:
                        dynamic_ratio = CSI300_RATIO
                        ratio_signal = "STATIC"

                    # 构建Top-N组合 (分层选股: CSI300 + 非CSI300)
                    csi300_codes = fetcher.get_csi300_constituents()
                    new_weights = portfolio_builder.build(
                        ml_scores, vol_dict,
                        csi300_codes=csi300_codes,
                        csi300_ratio=dynamic_ratio)

                    # 换手率控制
                    new_weights = apply_turnover_damping(new_weights, current_weights)

                    # 计算交易成本
                    all_c = set(list(new_weights.keys()) + list(current_weights.keys()))
                    turnover_ratio = sum(
                        abs(new_weights.get(c, 0) - current_weights.get(c, 0))
                        for c in all_c
                    ) / 2
                    rebalance_cost = (turnover_ratio * TRANSACTION_COSTS['sell_commission'] +
                                      turnover_ratio * TRANSACTION_COSTS['buy_commission'])
                    total_tx_cost += rebalance_cost
                    monthly_turnovers.append(turnover_ratio)

                    prev_weights = current_weights.copy()
                    current_weights = new_weights

                    n_held = len([w for w in current_weights.values() if w > 0.001])
                    top5 = sorted(current_weights.values(), reverse=True)[:5]
                    print(f"  [{rb_date}] 持股:{n_held} | 换手:{turnover_ratio:.1%} | "
                          f"成本:{rebalance_cost:.4%} | Top5权重:{sum(top5):.1%}")

                    # 特征重要性 (每6个月打印)
                    if ml_scorer.train_count % 6 == 1:
                        fi = ml_scorer.get_feature_importance(10)
                        logger.info(f"  Top10因子: {[f'{n}={v:.0f}' for n, v in fi]}")

            except Exception as e:
                logger.error(f"  [{rb_date}] 调仓错误: {e}")
                traceback.print_exc()

            rb_idx += 1

        # 日收益计算
        if not current_weights:
            # ML尚未生成信号, 用等权R5
            enhanced_returns.append(0.0)
            curr = price_pivot.loc[curr_date]
            prev_p = price_pivot.loc[prev_date]
            orig_ret = 0
            for code in codes:
                if code in curr.index and pd.notna(curr[code]) and pd.notna(prev_p[code]) and prev_p[code] > 0:
                    ret = curr[code] / prev_p[code] - 1
                    orig_ret += ret * original_weights.get(code, 0)
            original_returns.append(orig_ret)
            return_dates.append(curr_date)
            continue

        # 市场止损
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

        curr_p = price_pivot.loc[curr_date]
        prev_p = price_pivot.loc[prev_date]

        enh_ret = 0
        orig_ret = 0
        for code in set(list(current_weights.keys()) + codes):
            if code in curr_p.index and pd.notna(curr_p[code]) and pd.notna(prev_p.get(code)) and prev_p.get(code, 0) > 0:
                ret = curr_p[code] / prev_p[code] - 1
                enh_ret += ret * current_weights.get(code, 0) * market_position
                orig_ret += ret * original_weights.get(code, 0)

        if rebalance_cost > 0:
            enh_ret -= rebalance_cost

        enhanced_returns.append(enh_ret)
        original_returns.append(orig_ret)
        return_dates.append(curr_date)

    # ================================================================
    # 绩效报告
    # ================================================================
    print(f"\n{'='*70}")
    print("[5/5] 绩效报告")
    print(f"{'='*70}")

    enh = calc_performance(enhanced_returns)
    orig = calc_performance(original_returns)

    print(f"\n{'指标':<20} {'v5.0 ML驱动':>15} {'R5等权基线':>15}")
    print("-" * 55)
    print(f"{'年化收益':.<20} {enh['annual_return']*100:>14.2f}% {orig['annual_return']*100:>14.2f}%")
    print(f"{'夏普比率':.<20} {enh['sharpe']:>15.2f} {orig['sharpe']:>15.2f}")
    print(f"{'最大回撤':.<20} {enh['max_drawdown']*100:>14.2f}% {orig['max_drawdown']*100:>14.2f}%")
    print(f"{'卡尔玛比率':.<20} {enh['calmar']:>15.2f} {orig['calmar']:>15.2f}")
    print(f"{'总交易成本':.<20} {total_tx_cost*100:>14.2f}%")
    print(f"{'止损触发次数':.<20} {stop_loss_count:>15d}")

    if monthly_turnovers:
        print(f"{'平均月度换手':.<20} {np.mean(monthly_turnovers)*100:>14.1f}%")
    if monthly_ics:
        print(f"{'平均月度IC':.<20} {np.mean(monthly_ics):>15.4f}")
        print(f"{'IC>0占比':.<20} {np.mean([ic>0 for ic in monthly_ics])*100:>14.1f}%")

    # 年度分解
    yearly_enh = calc_yearly_returns(return_dates, enhanced_returns)
    yearly_orig = calc_yearly_returns(return_dates, original_returns)

    if yearly_enh:
        print(f"\n{'年度分解':}")
        print(f"  {'年份':<8} {'v5.0':>12} {'R5等权':>12} {'超额':>12}")
        print(f"  {'-'*48}")
        for year in sorted(yearly_enh.keys()):
            e = yearly_enh.get(year, 0)
            o = yearly_orig.get(year, 0)
            print(f"  {year:<8} {e*100:>11.2f}% {o*100:>11.2f}% {(e-o)*100:>11.2f}%")

    # 保存结果
    result = {
        'version': 'v5.0',
        'config': {
            'n_stocks': N_STOCKS,
            'max_weight': MAX_WEIGHT,
            'max_turnover': MAX_TURNOVER,
            'min_market_cap': MIN_MARKET_CAP,
            'label_period': LABEL_PERIOD,
            'rebalance': 'monthly',
            'dynamic_ratio': DYNAMIC_RATIO,
            'csi300_ratio_base': CSI300_RATIO,
            'ratio_lookback_months': RATIO_LOOKBACK_MONTHS,
        },
        'v5_enhanced': {k: float(v) if isinstance(v, (np.floating, float)) else v
                        for k, v in enh.items()},
        'r5_baseline': {k: float(v) if isinstance(v, (np.floating, float)) else v
                        for k, v in orig.items()},
        'total_tx_cost': float(total_tx_cost),
        'stop_loss_count': stop_loss_count,
        'avg_monthly_turnover': float(np.mean(monthly_turnovers)) if monthly_turnovers else 0,
        'avg_monthly_ic': float(np.mean(monthly_ics)) if monthly_ics else 0,
        'yearly_returns': {
            'v5': {k: float(v) for k, v in yearly_enh.items()},
            'baseline': {k: float(v) for k, v in yearly_orig.items()},
        },
    }

    # 特征重要性
    if ml_scorer.train_count > 0:
        fi = ml_scorer.get_feature_importance(30)
        result['top_features'] = [{'name': n, 'importance': float(v)} for n, v in fi]
        print(f"\nTop-15 重要因子:")
        for name, imp in fi[:15]:
            print(f"  {name:<40} {imp:.0f}")

    output_path = os.path.join(BASE_DIR, 'factor_backtest_result_v5.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {output_path}")

    return result


# ============================================================================
# 入口
# ============================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='v5.0 ML驱动选股回测')
    parser.add_argument('--start', default='2022-01-01', help='回测开始日期')
    parser.add_argument('--end', default='2025-12-31', help='回测结束日期')
    args = parser.parse_args()

    result = run_v5_backtest(start_date=args.start, end_date=args.end)
