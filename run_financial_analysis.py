#!/usr/bin/env python3
"""
v5.1 财务分析: 200只标的的财务画像 + 行业因子

从ClickHouse已入库的Tushare数据生成:
  1. reports/stock_financials.csv  — 200只标的 × 财务指标矩阵
  2. reports/industry_distribution.csv — 行业分布统计
  3. reports/financial_summary.md  — 可读的财务画像报告

数据源:
  - tushare_income: 利润表
  - tushare_balancesheet: 资产负债表
  - tushare_cashflow: 现金流量表
  - tushare_daily_basic: 每日估值指标
  - stock_block: 行业/指数分类
"""

import os
import sys
import pandas as pd
import numpy as np
import clickhouse_connect
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ========== 配置 ==========
CLICKHOUSE_HOST = '192.168.0.74'
CLICKHOUSE_PORT = 8123
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
POOL_CSV = os.path.join(BASE_DIR, 'r5_weights_local.csv')


def get_client():
    return clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST, port=CLICKHOUSE_PORT,
        compress=False, query_limit=10000000
    )


def load_stock_pool():
    """加载R5股票池, 返回6位code列表"""
    df = pd.read_csv(POOL_CSV)
    codes = df['code'].astype(str).str.zfill(6).tolist()
    logger.info(f"股票池: {len(codes)}只标的")
    return codes


def code_to_ts(code):
    """6位code → Tushare ts_code (000001 → 000001.SZ)"""
    return f"{code}.SZ" if code.startswith(('0', '3')) else f"{code}.SH"


def ts_to_code(ts_code):
    """Tushare ts_code → 6位code"""
    return ts_code.split('.')[0]


# ========== Step 1: 财务数据查询 ==========

def fetch_income(client, ts_codes, report_date, yoy_report_date):
    """查询利润表 (最新期 + 同比期)"""
    ts_str = "','".join(ts_codes)

    # 最新期
    query = f"""
        SELECT ts_code, end_date, revenue, n_income_attr_p, operate_profit,
               total_revenue, basic_eps, oper_cost
        FROM tushare_income
        WHERE ts_code IN ('{ts_str}')
          AND end_date <= '{report_date}' AND report_type = '1'
        ORDER BY end_date DESC LIMIT 1 BY ts_code
    """
    res = client.query(query)
    latest = {}
    for row in res.result_rows:
        latest[row[0]] = {
            'end_date': row[1], 'revenue': row[2] or 0,
            'net_profit': row[3] or 0, 'oper_profit': row[4] or 0,
            'total_revenue': row[5] or 0, 'eps': row[6] or 0,
            'oper_cost': row[7] or 0,
        }

    # 同比期
    query_yoy = f"""
        SELECT ts_code, revenue, n_income_attr_p, operate_profit
        FROM tushare_income
        WHERE ts_code IN ('{ts_str}')
          AND end_date <= '{yoy_report_date}' AND report_type = '1'
        ORDER BY end_date DESC LIMIT 1 BY ts_code
    """
    res_yoy = client.query(query_yoy)
    yoy = {}
    for row in res_yoy.result_rows:
        yoy[row[0]] = {
            'revenue': row[1] or 0, 'net_profit': row[2] or 0,
            'oper_profit': row[3] or 0,
        }

    return latest, yoy


def fetch_balancesheet(client, ts_codes, report_date):
    """查询资产负债表"""
    ts_str = "','".join(ts_codes)
    query = f"""
        SELECT ts_code, total_assets, total_liab,
               total_hldr_eqy_exc_min_int, total_cur_assets,
               total_cur_liab, accounts_receiv, inventories, goodwill
        FROM tushare_balancesheet
        WHERE ts_code IN ('{ts_str}')
          AND end_date <= '{report_date}' AND report_type = '1'
        ORDER BY end_date DESC LIMIT 1 BY ts_code
    """
    res = client.query(query)
    data = {}
    for row in res.result_rows:
        data[row[0]] = {
            'total_assets': row[1] or 0, 'total_liab': row[2] or 0,
            'equity': row[3] or 0, 'cur_assets': row[4] or 0,
            'cur_liab': row[5] or 0, 'receivables': row[6] or 0,
            'inventories': row[7] or 0, 'goodwill': row[8] or 0,
        }
    return data


def fetch_cashflow(client, ts_codes, report_date):
    """查询现金流量表"""
    ts_str = "','".join(ts_codes)
    query = f"""
        SELECT ts_code, n_cashflow_act, free_cashflow, c_fr_sale_sg
        FROM tushare_cashflow
        WHERE ts_code IN ('{ts_str}')
          AND end_date <= '{report_date}' AND report_type = '1'
        ORDER BY end_date DESC LIMIT 1 BY ts_code
    """
    res = client.query(query)
    data = {}
    for row in res.result_rows:
        data[row[0]] = {
            'oper_cf': row[1] or 0, 'fcf': row[2] or 0,
            'cash_from_sales': row[3] or 0,
        }
    return data


def fetch_daily_basic(client, ts_codes, as_of_date):
    """查询每日估值指标 (最近交易日)"""
    ts_str = "','".join(ts_codes)
    as_of_str = as_of_date.replace('-', '')
    query = f"""
        SELECT ts_code, pe_ttm, pb, ps_ttm, turnover_rate_f,
               dv_ttm, total_mv, circ_mv
        FROM tushare_daily_basic
        WHERE ts_code IN ('{ts_str}')
          AND trade_date <= '{as_of_str}'
        ORDER BY trade_date DESC LIMIT 1 BY ts_code
    """
    res = client.query(query)
    data = {}
    for row in res.result_rows:
        data[row[0]] = {
            'pe_ttm': row[1] or 0, 'pb': row[2] or 0,
            'ps_ttm': row[3] or 0, 'turnover_rate': row[4] or 0,
            'dv_ttm': row[5] or 0, 'total_mv': row[6] or 0,
            'circ_mv': row[7] or 0,
        }
    return data


# ========== Step 2: 行业分类 ==========

def fetch_industry(client, codes):
    """
    从stock_block_em获取东方财富行业分类 (85个行业板块).
    回退到stock_block concept作为补充.
    """
    code_str = "','".join(codes)
    industry_map = {}

    # 优先: stock_block_em 的 industry 类型 (真正的行业分类)
    try:
        res = client.query(
            f"SELECT stock_code, block_name FROM stock_block_em "
            f"WHERE stock_code IN ('{code_str}') AND block_type = 'industry'"
        )
        for row in res.result_rows:
            code = str(row[0]).zfill(6)
            if code not in industry_map:
                industry_map[code] = row[1]
        logger.info(f"行业分类来源: stock_block_em(industry), 覆盖{len(industry_map)}只")
    except Exception as e:
        logger.warning(f"stock_block_em查询失败: {e}")

    # 补充: 对缺失的股票, 用stock_block concept中出现最少的概念作为近似行业
    missing = [c for c in codes if c not in industry_map]
    if missing:
        miss_str = "','".join(missing)
        try:
            res = client.query(
                f"SELECT stock_code, block_name FROM stock_block "
                f"WHERE stock_code IN ('{miss_str}') AND block_type = 'concept'"
            )
            # 每只股票可能有多个concept, 取第一个
            for row in res.result_rows:
                code = str(row[0]).zfill(6)
                if code not in industry_map:
                    industry_map[code] = row[1]
        except Exception:
            pass
        logger.info(f"补充concept后覆盖: {len(industry_map)}只 (缺失{len(codes) - len(industry_map)}只)")

    return industry_map


# ========== Step 3: 计算财务指标 ==========

def compute_financial_metrics(codes, ts_codes_map, income_latest, income_yoy,
                              bs_data, cf_data, daily_data, industry_map):
    """组装所有财务指标为DataFrame"""
    records = []

    for code in codes:
        ts_code = ts_codes_map[code]
        row = {'code': code, 'ts_code': ts_code}

        # 行业
        row['industry'] = industry_map.get(code, '未知')

        # --- 利润表指标 ---
        inc = income_latest.get(ts_code, {})
        inc_yoy = income_yoy.get(ts_code, {})

        row['report_date'] = inc.get('end_date', '')
        row['revenue'] = inc.get('revenue', np.nan)
        row['net_profit'] = inc.get('net_profit', np.nan)
        row['oper_profit'] = inc.get('oper_profit', np.nan)
        row['eps'] = inc.get('eps', np.nan)

        # 毛利率 = (营收 - 营业成本) / 营收
        rev = inc.get('total_revenue', 0)
        cost = inc.get('oper_cost', 0)
        row['gross_margin'] = (rev - cost) / rev * 100 if abs(rev) > 1e-6 else np.nan

        # 净利率
        row['net_margin'] = inc.get('net_profit', 0) / rev * 100 if abs(rev) > 1e-6 else np.nan

        # YoY增长
        if inc and inc_yoy:
            prev_rev = inc_yoy.get('revenue', 0)
            prev_np = inc_yoy.get('net_profit', 0)
            row['revenue_yoy'] = (inc['revenue'] / prev_rev - 1) * 100 if abs(prev_rev) > 1e-6 else np.nan
            row['net_profit_yoy'] = (inc['net_profit'] / prev_np - 1) * 100 if abs(prev_np) > 1e-6 else np.nan
        else:
            row['revenue_yoy'] = np.nan
            row['net_profit_yoy'] = np.nan

        # --- 资产负债表指标 ---
        bs = bs_data.get(ts_code, {})
        equity = bs.get('equity', 0)
        total_assets = bs.get('total_assets', 0)
        total_liab = bs.get('total_liab', 0)

        row['total_assets'] = total_assets if total_assets else np.nan
        row['equity'] = equity if equity else np.nan

        # ROE = 净利润 / 股东权益
        row['roe'] = inc.get('net_profit', 0) / equity * 100 if abs(equity) > 1e-6 else np.nan

        # ROA = 净利润 / 总资产
        row['roa'] = inc.get('net_profit', 0) / total_assets * 100 if abs(total_assets) > 1e-6 else np.nan

        # 资产负债率
        row['debt_ratio'] = total_liab / total_assets * 100 if abs(total_assets) > 1e-6 else np.nan

        # 流动比率
        cur_assets = bs.get('cur_assets', 0)
        cur_liab = bs.get('cur_liab', 0)
        row['current_ratio'] = cur_assets / cur_liab if abs(cur_liab) > 1e-6 else np.nan

        # 商誉占比
        row['goodwill_ratio'] = bs.get('goodwill', 0) / total_assets * 100 if abs(total_assets) > 1e-6 else np.nan

        # --- 现金流指标 ---
        cf = cf_data.get(ts_code, {})
        oper_cf = cf.get('oper_cf', 0)
        net_profit = inc.get('net_profit', 0)

        row['oper_cashflow'] = oper_cf if oper_cf else np.nan
        row['free_cashflow'] = cf.get('fcf', np.nan)

        # 经营现金流/净利润
        row['cf_to_profit'] = oper_cf / net_profit if abs(net_profit) > 1e-6 else np.nan

        # --- 估值指标 ---
        daily = daily_data.get(ts_code, {})
        row['pe_ttm'] = daily.get('pe_ttm', np.nan) or np.nan
        row['pb'] = daily.get('pb', np.nan) or np.nan
        row['ps_ttm'] = daily.get('ps_ttm', np.nan) or np.nan
        row['turnover_rate'] = daily.get('turnover_rate', np.nan) or np.nan
        row['dv_ttm'] = daily.get('dv_ttm', np.nan) or np.nan
        row['total_mv'] = daily.get('total_mv', np.nan) or np.nan
        row['circ_mv'] = daily.get('circ_mv', np.nan) or np.nan

        records.append(row)

    df = pd.DataFrame(records)
    return df


# ========== Step 4: 行业内排名 ==========

def add_industry_ranks(df):
    """在行业内对关键指标做排名 (百分位)"""
    rank_cols = ['roe', 'roa', 'revenue_yoy', 'net_profit_yoy', 'net_margin',
                 'debt_ratio', 'cf_to_profit', 'pe_ttm', 'pb']

    for col in rank_cols:
        if col in df.columns:
            # 行业内百分位排名
            rank_col = f'{col}_ind_rank'
            df[rank_col] = df.groupby('industry')[col].rank(pct=True, na_option='keep')

    return df


# ========== Step 5: 行业OneHot编码 ==========

def generate_industry_onehot(df):
    """生成行业OneHot编码矩阵"""
    if 'industry' not in df.columns or df['industry'].nunique() < 2:
        logger.warning("行业数据不足, 跳过OneHot编码")
        return pd.DataFrame()

    onehot = pd.get_dummies(df[['code', 'industry']], columns=['industry'], prefix='ind')
    onehot = onehot.set_index('code')
    return onehot


# ========== Step 6: 报告生成 ==========

def generate_summary_report(df, industry_dist, output_path):
    """生成Markdown格式的财务画像报告"""
    lines = []
    lines.append(f"# v5.1 财务画像报告")
    lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"\n标的数量: {len(df)}只")

    # 数据覆盖率
    lines.append(f"\n## 数据覆盖率\n")
    key_cols = ['roe', 'revenue_yoy', 'net_profit_yoy', 'net_margin', 'debt_ratio',
                'cf_to_profit', 'pe_ttm', 'pb', 'industry']
    for col in key_cols:
        if col in df.columns:
            valid = df[col].notna().sum() if col != 'industry' else (df[col] != '未知').sum()
            pct = valid / len(df) * 100
            lines.append(f"- {col}: {valid}/{len(df)} ({pct:.0f}%)")

    # 行业分布
    if not industry_dist.empty:
        lines.append(f"\n## 行业分布 (Top 20)\n")
        lines.append(f"| 行业 | 数量 | 占比 | 平均市值(亿) | 平均ROE(%) |")
        lines.append(f"|------|------|------|-------------|-----------|")
        top20 = industry_dist.head(20)
        for _, row in top20.iterrows():
            lines.append(f"| {row['industry']} | {row['count']:.0f} | {row['pct']:.1f}% | "
                         f"{row.get('avg_mv', 0):.0f} | {row.get('avg_roe', 0):.1f} |")

    # 关键指标统计
    lines.append(f"\n## 关键财务指标分布\n")
    stat_cols = {
        'roe': 'ROE(%)', 'roa': 'ROA(%)', 'revenue_yoy': '营收增长(%)',
        'net_profit_yoy': '净利润增长(%)', 'gross_margin': '毛利率(%)',
        'net_margin': '净利率(%)', 'debt_ratio': '资产负债率(%)',
        'cf_to_profit': '经营现金流/净利润', 'pe_ttm': 'PE(TTM)',
        'pb': 'PB', 'total_mv': '总市值(万元)',
    }
    lines.append(f"| 指标 | 均值 | 中位数 | P25 | P75 | 最小 | 最大 |")
    lines.append(f"|------|------|--------|-----|-----|------|------|")
    for col, label in stat_cols.items():
        if col in df.columns:
            s = df[col].dropna()
            if len(s) > 0:
                lines.append(
                    f"| {label} | {s.mean():.2f} | {s.median():.2f} | "
                    f"{s.quantile(0.25):.2f} | {s.quantile(0.75):.2f} | "
                    f"{s.min():.2f} | {s.max():.2f} |"
                )

    # Top/Bottom排名
    for metric, label, ascending in [
        ('roe', 'ROE', False), ('revenue_yoy', '营收增长率', False),
        ('net_profit_yoy', '净利润增长率', False), ('debt_ratio', '资产负债率', True),
    ]:
        if metric in df.columns:
            lines.append(f"\n### {label} Top 10\n")
            top = df.dropna(subset=[metric]).nsmallest(10, metric) if ascending else \
                  df.dropna(subset=[metric]).nlargest(10, metric)
            lines.append(f"| 代码 | 行业 | {label}(%) | 总市值(万) |")
            lines.append(f"|------|------|-----------|-----------|")
            for _, r in top.iterrows():
                lines.append(f"| {r['code']} | {r.get('industry', '')} | "
                             f"{r[metric]:.2f} | {r.get('total_mv', 0):.0f} |")

    report = '\n'.join(lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"报告已生成: {output_path}")
    return report


# ========== 主流程 ==========

def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    client = get_client()
    logger.info("ClickHouse连接成功")

    # 1. 加载股票池
    codes = load_stock_pool()
    ts_codes_map = {code: code_to_ts(code) for code in codes}
    ts_codes = list(ts_codes_map.values())

    # 2. 确定报告期 (使用最近完成的财报期)
    now = datetime.now()
    year, month = now.year, now.month
    if month <= 4:
        report_date = f"{year - 1}0930"  # Q3
    elif month <= 8:
        report_date = f"{year - 1}1231"  # 年报
    elif month <= 10:
        report_date = f"{year}0630"      # 中报
    else:
        report_date = f"{year}0930"       # Q3
    yoy_year = str(int(report_date[:4]) - 1)
    yoy_report_date = yoy_year + report_date[4:]
    as_of_date = now.strftime('%Y-%m-%d')
    logger.info(f"报告期: {report_date}, 同比期: {yoy_report_date}, 估值截面: {as_of_date}")

    # 3. 查询财务数据
    logger.info("查询利润表...")
    income_latest, income_yoy = fetch_income(client, ts_codes, report_date, yoy_report_date)
    logger.info(f"  利润表覆盖: 最新{len(income_latest)}只, 同比{len(income_yoy)}只")

    logger.info("查询资产负债表...")
    bs_data = fetch_balancesheet(client, ts_codes, report_date)
    logger.info(f"  资产负债表覆盖: {len(bs_data)}只")

    logger.info("查询现金流量表...")
    cf_data = fetch_cashflow(client, ts_codes, report_date)
    logger.info(f"  现金流量表覆盖: {len(cf_data)}只")

    logger.info("查询每日估值指标...")
    daily_data = fetch_daily_basic(client, ts_codes, as_of_date)
    logger.info(f"  每日指标覆盖: {len(daily_data)}只")

    # 4. 查询行业分类
    logger.info("查询行业分类...")
    industry_map = fetch_industry(client, codes)
    logger.info(f"  行业分类覆盖: {len(industry_map)}只")

    # 5. 计算财务指标
    logger.info("计算财务指标矩阵...")
    df = compute_financial_metrics(
        codes, ts_codes_map, income_latest, income_yoy,
        bs_data, cf_data, daily_data, industry_map
    )

    # 6. 行业内排名
    df = add_industry_ranks(df)

    # 7. 保存stock_financials.csv
    csv_path = os.path.join(REPORTS_DIR, 'stock_financials.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    logger.info(f"财务指标矩阵已保存: {csv_path} ({len(df)}行 × {len(df.columns)}列)")

    # 8. 行业分布统计
    if df['industry'].nunique() > 1 and (df['industry'] != '未知').any():
        ind_stats = df.groupby('industry').agg(
            count=('code', 'count'),
            avg_mv=('total_mv', 'mean'),
            avg_roe=('roe', 'mean'),
            avg_pe=('pe_ttm', 'mean'),
            avg_revenue_yoy=('revenue_yoy', 'mean'),
        ).reset_index()
        ind_stats['pct'] = ind_stats['count'] / len(df) * 100
        ind_stats = ind_stats.sort_values('count', ascending=False)
        ind_csv = os.path.join(REPORTS_DIR, 'industry_distribution.csv')
        ind_stats.to_csv(ind_csv, index=False, encoding='utf-8-sig')
        logger.info(f"行业分布已保存: {ind_csv} ({len(ind_stats)}个行业)")
    else:
        ind_stats = pd.DataFrame()
        logger.warning("行业数据不足, 跳过行业分布统计")

    # 9. 行业OneHot编码 (可作为ML特征)
    onehot = generate_industry_onehot(df)
    if not onehot.empty:
        onehot_path = os.path.join(REPORTS_DIR, 'industry_onehot.csv')
        onehot.to_csv(onehot_path, encoding='utf-8-sig')
        logger.info(f"行业OneHot编码已保存: {onehot_path} ({onehot.shape[1]}个行业特征)")

    # 10. 生成报告
    report_path = os.path.join(REPORTS_DIR, 'financial_summary.md')
    generate_summary_report(df, ind_stats, report_path)

    # 打印摘要
    print("\n" + "=" * 60)
    print("v5.1 财务分析完成")
    print("=" * 60)
    print(f"标的数: {len(df)}")
    print(f"报告期: {report_date}")
    print(f"\n关键指标中位数:")
    for col, label in [('roe', 'ROE(%)'), ('revenue_yoy', '营收增长(%)'),
                        ('net_margin', '净利率(%)'), ('debt_ratio', '资产负债率(%)'),
                        ('pe_ttm', 'PE(TTM)'), ('pb', 'PB')]:
        if col in df.columns:
            v = df[col].median()
            print(f"  {label}: {v:.2f}" if pd.notna(v) else f"  {label}: N/A")

    known = (df['industry'] != '未知').sum()
    print(f"\n行业覆盖: {known}/{len(df)}")
    if known > 0:
        top3 = df[df['industry'] != '未知']['industry'].value_counts().head(3)
        for ind, cnt in top3.items():
            print(f"  {ind}: {cnt}只")

    print(f"\n输出文件:")
    print(f"  {csv_path}")
    if not ind_stats.empty:
        print(f"  {os.path.join(REPORTS_DIR, 'industry_distribution.csv')}")
    if not onehot.empty:
        print(f"  {os.path.join(REPORTS_DIR, 'industry_onehot.csv')}")
    print(f"  {report_path}")


if __name__ == '__main__':
    main()
