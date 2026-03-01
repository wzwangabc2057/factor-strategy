"""
使用本地ClickHouse数据计算PE、PB估值
PE = 当前价格 / EPS
PB = 当前价格 / BVPS
"""

import pandas as pd
import numpy as np
import clickhouse_connect
import akshare as ak
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def get_valuation_from_local(codes: list) -> pd.DataFrame:
    """
    从本地ClickHouse计算PE和PB
    """
    print("=" * 60)
    print("从本地ClickHouse计算估值数据")
    print(f"目标股票: {len(codes)} 只")
    print("=" * 60)

    client = clickhouse_connect.get_client(host='127.0.0.1', port=8123)
    codes_str = "','".join(codes)

    # 1. 获取最新价格
    print("\n获取最新价格...")
    price_query = f"""
        SELECT code,
               max(date) as latest_date,
               argMax(toFloat64OrNull(close), date) as close
        FROM default.stock_data
        WHERE code IN ('{codes_str}')
        GROUP BY code
    """
    price_result = client.query(price_query)

    price_data = []
    for row in price_result.result_rows:
        price_data.append({
            'code': row[0],
            'latest_date': row[1],
            'price': float(row[2]) if row[2] else None
        })
    price_df = pd.DataFrame(price_data)
    print(f"  获取到 {len(price_df)} 只股票价格")

    # 2. 获取财务数据 (2024年报)
    print("\n获取财务数据 (2024年报)...")
    fin_query = f"""
        SELECT code, eps, bvps, roe,
               net_profit, total_shares,
               shareholders_equity
        FROM default.stock_financial
        WHERE code IN ('{codes_str}')
          AND report_date = '2024-12-31'
    """
    fin_result = client.query(fin_query)

    fin_data = []
    for row in fin_result.result_rows:
        fin_data.append({
            'code': row[0],
            'eps': float(row[1]) if row[1] else None,
            'bvps': float(row[2]) if row[2] else None,
            'roe': float(row[3]) if row[3] else None,
            'net_profit': float(row[4]) if row[4] else None,
            'total_shares': float(row[5]) if row[5] else None,
            'shareholders_equity': float(row[6]) if row[6] else None
        })
    fin_df = pd.DataFrame(fin_data)
    # 去重
    fin_df = fin_df.drop_duplicates(subset=['code'], keep='first')
    print(f"  获取到 {len(fin_df)} 只股票财务数据")

    # 3. 合并数据
    print("\n合并并计算PE/PB...")
    result = price_df.merge(fin_df, on='code', how='left')

    # 计算PE和PB
    result['pe_ttm'] = result.apply(
        lambda row: row['price'] / row['eps'] if row['eps'] and row['eps'] > 0 and row['price'] else None,
        axis=1
    )
    result['pb'] = result.apply(
        lambda row: row['price'] / row['bvps'] if row['bvps'] and row['bvps'] > 0 and row['price'] else None,
        axis=1
    )

    # 计算市值 (亿)
    result['total_mv'] = result.apply(
        lambda row: row['price'] * row['total_shares'] / 1e8 if row['price'] and row['total_shares'] else None,
        axis=1
    )

    # 显示PE分布
    valid_pe = result[(result['pe_ttm'] > 0) & (result['pe_ttm'] < 200)]
    if len(valid_pe) > 0:
        print(f"\n  PE分布 (有效数据 {len(valid_pe)} 只):")
        print(f"    最小: {valid_pe['pe_ttm'].min():.1f}")
        print(f"    25%:  {valid_pe['pe_ttm'].quantile(0.25):.1f}")
        print(f"    中位: {valid_pe['pe_ttm'].median():.1f}")
        print(f"    75%:  {valid_pe['pe_ttm'].quantile(0.75):.1f}")
        print(f"    最大: {valid_pe['pe_ttm'].max():.1f}")

    # 显示PB分布
    valid_pb = result[(result['pb'] > 0) & (result['pb'] < 20)]
    if len(valid_pb) > 0:
        print(f"\n  PB分布 (有效数据 {len(valid_pb)} 只):")
        print(f"    最小: {valid_pb['pb'].min():.2f}")
        print(f"    中位: {valid_pb['pb'].median():.2f}")
        print(f"    最大: {valid_pb['pb'].max():.2f}")

    return result


def get_dividend_yield(codes: list) -> pd.DataFrame:
    """
    尝试从akshare获取股息率数据
    """
    print("\n" + "=" * 60)
    print("获取股息率数据")
    print("=" * 60)

    try:
        # 方法1: 从东方财富获取股息率
        print("\n尝试方法1: stock_dividend_cninfo...")
        df_div = ak.stock_dividend_cninfo()
        df_div = df_div.rename(columns={
            '证券代码': 'code',
            '股息率': 'dividend_yield'
        })
        df_div['code'] = df_div['code'].astype(str).str.zfill(6)
        df_div['dividend_yield'] = pd.to_numeric(df_div['dividend_yield'], errors='coerce')
        df_div['dividend_yield'] = df_div['dividend_yield'] / 100  # 转为小数
        df_div = df_div.drop_duplicates(subset=['code'], keep='first')
        df_div = df_div[df_div['code'].isin(codes)]

        if len(df_div) > 0:
            print(f"  获取到 {len(df_div)} 只股票股息率")
            valid_div = df_div[df_div['dividend_yield'] > 0]
            if len(valid_div) > 0:
                print(f"\n  股息率分布 (有效数据 {len(valid_div)} 只):")
                print(f"    最小: {valid_div['dividend_yield'].min()*100:.2f}%")
                print(f"    中位: {valid_div['dividend_yield'].median()*100:.2f}%")
                print(f"    最大: {valid_div['dividend_yield'].max()*100:.2f}%")
            return df_div[['code', 'dividend_yield']]
    except Exception as e:
        print(f"  方法1失败: {e}")

    try:
        # 方法2: 从新浪获取分红数据
        print("\n尝试方法2: 从个股分红数据估算...")
        dividend_data = []
        sample_codes = codes[:10]  # 只测试10只

        for code in sample_codes:
            try:
                df = ak.stock_history_dividend_detail(symbol=code, indicator="分红")
                if not df.empty and '每股股利' in df.columns:
                    latest_div = df.iloc[0]['每股股利']
                    dividend_data.append({
                        'code': code,
                        'dividend_per_share': float(latest_div) if latest_div else 0
                    })
            except:
                pass

        if dividend_data:
            print(f"  获取到 {len(dividend_data)} 只股票分红数据")
            return pd.DataFrame(dividend_data)
    except Exception as e:
        print(f"  方法2失败: {e}")

    print("  无法获取股息率数据，将使用默认值")
    return pd.DataFrame()


def main():
    # 读取持仓
    portfolio_file = "/Users/kangbing/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/wxid_gusa1p3piit022_6ca9/msg/file/2026-01/四大类策略持仓20251231.xlsx"

    print("读取持仓数据...")
    r4 = pd.read_excel(portfolio_file, sheet_name='稳健型股票组合(风险等级R4)')
    r5 = pd.read_excel(portfolio_file, sheet_name='进取型股票组合(风险等级R5)')
    r4.columns = ['code', 'weight']
    r5.columns = ['code', 'weight']

    all_codes = list(set(
        [c.split('.')[0] for c in r4['code'].tolist()] +
        [c.split('.')[0] for c in r5['code'].tolist()]
    ))
    print(f"共 {len(all_codes)} 只股票\n")

    # 1. 从本地计算PE/PB
    valuation_df = get_valuation_from_local(all_codes)

    # 2. 获取股息率
    dividend_df = get_dividend_yield(all_codes)

    # 3. 合并股息率
    if not dividend_df.empty and 'dividend_yield' in dividend_df.columns:
        valuation_df = valuation_df.merge(dividend_df[['code', 'dividend_yield']], on='code', how='left')
    else:
        valuation_df['dividend_yield'] = None

    # 填充缺失值
    valuation_df['dividend_yield'] = valuation_df['dividend_yield'].fillna(0)

    # 4. 保存到ClickHouse
    print("\n" + "=" * 60)
    print("保存到ClickHouse")
    print("=" * 60)

    client = clickhouse_connect.get_client(host='127.0.0.1', port=8123)

    # 删除旧表
    client.command("DROP TABLE IF EXISTS default.valuation_local")

    # 创建新表
    client.command("""
        CREATE TABLE default.valuation_local (
            code String,
            price Nullable(Float32),
            eps Nullable(Float32),
            bvps Nullable(Float32),
            roe Nullable(Float32),
            pe_ttm Nullable(Float32),
            pb Nullable(Float32),
            total_mv Nullable(Float64),
            dividend_yield Nullable(Float32),
            update_time DateTime DEFAULT now()
        ) ENGINE = MergeTree()
        ORDER BY code
    """)

    # 选择需要的列
    cols = ['code', 'price', 'eps', 'bvps', 'roe', 'pe_ttm', 'pb', 'total_mv', 'dividend_yield']
    df_save = valuation_df[[c for c in cols if c in valuation_df.columns]].copy()

    # 确保数据类型
    for col in df_save.columns:
        if col != 'code':
            df_save[col] = pd.to_numeric(df_save[col], errors='coerce')

    client.insert_df('default.valuation_local', df_save)
    print(f"  保存 {len(df_save)} 条数据到 valuation_local 表")

    # 5. 保存CSV备份
    output_file = "/Users/kangbing/112/pythontest/quantriji/enhanced_strategy_v1/valuation_local.csv"
    valuation_df.to_csv(output_file, index=False)
    print(f"\nCSV备份已保存到: {output_file}")

    # 6. 验证
    print("\n" + "=" * 60)
    print("数据验证")
    print("=" * 60)

    result = client.query("""
        SELECT count(*),
               avg(pe_ttm), avg(pb), avg(roe), avg(dividend_yield)
        FROM default.valuation_local
        WHERE pe_ttm > 0 AND pe_ttm < 200
    """)
    row = result.result_rows[0]
    print(f"有效数据: {row[0]} 只")
    print(f"平均PE: {row[1]:.1f}")
    print(f"平均PB: {row[2]:.2f}")
    print(f"平均ROE: {row[3]:.2f}%")
    print(f"平均股息率: {row[4]*100:.2f}%")

    # 显示样本
    print("\n样本数据:")
    sample = client.query("""
        SELECT code, price, eps, pe_ttm, pb, roe, dividend_yield
        FROM default.valuation_local
        WHERE pe_ttm > 0 AND pe_ttm < 100
        ORDER BY total_mv DESC
        LIMIT 10
    """)
    print(f"{'代码':<8} {'价格':>8} {'EPS':>8} {'PE':>8} {'PB':>8} {'ROE':>8} {'股息率':>8}")
    for row in sample.result_rows:
        div_pct = row[6]*100 if row[6] else 0
        print(f"{row[0]:<8} {row[1]:>8.2f} {row[2]:>8.2f} {row[3]:>8.1f} {row[4]:>8.2f} {row[5]:>8.2f}% {div_pct:>7.2f}%")

    print("\n完成!")


if __name__ == '__main__':
    main()
