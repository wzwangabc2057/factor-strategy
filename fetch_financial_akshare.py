"""
从 akshare 批量获取财务数据，存为本地 CSV 缓存
在 Mac Studio 上运行: python3 fetch_financial_akshare.py
"""
import pandas as pd
import numpy as np
import time
import os
import sys

import akshare as ak


def fetch_financial_data(codes: list, output_path: str):
    """批量获取财务摘要数据"""
    results = []
    failed = []
    total = len(codes)

    print(f"开始获取 {total} 只股票财务数据...")
    print(f"预计耗时: {total * 2 / 60:.1f} 分钟\n")

    for i, code in enumerate(codes):
        try:
            df = ak.stock_financial_abstract_ths(symbol=code, indicator='按年度')
            if len(df) == 0:
                failed.append((code, 'empty'))
                continue

            # 取最新两年数据用于计算同比
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) >= 2 else None

            def parse_pct(val):
                """解析百分比字符串，如 '15.08%' -> 15.08"""
                if isinstance(val, str) and val.endswith('%'):
                    try:
                        return float(val.replace('%', ''))
                    except:
                        return np.nan
                if isinstance(val, (int, float)):
                    return float(val)
                return np.nan

            def parse_number(val):
                """解析数字，如 '1380.12亿' -> 138012000000"""
                if isinstance(val, (int, float)):
                    return float(val)
                if isinstance(val, str):
                    val = val.strip()
                    if val in ('False', '', '-', '--'):
                        return np.nan
                    try:
                        if '亿' in val:
                            return float(val.replace('亿', '')) * 1e8
                        elif '万' in val:
                            return float(val.replace('万', '')) * 1e4
                        else:
                            return float(val)
                    except:
                        return np.nan
                return np.nan

            row = {
                'code': code,
                'report_year': str(latest['报告期']),
                'roe': parse_pct(latest['净资产收益率']),
                'roe_diluted': parse_pct(latest.get('净资产收益率-摊薄', np.nan)),
                'eps': parse_number(latest['基本每股收益']),
                'bvps': parse_number(latest['每股净资产']),
                'net_profit': parse_number(latest['净利润']),
                'net_profit_yoy': parse_pct(latest['净利润同比增长率']),
                'revenue_yoy': parse_pct(latest['营业总收入同比增长率']),
                'ocfps': parse_number(latest['每股经营现金流']),
                'gross_margin': parse_pct(latest['销售净利率']),
                'asset_liability_ratio': parse_pct(latest['资产负债率']),
            }

            # 计算 ROE 稳定性 (最近3年 ROE 标准差)
            if len(df) >= 3:
                recent_roe = [parse_pct(df.iloc[j]['净资产收益率']) for j in range(-3, 0)]
                recent_roe = [x for x in recent_roe if not np.isnan(x)]
                if len(recent_roe) >= 2:
                    row['roe_std_3y'] = np.std(recent_roe)
                else:
                    row['roe_std_3y'] = np.nan
            else:
                row['roe_std_3y'] = np.nan

            # 上一年数据
            if prev is not None:
                row['prev_eps'] = parse_number(prev['基本每股收益'])
                row['prev_revenue_yoy'] = parse_pct(prev['营业总收入同比增长率'])
            else:
                row['prev_eps'] = np.nan
                row['prev_revenue_yoy'] = np.nan

            results.append(row)

            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{total}] 已完成 {code}, 成功 {len(results)} 只, 失败 {len(failed)} 只")

        except Exception as e:
            failed.append((code, str(e)))
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{total}] 失败 {code}: {e}")

        # 控制请求频率
        time.sleep(0.5)

    # 保存结果
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False)

    print(f"\n{'=' * 50}")
    print(f"完成! 成功: {len(results)}/{total}, 失败: {len(failed)}")
    print(f"保存到: {output_path}")
    if failed:
        print(f"\n失败列表 ({len(failed)} 只):")
        for code, reason in failed[:20]:
            print(f"  {code}: {reason}")
        if len(failed) > 20:
            print(f"  ... 还有 {len(failed) - 20} 只")

    return result_df


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 加载持仓代码
    r4 = pd.read_csv(os.path.join(base_dir, 'r4_weights_local.csv'))
    r5 = pd.read_csv(os.path.join(base_dir, 'r5_weights_local.csv'))

    r4['code'] = r4['code'].astype(str).str.zfill(6)
    r5['code'] = r5['code'].astype(str).str.zfill(6)

    all_codes = sorted(set(r4['code'].tolist() + r5['code'].tolist()))
    print(f"共 {len(all_codes)} 只股票需要获取财务数据\n")

    output_path = os.path.join(base_dir, 'financial_data_akshare.csv')

    # 如果已有缓存且较新，询问是否重新获取
    if os.path.exists(output_path):
        mtime = os.path.getmtime(output_path)
        age_hours = (time.time() - mtime) / 3600
        existing = pd.read_csv(output_path)
        print(f"已有缓存: {len(existing)} 只股票, {age_hours:.1f} 小时前更新")
        if age_hours < 24:
            print("缓存较新(<24h), 跳过获取。删除文件可强制重新获取。")
            return

    fetch_financial_data(all_codes, output_path)


if __name__ == '__main__':
    main()
