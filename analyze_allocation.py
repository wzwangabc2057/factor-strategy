"""分析CSI300 vs CSI500配比问题"""
import clickhouse_driver
import csv
from datetime import date

client = clickhouse_driver.Client(host="192.168.0.74")

# Get CSI300 constituents
csi300 = set()
rows = client.execute("SELECT DISTINCT stock_code FROM stock_block WHERE block_name = '沪深300' AND block_type = 'index'")
for r in rows:
    csi300.add(r[0])
print(f"CSI300 constituents in DB: {len(csi300)}")

# Try to find CSI500
for name in ["中证500", "中证 500", "CSI500"]:
    rows = client.execute(f"SELECT count() FROM stock_block WHERE block_name = '{name}' AND block_type = 'index'")
    print(f"  '{name}': {rows[0][0]} stocks")

# Search for anything with 500
rows = client.execute("SELECT DISTINCT block_name FROM stock_block WHERE block_name LIKE '%500%' AND block_type = 'index'")
print(f"\nIndex blocks containing '500': {[r[0] for r in rows]}")

# Load R5 pool
r5_codes = []
with open("/Users/macstudio/112/pythontest/r5_weights_local.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        r5_codes.append(row["code"])

in300 = [c for c in r5_codes if c in csi300]
not300 = [c for c in r5_codes if c not in csi300]
print(f"\nR5 pool: {len(r5_codes)} total")
print(f"  CSI300成分: {len(in300)} ({len(in300)/len(r5_codes)*100:.1f}%)")
print(f"  非CSI300:   {len(not300)} ({len(not300)/len(r5_codes)*100:.1f}%)")

# Market cap for each group
for label, codes in [("CSI300成分", in300), ("非CSI300", not300)]:
    if not codes:
        continue
    code_list = "','".join(codes)
    sql = f"SELECT avg(market_cap), min(market_cap), max(market_cap), median(market_cap) FROM stock_financial WHERE code IN ('{code_list}') AND date = (SELECT max(date) FROM stock_financial WHERE code = '000001')"
    rows = client.execute(sql)
    if rows and rows[0][0]:
        print(f"  {label} ({len(codes)}只): avg={rows[0][0]:.0f}亿, median={rows[0][3]:.0f}亿, min={rows[0][1]:.0f}亿, max={rows[0][2]:.0f}亿")

# Annual performance: CSI300 vs CSI500
print("\n=== 年度指数表现对比 ===")
print(f"{'年份':>6} {'CSI300':>10} {'CSI500':>10} {'差值(500-300)':>14}")
for year in range(2021, 2026):
    results = {}
    for idx_code, idx_name in [("000300", "CSI300"), ("399905", "CSI500")]:
        rows = client.execute(f"""
            SELECT argMin(close, date) as first_close, argMax(close, date) as last_close
            FROM stock_index
            WHERE code = '{idx_code}' AND toYear(date) = {year}
        """)
        if rows and rows[0][0] and rows[0][1]:
            ret = (rows[0][1] / rows[0][0] - 1) * 100
            results[idx_name] = ret
    if len(results) == 2:
        diff = results["CSI500"] - results["CSI300"]
        print(f"{year:>6} {results['CSI300']:>+9.1f}% {results['CSI500']:>+9.1f}% {diff:>+13.1f}%")

# v5 backtest result by year
print("\n=== v5.0 r3 年度表现 vs 基线 ===")
v5_yearly = {"2022": 0.52, "2023": 10.32, "2024": 18.11, "2025": 47.82}
bl_yearly = {"2022": -9.11, "2023": 8.62, "2024": 14.16, "2025": 44.28}
for yr in ["2022", "2023", "2024", "2025"]:
    print(f"  {yr}: v5={v5_yearly[yr]:+.2f}%, 基线={bl_yearly[yr]:+.2f}%, 超额={v5_yearly[yr]-bl_yearly[yr]:+.2f}%")

print("\n=== 配比分析结论 ===")
print("当前v5策略: 纯ML打分选Top-50, 不区分大盘/中盘配比")
print("问题: 不同年份CSI300和CSI500表现差异巨大:")
print("  2024年: CSI300大幅跑赢CSI500 → 多配大盘有利")
print("  2025年: CSI500大幅跑赢CSI300 → 多配中盘有利")
print("\n优化方向:")
print("  1. 动态配比: 根据市场信号调整大盘/中盘比例")
print("  2. 分层选股: CSI300内选N只 + 非CSI300内选M只")
print("  3. 市值因子作为ML特征已有, 但无显式约束")
