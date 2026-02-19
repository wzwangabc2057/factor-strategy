#!/usr/bin/env python3
"""Check R5 industry coverage from stock_block_em"""
import clickhouse_driver
import csv

ch = clickhouse_driver.Client(host='192.168.0.74')

# Load R5 codes
codes = []
with open('/Users/macstudio/112/pythontest/r5_weights_local.csv') as f:
    for row in csv.DictReader(f):
        codes.append(row['code'].zfill(6))

code_str = "','".join(codes)

# Check coverage
res = ch.execute(
    f"SELECT count(DISTINCT stock_code) FROM stock_block_em "
    f"WHERE block_type='industry' AND stock_code IN ('{code_str}')"
)
print(f"R5 industry coverage: {res[0][0]}/{len(codes)}")

# All industry sectors
res = ch.execute(
    "SELECT block_name, count() FROM stock_block_em "
    "WHERE block_type='industry' GROUP BY block_name ORDER BY block_name"
)
print(f"\nTotal industry sectors: {len(res)}")
for r in res:
    print(f"  {r[0]}: {r[1]}")

# R5 distribution
print("\n=== R5 industry distribution ===")
res = ch.execute(
    f"SELECT block_name, count() FROM stock_block_em "
    f"WHERE block_type='industry' AND stock_code IN ('{code_str}') "
    f"GROUP BY block_name ORDER BY count() DESC"
)
for r in res:
    print(f"  {r[0]}: {r[1]}")

# Missing stocks
res = ch.execute(
    f"SELECT DISTINCT stock_code FROM stock_block_em "
    f"WHERE block_type='industry' AND stock_code IN ('{code_str}')"
)
covered = {r[0] for r in res}
missing = [c for c in codes if c not in covered]
if missing:
    print(f"\nMissing stocks ({len(missing)}):")
    for m in missing:
        print(f"  {m}")

# Also check BS/CF gaps
print("\n=== BS/CF data gaps ===")
for table in ['tushare_balancesheet', 'tushare_cashflow']:
    ts_codes = []
    for c in codes:
        if c.startswith(('6', '9')):
            ts_codes.append(f"{c}.SH")
        else:
            ts_codes.append(f"{c}.SZ")
    ts_str = "','".join(ts_codes)
    res = ch.execute(
        f"SELECT count(DISTINCT ts_code) FROM {table} WHERE ts_code IN ('{ts_str}')"
    )
    covered_cnt = res[0][0]
    if covered_cnt < len(codes):
        # Find missing
        res2 = ch.execute(
            f"SELECT DISTINCT ts_code FROM {table} WHERE ts_code IN ('{ts_str}')"
        )
        have = {r[0] for r in res2}
        miss = [t for t in ts_codes if t not in have]
        print(f"  {table}: {covered_cnt}/{len(codes)}, missing: {miss}")
    else:
        print(f"  {table}: {covered_cnt}/{len(codes)} OK")
