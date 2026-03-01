"""分析大盘/中小盘风格轮动信号"""
import clickhouse_driver
from datetime import date, timedelta

client = clickhouse_driver.Client(host="192.168.0.74")

print("=== CSI300 vs CSI500 vs CSI1000 年度表现 ===")
print(f"{'Year':>6} {'CSI300':>10} {'CSI500':>10} {'CSI1000':>10} {'300vs1000':>12}")
for year in range(2021, 2026):
    results = {}
    for code, name in [("000300", "CSI300"), ("399905", "CSI500"), ("000852", "CSI1000")]:
        rows = client.execute(
            f"SELECT argMin(close,date), argMax(close,date) FROM stock_index "
            f"WHERE code='{code}' AND toYear(date)={year}"
        )
        if rows and rows[0][0] and rows[0][1]:
            results[name] = (rows[0][1] / rows[0][0] - 1) * 100
    if results:
        diff = results.get("CSI300", 0) - results.get("CSI1000", 0)
        print(f"{year:>6} {results.get('CSI300',0):>+9.1f}% {results.get('CSI500',0):>+9.1f}% "
              f"{results.get('CSI1000',0):>+9.1f}% {diff:>+11.1f}%")

print("\n=== CSI300 vs CSI1000 月度相对强弱 (正=大盘强) ===")
for year in range(2022, 2026):
    for month in range(1, 13):
        m_start = date(year, month, 1)
        if month == 12:
            m_end = date(year + 1, 1, 1)
        else:
            m_end = date(year, month + 1, 1)
        rets = {}
        for code, name in [("000300", "300"), ("000852", "1000")]:
            rows = client.execute(
                f"SELECT argMin(close,date), argMax(close,date) FROM stock_index "
                f"WHERE code='{code}' AND date>=toDate('{m_start}') AND date<toDate('{m_end}')"
            )
            if rows and rows[0][0] and rows[0][1]:
                rets[name] = (rows[0][1] / rows[0][0] - 1) * 100
        if "300" in rets and "1000" in rets:
            diff = rets["300"] - rets["1000"]
            signal = "LARGE" if diff > 3 else ("SMALL" if diff < -3 else "NEUT")
            print(f"  {year}-{month:02d}: 300={rets['300']:>+6.1f}% 1000={rets['1000']:>+6.1f}% "
                  f"diff={diff:>+6.1f}% [{signal}]")

# 计算滚动3个月相对强弱信号
print("\n=== 滚动3个月相对强弱信号 (用于动态配比) ===")
# 获取月度收盘数据
for code, name in [("000300", "CSI300"), ("000852", "CSI1000")]:
    rows = client.execute(
        f"SELECT toStartOfMonth(date) as m, argMax(close, date) as mc "
        f"FROM stock_index WHERE code='{code}' AND date >= toDate('2021-10-01') "
        f"GROUP BY m ORDER BY m"
    )
    print(f"\n{name} monthly close (last 5):")
    for r in rows[-5:]:
        print(f"  {r[0]}: {r[1]:.2f}")

# 计算动态配比建议
print("\n=== 动态配比建议 ===")
print("逻辑: ")
print("  1. 计算过去3个月CSI300 vs CSI1000的相对收益")
print("  2. 大盘跑赢 -> CSI300_RATIO 提高到 50-60%")
print("  3. 小盘跑赢 -> CSI300_RATIO 降低到 20-30%")
print("  4. 中性 -> CSI300_RATIO 维持 40%")
print("  这就是动态风格轮动!")
