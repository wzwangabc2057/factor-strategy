#!/usr/bin/env python3
"""Analyze revenue & profit growth distribution of R5 232 stocks"""
import pandas as pd
import numpy as np

df = pd.read_csv('reports/stock_financials.csv')
print(f"总标的: {len(df)}只\n")

# === 1. 营收增长分布 ===
rev = df['revenue_yoy'].dropna()
print("=" * 60)
print("一、营收增长率分布")
print("=" * 60)
bins = [(-np.inf, -50), (-50, -30), (-30, -10), (-10, 0), (0, 10), (10, 30), (30, 50), (50, np.inf)]
labels = ['<-50%', '-50~-30%', '-30~-10%', '-10~0%', '0~10%', '10~30%', '30~50%', '>50%']
rev_bins = pd.cut(rev, bins=[b[0] for b in bins] + [np.inf], labels=labels, right=True)
# Manual binning
counts = {}
for label, (lo, hi) in zip(labels, bins):
    mask = (rev > lo) & (rev <= hi)
    counts[label] = mask.sum()

total = len(rev)
print(f"{'区间':<12} {'数量':>6} {'占比':>8} {'累计':>8}")
print("-" * 40)
cum = 0
for label in labels:
    c = counts[label]
    cum += c
    print(f"{label:<12} {c:>6} {c/total*100:>7.1f}% {cum/total*100:>7.1f}%")

neg_rev = (rev < 0).sum()
pos_rev = (rev > 0).sum()
print(f"\n营收正增长: {pos_rev}只 ({pos_rev/total*100:.1f}%)")
print(f"营收负增长: {neg_rev}只 ({neg_rev/total*100:.1f}%)")
print(f"中位数: {rev.median():.2f}%  均值: {rev.mean():.2f}%")

# === 2. 净利润增长分布 ===
np_yoy = df['net_profit_yoy'].dropna()
print(f"\n{'=' * 60}")
print("二、净利润增长率分布")
print("=" * 60)
counts2 = {}
for label, (lo, hi) in zip(labels, bins):
    mask = (np_yoy > lo) & (np_yoy <= hi)
    counts2[label] = mask.sum()

total2 = len(np_yoy)
print(f"{'区间':<12} {'数量':>6} {'占比':>8} {'累计':>8}")
print("-" * 40)
cum = 0
for label in labels:
    c = counts2[label]
    cum += c
    print(f"{label:<12} {c:>6} {c/total2*100:>7.1f}% {cum/total2*100:>7.1f}%")

neg_np = (np_yoy < 0).sum()
pos_np = (np_yoy > 0).sum()
print(f"\n净利润正增长: {pos_np}只 ({pos_np/total2*100:.1f}%)")
print(f"净利润负增长: {neg_np}只 ({neg_np/total2*100:.1f}%)")
print(f"中位数: {np_yoy.median():.2f}%  均值: {np_yoy.mean():.2f}%")

# === 3. 双增长（营收+净利润都正增长）===
print(f"\n{'=' * 60}")
print("三、成长股筛选（营收+净利润双增长）")
print("=" * 60)
both = df[(df['revenue_yoy'] > 0) & (df['net_profit_yoy'] > 0)].copy()
print(f"双增长标的: {len(both)}只 / {len(df)}只 ({len(both)/len(df)*100:.1f}%)")

if len(both) > 0:
    both = both.sort_values('net_profit_yoy', ascending=False)
    print(f"\n{'代码':<10} {'行业':<12} {'营收增长%':>10} {'净利润增长%':>12} {'ROE%':>8} {'PE':>8} {'市值(亿)':>10}")
    print("-" * 75)
    for _, r in both.iterrows():
        mv = r['total_mv'] / 10000 if pd.notna(r['total_mv']) else 0
        print(f"{r['code']:<10} {str(r['industry']):<12} {r['revenue_yoy']:>10.2f} {r['net_profit_yoy']:>12.2f} {r.get('roe', 0):>8.2f} {r.get('pe_ttm', 0):>8.2f} {mv:>10.1f}")

# === 4. 低估成长股（双增长 + PE<30 + ROE>10）===
print(f"\n{'=' * 60}")
print("四、低估成长股（双增长 + PE<30 + ROE>10%）")
print("=" * 60)
growth = df[
    (df['revenue_yoy'] > 0) &
    (df['net_profit_yoy'] > 0) &
    (df['pe_ttm'] > 0) & (df['pe_ttm'] < 30) &
    (df['roe'] > 10)
].copy()
print(f"低估成长股: {len(growth)}只 / {len(df)}只 ({len(growth)/len(df)*100:.1f}%)")

if len(growth) > 0:
    growth = growth.sort_values('roe', ascending=False)
    print(f"\n{'代码':<10} {'行业':<12} {'营收增长%':>10} {'净利润增长%':>12} {'ROE%':>8} {'PE':>8} {'PB':>6} {'市值(亿)':>10}")
    print("-" * 85)
    for _, r in growth.iterrows():
        mv = r['total_mv'] / 10000 if pd.notna(r['total_mv']) else 0
        print(f"{r['code']:<10} {str(r['industry']):<12} {r['revenue_yoy']:>10.2f} {r['net_profit_yoy']:>12.2f} {r.get('roe', 0):>8.2f} {r.get('pe_ttm', 0):>8.2f} {r.get('pb', 0):>6.2f} {mv:>10.1f}")

# === 5. ROE分布 ===
print(f"\n{'=' * 60}")
print("五、ROE分布")
print("=" * 60)
roe = df['roe'].dropna()
roe_bins = [(-np.inf, 0), (0, 5), (5, 10), (10, 15), (15, 20), (20, np.inf)]
roe_labels = ['<0%', '0~5%', '5~10%', '10~15%', '15~20%', '>20%']
print(f"{'区间':<12} {'数量':>6} {'占比':>8}")
print("-" * 30)
for label, (lo, hi) in zip(roe_labels, roe_bins):
    c = ((roe > lo) & (roe <= hi)).sum()
    print(f"{label:<12} {c:>6} {c/len(roe)*100:>7.1f}%")

# === 6. PE分布 ===
print(f"\n{'=' * 60}")
print("六、PE(TTM)分布")
print("=" * 60)
pe = df['pe_ttm'].dropna()
pe = pe[pe > 0]  # 排除亏损股
pe_bins = [(0, 10), (10, 15), (15, 20), (20, 30), (30, 50), (50, 100), (100, np.inf)]
pe_labels = ['0~10', '10~15', '15~20', '20~30', '30~50', '50~100', '>100']
print(f"盈利股PE分布 ({len(pe)}只):")
print(f"{'区间':<12} {'数量':>6} {'占比':>8}")
print("-" * 30)
for label, (lo, hi) in zip(pe_labels, pe_bins):
    c = ((pe > lo) & (pe <= hi)).sum()
    print(f"{label:<12} {c:>6} {c/len(pe)*100:>7.1f}%")

# === 7. 综合标签 ===
print(f"\n{'=' * 60}")
print("七、标的风格标签统计")
print("=" * 60)
styles = {
    '高成长(营收>20%且利润>20%)': len(df[(df['revenue_yoy'] > 20) & (df['net_profit_yoy'] > 20)]),
    '稳健成长(营收>0%且利润>0%)': len(df[(df['revenue_yoy'] > 0) & (df['net_profit_yoy'] > 0)]),
    '高ROE(>15%)': len(df[df['roe'] > 15]),
    '低估值(PE<15)': len(df[(df['pe_ttm'] > 0) & (df['pe_ttm'] < 15)]),
    '现金牛(CF/利润>1.5)': len(df[df['cf_to_profit'] > 1.5]),
    '高杠杆(负债率>70%)': len(df[df['debt_ratio'] > 70]),
    '低杠杆(负债率<30%)': len(df[df['debt_ratio'] < 30]),
    '亏损股(ROE<0)': len(df[df['roe'] < 0]),
    '收入下滑(营收<-20%)': len(df[df['revenue_yoy'] < -20]),
    '利润下滑(净利润<-20%)': len(df[df['net_profit_yoy'] < -20]),
}
for label, cnt in styles.items():
    print(f"  {label}: {cnt}只 ({cnt/len(df)*100:.1f}%)")

print(f"\n{'=' * 60}")
print("结论")
print("=" * 60)
print(f"这232只标的中:")
print(f"  - 真正的'低估成长股'(双增长+PE<30+ROE>10): {len(growth)}只 ({len(growth)/len(df)*100:.1f}%)")
print(f"  - 收入正增长: {pos_rev}只 ({pos_rev/total*100:.1f}%)")
print(f"  - 利润正增长: {pos_np}只 ({pos_np/total2*100:.1f}%)")
print(f"  - 大部分标的更像是: 低估值+收入承压 的价值型标的")
