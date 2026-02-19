#!/usr/bin/env python3
"""Analyze technical characteristics of R5 232 stocks - find common patterns"""
import csv
import numpy as np
import clickhouse_connect

CH_HOST = "192.168.0.74"

def main():
    ch = clickhouse_connect.get_client(host=CH_HOST, port=8123)

    # Load R5 codes
    codes = []
    with open('/Users/macstudio/112/pythontest/r5_weights_local.csv') as f:
        for row in csv.DictReader(f):
            codes.append(row['code'].zfill(6))

    ts_codes = []
    for c in codes:
        if c.startswith(('6', '9')):
            ts_codes.append(f"{c}.SH")
        else:
            ts_codes.append(f"{c}.SZ")
    ts_str = "','".join(ts_codes)

    # Get latest trade date
    max_date = ch.query("SELECT max(trade_date) FROM tushare_daily_basic").result_rows[0][0]
    print(f"Latest date: {max_date}")
    print(f"R5 stocks: {len(ts_codes)}")

    # === 1. Get daily_basic data for latest date ===
    res = ch.query(
        f"SELECT ts_code, close, turnover_rate_f, volume_ratio, pe_ttm, pb, total_mv "
        f"FROM tushare_daily_basic "
        f"WHERE ts_code IN ('{ts_str}') AND trade_date = '{max_date}'"
    )
    latest = {}
    for r in res.result_rows:
        latest[r[0]] = {'close': r[1], 'turnover': r[2], 'vol_ratio': r[3],
                        'pe': r[4], 'pb': r[5], 'mv': r[6]}
    print(f"Latest data: {len(latest)} stocks")

    # === 2. Get price history (last 250 trading days) for momentum/volatility ===
    # Get all available dates
    dates_res = ch.query(
        f"SELECT DISTINCT trade_date FROM tushare_daily_basic "
        f"WHERE ts_code = '000001.SZ' ORDER BY trade_date DESC LIMIT 250"
    )
    all_dates = [r[0] for r in dates_res.result_rows]
    print(f"Available dates: {len(all_dates)} trading days")

    if len(all_dates) < 20:
        print("Not enough history")
        return

    # Key dates for momentum calculation
    date_latest = all_dates[0]
    date_5d = all_dates[min(4, len(all_dates)-1)]
    date_10d = all_dates[min(9, len(all_dates)-1)]
    date_20d = all_dates[min(19, len(all_dates)-1)]
    date_60d = all_dates[min(59, len(all_dates)-1)] if len(all_dates) > 59 else all_dates[-1]
    date_120d = all_dates[min(119, len(all_dates)-1)] if len(all_dates) > 119 else all_dates[-1]
    date_250d = all_dates[-1]

    # Get close prices at key dates
    key_dates = [date_latest, date_5d, date_10d, date_20d, date_60d, date_120d, date_250d]
    date_str = "','".join(key_dates)

    price_data = {}  # ts_code -> {date: close}
    res = ch.query(
        f"SELECT ts_code, trade_date, close FROM tushare_daily_basic "
        f"WHERE ts_code IN ('{ts_str}') AND trade_date IN ('{date_str}')"
    )
    for r in res.result_rows:
        if r[0] not in price_data:
            price_data[r[0]] = {}
        price_data[r[0]][r[1]] = r[2]

    # === 3. Get full 60-day history for volatility/RSI ===
    date_60d_all = all_dates[:60]
    date_60_str = "','".join(date_60d_all)

    daily_returns = {}  # ts_code -> [returns]
    daily_closes = {}   # ts_code -> [close prices sorted by date]
    daily_turnover = {} # ts_code -> [turnover rates]

    res = ch.query(
        f"SELECT ts_code, trade_date, close, turnover_rate_f FROM tushare_daily_basic "
        f"WHERE ts_code IN ('{ts_str}') AND trade_date IN ('{date_60_str}') "
        f"ORDER BY ts_code, trade_date"
    )

    current_ts = None
    closes = []
    turnovers = []
    for r in res.result_rows:
        if r[0] != current_ts:
            if current_ts and len(closes) > 1:
                daily_closes[current_ts] = closes
                daily_turnover[current_ts] = turnovers
                rets = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes)) if closes[i-1] > 0]
                daily_returns[current_ts] = rets
            current_ts = r[0]
            closes = []
            turnovers = []
        closes.append(r[2])
        turnovers.append(r[3])
    # Last stock
    if current_ts and len(closes) > 1:
        daily_closes[current_ts] = closes
        daily_turnover[current_ts] = turnovers
        rets = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes)) if closes[i-1] > 0]
        daily_returns[current_ts] = rets

    # === 4. Compute technical indicators for each stock ===
    results = []
    for ts_code in ts_codes:
        prices = price_data.get(ts_code, {})
        p_now = prices.get(date_latest, 0)
        if p_now <= 0:
            continue

        # Momentum
        p_5d = prices.get(date_5d, 0)
        p_10d = prices.get(date_10d, 0)
        p_20d = prices.get(date_20d, 0)
        p_60d = prices.get(date_60d, 0)
        p_120d = prices.get(date_120d, 0)
        p_250d = prices.get(date_250d, 0)

        mom_5d = (p_now - p_5d) / p_5d * 100 if p_5d > 0 else None
        mom_10d = (p_now - p_10d) / p_10d * 100 if p_10d > 0 else None
        mom_20d = (p_now - p_20d) / p_20d * 100 if p_20d > 0 else None
        mom_60d = (p_now - p_60d) / p_60d * 100 if p_60d > 0 else None
        mom_120d = (p_now - p_120d) / p_120d * 100 if p_120d > 0 else None
        mom_250d = (p_now - p_250d) / p_250d * 100 if p_250d > 0 else None

        # Volatility (60d)
        rets = daily_returns.get(ts_code, [])
        vol_60d = np.std(rets) * np.sqrt(250) * 100 if len(rets) > 5 else None

        # Average turnover (60d)
        turnover_list = daily_turnover.get(ts_code, [])
        avg_turnover = np.mean(turnover_list) if turnover_list else None

        # RSI-like (14d)
        if len(rets) >= 14:
            recent_rets = rets[-14:]
            gains = [r for r in recent_rets if r > 0]
            losses = [-r for r in recent_rets if r < 0]
            avg_gain = np.mean(gains) if gains else 0
            avg_loss = np.mean(losses) if losses else 0.0001
            rs = avg_gain / avg_loss if avg_loss > 0 else 100
            rsi = 100 - 100 / (1 + rs)
        else:
            rsi = None

        # MA position (close vs MA20, MA60)
        closes_list = daily_closes.get(ts_code, [])
        ma20 = np.mean(closes_list[-20:]) if len(closes_list) >= 20 else None
        ma60 = np.mean(closes_list) if len(closes_list) >= 40 else None
        close_vs_ma20 = (p_now / ma20 - 1) * 100 if ma20 and ma20 > 0 else None
        close_vs_ma60 = (p_now / ma60 - 1) * 100 if ma60 and ma60 > 0 else None

        # Price position in 60d range
        if len(closes_list) >= 20:
            high_60d = max(closes_list)
            low_60d = min(closes_list)
            price_pos = (p_now - low_60d) / (high_60d - low_60d) * 100 if high_60d > low_60d else 50
        else:
            price_pos = None

        # Market cap
        mv = latest.get(ts_code, {}).get('mv', 0)

        results.append({
            'ts_code': ts_code,
            'price': p_now,
            'mv_wan': mv,
            'mom_5d': mom_5d,
            'mom_10d': mom_10d,
            'mom_20d': mom_20d,
            'mom_60d': mom_60d,
            'mom_120d': mom_120d,
            'mom_250d': mom_250d,
            'vol_60d': vol_60d,
            'avg_turnover': avg_turnover,
            'rsi_14': rsi,
            'close_vs_ma20': close_vs_ma20,
            'close_vs_ma60': close_vs_ma60,
            'price_pos_60d': price_pos,
        })

    print(f"\nComputed technicals for {len(results)} stocks")

    # === 5. Statistical Summary ===
    def stats(values, name):
        v = [x for x in values if x is not None]
        if not v:
            return
        v = np.array(v)
        print(f"  {name:<20} avg={np.mean(v):>8.2f}  med={np.median(v):>8.2f}  "
              f"P25={np.percentile(v,25):>8.2f}  P75={np.percentile(v,75):>8.2f}  "
              f"min={np.min(v):>8.2f}  max={np.max(v):>8.2f}")

    print(f"\n{'='*80}")
    print("一、动量(涨跌幅%)分布")
    print('='*80)
    stats([r['mom_5d'] for r in results], '5日涨跌幅')
    stats([r['mom_10d'] for r in results], '10日涨跌幅')
    stats([r['mom_20d'] for r in results], '20日涨跌幅')
    stats([r['mom_60d'] for r in results], '60日涨跌幅')
    stats([r['mom_120d'] for r in results], '120日涨跌幅')
    stats([r['mom_250d'] for r in results], '250日涨跌幅')

    # Momentum distribution
    print(f"\n--- 20日动量分布 ---")
    mom20 = [r['mom_20d'] for r in results if r['mom_20d'] is not None]
    bins = [(-100, -10), (-10, -5), (-5, 0), (0, 5), (5, 10), (10, 20), (20, 100)]
    labels = ['<-10%', '-10~-5%', '-5~0%', '0~5%', '5~10%', '10~20%', '>20%']
    for label, (lo, hi) in zip(labels, bins):
        c = sum(1 for m in mom20 if lo < m <= hi)
        print(f"  {label:<10} {c:>4}只 ({c/len(mom20)*100:>5.1f}%)")

    print(f"\n--- 60日动量分布 ---")
    mom60 = [r['mom_60d'] for r in results if r['mom_60d'] is not None]
    for label, (lo, hi) in zip(labels, bins):
        c = sum(1 for m in mom60 if lo < m <= hi)
        print(f"  {label:<10} {c:>4}只 ({c/len(mom60)*100:>5.1f}%)")

    print(f"\n{'='*80}")
    print("二、波动率分布 (年化%)")
    print('='*80)
    stats([r['vol_60d'] for r in results], '60日年化波动率')
    vol = [r['vol_60d'] for r in results if r['vol_60d'] is not None]
    vol_bins = [(0, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 80), (80, 200)]
    vol_labels = ['<20%', '20~30%', '30~40%', '40~50%', '50~60%', '60~80%', '>80%']
    for label, (lo, hi) in zip(vol_labels, vol_bins):
        c = sum(1 for v in vol if lo < v <= hi)
        print(f"  {label:<10} {c:>4}只 ({c/len(vol)*100:>5.1f}%)")

    print(f"\n{'='*80}")
    print("三、换手率分布 (日均%)")
    print('='*80)
    stats([r['avg_turnover'] for r in results], '60日均换手率')
    turn = [r['avg_turnover'] for r in results if r['avg_turnover'] is not None]
    t_bins = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 10), (10, 100)]
    t_labels = ['<1%', '1~2%', '2~3%', '3~5%', '5~10%', '>10%']
    for label, (lo, hi) in zip(t_labels, t_bins):
        c = sum(1 for t in turn if lo < t <= hi)
        print(f"  {label:<10} {c:>4}只 ({c/len(turn)*100:>5.1f}%)")

    print(f"\n{'='*80}")
    print("四、RSI(14)分布")
    print('='*80)
    stats([r['rsi_14'] for r in results], 'RSI(14)')
    rsi_list = [r['rsi_14'] for r in results if r['rsi_14'] is not None]
    rsi_bins = [(0, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 100)]
    rsi_labels = ['<30超卖', '30~40偏弱', '40~50中性偏弱', '50~60中性偏强', '60~70偏强', '>70超买']
    for label, (lo, hi) in zip(rsi_labels, rsi_bins):
        c = sum(1 for r in rsi_list if lo < r <= hi)
        print(f"  {label:<16} {c:>4}只 ({c/len(rsi_list)*100:>5.1f}%)")

    print(f"\n{'='*80}")
    print("五、均线位置")
    print('='*80)
    stats([r['close_vs_ma20'] for r in results], '收盘vs MA20 (%)')
    stats([r['close_vs_ma60'] for r in results], '收盘vs MA60 (%)')

    above_ma20 = sum(1 for r in results if r['close_vs_ma20'] is not None and r['close_vs_ma20'] > 0)
    below_ma20 = sum(1 for r in results if r['close_vs_ma20'] is not None and r['close_vs_ma20'] <= 0)
    above_ma60 = sum(1 for r in results if r['close_vs_ma60'] is not None and r['close_vs_ma60'] > 0)
    below_ma60 = sum(1 for r in results if r['close_vs_ma60'] is not None and r['close_vs_ma60'] <= 0)
    total_ma = above_ma20 + below_ma20
    total_ma60 = above_ma60 + below_ma60
    print(f"\n  站上MA20: {above_ma20}只 ({above_ma20/total_ma*100:.1f}%)  |  跌破MA20: {below_ma20}只 ({below_ma20/total_ma*100:.1f}%)")
    print(f"  站上MA60: {above_ma60}只 ({above_ma60/total_ma60*100:.1f}%)  |  跌破MA60: {below_ma60}只 ({below_ma60/total_ma60*100:.1f}%)")

    print(f"\n{'='*80}")
    print("六、60日价格位置 (0=最低 100=最高)")
    print('='*80)
    stats([r['price_pos_60d'] for r in results], '60日价格位置')
    pos = [r['price_pos_60d'] for r in results if r['price_pos_60d'] is not None]
    pos_bins = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
    pos_labels = ['0~20%底部', '20~40%偏低', '40~60%中间', '60~80%偏高', '80~100%顶部']
    for label, (lo, hi) in zip(pos_labels, pos_bins):
        c = sum(1 for p in pos if lo <= p < hi) if hi < 100 else sum(1 for p in pos if lo <= p <= hi)
        print(f"  {label:<16} {c:>4}只 ({c/len(pos)*100:>5.1f}%)")

    print(f"\n{'='*80}")
    print("七、市值分布 (亿元)")
    print('='*80)
    mvs = [r['mv_wan']/10000 for r in results if r['mv_wan'] > 0]
    stats(mvs, '总市值(亿)')
    mv_bins = [(0, 50), (50, 100), (100, 200), (200, 500), (500, 1000), (1000, 5000), (5000, 100000)]
    mv_labels = ['<50亿', '50~100亿', '100~200亿', '200~500亿', '500~1000亿', '1000~5000亿', '>5000亿']
    for label, (lo, hi) in zip(mv_labels, mv_bins):
        c = sum(1 for m in mvs if lo < m <= hi)
        print(f"  {label:<14} {c:>4}只 ({c/len(mvs)*100:>5.1f}%)")

    # === 6. Summary / Common Profile ===
    print(f"\n{'='*80}")
    print("八、技术面共性总结")
    print('='*80)

    mom20_med = np.median([r['mom_20d'] for r in results if r['mom_20d'] is not None])
    mom60_med = np.median([r['mom_60d'] for r in results if r['mom_60d'] is not None])
    mom250_med = np.median([r['mom_250d'] for r in results if r['mom_250d'] is not None])
    vol_med = np.median([r['vol_60d'] for r in results if r['vol_60d'] is not None])
    turn_med = np.median([r['avg_turnover'] for r in results if r['avg_turnover'] is not None])
    rsi_med = np.median([r['rsi_14'] for r in results if r['rsi_14'] is not None])
    pos_med = np.median([r['price_pos_60d'] for r in results if r['price_pos_60d'] is not None])
    mv_med = np.median(mvs)

    print(f"  20日动量中位数: {mom20_med:+.2f}%")
    print(f"  60日动量中位数: {mom60_med:+.2f}%")
    print(f"  250日动量中位数: {mom250_med:+.2f}%")
    print(f"  年化波动率中位数: {vol_med:.1f}%")
    print(f"  日均换手率中位数: {turn_med:.2f}%")
    print(f"  RSI(14)中位数: {rsi_med:.1f}")
    print(f"  60日价格位置中位数: {pos_med:.1f}%")
    print(f"  市值中位数: {mv_med:.0f}亿")

    # Positive momentum ratio
    mom20_pos = sum(1 for r in results if r['mom_20d'] is not None and r['mom_20d'] > 0)
    mom60_pos = sum(1 for r in results if r['mom_60d'] is not None and r['mom_60d'] > 0)
    total_m = len([r for r in results if r['mom_20d'] is not None])
    print(f"\n  20日正涨幅比例: {mom20_pos}/{total_m} ({mom20_pos/total_m*100:.1f}%)")
    print(f"  60日正涨幅比例: {mom60_pos}/{total_m} ({mom60_pos/total_m*100:.1f}%)")

if __name__ == '__main__':
    main()
