#!/usr/bin/env python3
"""Fetch dividend data for R5 232 stocks and calculate dividend yield"""
import json, time, queue, threading, re, csv, requests
import clickhouse_connect

MCP_BASE_URL = "http://156.254.5.155:8092"
MCP_TOKEN = "lhjy.653653a5ac6d4f348932d3365abcdeca"
CH_HOST = "192.168.0.74"

session_id = None
response_queue = queue.Queue()

def sse_listener():
    global session_id
    resp = requests.get(f'{MCP_BASE_URL}/sse',
                        headers={'Authorization': f'Bearer {MCP_TOKEN}'},
                        stream=True, timeout=600)
    for line in resp.iter_lines(decode_unicode=True):
        if not line: continue
        if line.startswith('data: '):
            data = line[6:]
            if 'sessionId=' in data:
                m = re.search(r'sessionId=([a-f0-9-]+)', data)
                if m: session_id = m.group(1)
            elif data.startswith('{'):
                try:
                    msg = json.loads(data)
                    if 'result' in msg or 'error' in msg:
                        response_queue.put(msg)
                except: pass

def call_tool(tool_name, arguments, timeout=30):
    while not response_queue.empty():
        try: response_queue.get_nowait()
        except: break
    payload = {
        'jsonrpc': '2.0', 'method': 'tools/call',
        'params': {'name': tool_name, 'arguments': arguments},
        'id': int(time.time() * 1000)
    }
    try:
        requests.post(f'{MCP_BASE_URL}/messages?sessionId={session_id}',
                      json=payload,
                      headers={'Content-Type': 'application/json',
                               'Authorization': f'Bearer {MCP_TOKEN}'},
                      timeout=10)
    except:
        return None
    try:
        msg = response_queue.get(timeout=timeout)
        if 'result' in msg:
            items = []
            for c in msg['result'].get('content', []):
                if c['type'] == 'text':
                    try: items.append(json.loads(c['text']))
                    except: pass
            return items
    except queue.Empty:
        pass
    return None

def main():
    # Load R5 codes
    codes = []
    with open('/Users/macstudio/112/pythontest/r5_weights_local.csv') as f:
        for row in csv.DictReader(f):
            codes.append(row['code'].zfill(6))
    print(f"R5 stocks: {len(codes)}")

    # Get latest price from ClickHouse
    ch = clickhouse_connect.get_client(host=CH_HOST, port=8123)
    ts_codes = []
    for c in codes:
        if c.startswith(('6', '9')):
            ts_codes.append(f"{c}.SH")
        else:
            ts_codes.append(f"{c}.SZ")

    # Get latest close price
    price_map = {}
    ts_str = "','".join(ts_codes)
    res = ch.query(
        f"SELECT ts_code, close FROM tushare_daily_basic "
        f"WHERE ts_code IN ('{ts_str}') AND trade_date = "
        f"(SELECT max(trade_date) FROM tushare_daily_basic)"
    )
    for row in res.result_rows:
        price_map[row[0]] = row[1]
    print(f"Price data: {len(price_map)} stocks")

    # Connect MCP
    t = threading.Thread(target=sse_listener, daemon=True)
    t.start()
    time.sleep(2)
    if not session_id:
        print("MCP connect failed")
        return
    print(f"MCP connected: {session_id[:8]}...")

    # Fetch dividend for each stock
    dividend_data = {}  # ts_code -> {cash_div_ttm, dv_yield}
    batch_size = 5
    for i, ts_code in enumerate(ts_codes):
        print(f"[{i+1}/{len(ts_codes)}] {ts_code}...", end=" ", flush=True)
        items = call_tool("tushare_dividend_query", {
            "ts_code": ts_code,
            "fields": "ts_code,end_date,ann_date,div_proc,cash_div,cash_div_tax,record_date,ex_date"
        }, timeout=20)

        if items:
            # Find implemented dividends (cashDiv > 0) in recent periods
            # Note: divProc has encoding issues in SSE, so use cashDiv > 0 as filter
            total_cash_div = 0
            div_count = 0
            seen_periods = set()  # avoid double counting same period
            for item in items:
                cash_div = float(item.get('cashDiv', item.get('cash_div', 0)) or 0)
                end_date = item.get('endDate', item.get('end_date', ''))
                # Only count dividends with actual cash and from 2024+ periods
                if cash_div > 0 and end_date >= '2024':
                    period_key = f"{end_date}_{cash_div}"
                    if period_key not in seen_periods:
                        seen_periods.add(period_key)
                        total_cash_div += cash_div
                        div_count += 1

            price = price_map.get(ts_code, 0)
            if total_cash_div > 0 and price > 0:
                # cash_div is per 10 shares, convert to per share
                dv_yield = (total_cash_div / 10) / price * 100
                dividend_data[ts_code] = {
                    'cash_div_per10': total_cash_div,
                    'price': price,
                    'dv_yield': dv_yield,
                    'div_count': div_count
                }
                print(f"div={total_cash_div:.2f}/10股, yield={dv_yield:.2f}%")
            else:
                print(f"no recent div (items={len(items)})")
        else:
            print("no data")
        time.sleep(0.3)

    # Summary
    print(f"\n{'='*60}")
    print(f"分红数据汇总")
    print(f"{'='*60}")
    print(f"有分红数据: {len(dividend_data)}/{len(ts_codes)}")

    if dividend_data:
        yields = [v['dv_yield'] for v in dividend_data.values()]
        yields.sort(reverse=True)
        print(f"股息率: avg={sum(yields)/len(yields):.2f}%, median={yields[len(yields)//2]:.2f}%")
        print(f"股息率>3%: {sum(1 for y in yields if y > 3)}只")
        print(f"股息率>5%: {sum(1 for y in yields if y > 5)}只")

        # Save to CSV
        with open('/Users/macstudio/112/pythontest/reports/dividend_data.csv', 'w') as f:
            w = csv.writer(f)
            w.writerow(['ts_code', 'code', 'cash_div_per10', 'price', 'dv_yield_pct', 'div_count'])
            for ts_code in sorted(dividend_data.keys(), key=lambda x: dividend_data[x]['dv_yield'], reverse=True):
                d = dividend_data[ts_code]
                code = ts_code.split('.')[0]
                w.writerow([ts_code, code, f"{d['cash_div_per10']:.4f}", f"{d['price']:.2f}",
                           f"{d['dv_yield']:.2f}", d['div_count']])

        # Top 30 by yield
        print(f"\n{'='*60}")
        print(f"股息率 Top 30")
        print(f"{'='*60}")
        sorted_div = sorted(dividend_data.items(), key=lambda x: x[1]['dv_yield'], reverse=True)
        print(f"{'代码':<12} {'每10股分红':>10} {'股价':>8} {'股息率%':>8}")
        print("-" * 45)
        for ts_code, d in sorted_div[:30]:
            print(f"{ts_code:<12} {d['cash_div_per10']:>10.2f} {d['price']:>8.2f} {d['dv_yield']:>8.2f}")

if __name__ == '__main__':
    main()
