#!/usr/bin/env python3
"""Fetch missing BS/CF data for 3 stocks via MCP"""
import json
import time
import logging
import requests
import threading
import queue
import re
import clickhouse_driver

MCP_BASE_URL = "http://156.254.5.155:8092"
MCP_TOKEN = "lhjy.653653a5ac6d4f348932d3365abcdeca"
CH_HOST = "192.168.0.74"

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


class MCPClient:
    def __init__(self):
        self.session_id = None
        self.response_queue = queue.Queue()
        self._running = False

    def connect(self):
        self._running = True
        t = threading.Thread(target=self._sse_listener, daemon=True)
        t.start()
        for _ in range(50):
            if self.session_id:
                logger.info(f"MCP connected: {self.session_id[:8]}...")
                return True
            time.sleep(0.1)
        return False

    def _sse_listener(self):
        try:
            resp = requests.get(f"{MCP_BASE_URL}/sse",
                                headers={"Authorization": f"Bearer {MCP_TOKEN}"},
                                stream=True, timeout=600)
            for line in resp.iter_lines(decode_unicode=True):
                if not self._running:
                    break
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[6:]
                    if "sessionId=" in data:
                        m = re.search(r'sessionId=([a-f0-9-]+)', data)
                        if m:
                            self.session_id = m.group(1)
                    elif data.startswith("{"):
                        try:
                            msg = json.loads(data)
                            if "result" in msg or "error" in msg:
                                self.response_queue.put(msg)
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            logger.error(f"SSE error: {e}")

    def call_tool(self, tool_name, arguments, timeout=30):
        if not self.session_id:
            return None
        while not self.response_queue.empty():
            try:
                self.response_queue.get_nowait()
            except:
                break
        call_id = int(time.time() * 1000)
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
            "id": call_id
        }
        try:
            requests.post(f"{MCP_BASE_URL}/messages?sessionId={self.session_id}",
                          json=payload,
                          headers={"Content-Type": "application/json",
                                   "Authorization": f"Bearer {MCP_TOKEN}"},
                          timeout=10)
        except Exception as e:
            logger.error(f"POST failed: {e}")
            return None
        try:
            msg = self.response_queue.get(timeout=timeout)
            if "result" in msg and "content" in msg["result"]:
                items = []
                for c in msg["result"]["content"]:
                    if c["type"] == "text":
                        try:
                            items.append(json.loads(c["text"]))
                        except:
                            pass
                return items
        except queue.Empty:
            logger.warning(f"Timeout: {tool_name}")
        return None

    def close(self):
        self._running = False


def main():
    ch = clickhouse_driver.Client(host=CH_HOST)
    mcp = MCPClient()
    if not mcp.connect():
        logger.error("MCP connection failed")
        return

    # BS missing: 300193.SZ, 600853.SH
    bs_missing = [("300193", "300193.SZ"), ("600853", "600853.SH")]
    for code, ts_code in bs_missing:
        logger.info(f"Fetching BS for {ts_code}...")
        items = mcp.call_tool("tushare_balancesheet_query", {
            "ts_code": ts_code,
            "start_date": "20200101", "end_date": "20251231",
            "fields": "ts_code,ann_date,end_date,report_type,total_assets,total_liab,"
                      "total_hldr_eqy_exc_min_int,total_cur_assets,total_nca,"
                      "total_cur_liab,total_ncl,money_cap,accounts_receiv,inventories,goodwill"
        })
        if items:
            rows = []
            for item in items:
                if isinstance(item, dict):
                    rows.append((
                        item.get('tsCode', item.get('ts_code', ts_code)),
                        item.get('annDate', item.get('ann_date', '')),
                        item.get('endDate', item.get('end_date', '')),
                        item.get('reportType', item.get('report_type', '')),
                        float(item.get('totalAssets', item.get('total_assets', 0)) or 0),
                        float(item.get('totalLiab', item.get('total_liab', 0)) or 0),
                        float(item.get('totalHldrEqyExcMinInt', item.get('total_hldr_eqy_exc_min_int', 0)) or 0),
                        float(item.get('totalCurAssets', item.get('total_cur_assets', 0)) or 0),
                        float(item.get('totalNca', item.get('total_nca', 0)) or 0),
                        float(item.get('totalCurLiab', item.get('total_cur_liab', 0)) or 0),
                        float(item.get('totalNcl', item.get('total_ncl', 0)) or 0),
                        float(item.get('moneyCap', item.get('money_cap', 0)) or 0),
                        float(item.get('accountsReceiv', item.get('accounts_receiv', 0)) or 0),
                        float(item.get('inventories', 0) or 0),
                        float(item.get('goodwill', 0) or 0),
                    ))
            if rows:
                ch.execute(
                    "INSERT INTO tushare_balancesheet (ts_code, ann_date, end_date, report_type, "
                    "total_assets, total_liab, total_hldr_eqy_exc_min_int, total_cur_assets, "
                    "total_nca, total_cur_liab, total_ncl, money_cap, accounts_receiv, "
                    "inventories, goodwill) VALUES",
                    rows
                )
                logger.info(f"  {ts_code} BS: {len(rows)} records inserted")
            else:
                logger.warning(f"  {ts_code} BS: parsed 0 rows from {len(items)} items")
        else:
            logger.warning(f"  {ts_code} BS: no data")
        time.sleep(0.5)

    # CF missing: 002171.SZ, 300193.SZ
    cf_missing = [("002171", "002171.SZ"), ("300193", "300193.SZ")]
    for code, ts_code in cf_missing:
        logger.info(f"Fetching CF for {ts_code}...")
        items = mcp.call_tool("tushare_cashflow_query", {
            "ts_code": ts_code,
            "start_date": "20200101", "end_date": "20251231",
            "fields": "ts_code,ann_date,end_date,report_type,n_cashflow_act,"
                      "n_cashflow_inv_act,n_cash_flows_fnc_act,c_fr_sale_sg,free_cashflow"
        })
        if items:
            rows = []
            for item in items:
                if isinstance(item, dict):
                    rows.append((
                        item.get('tsCode', item.get('ts_code', ts_code)),
                        item.get('annDate', item.get('ann_date', '')),
                        item.get('endDate', item.get('end_date', '')),
                        item.get('reportType', item.get('report_type', '')),
                        float(item.get('nCashflowAct', item.get('n_cashflow_act', 0)) or 0),
                        float(item.get('nCashflowInvAct', item.get('n_cashflow_inv_act', 0)) or 0),
                        float(item.get('nCashFlowsFncAct', item.get('n_cash_flows_fnc_act', 0)) or 0),
                        float(item.get('cFrSaleSg', item.get('c_fr_sale_sg', 0)) or 0),
                        float(item.get('freeCashflow', item.get('free_cashflow', 0)) or 0),
                    ))
            if rows:
                ch.execute(
                    "INSERT INTO tushare_cashflow (ts_code, ann_date, end_date, report_type, "
                    "n_cashflow_act, n_cashflow_inv_act, n_cash_flows_fnc_act, c_fr_sale_sg, "
                    "free_cashflow) VALUES",
                    rows
                )
                logger.info(f"  {ts_code} CF: {len(rows)} records inserted")
            else:
                logger.warning(f"  {ts_code} CF: parsed 0 rows from {len(items)} items")
        else:
            logger.warning(f"  {ts_code} CF: no data")
        time.sleep(0.5)

    # Verify
    logger.info("=== Verification ===")
    for table in ['tushare_balancesheet', 'tushare_cashflow']:
        for ts_code in ['300193.SZ', '600853.SH', '002171.SZ']:
            cnt = ch.execute(f"SELECT count() FROM {table} WHERE ts_code = '{ts_code}'")
            if cnt[0][0] > 0:
                logger.info(f"  {table} {ts_code}: {cnt[0][0]} records")

    mcp.close()
    logger.info("Done!")


if __name__ == '__main__':
    main()
