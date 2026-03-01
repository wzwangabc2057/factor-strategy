#!/usr/bin/env python3
"""
从Tushare MCP采集申万行业分类 + 补采缺失的BS/CF数据
然后存入ClickHouse的 tushare_stock_industry 表
"""

import json
import time
import logging
import requests
import threading
import queue
import csv

import clickhouse_driver

MCP_BASE_URL = "http://156.254.5.155:8092"
MCP_TOKEN = "lhjy.653653a5ac6d4f348932d3365abcdeca"
CH_HOST = "192.168.0.74"

REQUEST_DELAY = 0.3

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


class MCPClient:
    """MCP SSE传输客户端 (复用fetch_tushare_financial.py的逻辑)"""

    def __init__(self, base_url=MCP_BASE_URL, token=MCP_TOKEN):
        self.base_url = base_url
        self.token = token
        self.session_id = None
        self.response_queue = queue.Queue()
        self._sse_thread = None
        self._running = False

    def connect(self):
        self._running = True
        self._sse_thread = threading.Thread(target=self._sse_listener, daemon=True)
        self._sse_thread.start()
        for _ in range(50):
            if self.session_id:
                logger.info(f"MCP连接成功, session={self.session_id[:8]}...")
                return True
            time.sleep(0.1)
        logger.error("MCP连接超时")
        return False

    def _sse_listener(self):
        try:
            resp = requests.get(
                f"{self.base_url}/sse",
                headers={"Authorization": f"Bearer {self.token}"},
                stream=True, timeout=600
            )
            for line in resp.iter_lines(decode_unicode=True):
                if not self._running:
                    break
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[6:]
                    # Extract sessionId from endpoint data
                    if "sessionId=" in data:
                        import re
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
            logger.error(f"SSE监听异常: {e}")

    def call_tool(self, tool_name, arguments, timeout=30):
        if not self.session_id:
            return None
        while not self.response_queue.empty():
            try:
                self.response_queue.get_nowait()
            except queue.Empty:
                break

        call_id = int(time.time() * 1000)
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
            "id": call_id
        }
        try:
            resp = requests.post(
                f"{self.base_url}/messages?sessionId={self.session_id}",
                json=payload,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {self.token}"},
                timeout=10
            )
        except Exception as e:
            logger.error(f"POST失败: {e}")
            return None

        try:
            msg = self.response_queue.get(timeout=timeout)
            if "result" in msg and "content" in msg["result"]:
                items = []
                for c in msg["result"]["content"]:
                    if c["type"] == "text":
                        try:
                            items.append(json.loads(c["text"]))
                        except json.JSONDecodeError:
                            pass
                return items
        except queue.Empty:
            logger.warning(f"工具调用超时: {tool_name}")
        return None

    def close(self):
        self._running = False
        self.session_id = None


def convert_code_to_ts(code):
    if code.startswith(('6', '9')):
        return f"{code}.SH"
    return f"{code}.SZ"


def create_industry_table(ch):
    ch.execute("""
        CREATE TABLE IF NOT EXISTS tushare_stock_industry (
            ts_code String,
            name String,
            industry String,
            market String,
            list_date String,
            update_time DateTime DEFAULT now()
        ) ENGINE = ReplacingMergeTree(update_time)
        ORDER BY (ts_code)
    """)
    logger.info("tushare_stock_industry 表已就绪")


def fetch_stock_basic(mcp, ts_code):
    """通过stock_basic接口获取单只股票的行业分类"""
    items = mcp.call_tool("tushare_stock_basic_query", {
        "ts_code": ts_code,
        "fields": "ts_code,name,industry,market,list_date"
    })
    if items:
        return items
    # 备用: 不带fields参数
    items = mcp.call_tool("tushare_stock_basic_query", {"ts_code": ts_code})
    return items


def fetch_stock_basic_all(mcp):
    """批量获取全市场股票基本信息(含行业)"""
    # 尝试不传ts_code获取全部
    items = mcp.call_tool("tushare_stock_basic_query", {
        "exchange": "",
        "fields": "ts_code,name,industry,market,list_date"
    }, timeout=60)
    if items and len(items) > 100:
        return items
    # 按交易所分批
    all_items = []
    for exchange in ["SSE", "SZSE"]:
        items = mcp.call_tool("tushare_stock_basic_query", {
            "exchange": exchange,
            "fields": "ts_code,name,industry,market,list_date"
        }, timeout=60)
        if items:
            all_items.extend(items)
            logger.info(f"  {exchange}: {len(items)}只")
        time.sleep(REQUEST_DELAY)
    return all_items if all_items else None


def fetch_balancesheet(mcp, ts_code):
    items = mcp.call_tool("tushare_balancesheet_query", {
        "ts_code": ts_code,
        "start_date": "20200101", "end_date": "20251231",
        "fields": "ts_code,ann_date,end_date,report_type,total_assets,total_liab,"
                  "total_hldr_eqy_exc_min_int,total_cur_assets,total_nca,"
                  "total_cur_liab,total_ncl,money_cap,accounts_receiv,inventories,goodwill"
    })
    if not items:
        return []
    rows = []
    for item in items:
        if isinstance(item, dict):
            rows.append((
                item.get('ts_code', ts_code),
                item.get('ann_date', ''), item.get('end_date', ''),
                item.get('report_type', ''),
                float(item.get('total_assets', 0) or 0),
                float(item.get('total_liab', 0) or 0),
                float(item.get('total_hldr_eqy_exc_min_int', 0) or 0),
                float(item.get('total_cur_assets', 0) or 0),
                float(item.get('total_nca', 0) or 0),
                float(item.get('total_cur_liab', 0) or 0),
                float(item.get('total_ncl', 0) or 0),
                float(item.get('money_cap', 0) or 0),
                float(item.get('accounts_receiv', 0) or 0),
                float(item.get('inventories', 0) or 0),
                float(item.get('goodwill', 0) or 0),
            ))
    return rows


def fetch_cashflow(mcp, ts_code):
    items = mcp.call_tool("tushare_cashflow_query", {
        "ts_code": ts_code,
        "start_date": "20200101", "end_date": "20251231",
        "fields": "ts_code,ann_date,end_date,report_type,n_cashflow_act,"
                  "n_cashflow_inv_act,n_cash_flows_fnc_act,c_fr_sale_sg,free_cashflow"
    })
    if not items:
        return []
    rows = []
    for item in items:
        if isinstance(item, dict):
            rows.append((
                item.get('ts_code', ts_code),
                item.get('ann_date', ''), item.get('end_date', ''),
                item.get('report_type', ''),
                float(item.get('n_cashflow_act', 0) or 0),
                float(item.get('n_cashflow_inv_act', 0) or 0),
                float(item.get('n_cash_flows_fnc_act', 0) or 0),
                float(item.get('c_fr_sale_sg', 0) or 0),
                float(item.get('free_cashflow', 0) or 0),
            ))
    return rows


def main():
    import os
    # 加载R5股票池
    base = os.path.dirname(os.path.abspath(__file__))
    pool_csv = os.path.join(base, 'r5_weights_local.csv')
    if not os.path.exists(pool_csv):
        pool_csv = '/Users/macstudio/112/pythontest/r5_weights_local.csv'

    codes = []
    with open(pool_csv) as f:
        for row in csv.DictReader(f):
            codes.append(row['code'].zfill(6))
    logger.info(f"R5股票池: {len(codes)}只")

    ch = clickhouse_driver.Client(host=CH_HOST)
    create_industry_table(ch)

    mcp = MCPClient()
    if not mcp.connect():
        logger.error("MCP连接失败")
        return

    # ===== 1. 采集申万行业分类 =====
    logger.info("=" * 60)
    logger.info("采集申万行业分类 (stock_basic接口)...")
    logger.info("=" * 60)

    # 先尝试批量获取
    all_basic = fetch_stock_basic_all(mcp)
    if all_basic and len(all_basic) > 100:
        logger.info(f"批量获取成功: {len(all_basic)}只")
        rows = []
        for item in all_basic:
            if isinstance(item, dict) and item.get('ts_code'):
                rows.append((
                    item.get('ts_code', ''),
                    item.get('name', ''),
                    item.get('industry', ''),
                    item.get('market', ''),
                    item.get('list_date', ''),
                ))
        if rows:
            ch.execute("TRUNCATE TABLE IF EXISTS tushare_stock_industry")
            ch.execute(
                "INSERT INTO tushare_stock_industry (ts_code, name, industry, market, list_date) VALUES",
                rows
            )
            logger.info(f"已写入 {len(rows)} 只股票的行业信息")
    else:
        # 逐只获取
        logger.info("批量失败, 逐只获取...")
        rows = []
        for i, code in enumerate(codes):
            ts_code = convert_code_to_ts(code)
            items = fetch_stock_basic(mcp, ts_code)
            if items:
                for item in items:
                    if isinstance(item, dict):
                        rows.append((
                            item.get('ts_code', ts_code),
                            item.get('name', ''),
                            item.get('industry', ''),
                            item.get('market', ''),
                            item.get('list_date', ''),
                        ))
            if (i + 1) % 50 == 0:
                logger.info(f"  进度: {i+1}/{len(codes)}, 已获取{len(rows)}只")
            time.sleep(REQUEST_DELAY)

        if rows:
            ch.execute(
                "INSERT INTO tushare_stock_industry (ts_code, name, industry, market, list_date) VALUES",
                rows
            )
            logger.info(f"已写入 {len(rows)} 只股票的行业信息")

    # 统计行业覆盖
    ts_codes = [convert_code_to_ts(c) for c in codes]
    ts_str = "','".join(ts_codes)
    cnt = ch.execute(
        f"SELECT count() FROM tushare_stock_industry WHERE ts_code IN ('{ts_str}') AND industry != ''"
    )
    logger.info(f"R5池行业覆盖: {cnt[0][0]}/{len(codes)}")

    # ===== 2. 补采缺失的BS/CF =====
    logger.info("=" * 60)
    logger.info("补采缺失的资产负债表和现金流...")
    logger.info("=" * 60)

    bs_missing = ['300193', '600853']
    cf_missing = ['002171', '300193']

    for code in bs_missing:
        ts_code = convert_code_to_ts(code)
        rows = fetch_balancesheet(mcp, ts_code)
        if rows:
            ch.execute(
                "INSERT INTO tushare_balancesheet (ts_code, ann_date, end_date, report_type, "
                "total_assets, total_liab, total_hldr_eqy_exc_min_int, total_cur_assets, "
                "total_nca, total_cur_liab, total_ncl, money_cap, accounts_receiv, "
                "inventories, goodwill) VALUES",
                rows
            )
            logger.info(f"  {ts_code} BS: {len(rows)}条")
        else:
            logger.warning(f"  {ts_code} BS: 无数据")
        time.sleep(REQUEST_DELAY)

    for code in cf_missing:
        ts_code = convert_code_to_ts(code)
        rows = fetch_cashflow(mcp, ts_code)
        if rows:
            ch.execute(
                "INSERT INTO tushare_cashflow (ts_code, ann_date, end_date, report_type, "
                "n_cashflow_act, n_cashflow_inv_act, n_cash_flows_fnc_act, c_fr_sale_sg, "
                "free_cashflow) VALUES",
                rows
            )
            logger.info(f"  {ts_code} CF: {len(rows)}条")
        else:
            logger.warning(f"  {ts_code} CF: 无数据")
        time.sleep(REQUEST_DELAY)

    # ===== 3. 验证 =====
    logger.info("=" * 60)
    logger.info("数据完整性验证")
    logger.info("=" * 60)
    for table in ['tushare_income', 'tushare_balancesheet', 'tushare_cashflow',
                  'tushare_daily_basic', 'tushare_stock_industry']:
        cnt = ch.execute(f"SELECT count(DISTINCT ts_code) FROM {table} WHERE ts_code IN ('{ts_str}')")
        logger.info(f"  {table}: {cnt[0][0]}/232")

    # 行业分布预览
    res = ch.execute(
        f"SELECT industry, count() FROM tushare_stock_industry "
        f"WHERE ts_code IN ('{ts_str}') AND industry != '' "
        f"GROUP BY industry ORDER BY count() DESC LIMIT 15"
    )
    if res:
        logger.info("\n申万行业分布 (Top 15):")
        for r in res:
            logger.info(f"  {r[0]}: {r[1]}只")

    mcp.close()
    logger.info("采集完成!")


if __name__ == '__main__':
    main()
