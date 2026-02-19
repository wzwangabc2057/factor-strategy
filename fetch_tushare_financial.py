#!/usr/bin/env python3
"""
从Tushare MCP服务批量采集财务数据，存入ClickHouse
数据源: http://156.254.5.155:8092/mcp
采集内容:
  1. 利润表 (income) - 营收、净利润、营业利润
  2. 资产负债表 (balancesheet) - 总资产、股东权益、负债
  3. 现金流量表 (cashflow) - 经营/投资/筹资现金流
  4. 每日基本指标 (daily_basic) - PE/PB/PS/换手率/市值
  5. 业绩预告 (forecast) - 预告类型和净利润范围
"""

import json
import time
import logging
import requests
import threading
import queue
from datetime import datetime

import clickhouse_driver

# ========== 配置 ==========
MCP_BASE_URL = "http://156.254.5.155:8092"
MCP_TOKEN = "lhjy.653653a5ac6d4f348932d3365abcdeca"
CH_HOST = "192.168.0.74"

# 采集参数
FINANCIAL_START = "20200101"  # 财报起始 (需要历史数据算增速)
FINANCIAL_END = "20251231"
DAILY_START = "20210101"      # 每日指标起始
DAILY_END = "20260219"

# 请求间隔 (避免限流)
REQUEST_DELAY = 0.3  # 秒

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# ========== MCP客户端 ==========
class MCPClient:
    """MCP SSE传输客户端"""

    def __init__(self, base_url=MCP_BASE_URL, token=MCP_TOKEN):
        self.base_url = base_url
        self.token = token
        self.session_id = None
        self.response_queue = queue.Queue()
        self._sse_thread = None
        self._running = False

    def connect(self):
        """建立SSE连接，获取session_id"""
        self._running = True
        self._sse_thread = threading.Thread(target=self._sse_listener, daemon=True)
        self._sse_thread.start()
        # 等待session_id
        for _ in range(50):
            if self.session_id:
                logger.info(f"MCP连接成功, session={self.session_id[:8]}...")
                return True
            time.sleep(0.1)
        logger.error("MCP连接超时")
        return False

    def _sse_listener(self):
        """SSE监听线程"""
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
                    if "sessionId=" in data:
                        self.session_id = data.split("sessionId=")[1]
                    else:
                        try:
                            msg = json.loads(data)
                            if "result" in msg:
                                self.response_queue.put(msg)
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            logger.error(f"SSE连接断开: {e}")
            self._running = False

    def call_tool(self, tool_name, arguments, timeout=30):
        """调用MCP工具并等待结果"""
        if not self.session_id:
            return None

        # 清空队列
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
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.token}"
                },
                timeout=10
            )
        except Exception as e:
            logger.error(f"POST失败: {e}")
            return None

        # 等待SSE响应
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

    def reconnect(self):
        """重新连接"""
        self.close()
        time.sleep(1)
        return self.connect()

    def close(self):
        self._running = False
        self.session_id = None


# ========== ClickHouse建表 ==========
def create_tables(client):
    """创建ClickHouse表"""

    # 利润表
    client.execute("""
        CREATE TABLE IF NOT EXISTS tushare_income (
            ts_code String,
            ann_date String,
            end_date String,
            report_type String,
            basic_eps Float64,
            total_revenue Float64,
            revenue Float64,
            oper_cost Float64,
            operate_profit Float64,
            total_profit Float64,
            n_income Float64,
            n_income_attr_p Float64,
            rd_exp Float64,
            update_time DateTime DEFAULT now()
        ) ENGINE = ReplacingMergeTree(update_time)
        ORDER BY (ts_code, end_date, report_type)
    """)

    # 资产负债表
    client.execute("""
        CREATE TABLE IF NOT EXISTS tushare_balancesheet (
            ts_code String,
            ann_date String,
            end_date String,
            report_type String,
            total_assets Float64,
            total_liab Float64,
            total_hldr_eqy_exc_min_int Float64,
            total_cur_assets Float64,
            total_nca Float64,
            total_cur_liab Float64,
            total_ncl Float64,
            money_cap Float64,
            accounts_receiv Float64,
            inventories Float64,
            goodwill Float64,
            update_time DateTime DEFAULT now()
        ) ENGINE = ReplacingMergeTree(update_time)
        ORDER BY (ts_code, end_date, report_type)
    """)

    # 现金流量表
    client.execute("""
        CREATE TABLE IF NOT EXISTS tushare_cashflow (
            ts_code String,
            ann_date String,
            end_date String,
            report_type String,
            n_cashflow_act Float64,
            n_cashflow_inv_act Float64,
            n_cash_flows_fnc_act Float64,
            c_fr_sale_sg Float64,
            free_cashflow Float64,
            update_time DateTime DEFAULT now()
        ) ENGINE = ReplacingMergeTree(update_time)
        ORDER BY (ts_code, end_date, report_type)
    """)

    # 每日基本指标
    client.execute("""
        CREATE TABLE IF NOT EXISTS tushare_daily_basic (
            ts_code String,
            trade_date String,
            close Float64,
            turnover_rate Float64,
            turnover_rate_f Float64,
            volume_ratio Float64,
            pe Float64,
            pe_ttm Float64,
            pb Float64,
            ps Float64,
            ps_ttm Float64,
            dv_ratio Float64,
            dv_ttm Float64,
            total_share Float64,
            float_share Float64,
            free_share Float64,
            total_mv Float64,
            circ_mv Float64,
            update_time DateTime DEFAULT now()
        ) ENGINE = ReplacingMergeTree(update_time)
        ORDER BY (ts_code, trade_date)
    """)

    # 业绩预告
    client.execute("""
        CREATE TABLE IF NOT EXISTS tushare_forecast (
            ts_code String,
            ann_date String,
            end_date String,
            type String,
            p_change_min Float64,
            p_change_max Float64,
            net_profit_min Float64,
            net_profit_max Float64,
            summary String,
            update_time DateTime DEFAULT now()
        ) ENGINE = ReplacingMergeTree(update_time)
        ORDER BY (ts_code, end_date)
    """)

    logger.info("ClickHouse表创建完成")


# ========== 数据采集函数 ==========
def convert_code_to_ts(code):
    """000001 -> 000001.SZ / 600000 -> 600000.SH"""
    if code.startswith(('6', '9')):
        return f"{code}.SH"
    else:
        return f"{code}.SZ"


def fetch_income(mcp, ts_code):
    """采集利润表"""
    items = mcp.call_tool("tushare_income_query", {
        "ts_code": ts_code,
        "start_end_date": FINANCIAL_START,
        "end_end_date": FINANCIAL_END,
        "report_type": "1"
    })
    if items is None:
        return None  # 超时返回None, 区别于空数据返回[]
    if not items:
        return []

    rows = []
    for item in items:
        rows.append((
            item.get("tsCode", ts_code),
            item.get("annDate", ""),
            item.get("endDate", ""),
            item.get("reportType", "1"),
            item.get("basicEps", 0) or 0,
            item.get("totalRevenue", 0) or 0,
            item.get("revenue", 0) or 0,
            item.get("operCost", 0) or 0,
            item.get("operateProfit", 0) or 0,
            item.get("totalProfit", 0) or 0,
            item.get("nIncome", 0) or 0,
            item.get("nIncomeAttrP", 0) or 0,
            item.get("rdExp", 0) or 0,
        ))
    return rows


def fetch_balancesheet(mcp, ts_code):
    """采集资产负债表"""
    items = mcp.call_tool("tushare_balancesheet_query", {
        "ts_code": ts_code,
        "start_end_date": FINANCIAL_START,
        "end_end_date": FINANCIAL_END,
        "report_type": "1"
    })
    if items is None:
        return None
    if not items:
        return []

    rows = []
    for item in items:
        rows.append((
            item.get("tsCode", ts_code),
            item.get("annDate", ""),
            item.get("endDate", ""),
            item.get("reportType", "1"),
            item.get("totalAssets", 0) or 0,
            item.get("totalLiab", 0) or 0,
            item.get("totalHldrEqyExcMinInt", 0) or 0,
            item.get("totalCurAssets", 0) or 0,
            item.get("totalNca", 0) or 0,
            item.get("totalCurLiab", 0) or 0,
            item.get("totalNcl", 0) or 0,
            item.get("moneyCap", 0) or 0,
            item.get("accountsReceiv", 0) or 0,
            item.get("inventories", 0) or 0,
            item.get("goodwill", 0) or 0,
        ))
    return rows


def fetch_cashflow(mcp, ts_code):
    """采集现金流量表"""
    items = mcp.call_tool("tushare_cashflow_query", {
        "ts_code": ts_code,
        "start_end_date": FINANCIAL_START,
        "end_end_date": FINANCIAL_END,
        "report_type": "1"
    })
    if items is None:
        return None
    if not items:
        return []

    rows = []
    for item in items:
        # 自由现金流 = 经营现金流 - 资本支出(近似用投资现金流)
        n_cf_act = item.get("nCashflowAct", 0) or 0
        n_cf_inv = item.get("nCashflowInvAct", 0) or 0
        fcf = n_cf_act + n_cf_inv  # 投资现金流通常为负

        rows.append((
            item.get("tsCode", ts_code),
            item.get("annDate", ""),
            item.get("endDate", ""),
            item.get("reportType", "1"),
            n_cf_act,
            n_cf_inv,
            item.get("nCashFlowsFncAct", 0) or 0,
            item.get("cFrSaleSg", 0) or 0,
            fcf,
        ))
    return rows


def fetch_daily_basic_by_date(mcp, trade_date):
    """按日期采集全市场每日指标（更高效）"""
    items = mcp.call_tool("tushare_daily_basic_query", {
        "trade_date": trade_date,
        "page_size": 5000
    }, timeout=60)
    if not items:
        return []

    rows = []
    for item in items:
        rows.append((
            item.get("tsCode", ""),
            item.get("tradeDate", trade_date),
            item.get("close", 0) or 0,
            item.get("turnoverRate", 0) or 0,
            item.get("turnoverRateF", 0) or 0,
            item.get("volumeRatio", 0) or 0,
            item.get("pe", 0) or 0,
            item.get("peTtm", 0) or 0,
            item.get("pb", 0) or 0,
            item.get("ps", 0) or 0,
            item.get("psTtm", 0) or 0,
            item.get("dvRatio", 0) or 0,
            item.get("dvTtm", 0) or 0,
            item.get("totalShare", 0) or 0,
            item.get("floatShare", 0) or 0,
            item.get("freeShare", 0) or 0,
            item.get("totalMv", 0) or 0,
            item.get("circMv", 0) or 0,
        ))
    return rows


def fetch_forecast(mcp, ts_code):
    """采集业绩预告"""
    items = mcp.call_tool("tushare_forecast_query", {
        "ts_code": ts_code,
        "start_end_date": FINANCIAL_START,
        "end_end_date": FINANCIAL_END,
    })
    if not items:
        return []

    rows = []
    for item in items:
        rows.append((
            item.get("tsCode", ts_code),
            item.get("annDate", ""),
            item.get("endDate", ""),
            item.get("type", ""),
            item.get("pChangeMin", 0) or 0,
            item.get("pChangeMax", 0) or 0,
            item.get("netProfitMin", 0) or 0,
            item.get("netProfitMax", 0) or 0,
            item.get("summary", ""),
        ))
    return rows


# ========== 主函数 ==========
def get_stock_codes(ch_client):
    """获取需要采集的股票代码列表"""
    # 从stock_data_qfq获取所有活跃股票
    rows = ch_client.execute("""
        SELECT DISTINCT code FROM stock_data_qfq
        WHERE date >= '2024-01-01'
        ORDER BY code
    """)
    return [r[0] for r in rows]


def get_trade_dates(ch_client, start='2021-01-01', end='2026-02-19'):
    """获取交易日列表"""
    rows = ch_client.execute(f"""
        SELECT DISTINCT toString(date) FROM stock_data_qfq
        WHERE code = '000001' AND date >= '{start}' AND date <= '{end}'
        ORDER BY date
    """)
    return [r[0].replace('-', '') for r in rows]


def main():
    import argparse
    parser = argparse.ArgumentParser(description='采集Tushare财务数据到ClickHouse')
    parser.add_argument('--mode', choices=['all', 'financial', 'daily', 'forecast'],
                        default='all', help='采集模式')
    parser.add_argument('--stocks', type=str, default='',
                        help='指定股票代码(逗号分隔), 留空=全部')
    parser.add_argument('--daily-freq', type=int, default=5,
                        help='每日指标采集频率(每N个交易日采一次)')
    args = parser.parse_args()

    # 连接ClickHouse
    ch = clickhouse_driver.Client(host=CH_HOST)
    create_tables(ch)

    # 获取股票列表
    if args.stocks:
        codes = args.stocks.split(',')
    else:
        codes = get_stock_codes(ch)
    logger.info(f"待采集股票: {len(codes)}只")

    # 连接MCP
    mcp = MCPClient()
    if not mcp.connect():
        logger.error("MCP连接失败，退出")
        return

    total_start = time.time()

    # ===== 1. 财务报表采集 (按股票逐只) =====
    if args.mode in ('all', 'financial'):
        logger.info("=" * 60)
        logger.info("开始采集财务报表...")
        logger.info("=" * 60)

        # 断点续传: 查已采集的股票跳过
        done_codes = set()
        try:
            done_rows = ch.execute("SELECT DISTINCT ts_code FROM tushare_income")
            done_codes = set(r[0] for r in done_rows)
            if done_codes:
                logger.info(f"已采集 {len(done_codes)} 只股票, 断点续传")
        except:
            pass

        income_total = 0
        bs_total = 0
        cf_total = 0
        consecutive_fails = 0
        MAX_CONSECUTIVE_FAILS = 10

        for i, code in enumerate(codes):
            ts_code = convert_code_to_ts(code)

            # 跳过已采集
            if ts_code in done_codes:
                continue

            if (i + 1) % 50 == 0:
                logger.info(f"  进度: {i+1}/{len(codes)} ({(i+1)/len(codes)*100:.0f}%)"
                            f" income={income_total} bs={bs_total} cf={cf_total}")

            # 利润表
            try:
                rows = fetch_income(mcp, ts_code)
                if rows:
                    ch.execute(
                        "INSERT INTO tushare_income (ts_code, ann_date, end_date, report_type, basic_eps, total_revenue, revenue, oper_cost, operate_profit, total_profit, n_income, n_income_attr_p, rd_exp) VALUES",
                        rows
                    )
                    income_total += len(rows)
                    consecutive_fails = 0
                elif rows is None:
                    consecutive_fails += 1
            except Exception as e:
                logger.warning(f"  {ts_code} income失败: {e}")
                consecutive_fails += 1

            time.sleep(REQUEST_DELAY)

            # 资产负债表
            try:
                rows = fetch_balancesheet(mcp, ts_code)
                if rows:
                    ch.execute(
                        "INSERT INTO tushare_balancesheet (ts_code, ann_date, end_date, report_type, total_assets, total_liab, total_hldr_eqy_exc_min_int, total_cur_assets, total_nca, total_cur_liab, total_ncl, money_cap, accounts_receiv, inventories, goodwill) VALUES",
                        rows
                    )
                    bs_total += len(rows)
                    consecutive_fails = 0
                elif rows is None:
                    consecutive_fails += 1
            except Exception as e:
                logger.warning(f"  {ts_code} balancesheet失败: {e}")
                consecutive_fails += 1

            time.sleep(REQUEST_DELAY)

            # 现金流量表
            try:
                rows = fetch_cashflow(mcp, ts_code)
                if rows:
                    ch.execute(
                        "INSERT INTO tushare_cashflow (ts_code, ann_date, end_date, report_type, n_cashflow_act, n_cashflow_inv_act, n_cash_flows_fnc_act, c_fr_sale_sg, free_cashflow) VALUES",
                        rows
                    )
                    cf_total += len(rows)
                    consecutive_fails = 0
                elif rows is None:
                    consecutive_fails += 1
            except Exception as e:
                logger.warning(f"  {ts_code} cashflow失败: {e}")
                consecutive_fails += 1

            time.sleep(REQUEST_DELAY)

            # 连续失败超过阈值 → 强制重连
            if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
                logger.warning(f"  连续失败{consecutive_fails}次, 强制重连...")
                time.sleep(3)
                if not mcp.reconnect():
                    logger.error("重连失败, 等待10秒再试...")
                    time.sleep(10)
                    if not mcp.reconnect():
                        logger.error("二次重连失败, 退出财报采集")
                        break
                consecutive_fails = 0

            # 每50只股票重连一次MCP (防止SSE超时, 从100改为50)
            if (i + 1) % 50 == 0:
                logger.info("  定时重连MCP...")
                mcp.reconnect()

        logger.info(f"财务报表采集完成: income={income_total} bs={bs_total} cf={cf_total}")

    # ===== 2. 每日基本指标采集 (按日期) =====
    if args.mode in ('all', 'daily'):
        logger.info("=" * 60)
        logger.info("开始采集每日基本指标...")
        logger.info("=" * 60)

        # 检查已有数据，避免重复
        existing = ch.execute(
            "SELECT DISTINCT trade_date FROM tushare_daily_basic ORDER BY trade_date"
        )
        existing_dates = set(r[0] for r in existing)

        trade_dates = get_trade_dates(ch)
        # 按频率采样 (每日指标数据量大，先按周采集)
        sampled_dates = trade_dates[::args.daily_freq]
        # 确保最后一个日期也包含
        if trade_dates and trade_dates[-1] not in sampled_dates:
            sampled_dates.append(trade_dates[-1])

        new_dates = [d for d in sampled_dates if d not in existing_dates]
        logger.info(f"交易日总数: {len(trade_dates)}, 采样: {len(sampled_dates)}, "
                    f"已有: {len(existing_dates)}, 待采: {len(new_dates)}")

        daily_total = 0
        consecutive_fails = 0
        for i, td in enumerate(new_dates):
            if (i + 1) % 20 == 0:
                logger.info(f"  进度: {i+1}/{len(new_dates)} ({td}) total={daily_total}")

            try:
                rows = fetch_daily_basic_by_date(mcp, td)
                if rows:
                    ch.execute("INSERT INTO tushare_daily_basic (ts_code, trade_date, close, turnover_rate, turnover_rate_f, volume_ratio, pe, pe_ttm, pb, ps, ps_ttm, dv_ratio, dv_ttm, total_share, float_share, free_share, total_mv, circ_mv) VALUES", rows)
                    daily_total += len(rows)
                    consecutive_fails = 0
                elif rows is None:
                    consecutive_fails += 1
            except Exception as e:
                logger.warning(f"  {td} daily_basic失败: {e}")
                consecutive_fails += 1

            if consecutive_fails >= 5:
                logger.warning(f"  连续失败{consecutive_fails}次, 强制重连...")
                time.sleep(3)
                mcp.reconnect()
                consecutive_fails = 0

            time.sleep(REQUEST_DELAY)

            if (i + 1) % 30 == 0:
                mcp.reconnect()

        logger.info(f"每日指标采集完成: {daily_total}条")

    # ===== 3. 业绩预告采集 =====
    if args.mode in ('all', 'forecast'):
        logger.info("=" * 60)
        logger.info("开始采集业绩预告...")
        logger.info("=" * 60)

        fc_total = 0
        for i, code in enumerate(codes):
            ts_code = convert_code_to_ts(code)

            if (i + 1) % 100 == 0:
                logger.info(f"  进度: {i+1}/{len(codes)} forecast={fc_total}")

            try:
                rows = fetch_forecast(mcp, ts_code)
                if rows:
                    ch.execute("INSERT INTO tushare_forecast (ts_code, ann_date, end_date, type, p_change_min, p_change_max, net_profit_min, net_profit_max, summary) VALUES", rows)
                    fc_total += len(rows)
            except Exception as e:
                logger.warning(f"  {ts_code} forecast失败: {e}")

            time.sleep(REQUEST_DELAY)

            if (i + 1) % 100 == 0:
                mcp.reconnect()

        logger.info(f"业绩预告采集完成: {fc_total}条")

    # ===== 汇总 =====
    elapsed = time.time() - total_start
    logger.info("=" * 60)
    logger.info(f"全部采集完成! 耗时: {elapsed/60:.1f}分钟")
    logger.info("=" * 60)

    # 打印各表数据量
    for table in ['tushare_income', 'tushare_balancesheet', 'tushare_cashflow',
                  'tushare_daily_basic', 'tushare_forecast']:
        cnt = ch.execute(f"SELECT count() FROM {table}")
        logger.info(f"  {table}: {cnt[0][0]}条")

    mcp.close()


if __name__ == '__main__':
    main()
