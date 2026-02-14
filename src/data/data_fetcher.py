"""
数据获取模块
从ClickHouse获取股价数据，从akshare获取财务数据
"""

import pandas as pd
import numpy as np
from clickhouse_driver import Client
import akshare as ak
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataFetcher:
    """数据获取器"""

    def __init__(self, clickhouse_host='localhost', clickhouse_port=9000):
        """
        初始化数据获取器

        Args:
            clickhouse_host: ClickHouse主机地址
            clickhouse_port: ClickHouse端口
        """
        self.ch_client = Client(host=clickhouse_host, port=clickhouse_port)
        logger.info(f"连接ClickHouse: {clickhouse_host}:{clickhouse_port}")

    def get_stock_prices(self,
                        stock_codes: List[str],
                        start_date: str,
                        end_date: str) -> pd.DataFrame:
        """
        从ClickHouse获取股票价格数据（复权价格）

        Args:
            stock_codes: 股票代码列表 ['000001', '000002', ...]
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'

        Returns:
            DataFrame with columns: date, code, close, volume, amount
        """
        logger.info(f"获取 {len(stock_codes)} 只股票的价格数据...")

        # 构建SQL查询
        codes_str = "','".join(stock_codes)
        query = f"""
        SELECT
            toDate(trade_date) as date,
            stock_code as code,
            adj_close as close,
            volume,
            amount
        FROM stock_daily_hfq
        WHERE stock_code IN ('{codes_str}')
          AND trade_date >= toDate('{start_date}')
          AND trade_date <= toDate('{end_date}')
        ORDER BY trade_date, stock_code
        """

        # 执行查询
        result = self.ch_client.execute(query)

        # 转换为DataFrame
        df = pd.DataFrame(result, columns=['date', 'code', 'close', 'volume', 'amount'])

        logger.info(f"  获取到 {len(df)} 条价格数据")
        return df

    def get_market_cap(self, stock_codes: List[str], date: str) -> pd.DataFrame:
        """
        获取股票市值数据

        Args:
            stock_codes: 股票代码列表
            date: 日期

        Returns:
            DataFrame with columns: code, market_cap (单位：亿元)
        """
        logger.info(f"获取 {len(stock_codes)} 只股票的市值数据...")

        codes_str = "','".join(stock_codes)
        query = f"""
        SELECT
            stock_code as code,
            total_market_cap / 100000000 as market_cap
        FROM stock_daily_hfq
        WHERE stock_code IN ('{codes_str}')
          AND trade_date = toDate('{date}')
        """

        result = self.ch_client.execute(query)
        df = pd.DataFrame(result, columns=['code', 'market_cap'])

        logger.info(f"  获取到 {len(df)} 条市值数据")
        return df

    def get_financial_data(self, stock_codes: List[str]) -> Dict[str, pd.DataFrame]:
        """
        从akshare获取财务数据

        Args:
            stock_codes: 股票代码列表

        Returns:
            字典，包含各类财务数据
            {
                'dividend': DataFrame,  # 股息数据
                'roe': DataFrame,       # ROE数据
                'profit': DataFrame     # 利润数据
            }
        """
        logger.info(f"获取 {len(stock_codes)} 只股票的财务数据...")

        dividend_data = []
        roe_data = []
        profit_data = []

        for i, code in enumerate(stock_codes):
            if (i + 1) % 50 == 0:
                logger.info(f"  进度: {i+1}/{len(stock_codes)}")

            try:
                # 转换股票代码格式
                if code.startswith('6'):
                    symbol = code + '.SH'
                else:
                    symbol = code + '.SZ'

                # 获取股息数据
                try:
                    div_df = ak.stock_divyield_cninfo(symbol=symbol)
                    if div_df is not None and len(div_df) > 0:
                        latest_div = div_df.iloc[-1]
                        dividend_data.append({
                            'code': code,
                            'dividend_yield': latest_div.get('股息率(%)', 0) / 100,
                            'dividend_amount': latest_div.get('每股派息(元)', 0)
                        })
                except Exception as e:
                    dividend_data.append({
                        'code': code,
                        'dividend_yield': 0,
                        'dividend_amount': 0
                    })

                # 获取ROE和利润数据
                try:
                    indicator_df = ak.stock_financial_analysis_indicator(symbol=symbol)
                    if indicator_df is not None and len(indicator_df) > 0:
                        latest = indicator_df.iloc[-1]
                        roe_data.append({
                            'code': code,
                            'roe': latest.get('净资产收益率', 0),
                            'roe_diluted': latest.get('净资产收益率-摊薄', 0)
                        })

                        profit_data.append({
                            'code': code,
                            'net_profit': latest.get('净利润', 0),
                            'net_profit_yoy': latest.get('净利润同比增长率', 0)
                        })
                except Exception as e:
                    roe_data.append({
                        'code': code,
                        'roe': 0,
                        'roe_diluted': 0
                    })
                    profit_data.append({
                        'code': code,
                        'net_profit': 0,
                        'net_profit_yoy': 0
                    })

            except Exception as e:
                logger.warning(f"获取 {code} 财务数据失败: {e}")
                continue

        result = {
            'dividend': pd.DataFrame(dividend_data),
            'roe': pd.DataFrame(roe_data),
            'profit': pd.DataFrame(profit_data)
        }

        logger.info(f"  财务数据获取完成")
        return result

    def calculate_momentum(self,
                          prices_df: pd.DataFrame,
                          period_days: int = 20) -> pd.DataFrame:
        """
        计算动量因子（近N日涨跌幅）

        Args:
            prices_df: 价格数据 DataFrame (date, code, close)
            period_days: 计算周期（天数）

        Returns:
            DataFrame with columns: code, momentum
        """
        logger.info(f"计算 {period_days} 日动量...")

        # 转为宽表
        prices_wide = prices_df.pivot(index='date', columns='code', values='close')
        prices_wide = prices_wide.sort_index()

        # 计算收益率
        momentum = {}
        for code in prices_wide.columns:
            prices = prices_wide[code].dropna()
            if len(prices) >= period_days:
                # 近N日收益率
                ret = (prices.iloc[-1] / prices.iloc[-period_days] - 1) * 100
                momentum[code] = ret
            else:
                momentum[code] = 0

        momentum_df = pd.DataFrame(list(momentum.items()), columns=['code', 'momentum'])

        logger.info(f"  动量计算完成")
        return momentum_df

    def get_all_factor_data(self,
                           stock_codes: List[str],
                           start_date: str,
                           end_date: str,
                           momentum_period: int = 20) -> Dict[str, pd.DataFrame]:
        """
        获取所有因子所需的数据

        Args:
            stock_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            momentum_period: 动量计算周期

        Returns:
            字典，包含所有因子数据
        """
        logger.info("="*60)
        logger.info("开始获取所有因子数据")
        logger.info("="*60)

        # 1. 获取价格数据
        prices_df = self.get_stock_prices(stock_codes, start_date, end_date)

        # 2. 获取市值数据
        market_cap_df = self.get_market_cap(stock_codes, end_date)

        # 3. 获取财务数据
        financial_data = self.get_financial_data(stock_codes)

        # 4. 计算动量
        momentum_df = self.calculate_momentum(prices_df, momentum_period)

        result = {
            'prices': prices_df,
            'market_cap': market_cap_df,
            'dividend': financial_data['dividend'],
            'roe': financial_data['roe'],
            'profit': financial_data['profit'],
            'momentum': momentum_df
        }

        logger.info("="*60)
        logger.info("所有因子数据获取完成")
        logger.info("="*60)

        return result


if __name__ == '__main__':
    # 测试代码
    fetcher = DataFetcher()

    # 测试股票列表
    test_codes = ['000001', '000002', '600000', '600519']

    # 获取数据
    data = fetcher.get_all_factor_data(
        stock_codes=test_codes,
        start_date='2024-01-01',
        end_date='2024-12-31',
        momentum_period=20
    )

    print("\n价格数据:")
    print(data['prices'].head())

    print("\n市值数据:")
    print(data['market_cap'])

    print("\n股息数据:")
    print(data['dividend'])

    print("\n动量数据:")
    print(data['momentum'])
