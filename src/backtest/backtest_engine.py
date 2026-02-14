"""
回测引擎
运行完整的多因子增强策略回测
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BacktestEngine:
    """回测引擎"""

    def __init__(self,
                 initial_capital: float = 1000000,
                 rebalance_frequency: str = 'monthly',
                 transaction_costs: Dict[str, float] = None):
        """
        初始化回测引擎

        Args:
            initial_capital: 初始资金
            rebalance_frequency: 调仓频率 (monthly/quarterly)
            transaction_costs: 交易成本
                {
                    'buy_commission': 0.00026,   # 买入佣金0.026%
                    'sell_commission': 0.00126   # 卖出佣金+印花税0.126%
                }
        """
        self.initial_capital = initial_capital
        self.rebalance_frequency = rebalance_frequency

        if transaction_costs is None:
            self.transaction_costs = {
                'buy_commission': 0.00026,
                'sell_commission': 0.00126
            }
        else:
            self.transaction_costs = transaction_costs

        logger.info("回测引擎初始化:")
        logger.info(f"  初始资金: {self.initial_capital:,.0f}元")
        logger.info(f"  调仓频率: {self.rebalance_frequency}")
        logger.info(f"  买入成本: {self.transaction_costs['buy_commission']:.4%}")
        logger.info(f"  卖出成本: {self.transaction_costs['sell_commission']:.4%}")

    def _get_rebalance_dates(self, all_dates: List[datetime]) -> List[datetime]:
        """
        获取调仓日期列表

        Args:
            all_dates: 所有交易日列表

        Returns:
            调仓日期列表
        """
        if self.rebalance_frequency == 'monthly':
            # 每月第一个交易日
            rebalance_dates = []
            current_month = None

            for date in all_dates:
                if current_month != date.month:
                    rebalance_dates.append(date)
                    current_month = date.month

        elif self.rebalance_frequency == 'quarterly':
            # 每季度第一个交易日
            rebalance_dates = []
            current_quarter = None

            for date in all_dates:
                quarter = (date.month - 1) // 3 + 1
                if current_quarter != (date.year, quarter):
                    rebalance_dates.append(date)
                    current_quarter = (date.year, quarter)

        else:
            raise ValueError(f"不支持的调仓频率: {self.rebalance_frequency}")

        return rebalance_dates

    def _calculate_transaction_cost(self, orders: List[Dict], order_type: str) -> float:
        """
        计算交易成本

        Args:
            orders: 订单列表 [{code, shares, price, value}]
            order_type: 订单类型 'buy' or 'sell'

        Returns:
            总交易成本
        """
        total_cost = 0

        commission_rate = (self.transaction_costs['buy_commission']
                          if order_type == 'buy'
                          else self.transaction_costs['sell_commission'])

        for order in orders:
            total_cost += order['value'] * commission_rate

        return total_cost

    def run_backtest(self,
                    prices_df: pd.DataFrame,
                    portfolio_adjustments: Dict[datetime, pd.DataFrame],
                    strategy_name: str = 'Enhanced') -> Dict:
        """
        运行回测

        Args:
            prices_df: 价格数据 DataFrame [date, code, close]
            portfolio_adjustments: 每个调仓日的目标持仓
                {
                    date: DataFrame[code, adjusted_weight]
                }
            strategy_name: 策略名称

        Returns:
            回测结果字典
        """
        logger.info("="*80)
        logger.info(f"开始运行 {strategy_name} 策略回测")
        logger.info("="*80)

        # 转为宽表
        prices_wide = prices_df.pivot(index='date', columns='code', values='close')
        prices_wide = prices_wide.sort_index()

        # 获取交易日
        trade_dates = prices_wide.index.tolist()
        logger.info(f"回测周期: {trade_dates[0]} 至 {trade_dates[-1]}")
        logger.info(f"交易日数: {len(trade_dates)}")

        # 获取调仓日期
        rebalance_dates = sorted(list(portfolio_adjustments.keys()))
        logger.info(f"调仓次数: {len(rebalance_dates)}")

        # 初始化回测状态
        capital = self.initial_capital  # 现金
        holdings = {}  # 持仓 {code: shares}
        nav_history = []
        trades_history = []
        total_transaction_costs = 0

        # 回测循环
        for i, date in enumerate(trade_dates):
            # 计算当前市值
            holdings_value = 0
            for code, shares in holdings.items():
                if code in prices_wide.columns and pd.notna(prices_wide.loc[date, code]):
                    holdings_value += shares * prices_wide.loc[date, code]

            total_value = capital + holdings_value

            # 记录净值
            nav_history.append({
                'date': date,
                'nav': total_value / self.initial_capital,
                'total_value': total_value,
                'capital': capital,
                'holdings_value': holdings_value
            })

            # 调仓
            if date in rebalance_dates:
                logger.info(f"\n调仓: {date}")

                target_portfolio = portfolio_adjustments[date]

                # 1. 清仓（卖出所有持仓）
                sell_value = 0
                for code, shares in list(holdings.items()):
                    if code in prices_wide.columns and pd.notna(prices_wide.loc[date, code]):
                        price = prices_wide.loc[date, code]
                        value = shares * price
                        sell_value += value

                        trades_history.append({
                            'date': date,
                            'code': code,
                            'action': 'sell',
                            'shares': shares,
                            'price': price,
                            'value': value
                        })

                holdings = {}

                # 计算卖出成本
                sell_cost = sell_value * self.transaction_costs['sell_commission']
                capital += sell_value - sell_cost
                total_transaction_costs += sell_cost

                # 2. 按目标权重买入
                buy_value = 0
                buy_orders = []

                for _, row in target_portfolio.iterrows():
                    code = row['code']
                    weight = row['adjusted_weight']

                    if code in prices_wide.columns and pd.notna(prices_wide.loc[date, code]):
                        price = prices_wide.loc[date, code]
                        if price > 0:
                            target_value = total_value * weight
                            shares = int(target_value / price / 100) * 100  # 整百股

                            if shares > 0:
                                actual_value = shares * price
                                buy_orders.append({
                                    'code': code,
                                    'shares': shares,
                                    'price': price,
                                    'value': actual_value
                                })
                                buy_value += actual_value

                # 计算买入成本
                buy_cost = buy_value * self.transaction_costs['buy_commission']
                total_transaction_costs += buy_cost

                # 执行买入
                if capital >= buy_value + buy_cost:
                    for order in buy_orders:
                        holdings[order['code']] = order['shares']
                        capital -= order['value']

                        trades_history.append({
                            'date': date,
                            'code': order['code'],
                            'action': 'buy',
                            'shares': order['shares'],
                            'price': order['price'],
                            'value': order['value']
                        })

                    capital -= buy_cost

                    logger.info(f"  卖出市值: {sell_value:,.0f}元, 成本: {sell_cost:,.0f}元")
                    logger.info(f"  买入市值: {buy_value:,.0f}元, 成本: {buy_cost:,.0f}元")
                    logger.info(f"  剩余现金: {capital:,.0f}元")
                else:
                    logger.warning(f"  现金不足，跳过本次调仓")

        # 计算回测指标
        metrics = self._calculate_metrics(
            pd.DataFrame(nav_history),
            pd.DataFrame(trades_history),
            total_transaction_costs
        )

        logger.info("="*80)
        logger.info("回测完成")
        logger.info("="*80)

        return {
            'strategy_name': strategy_name,
            'nav_history': pd.DataFrame(nav_history),
            'trades_history': pd.DataFrame(trades_history),
            'metrics': metrics,
            'total_transaction_costs': total_transaction_costs
        }

    def _calculate_metrics(self,
                          nav_df: pd.DataFrame,
                          trades_df: pd.DataFrame,
                          total_costs: float) -> Dict:
        """
        计算回测指标

        Args:
            nav_df: 净值历史
            trades_df: 交易历史
            total_costs: 总交易成本

        Returns:
            指标字典
        """
        # 基本指标
        final_nav = nav_df['nav'].iloc[-1]
        total_return = (final_nav - 1) * 100

        # 年化收益
        days = len(nav_df)
        years = days / 252
        annual_return = (final_nav ** (1/years) - 1) * 100 if years > 0 else 0

        # 最大回撤
        cummax = nav_df['nav'].cummax()
        drawdown = (nav_df['nav'] - cummax) / cummax * 100
        max_drawdown = drawdown.min()

        # 年化波动率
        daily_returns = nav_df['nav'].pct_change().dropna()
        annual_volatility = daily_returns.std() * np.sqrt(252) * 100

        # 夏普比率 (假设无风险利率3%)
        risk_free_rate = 0.03
        excess_return = annual_return / 100 - risk_free_rate
        sharpe = excess_return / (annual_volatility / 100) if annual_volatility > 0 else 0

        # 胜率
        win_days = len(daily_returns[daily_returns > 0])
        total_days = len(daily_returns)
        win_rate = win_days / total_days * 100 if total_days > 0 else 0

        # 卡尔玛比率
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        metrics = {
            'final_nav': f"{final_nav:.4f}",
            'total_return': f"{total_return:.2f}%",
            'annual_return': f"{annual_return:.2f}%",
            'max_drawdown': f"{max_drawdown:.2f}%",
            'annual_volatility': f"{annual_volatility:.2f}%",
            'sharpe': f"{sharpe:.2f}",
            'win_rate': f"{win_rate:.2f}%",
            'calmar': f"{calmar:.2f}",
            'total_transaction_costs': f"{total_costs:,.0f}",
            'cost_percentage': f"{total_costs/self.initial_capital*100:.2f}%"
        }

        # 打印指标
        logger.info("\n回测指标:")
        logger.info(f"  最终净值: {metrics['final_nav']}")
        logger.info(f"  总收益: {metrics['total_return']}")
        logger.info(f"  年化收益: {metrics['annual_return']}")
        logger.info(f"  最大回撤: {metrics['max_drawdown']}")
        logger.info(f"  年化波动: {metrics['annual_volatility']}")
        logger.info(f"  夏普比率: {metrics['sharpe']}")
        logger.info(f"  胜率: {metrics['win_rate']}")
        logger.info(f"  卡尔玛比率: {metrics['calmar']}")
        logger.info(f"  总交易成本: {metrics['total_transaction_costs']} ({metrics['cost_percentage']})")

        return metrics


if __name__ == '__main__':
    # 测试代码
    engine = BacktestEngine(initial_capital=1000000)

    # 创建测试价格数据
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    codes = ['000001', '000002', '600000']

    price_data = []
    for date in dates:
        for code in codes:
            price = 10 + np.random.randn() * 0.5
            price_data.append({'date': date, 'code': code, 'close': price})

    prices_df = pd.DataFrame(price_data)

    # 创建调仓计划
    portfolio_adjustments = {}
    for date in dates[::30]:  # 每30天调仓
        portfolio_adjustments[date] = pd.DataFrame({
            'code': codes,
            'adjusted_weight': [0.3, 0.4, 0.3]
        })

    # 运行回测
    result = engine.run_backtest(prices_df, portfolio_adjustments, 'Test Strategy')

    print("\n净值曲线:")
    print(result['nav_history'].head())
