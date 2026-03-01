"""
快速回测 - 使用因子IC加权直接选股
更快、更简单的方法验证因子有效性
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import clickhouse_connect
import logging
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ClickHouse 配置
CLICKHOUSE_HOST = '192.168.0.74'
CLICKHOUSE_PORT = 8123

# 导入扩展因子计算器
from src.data.extended_factors import ExtendedFactorCalculator


class QuickBacktest:
    """快速回测 - 使用因子IC加权"""

    def __init__(self):
        self.client = clickhouse_connect.get_client(
            host=CLICKHOUSE_HOST,
            port=CLICKHOUSE_PORT,
            compress=False,
            query_limit=0
        )
        logger.info(f"连接 ClickHouse: {CLICKHOUSE_HOST}:{CLICKHOUSE_PORT}")

    def get_stock_pool(self) -> list:
        query = """
        SELECT DISTINCT code FROM stock_data
        WHERE date >= today() - 30 AND vol > 0 AND code NOT LIKE '688%'
        ORDER BY code
        """
        result = self.client.query(query)
        return [row[0] for row in result.result_rows]

    def get_prices(self, codes: list, start_date: str, end_date: str) -> pd.DataFrame:
        batch_size = 500
        all_dfs = []
        for i in range(0, len(codes), batch_size):
            batch_codes = codes[i:i+batch_size]
            codes_str = "','".join(batch_codes)
            query = f"""
            SELECT toDate(date) as date, code, open, high, low, close, vol as volume, amount
            FROM stock_data WHERE code IN ('{codes_str}')
            AND date >= toDate('{start_date}') AND date <= toDate('{end_date}')
            ORDER BY date, code
            """
            result = self.client.query(query)
            df = pd.DataFrame(result.result_rows,
                             columns=['date', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount'])
            all_dfs.append(df)

        if not all_dfs:
            return pd.DataFrame()
        df = pd.concat(all_dfs, ignore_index=True)
        df['date'] = pd.to_datetime(df['date'])
        return df

    def run(self, prices_df: pd.DataFrame, factor_ic_weights: dict,
            rebalance_freq: int = 5, top_n: int = 30, forward_days: int = 5):
        """
        快速回测

        Args:
            prices_df: 价格数据
            factor_ic_weights: 因子IC权重 {factor_name: ic_value}
            rebalance_freq: 调仓频率（天）
            top_n: 选股数量
            forward_days: 预测天数
        """

        logger.info("="*60)
        logger.info("快速回测 - IC加权因子选股")
        logger.info(f"调仓频率: {rebalance_freq}天, 选股数量: {top_n}")
        logger.info("="*60)

        prices_df = prices_df.sort_values('date')
        all_dates = np.sort(prices_df['date'].unique())

        factor_calculator = ExtendedFactorCalculator()

        # 使用因子IC绝对值作为权重
        factor_names = list(factor_ic_weights.keys())
        ic_weights = {k: abs(v) for k, v in factor_ic_weights.items()}

        logger.info(f"使用 {len(factor_names)} 个因子")

        period_returns = []
        signal_ics = []

        # 计算起始日期（需要60天历史数据）
        start_idx = 60

        while start_idx < len(all_dates) - forward_days:
            rebalance_date = all_dates[start_idx]
            eval_date = all_dates[min(start_idx + forward_days, len(all_dates) - 1)]

            # 计算因子
            current_prices = prices_df[prices_df['date'] <= rebalance_date].copy()
            factors = factor_calculator.calculate_all_factors(current_prices)

            if factors.empty:
                start_idx += rebalance_freq
                continue

            # 计算加权综合得分
            scores = np.zeros(len(factors))
            valid_factors = 0

            for factor in factor_names:
                if factor in factors.columns:
                    # 标准化因子
                    factor_values = factors[factor].values
                    factor_values = np.nan_to_num(factor_values, nan=0, posinf=0, neginf=0)
                    if np.std(factor_values) > 0:
                        factor_values = (factor_values - np.mean(factor_values)) / np.std(factor_values)

                    # 根据IC符号决定方向
                    ic = factor_ic_weights[factor]
                    if ic < 0:
                        factor_values = -factor_values  # 反转

                    # 加权
                    scores += factor_values * ic_weights[factor]
                    valid_factors += 1

            if valid_factors < 5:
                start_idx += rebalance_freq
                continue

            factors['score'] = scores

            # 选股
            selected_stocks = factors.nlargest(top_n, 'score')['code'].tolist()

            # 计算实际收益
            actual_returns = []
            for code in selected_stocks:
                try:
                    buy_price = prices_df[(prices_df['code'] == code) &
                                         (prices_df['date'] == rebalance_date)]['close'].values
                    sell_price = prices_df[(prices_df['code'] == code) &
                                          (prices_df['date'] == eval_date)]['close'].values
                    if len(buy_price) > 0 and len(sell_price) > 0:
                        ret = (sell_price[0] / buy_price[0] - 1) * 100
                        actual_returns.append(ret)
                except:
                    continue

            if actual_returns:
                period_return = np.mean(actual_returns)
                period_returns.append({
                    'date': pd.Timestamp(rebalance_date),
                    'return': period_return,
                    'n_stocks': len(actual_returns)
                })

                # 计算信号IC
                all_actual = []
                all_scores = []
                for idx, row in factors.iterrows():
                    code = row['code']
                    try:
                        buy_price = prices_df[(prices_df['code'] == code) &
                                             (prices_df['date'] == rebalance_date)]['close'].values
                        sell_price = prices_df[(prices_df['code'] == code) &
                                              (prices_df['date'] == eval_date)]['close'].values
                        if len(buy_price) > 0 and len(sell_price) > 0:
                            all_actual.append((sell_price[0] / buy_price[0] - 1) * 100)
                            all_scores.append(row['score'])
                    except:
                        continue

                if len(all_actual) > 10:
                    ic, _ = spearmanr(all_actual, all_scores)
                    signal_ics.append(ic)
                    logger.info(f"{pd.Timestamp(rebalance_date).strftime('%Y-%m-%d')}: IC={ic:.4f}, Return={period_return:.2f}%")

            start_idx += rebalance_freq

        # 计算累计收益
        if not period_returns:
            logger.error("没有有效的回测结果!")
            return None

        returns_df = pd.DataFrame(period_returns)
        returns_df['cumulative'] = (1 + returns_df['return'] / 100).cumprod()

        total_return = returns_df['cumulative'].iloc[-1] - 1
        n_days = (returns_df['date'].iloc[-1] - returns_df['date'].iloc[0]).days
        annual_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1

        mean_return = returns_df['return'].mean()
        std_return = returns_df['return'].std()
        sharpe = mean_return / std_return * np.sqrt(252 / rebalance_freq) if std_return > 0 else 0

        cumulative = returns_df['cumulative']
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        logger.info("\n" + "="*60)
        logger.info("回测结果")
        logger.info("="*60)
        logger.info(f"回测区间: {returns_df['date'].iloc[0].strftime('%Y-%m-%d')} ~ {returns_df['date'].iloc[-1].strftime('%Y-%m-%d')}")
        logger.info(f"总收益: {total_return*100:.2f}%")
        logger.info(f"年化收益: {annual_return*100:.2f}%")
        logger.info(f"夏普比率: {sharpe:.2f}")
        logger.info(f"最大回撤: {max_drawdown*100:.2f}%")
        logger.info(f"平均信号IC: {np.mean(signal_ics):.4f}")
        logger.info(f"胜率: {(returns_df['return'] > 0).sum() / len(returns_df) * 100:.1f}%")
        logger.info(f"回测期数: {len(returns_df)}")
        logger.info("="*60)

        return {
            'returns_df': returns_df,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'mean_ic': np.mean(signal_ics),
            'win_rate': (returns_df['return'] > 0).sum() / len(returns_df)
        }


def main():
    logger.info("="*60)
    logger.info("快速回测验证")
    logger.info("="*60)

    # 从IC分数文件读取因子权重
    ic_file = os.path.expanduser('~/112/pythontest/factor-strategy/factor_ic_scores.csv')
    if os.path.exists(ic_file):
        ic_df = pd.read_csv(ic_file)
        factor_ic_weights = dict(zip(ic_df['factor'], ic_df['ic']))
        # 只使用IC绝对值前20的因子
        ic_df['abs_ic'] = ic_df['ic'].abs()
        top_factors = ic_df.nlargest(20, 'abs_ic')
        factor_ic_weights = dict(zip(top_factors['factor'], top_factors['ic']))
        logger.info(f"使用前20个IC因子")
    else:
        # 默认因子
        factor_ic_weights = {
            'momentum_accel': -0.25,
            'vol_of_vol_10d': -0.14,
            'momentum_diff': -0.13,
            'volatility_10d': -0.12,
            'momentum_10d': 0.12,
            'trend_strength': -0.12,
            'parkinson_vol': -0.11,
            'return_skew': -0.11,
            'price_slope': 0.11,
            'macd_hist': 0.10,
            'volume_ratio_5d': -0.10,
            'price_efficiency': -0.09,
            'is_bullish': -0.09,
            'distance_high_10d': -0.09,
            'momentum_5d': -0.08,
            'kdj_k': 0.08,
            'kdj_d': 0.08,
            'kdj_j': 0.08,
            'price_position_20d': 0.08,
            'upper_shadow': 0.08
        }

    backtest = QuickBacktest()
    codes = backtest.get_stock_pool()
    logger.info(f"股票池: {len(codes)} 只")

    # 使用最近1年数据
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')
    prices_df = backtest.get_prices(codes, start_date, end_date)
    logger.info(f"获取 {len(prices_df)} 条价格数据")

    # 运行回测
    results = backtest.run(
        prices_df,
        factor_ic_weights,
        rebalance_freq=5,
        top_n=30,
        forward_days=5
    )

    if results:
        output_dir = os.path.expanduser('~/112/pythontest/factor-strategy')
        results['returns_df'].to_csv(f'{output_dir}/quick_backtest_returns.csv', index=False)

        logger.info("\n" + "="*60)
        logger.info("最终结果")
        logger.info("="*60)
        if results['annual_return'] >= 0.30:
            logger.info(f"✓ 年化收益 {results['annual_return']*100:.2f}% >= 30% 目标达成!")
        else:
            logger.info(f"✗ 年化收益 {results['annual_return']*100:.2f}% < 30% 目标未达成")
            logger.info(f"  距离目标还差 {(0.30 - results['annual_return'])*100:.2f}%")
        logger.info("="*60)


if __name__ == '__main__':
    main()
