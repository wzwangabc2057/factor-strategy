"""
回测验证 - 使用训练好的模型进行回测
目标: 验证年化收益 > 30%
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
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ClickHouse 配置
CLICKHOUSE_HOST = '192.168.0.74'
CLICKHOUSE_PORT = 8123

# 导入扩展因子计算器
from src.data.extended_factors import ExtendedFactorCalculator


class BacktestDataFetcher:
    """回测数据获取器"""

    def __init__(self):
        self.client = clickhouse_connect.get_client(
            host=CLICKHOUSE_HOST,
            port=CLICKHOUSE_PORT,
            compress=False,
            query_limit=0
        )
        logger.info(f"连接 ClickHouse: {CLICKHOUSE_HOST}:{CLICKHOUSE_PORT}")

    def get_stock_pool(self) -> list:
        """获取股票池"""
        query = """
        SELECT DISTINCT code
        FROM stock_data
        WHERE date >= today() - 30
          AND vol > 0
          AND code NOT LIKE '688%'
        ORDER BY code
        """
        result = self.client.query(query)
        codes = [row[0] for row in result.result_rows]
        logger.info(f"股票池: {len(codes)} 只")
        return codes

    def get_prices(self, codes: list, start_date: str, end_date: str) -> pd.DataFrame:
        """获取价格数据"""
        batch_size = 500
        all_dfs = []

        for i in range(0, len(codes), batch_size):
            batch_codes = codes[i:i+batch_size]
            codes_str = "','".join(batch_codes)
            query = f"""
            SELECT
                toDate(date) as date,
                code,
                open,
                high,
                low,
                close,
                vol as volume,
                amount
            FROM stock_data
            WHERE code IN ('{codes_str}')
              AND date >= toDate('{start_date}')
              AND date <= toDate('{end_date}')
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
        logger.info(f"获取 {len(df)} 条价格数据 ({start_date} ~ {end_date})")
        return df


def run_rolling_backtest(prices_df: pd.DataFrame, selected_factors: list,
                          train_window: int = 120, rebalance_freq: int = 5,
                          top_n: int = 30) -> dict:
    """
    滚动回测

    Args:
        prices_df: 价格数据
        selected_factors: 选择的因子列表
        train_window: 训练窗口（天）
        rebalance_freq: 调仓频率（天）
        top_n: 选股数量

    Returns:
        回测结果
    """

    logger.info("="*60)
    logger.info("开始滚动回测")
    logger.info(f"训练窗口: {train_window}天, 调仓频率: {rebalance_freq}天, 选股数量: {top_n}")
    logger.info("="*60)

    prices_df = prices_df.sort_values('date')
    all_dates = prices_df['date'].unique()
    all_dates = np.sort(all_dates)

    factor_calculator = ExtendedFactorCalculator()

    # 存储每期收益
    period_returns = []
    signal_ics = []

    # 从train_window开始回测
    start_idx = train_window

    while start_idx < len(all_dates) - rebalance_freq:
        # 训练数据截止日期
        train_end_date = all_dates[start_idx]
        # 调仓日期
        rebalance_date = all_dates[start_idx]
        # 下一期评估日期
        eval_date = all_dates[min(start_idx + rebalance_freq, len(all_dates) - 1)]

        logger.info(f"\n{'='*50}")
        logger.info(f"回测日期: {pd.Timestamp(rebalance_date).strftime('%Y-%m-%d')}")

        # 1. 使用训练窗口内的数据训练模型
        train_start_idx = max(0, start_idx - train_window)
        train_dates = all_dates[train_start_idx:start_idx]

        # 准备训练数据
        train_samples = []

        for i, sample_date in enumerate(train_dates[::20]):  # 每20天采样一次
            sample_date = pd.Timestamp(sample_date)
            sample_prices = prices_df[prices_df['date'] <= sample_date].copy()

            if len(sample_prices) < 1000:
                continue

            factors = factor_calculator.calculate_all_factors(sample_prices)

            if factors.empty:
                continue

            # 计算未来5日收益
            future_dates = all_dates[all_dates > sample_date]
            if len(future_dates) < 5:
                continue
            future_date = future_dates[4]

            for idx, row in factors.iterrows():
                code = row['code']
                try:
                    current_price = prices_df[(prices_df['code'] == code) &
                                             (prices_df['date'] == sample_date)]['close'].values
                    future_price = prices_df[(prices_df['code'] == code) &
                                            (prices_df['date'] == future_date)]['close'].values

                    if len(current_price) == 0 or len(future_price) == 0:
                        continue

                    forward_return = (future_price[0] / current_price[0] - 1) * 100

                    sample = {'forward_return': forward_return}
                    for f in selected_factors:
                        if f in row:
                            sample[f] = row[f]

                    train_samples.append(sample)
                except:
                    continue

        if len(train_samples) < 500:
            logger.warning(f"训练样本不足: {len(train_samples)}")
            start_idx += rebalance_freq
            continue

        # 训练模型
        train_df = pd.DataFrame(train_samples)
        train_df = train_df.dropna()

        X_train = train_df[selected_factors].values
        y_train = train_df['forward_return'].values
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = Ridge(alpha=1.0)
        model.fit(X_train_scaled, y_train)

        # 2. 使用模型预测当前股票收益
        current_prices = prices_df[prices_df['date'] <= rebalance_date].copy()

        if len(current_prices) < 1000:
            start_idx += rebalance_freq
            continue

        current_factors = factor_calculator.calculate_all_factors(current_prices)

        if current_factors.empty:
            start_idx += rebalance_freq
            continue

        X_pred = current_factors[selected_factors].values
        X_pred = np.nan_to_num(X_pred, nan=0.0, posinf=0.0, neginf=0.0)
        X_pred_scaled = scaler.transform(X_pred)

        predicted_returns = model.predict(X_pred_scaled)

        # 3. 选股: 选择预测收益最高的top_n只股票
        current_factors['predicted_return'] = predicted_returns
        selected_stocks = current_factors.nlargest(top_n, 'predicted_return')['code'].tolist()

        # 4. 计算实际收益
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
                'date': rebalance_date,
                'return': period_return,
                'n_stocks': len(actual_returns)
            })

            # 计算信号IC
            all_actual = []
            all_pred = []
            for idx, row in current_factors.iterrows():
                code = row['code']
                try:
                    buy_price = prices_df[(prices_df['code'] == code) &
                                         (prices_df['date'] == rebalance_date)]['close'].values
                    sell_price = prices_df[(prices_df['code'] == code) &
                                          (prices_df['date'] == eval_date)]['close'].values

                    if len(buy_price) > 0 and len(sell_price) > 0:
                        actual_ret = (sell_price[0] / buy_price[0] - 1) * 100
                        all_actual.append(actual_ret)
                        all_pred.append(row['predicted_return'])
                except:
                    continue

            if len(all_actual) > 10:
                ic, _ = spearmanr(all_actual, all_pred)
                signal_ics.append(ic)
                logger.info(f"信号IC: {ic:.4f}")

        logger.info(f"本期收益: {period_return:.2f}%")

        # 移动到下一期
        start_idx += rebalance_freq

    # 计算累计收益和年化收益
    if not period_returns:
        logger.error("没有有效的回测结果!")
        return None

    returns_df = pd.DataFrame(period_returns)
    returns_df['cumulative'] = (1 + returns_df['return'] / 100).cumprod()

    total_return = returns_df['cumulative'].iloc[-1] - 1
    n_days = (returns_df['date'].iloc[-1] - returns_df['date'].iloc[0]).days
    annual_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1

    # 计算夏普比率
    mean_return = returns_df['return'].mean()
    std_return = returns_df['return'].std()
    sharpe = mean_return / std_return * np.sqrt(252 / rebalance_freq) if std_return > 0 else 0

    # 最大回撤
    cumulative = returns_df['cumulative']
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    logger.info("\n" + "="*60)
    logger.info("回测结果")
    logger.info("="*60)
    logger.info(f"回测区间: {pd.Timestamp(returns_df['date'].iloc[0]).strftime('%Y-%m-%d')} ~ {pd.Timestamp(returns_df['date'].iloc[-1]).strftime('%Y-%m-%d')}")
    logger.info(f"总收益: {total_return*100:.2f}%")
    logger.info(f"年化收益: {annual_return*100:.2f}%")
    logger.info(f"夏普比率: {sharpe:.2f}")
    logger.info(f"最大回撤: {max_drawdown*100:.2f}%")
    logger.info(f"平均信号IC: {np.mean(signal_ics):.4f}")
    logger.info(f"胜率: {(returns_df['return'] > 0).sum() / len(returns_df) * 100:.1f}%")
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
    logger.info("回测验证 - 使用训练好的模型")
    logger.info("="*60)

    # 1. 读取选择的因子
    factors_file = os.path.expanduser('~/112/pythontest/factor-strategy/selected_factors.txt')
    if os.path.exists(factors_file):
        with open(factors_file, 'r') as f:
            selected_factors = [line.strip() for line in f if line.strip()]
    else:
        # 使用默认因子
        selected_factors = [
            'momentum_accel', 'vol_of_vol_10d', 'momentum_diff', 'volatility_10d',
            'momentum_10d', 'trend_strength', 'parkinson_vol', 'return_skew',
            'price_slope', 'macd_hist', 'volume_ratio_5d', 'price_efficiency',
            'is_bullish', 'distance_high_10d', 'momentum_5d'
        ]

    logger.info(f"使用 {len(selected_factors)} 个因子")

    # 2. 获取数据
    fetcher = BacktestDataFetcher()
    codes = fetcher.get_stock_pool()

    if len(codes) == 0:
        logger.error("股票池为空!")
        return

    # 使用最近1年的数据进行回测
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')

    prices_df = fetcher.get_prices(codes, start_date, end_date)

    if len(prices_df) == 0:
        logger.error("无法获取价格数据!")
        return

    # 3. 运行回测
    results = run_rolling_backtest(
        prices_df,
        selected_factors,
        train_window=120,
        rebalance_freq=5,
        top_n=30
    )

    if results:
        # 保存结果
        output_dir = os.path.expanduser('~/112/pythontest/factor-strategy')
        results['returns_df'].to_csv(f'{output_dir}/backtest_returns.csv', index=False)

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
