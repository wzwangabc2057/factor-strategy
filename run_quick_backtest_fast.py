"""
快速回测 - 优化版
核心优化：预建价格pivot表，向量化计算收益和IC
预计运行时间：10-15分钟（原版2-3小时）
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
import time
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CLICKHOUSE_HOST = '192.168.0.74'
CLICKHOUSE_PORT = 8123

from src.data.extended_factors import ExtendedFactorCalculator


def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("快速回测验证 (优化版)")
    logger.info("=" * 60)

    # 读取因子IC权重
    ic_file = os.path.expanduser('~/112/pythontest/factor-strategy/factor_ic_scores.csv')
    if os.path.exists(ic_file):
        ic_df = pd.read_csv(ic_file)
        ic_df['abs_ic'] = ic_df['ic'].abs()
        top_factors = ic_df.nlargest(20, 'abs_ic')
        factor_ic_weights = dict(zip(top_factors['factor'], top_factors['ic']))
        logger.info(f"使用前20个IC因子")
    else:
        factor_ic_weights = {
            'momentum_accel': -0.25, 'vol_of_vol_10d': -0.14,
            'momentum_diff': -0.13, 'volatility_10d': -0.12,
            'momentum_10d': 0.12, 'trend_strength': -0.12,
        }

    # 连接数据库
    client = clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST, port=CLICKHOUSE_PORT,
        compress=False, query_limit=0
    )
    logger.info(f"连接 ClickHouse: {CLICKHOUSE_HOST}:{CLICKHOUSE_PORT}")

    # 获取股票池
    codes_result = client.query("""
        SELECT DISTINCT code FROM stock_data
        WHERE date >= today() - 30 AND vol > 0 AND code NOT LIKE '688%'
        ORDER BY code
    """)
    codes = [row[0] for row in codes_result.result_rows]
    logger.info(f"股票池: {len(codes)} 只")

    # 获取价格数据
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')

    all_dfs = []
    batch_size = 500
    for i in range(0, len(codes), batch_size):
        batch = codes[i:i + batch_size]
        codes_str = "','".join(batch)
        query = f"""
        SELECT toDate(date) as date, code, open, high, low, close, vol as volume, amount
        FROM stock_data WHERE code IN ('{codes_str}')
        AND date >= toDate('{start_date}') AND date <= toDate('{end_date}')
        ORDER BY date, code
        """
        result = client.query(query)
        df = pd.DataFrame(result.result_rows,
                           columns=['date', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount'])
        all_dfs.append(df)

    prices_df = pd.concat(all_dfs, ignore_index=True)
    prices_df['date'] = pd.to_datetime(prices_df['date'])
    logger.info(f"获取 {len(prices_df)} 条价格数据")

    # ========== 核心优化：预建价格pivot表 ==========
    logger.info("预建价格pivot表...")
    close_pivot = prices_df.pivot_table(index='date', columns='code', values='close', aggfunc='last')
    close_pivot = close_pivot.sort_index()
    all_dates = close_pivot.index.values
    logger.info(f"Pivot表: {close_pivot.shape[0]} 日 x {close_pivot.shape[1]} 只股票")

    # 预计算所有日期的forward returns (向量化)
    logger.info("预计算forward returns...")
    forward_days = 5
    # 对每个日期，计算forward_days天后的收益
    forward_returns = {}
    for i in range(len(all_dates) - forward_days):
        date = all_dates[i]
        future_date = all_dates[i + forward_days]
        price_now = close_pivot.loc[date]
        price_future = close_pivot.loc[future_date]
        ret = (price_future / price_now - 1) * 100
        forward_returns[date] = ret
    logger.info(f"预计算完成: {len(forward_returns)} 期")

    # ========== 回测主循环 ==========
    logger.info("=" * 60)
    logger.info("开始回测")
    logger.info("=" * 60)

    factor_calculator = ExtendedFactorCalculator()
    factor_names = list(factor_ic_weights.keys())
    ic_abs_weights = {k: abs(v) for k, v in factor_ic_weights.items()}

    rebalance_freq = 5
    top_n = 30
    start_idx = 60
    period_returns = []
    signal_ics = []
    period_count = 0

    while start_idx < len(all_dates) - forward_days:
        rebalance_date = all_dates[start_idx]
        period_count += 1
        t1 = time.time()

        # 只传入到rebalance_date的数据给因子计算
        current_prices = prices_df[prices_df['date'] <= rebalance_date].copy()
        factors = factor_calculator.calculate_all_factors(current_prices)

        if factors.empty:
            start_idx += rebalance_freq
            continue

        # 计算加权综合得分 (向量化)
        scores = np.zeros(len(factors))
        valid_factors = 0

        for factor in factor_names:
            if factor in factors.columns:
                vals = factors[factor].values.astype(float)
                vals = np.nan_to_num(vals, nan=0, posinf=0, neginf=0)
                std = np.std(vals)
                if std > 0:
                    vals = (vals - np.mean(vals)) / std
                ic = factor_ic_weights[factor]
                if ic < 0:
                    vals = -vals
                scores += vals * ic_abs_weights[factor]
                valid_factors += 1

        if valid_factors < 5:
            start_idx += rebalance_freq
            continue

        factors['score'] = scores

        # 选股：得分前top_n
        selected = factors.nlargest(top_n, 'score')
        selected_codes = selected['code'].tolist()

        # ========== 用pivot表向量化计算收益 ==========
        if rebalance_date in forward_returns:
            fwd_ret = forward_returns[rebalance_date]

            # 选中股票的收益
            valid_selected = [c for c in selected_codes if c in fwd_ret.index and not np.isnan(fwd_ret[c])]
            if valid_selected:
                sel_returns = fwd_ret[valid_selected].values
                period_return = np.mean(sel_returns)
                period_returns.append({
                    'date': pd.Timestamp(rebalance_date),
                    'return': period_return,
                    'n_stocks': len(valid_selected)
                })

                # 信号IC：向量化计算
                factor_codes = factors['code'].values
                factor_scores = factors['score'].values
                valid_mask = np.array([c in fwd_ret.index and not np.isnan(fwd_ret.get(c, np.nan))
                                       for c in factor_codes])
                if valid_mask.sum() > 10:
                    valid_codes = factor_codes[valid_mask]
                    valid_scores = factor_scores[valid_mask]
                    actual_rets = fwd_ret[valid_codes].values
                    nan_mask = ~np.isnan(actual_rets)
                    if nan_mask.sum() > 10:
                        ic, _ = spearmanr(actual_rets[nan_mask], valid_scores[nan_mask])
                        signal_ics.append(ic)

                        elapsed = time.time() - t1
                        total_elapsed = time.time() - t0
                        date_str = pd.Timestamp(rebalance_date).strftime('%Y-%m-%d')
                        logger.info(
                            f"[{period_count}] {date_str}: IC={ic:.4f}, Return={period_return:+.2f}%, "
                            f"选{len(valid_selected)}只, 耗时{elapsed:.0f}s, 总{total_elapsed:.0f}s"
                        )

        start_idx += rebalance_freq

    # ========== 汇总结果 ==========
    if not period_returns:
        logger.error("没有有效的回测结果!")
        return

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

    total_time = time.time() - t0

    logger.info("\n" + "=" * 60)
    logger.info("回测结果")
    logger.info("=" * 60)
    logger.info(f"回测区间: {returns_df['date'].iloc[0].strftime('%Y-%m-%d')} ~ {returns_df['date'].iloc[-1].strftime('%Y-%m-%d')}")
    logger.info(f"总收益: {total_return * 100:.2f}%")
    logger.info(f"年化收益: {annual_return * 100:.2f}%")
    logger.info(f"夏普比率: {sharpe:.2f}")
    logger.info(f"最大回撤: {max_drawdown * 100:.2f}%")
    logger.info(f"平均信号IC: {np.mean(signal_ics):.4f}")
    logger.info(f"胜率: {(returns_df['return'] > 0).sum() / len(returns_df) * 100:.1f}%")
    logger.info(f"回测期数: {len(returns_df)}")
    logger.info(f"总耗时: {total_time:.0f}s ({total_time / 60:.1f}分钟)")
    logger.info("=" * 60)

    if annual_return >= 0.30:
        logger.info(f"✓ 年化收益 {annual_return * 100:.2f}% >= 30% 目标达成!")
    else:
        logger.info(f"✗ 年化收益 {annual_return * 100:.2f}% < 30% 目标未达成")
        logger.info(f"  距离目标还差 {(0.30 - annual_return) * 100:.2f}%")

    # 保存结果
    output_dir = os.path.expanduser('~/112/pythontest/factor-strategy')
    returns_df.to_csv(f'{output_dir}/quick_backtest_returns.csv', index=False)
    logger.info(f"结果已保存到 quick_backtest_returns.csv")

    # 打印每期明细
    logger.info("\n" + "=" * 60)
    logger.info("每期明细")
    logger.info("=" * 60)
    for _, row in returns_df.iterrows():
        logger.info(f"  {row['date'].strftime('%Y-%m-%d')}: {row['return']:+.2f}% (累计: {row['cumulative']:.4f})")


if __name__ == '__main__':
    main()
