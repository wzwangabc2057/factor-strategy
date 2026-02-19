"""
使用真实 ClickHouse 数据进行 ML 因子训练
适配 Mac Studio 的表结构
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import clickhouse_connect
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ClickHouse 配置
CLICKHOUSE_HOST = '192.168.0.74'
CLICKHOUSE_PORT = 8123


class RealDataFetcher:
    """从 ClickHouse 获取真实数据"""

    def __init__(self):
        self.client = clickhouse_connect.get_client(
            host=CLICKHOUSE_HOST,
            port=CLICKHOUSE_PORT,
            compress=False,
            query_limit=0
        )
        logger.info(f"连接 ClickHouse: {CLICKHOUSE_HOST}:{CLICKHOUSE_PORT}")

    def get_stock_pool(self, min_days: int = 60) -> list:
        """获取股票池（有足够历史数据的股票）"""
        query = """
        SELECT DISTINCT code
        FROM stock_data
        WHERE date >= today() - 30
          AND vol > 0
          AND code NOT LIKE '688%'  -- 排除科创板
          AND code NOT LIKE '300%'  -- 排除创业板（可选）
        ORDER BY code
        """
        result = self.client.query(query)
        codes = [row[0] for row in result.result_rows]
        logger.info(f"股票池: {len(codes)} 只")
        return codes

    def get_prices(self, codes: list, start_date: str, end_date: str) -> pd.DataFrame:
        """获取价格数据"""
        if len(codes) == 0:
            return pd.DataFrame()

        # 分批查询，避免查询太长
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

    def get_financial_data(self, codes: list) -> pd.DataFrame:
        """从 ClickHouse 获取财务数据"""
        if len(codes) == 0:
            return pd.DataFrame()

        codes_str = "','".join(codes)
        query = f"""
        SELECT
            code,
            roe,
            eps,
            bvps,
            net_profit,
            operating_revenue,
            gross_profit_margin,
            net_profit_margin,
            asset_liability_ratio,
            total_assets,
            shareholders_equity
        FROM stock_financial
        WHERE code IN ('{codes_str}')
        ORDER BY code, report_date DESC
        LIMIT 1 BY code
        """
        result = self.client.query(query)
        df = pd.DataFrame(result.result_rows,
                         columns=['code', 'roe', 'eps', 'bvps', 'net_profit',
                                 'operating_revenue', 'gross_profit_margin',
                                 'net_profit_margin', 'asset_liability_ratio',
                                 'total_assets', 'shareholders_equity'])
        logger.info(f"获取财务数据: {len(df)} 条")
        return df


def calculate_technical_factors(prices_df: pd.DataFrame) -> pd.DataFrame:
    """计算技术因子"""
    logger.info("计算技术因子...")

    # 按股票分组
    grouped = prices_df.groupby('code')

    factor_list = []

    for code, group in grouped:
        group = group.sort_values('date')
        close = group['close'].values
        volume = group['volume'].values
        high = group['high'].values
        low = group['low'].values

        if len(close) < 60:
            continue

        factors = {'code': code, 'date': group['date'].iloc[-1]}

        try:
            # 动量因子 (多周期)
            for period in [5, 10, 20, 60]:
                if len(close) > period:
                    factors[f'momentum_{period}d'] = (close[-1] / close[-period] - 1) * 100

            # RSI
            for period in [6, 14, 28]:
                if len(close) > period + 1:
                    delta = np.diff(close[-period-1:])
                    gain = np.mean(delta[delta > 0]) if np.any(delta > 0) else 0
                    loss = np.mean(-delta[delta < 0]) if np.any(delta < 0) else 0.001
                    factors[f'rsi_{period}'] = 100 - 100 / (1 + gain / loss) if loss != 0 else 50

            # 波动率
            for period in [10, 20, 60]:
                if len(close) > period:
                    factors[f'volatility_{period}d'] = np.std(np.diff(np.log(close[-period:]))) * np.sqrt(252) * 100

            # 成交量因子
            for period in [5, 10, 20]:
                if len(volume) > period and np.mean(volume[-period:]) > 0:
                    factors[f'volume_ratio_{period}d'] = volume[-1] / np.mean(volume[-period:])

            # 价格位置
            for period in [10, 20, 60]:
                if len(close) > period:
                    period_high = np.max(high[-period:])
                    period_low = np.min(low[-period:])
                    if period_high > period_low:
                        factors[f'price_position_{period}d'] = (close[-1] - period_low) / (period_high - period_low) * 100

            # 均线偏离
            for period in [10, 20, 60]:
                if len(close) > period:
                    ma = np.mean(close[-period:])
                    factors[f'ma_deviation_{period}d'] = (close[-1] - ma) / ma * 100

            factor_list.append(factors)

        except Exception as e:
            continue

    result = pd.DataFrame(factor_list)
    logger.info(f"计算完成: {len(result)} 只股票, {len(result.columns)-2} 个技术因子")
    return result


def prepare_training_data(prices_df: pd.DataFrame, financial_df: pd.DataFrame, forward_days: int = 5) -> tuple:
    """准备训练数据：因子 + 未来收益标签

    策略：使用每个月末作为一个时间切片，用该日期之前的数据计算因子，
    然后用该日期之后 N 天的收益作为标签。
    """

    logger.info(f"准备训练数据 (预测 {forward_days} 日收益)...")

    # 确保数据按日期排序
    prices_df = prices_df.sort_values('date')

    # 获取所有交易日
    all_dates = prices_df['date'].unique()
    all_dates = np.sort(all_dates)

    # 选择每月的最后一个交易日作为采样点（排除最后forward_days天）
    prices_df['month'] = prices_df['date'].dt.to_period('M')
    month_ends = prices_df.groupby('month')['date'].max().values

    # 排除最近的日期（没有足够的未来数据）
    min_end_date = all_dates[-forward_days - 5] if len(all_dates) > forward_days + 5 else all_dates[-1]
    month_ends = [d for d in month_ends if d < min_end_date]

    logger.info(f"采样点数量: {len(month_ends)}")

    all_samples = []

    for sample_date in month_ends:
        sample_date = pd.Timestamp(sample_date)

        # 只取到 sample_date 的数据计算因子
        sample_prices = prices_df[prices_df['date'] <= sample_date].copy()

        if len(sample_prices) < 1000:
            continue

        # 计算因子
        factors = calculate_technical_factors(sample_prices)

        if factors.empty:
            continue

        # 获取未来收益
        # 找到 sample_date 之后 forward_days 天的价格
        future_dates = all_dates[all_dates > sample_date]

        if len(future_dates) < forward_days:
            continue

        future_date = future_dates[forward_days - 1]

        for idx, row in factors.iterrows():
            code = row['code']

            try:
                current_price = prices_df[(prices_df['code'] == code) & (prices_df['date'] == sample_date)]['close'].values
                future_price = prices_df[(prices_df['code'] == code) & (prices_df['date'] == future_date)]['close'].values

                if len(current_price) == 0 or len(future_price) == 0:
                    continue

                current_price = current_price[0]
                future_price = future_price[0]

                if pd.isna(current_price) or pd.isna(future_price) or current_price <= 0:
                    continue

                forward_return = (future_price / current_price - 1) * 100

                sample = row.to_dict()
                sample['forward_return'] = forward_return
                sample['sample_date'] = sample_date

                # 移除非特征列
                for key in ['code', 'date']:
                    sample.pop(key, None)

                all_samples.append(sample)

            except Exception as e:
                continue

    if not all_samples:
        logger.error("没有有效的训练样本!")
        return None, None, None, None

    df = pd.DataFrame(all_samples)

    # 按日期排序
    df = df.sort_values('sample_date')

    # 分离特征和标签
    feature_cols = [c for c in df.columns if c not in ['forward_return', 'sample_date']]
    X = df[feature_cols].values
    y = df['forward_return'].values

    # 处理缺失值和无穷值
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(f"训练数据: {X.shape[0]} 样本, {X.shape[1]} 特征")

    return X, y, feature_cols, df


def train_model(X, y):
    """训练 ML 模型"""
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error
    from scipy.stats import spearmanr
    import warnings
    warnings.filterwarnings('ignore')

    logger.info("="*60)
    logger.info("开始模型训练")
    logger.info("="*60)

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 时间序列交叉验证
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)

    results = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if len(X_train) < 100 or len(X_test) < 10:
            continue

        # 训练多个模型
        models = {
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.01),
        }

        for model_name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # 评估
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                ic, _ = spearmanr(y_pred, y_test) if len(y_pred) > 5 else (0, 0)

                results.append({
                    'fold': fold,
                    'model': model_name,
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'rmse': rmse,
                    'ic': ic
                })

                logger.info(f"Fold {fold} {model_name}: Train={len(X_train)}, Test={len(X_test)}, RMSE={rmse:.4f}, IC={ic:.4f}")

            except Exception as e:
                logger.warning(f"Fold {fold} {model_name} 训练失败: {e}")

    if results:
        results_df = pd.DataFrame(results)
        logger.info("="*60)
        logger.info(f"平均 RMSE: {results_df['rmse'].mean():.4f}")
        logger.info(f"平均 IC: {results_df['ic'].mean():.4f}")
        logger.info(f"IC > 0 的比例: {(results_df['ic'] > 0).sum()}/{len(results_df)}")

        # 按模型分组统计
        for model_name in results_df['model'].unique():
            model_results = results_df[results_df['model'] == model_name]
            logger.info(f"  {model_name}: 平均 IC = {model_results['ic'].mean():.4f}")

        logger.info("="*60)

        return results_df

    return None


def main():
    logger.info("="*60)
    logger.info("ML 因子训练 - 真实数据版")
    logger.info("="*60)

    # 1. 连接数据源
    fetcher = RealDataFetcher()

    # 2. 获取股票池
    codes = fetcher.get_stock_pool()

    if len(codes) == 0:
        logger.error("股票池为空!")
        return

    # 3. 获取历史数据
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')  # 2年数据

    prices_df = fetcher.get_prices(codes, start_date, end_date)

    if len(prices_df) == 0:
        logger.error("无法获取价格数据!")
        return

    # 4. 获取财务数据
    financial_df = fetcher.get_financial_data(codes)

    # 5. 准备训练数据
    result = prepare_training_data(prices_df, financial_df, forward_days=5)

    if result[0] is None:
        return

    X, y, feature_cols, sample_df = result

    # 6. 训练模型
    results = train_model(X, y)

    if results is not None:
        # 保存结果
        results.to_csv(os.path.expanduser('~/112/pythontest/factor-strategy/ml_training_results.csv'), index=False)
        logger.info("结果已保存到 ml_training_results.csv")

        # 保存特征列表
        with open(os.path.expanduser('~/112/pythontest/factor-strategy/feature_cols.txt'), 'w') as f:
            f.write('\n'.join(feature_cols))
        logger.info(f"特征数量: {len(feature_cols)}")


if __name__ == '__main__':
    main()
