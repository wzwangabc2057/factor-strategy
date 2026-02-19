"""
优化版 ML 因子训练
- 50+ 扩展因子
- LightGBM 模型
- 因子筛选
目标：IC > 0.15，年化收益 > 30%
"""

import sys
import os

# 设置 libomp 库路径 (macOS)
os.environ['DYLD_LIBRARY_PATH'] = '/opt/homebrew/opt/libomp/lib:' + os.environ.get('DYLD_LIBRARY_PATH', '')

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


class OptimizedDataFetcher:
    """优化的数据获取器"""

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

    def get_financial_data(self, codes: list) -> pd.DataFrame:
        """获取财务数据"""
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


def prepare_training_data_optimized(prices_df: pd.DataFrame, financial_df: pd.DataFrame,
                                     forward_days: int = 5) -> tuple:
    """准备训练数据 - 使用扩展因子"""

    logger.info(f"准备训练数据 (预测 {forward_days} 日收益)...")

    prices_df = prices_df.sort_values('date')
    all_dates = prices_df['date'].unique()
    all_dates = np.sort(all_dates)

    # 采样点
    prices_df['month'] = prices_df['date'].dt.to_period('M')
    month_ends = prices_df.groupby('month')['date'].max().values

    min_end_date = all_dates[-forward_days - 5] if len(all_dates) > forward_days + 5 else all_dates[-1]
    month_ends = [d for d in month_ends if d < min_end_date]

    logger.info(f"采样点数量: {len(month_ends)}")

    factor_calculator = ExtendedFactorCalculator()
    all_samples = []

    for i, sample_date in enumerate(month_ends):
        sample_date = pd.Timestamp(sample_date)

        sample_prices = prices_df[prices_df['date'] <= sample_date].copy()

        if len(sample_prices) < 1000:
            continue

        # 计算扩展因子
        factors = factor_calculator.calculate_all_factors(sample_prices)

        if factors.empty:
            continue

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

                for key in ['code', 'date']:
                    sample.pop(key, None)

                all_samples.append(sample)

            except Exception as e:
                continue

        if (i + 1) % 5 == 0:
            logger.info(f"  已处理 {i+1}/{len(month_ends)} 个采样点")

    if not all_samples:
        logger.error("没有有效的训练样本!")
        return None, None, None, None

    df = pd.DataFrame(all_samples)
    df = df.sort_values('sample_date')

    feature_cols = [c for c in df.columns if c not in ['forward_return', 'sample_date']]
    X = df[feature_cols].values
    y = df['forward_return'].values

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(f"训练数据: {X.shape[0]} 样本, {X.shape[1]} 特征")

    return X, y, feature_cols, df


def select_factors_by_ic(X: np.ndarray, y: np.ndarray, feature_cols: list,
                         top_k: int = 30) -> tuple:
    """通过 IC 筛选因子"""

    logger.info("因子 IC 筛选...")

    ic_scores = []
    for i, col in enumerate(feature_cols):
        ic, _ = spearmanr(X[:, i], y)
        ic_scores.append({'factor': col, 'ic': ic if not np.isnan(ic) else 0})

    ic_df = pd.DataFrame(ic_scores)
    ic_df = ic_df.reindex(ic_df['ic'].abs().sort_values(ascending=False).index)

    # 选择 top_k 个因子
    selected_factors = ic_df.head(top_k)['factor'].tolist()
    selected_indices = [feature_cols.index(f) for f in selected_factors]

    logger.info(f"筛选后保留 {len(selected_factors)} 个因子")
    logger.info(f"Top 5 IC 因子: {ic_df.head(5)['factor'].tolist()}")

    return X[:, selected_indices], selected_factors, ic_df


def train_lightgbm_model(X: np.ndarray, y: np.ndarray) -> dict:
    """训练 LightGBM 模型"""

    try:
        import lightgbm as lgb
        # Test if library actually loads
        lgb.__version__
        HAS_LGB = True
    except (ImportError, OSError) as e:
        HAS_LGB = False
        logger.warning(f"LightGBM 不可用，使用 Ridge 替代: {e}")

    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error

    logger.info("="*60)
    logger.info("开始模型训练")
    logger.info("="*60)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)

    results = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if len(X_train) < 100 or len(X_test) < 10:
            continue

        # LightGBM
        if HAS_LGB:
            try:
                train_data = lgb.Dataset(X_train, label=y_train)
                valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

                params = {
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'verbose': -1,
                    'num_threads': 4
                }

                model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=500,
                    valid_sets=[valid_data],
                    callbacks=[lgb.log_evaluation(period=0)]
                )

                y_pred = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                ic, _ = spearmanr(y_pred, y_test)

                results.append({
                    'fold': fold,
                    'model': 'LightGBM',
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'rmse': rmse,
                    'ic': ic
                })

                logger.info(f"Fold {fold} LightGBM: Train={len(X_train)}, Test={len(X_test)}, RMSE={rmse:.4f}, IC={ic:.4f}")

            except Exception as e:
                logger.warning(f"LightGBM Fold {fold} 失败: {e}")

        # Ridge (备选)
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        y_pred_ridge = ridge.predict(X_test)
        rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
        ic_ridge, _ = spearmanr(y_pred_ridge, y_test)

        results.append({
            'fold': fold,
            'model': 'Ridge',
            'train_size': len(X_train),
            'test_size': len(X_test),
            'rmse': rmse_ridge,
            'ic': ic_ridge
        })

        if not HAS_LGB:
            logger.info(f"Fold {fold} Ridge: Train={len(X_train)}, Test={len(X_test)}, RMSE={rmse_ridge:.4f}, IC={ic_ridge:.4f}")

    if results:
        results_df = pd.DataFrame(results)
        logger.info("="*60)
        logger.info(f"平均 RMSE: {results_df['rmse'].mean():.4f}")
        logger.info(f"平均 IC: {results_df['ic'].mean():.4f}")
        logger.info(f"IC > 0 的比例: {(results_df['ic'] > 0).sum()}/{len(results_df)}")

        for model_name in results_df['model'].unique():
            model_results = results_df[results_df['model'] == model_name]
            logger.info(f"  {model_name}: 平均 IC = {model_results['ic'].mean():.4f}")

        logger.info("="*60)

        return results_df

    return None


def main():
    logger.info("="*60)
    logger.info("优化版 ML 因子训练")
    logger.info("目标: IC > 0.15, 年化收益 > 30%")
    logger.info("="*60)

    # 1. 获取数据
    fetcher = OptimizedDataFetcher()
    codes = fetcher.get_stock_pool()

    if len(codes) == 0:
        logger.error("股票池为空!")
        return

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

    prices_df = fetcher.get_prices(codes, start_date, end_date)

    if len(prices_df) == 0:
        logger.error("无法获取价格数据!")
        return

    financial_df = fetcher.get_financial_data(codes)

    # 2. 准备训练数据 (扩展因子)
    result = prepare_training_data_optimized(prices_df, financial_df, forward_days=5)

    if result[0] is None:
        return

    X, y, feature_cols, sample_df = result

    # 3. 因子筛选
    X_selected, selected_factors, ic_df = select_factors_by_ic(X, y, feature_cols, top_k=30)

    # 4. 训练模型
    results = train_lightgbm_model(X_selected, y)

    if results is not None:
        # 保存结果
        output_dir = os.path.expanduser('~/112/pythontest/factor-strategy')
        results.to_csv(f'{output_dir}/optimized_training_results.csv', index=False)
        ic_df.to_csv(f'{output_dir}/factor_ic_scores.csv', index=False)

        with open(f'{output_dir}/selected_factors.txt', 'w') as f:
            f.write('\n'.join(selected_factors))

        logger.info(f"结果已保存")
        logger.info(f"特征数量: {len(selected_factors)}")

        # 总结
        best_ic = results['ic'].max()
        avg_ic = results['ic'].mean()
        logger.info("="*60)
        logger.info(f"最终结果: 最佳 IC = {best_ic:.4f}, 平均 IC = {avg_ic:.4f}")
        logger.info("="*60)


if __name__ == '__main__':
    main()
