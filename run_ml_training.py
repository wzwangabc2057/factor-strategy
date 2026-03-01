"""
运行机器学习因子合成
使用因子数据训练ML模型，预测股票收益
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
import logging

from src.ml.feature_builder import FeatureBuilder
from src.ml.model_trainer import ModelTrainer, RollingTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_sample_data(n_stocks: int = 50, n_days: int = 500) -> tuple:
    """
    生成模拟数据用于测试

    Args:
        n_stocks: 股票数量
        n_days: 天数

    Returns:
        (prices_df, factors_df)
    """
    np.random.seed(42)

    dates = pd.date_range('2020-01-01', periods=n_days, freq='B')  # 工作日
    codes = [f'{i:06d}' for i in range(1, n_stocks + 1)]

    logger.info(f"生成模拟数据: {len(dates)} 天, {len(codes)} 只股票")

    # 价格数据
    prices_data = []
    base_prices = {code: np.random.uniform(10, 100) for code in codes}

    for date in dates:
        for code in codes:
            ret = np.random.randn() * 0.02
            base_prices[code] *= (1 + ret)
            prices_data.append({
                'date': date,
                'code': code,
                'close': base_prices[code]
            })

    prices_df = pd.DataFrame(prices_data)

    # 因子数据 - 包含一些有预测能力的因子
    factors_data = []
    factor_base = {code: np.random.randn() * 0.1 for code in codes}

    for date in dates:
        for code in codes:
            # 因子包含一些信号 + 噪声
            # 这些因子与未来收益有一定的相关性
            momentum = factor_base[code] + np.random.randn() * 0.05
            value = np.random.randn() * 0.08
            quality = np.random.randn() * 0.06
            size = np.random.randn() * 0.04
            volatility = np.random.randn() * 0.03

            factors_data.append({
                'date': date,
                'code': code,
                'momentum': momentum,
                'value': value,
                'quality': quality,
                'size': size,
                'volatility': volatility,
                'turnover': np.random.randn() * 0.05,
                'rsi_5d': np.random.randn() * 0.07,
                'rsi_20d': np.random.randn() * 0.06,
                'boll_pos': np.random.randn() * 0.04,
                'volume_ratio': np.random.randn() * 0.03
            })

            # 更新基础因子值
            factor_base[code] = factor_base[code] * 0.95 + np.random.randn() * 0.1

    factors_df = pd.DataFrame(factors_data)

    return prices_df, factors_df


def run_simple_training():
    """运行简单训练（单次训练/验证/测试划分）"""
    logger.info("="*60)
    logger.info("简单训练模式")
    logger.info("="*60)

    # 1. 生成数据
    prices_df, factors_df = generate_sample_data(n_stocks=100, n_days=300)

    # 2. 特征工程
    feature_cols = ['momentum', 'value', 'quality', 'size', 'volatility',
                   'turnover', 'rsi_5d', 'rsi_20d', 'boll_pos', 'volume_ratio']

    builder = FeatureBuilder()
    training_data, features, label = builder.prepare_training_data(
        factors_df, prices_df, feature_cols, label_period=5
    )

    # 3. 划分数据
    dates = sorted(training_data['date'].unique())
    train_end = int(len(dates) * 0.7)
    val_end = int(len(dates) * 0.85)

    train_dates = set(dates[:train_end])
    val_dates = set(dates[train_end:val_end])
    test_dates = set(dates[val_end:])

    train_df = training_data[training_data['date'].isin(train_dates)]
    val_df = training_data[training_data['date'].isin(val_dates)]
    test_df = training_data[training_data['date'].isin(test_dates)]

    logger.info(f"训练集: {len(train_df)} 样本")
    logger.info(f"验证集: {len(val_df)} 样本")
    logger.info(f"测试集: {len(test_df)} 样本")

    # 4. 准备数据
    from sklearn.impute import SimpleImputer

    X_train = train_df[features].values
    y_train = train_df[label].values
    X_val = val_df[features].values
    y_val = val_df[label].values
    X_test = test_df[features].values
    y_test = test_df[label].values

    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)

    # 5. 训练模型
    logger.info("\n训练 Ridge 模型...")
    trainer = ModelTrainer(model_type='ridge')
    trainer.train(X_train, y_train, X_val, y_val, features)

    # 6. 评估
    metrics = trainer.evaluate(X_test, y_test)

    # 7. 特征重要性
    if trainer.feature_importance is not None:
        print("\n特征重要性:")
        print(trainer.feature_importance)

    return metrics


def run_rolling_training():
    """运行滚动训练"""
    logger.info("\n" + "="*60)
    logger.info("滚动训练模式")
    logger.info("="*60)

    # 1. 生成数据
    prices_df, factors_df = generate_sample_data(n_stocks=50, n_days=500)

    # 2. 特征工程
    feature_cols = ['momentum', 'value', 'quality', 'size', 'volatility',
                   'turnover', 'rsi_5d', 'rsi_20d', 'boll_pos', 'volume_ratio']

    builder = FeatureBuilder()
    training_data, features, label = builder.prepare_training_data(
        factors_df, prices_df, feature_cols, label_period=5
    )

    # 3. 滚动训练
    rolling_trainer = RollingTrainer(
        model_type='ridge',
        train_window=100,  # 减小窗口以适应测试数据
        val_window=20,
        test_window=10,
        step_size=20
    )

    predictions_df = rolling_trainer.train_rolling(
        training_data, features, label
    )

    # 4. 性能汇总
    summary = rolling_trainer.get_performance_summary()
    print("\n各Fold性能汇总:")
    print(summary)

    # 5. 集成预测
    ensemble = rolling_trainer.get_ensemble_predictions()
    if not ensemble.empty:
        # 计算整体IC
        overall_ic = ensemble['pred'].corr(ensemble['actual'])
        print(f"\n整体预测IC: {overall_ic:.4f}")

    return summary


def main():
    """主函数"""
    logger.info("="*60)
    logger.info("机器学习因子合成")
    logger.info("="*60)

    # 运行简单训练
    metrics = run_simple_training()

    # 运行滚动训练
    summary = run_rolling_training()

    logger.info("\n" + "="*60)
    logger.info("ML因子合成完成!")
    logger.info("="*60)

    return metrics, summary


if __name__ == '__main__':
    metrics, summary = main()
