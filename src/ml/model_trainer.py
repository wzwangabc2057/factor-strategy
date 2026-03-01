"""
机器学习模型训练模块
支持 LightGBM, XGBoost, CatBoost
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
import json
import os

# 模型导入（可选，如果未安装则跳过）
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """机器学习模型训练器"""

    def __init__(self, model_type: str = 'lightgbm', model_params: Dict = None):
        """
        初始化模型训练器

        Args:
            model_type: 模型类型 ('lightgbm', 'xgboost', 'sklearn_gb', 'ridge')
            model_params: 模型参数
        """
        self.model_type = model_type
        self.model_params = model_params or self._get_default_params(model_type)
        self.model = None
        self.feature_importance = None
        self.training_history = []

        logger.info(f"模型训练器初始化: type={model_type}")

    def _get_default_params(self, model_type: str) -> Dict:
        """获取默认模型参数"""
        if model_type == 'lightgbm':
            return {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'n_estimators': 500,
                'early_stopping_rounds': 50
            }
        elif model_type == 'xgboost':
            return {
                'objective': 'reg:squarederror',
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'n_estimators': 500,
                'early_stopping_rounds': 50
            }
        elif model_type == 'sklearn_gb':
            return {
                'n_estimators': 200,
                'max_depth': 5,
                'learning_rate': 0.05,
                'subsample': 0.8
            }
        elif model_type == 'ridge':
            return {
                'alpha': 1.0
            }
        else:
            return {}

    def _create_model(self):
        """创建模型实例"""
        if self.model_type == 'lightgbm':
            if not HAS_LIGHTGBM:
                raise ImportError("LightGBM未安装，请运行: pip install lightgbm")
            params = {k: v for k, v in self.model_params.items()
                     if k not in ['n_estimators', 'early_stopping_rounds']}
            return lgb.LGBMRegressor(**params)

        elif self.model_type == 'xgboost':
            if not HAS_XGBOOST:
                raise ImportError("XGBoost未安装，请运行: pip install xgboost")
            params = {k: v for k, v in self.model_params.items()
                     if k not in ['early_stopping_rounds']}
            return xgb.XGBRegressor(**params)

        elif self.model_type == 'sklearn_gb':
            if not HAS_SKLEARN:
                raise ImportError("sklearn未安装")
            return GradientBoostingRegressor(**self.model_params)

        elif self.model_type == 'ridge':
            if not HAS_SKLEARN:
                raise ImportError("sklearn未安装")
            return Ridge(**self.model_params)

        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: np.ndarray = None,
              y_val: np.ndarray = None,
              feature_names: List[str] = None) -> Any:
        """
        训练模型

        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            feature_names: 特征名称

        Returns:
            训练好的模型
        """
        logger.info(f"开始训练 {self.model_type} 模型...")
        logger.info(f"  训练集: {len(X_train)} 样本")
        if X_val is not None:
            logger.info(f"  验证集: {len(X_val)} 样本")

        self.model = self._create_model()
        start_time = datetime.now()

        if self.model_type == 'lightgbm':
            callbacks = [lgb.log_evaluation(period=100)]
            if X_val is not None:
                callbacks.append(lgb.early_stopping(self.model_params.get('early_stopping_rounds', 50)))

            self.model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val) if X_val is not None else None,
                feature_name=feature_names,
                callbacks=callbacks
            )

        elif self.model_type == 'xgboost':
            eval_set = [(X_val, y_val)] if X_val is not None else None
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=100
            )

        else:
            self.model.fit(X_train, y_train)

        training_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"训练完成，耗时: {training_time:.2f}秒")

        # 保存特征重要性
        self._save_feature_importance(feature_names)

        # 记录训练历史
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'model_type': self.model_type,
            'n_samples': len(X_train),
            'training_time': training_time
        })

        return self.model

    def _save_feature_importance(self, feature_names: List[str]):
        """保存特征重要性"""
        if feature_names is None:
            return

        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'booster') and self.model_type == 'xgboost':
            importance = self.model.get_booster().get_score(importance_type='gain')
            importance = [importance.get(f'f{i}', 0) for i in range(len(feature_names))]
        else:
            return

        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        logger.info("Top 10 特征重要性:")
        for _, row in self.feature_importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测

        Args:
            X: 特征数组

        Returns:
            预测值
        """
        if self.model is None:
            raise ValueError("模型未训练")

        return self.model.predict(X)

    def evaluate(self,
                X_test: np.ndarray,
                y_test: np.ndarray) -> Dict[str, float]:
        """
        评估模型

        Args:
            X_test: 测试特征
            y_test: 测试标签

        Returns:
            评估指标字典
        """
        y_pred = self.predict(X_test)

        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'ic': np.corrcoef(y_pred, y_test)[0, 1]  # 预测值与真实值的相关性
        }

        logger.info("模型评估结果:")
        for name, value in metrics.items():
            logger.info(f"  {name}: {value:.4f}")

        return metrics

    def save_model(self, path: str):
        """保存模型"""
        import joblib
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'model_type': self.model_type,
            'model_params': self.model_params,
            'feature_importance': self.feature_importance,
            'training_history': self.training_history
        }, path)
        logger.info(f"模型已保存: {path}")

    def load_model(self, path: str):
        """加载模型"""
        import joblib
        data = joblib.load(path)
        self.model = data['model']
        self.model_type = data['model_type']
        self.model_params = data['model_params']
        self.feature_importance = data.get('feature_importance')
        self.training_history = data.get('training_history', [])
        logger.info(f"模型已加载: {path}")


class RollingTrainer:
    """滚动训练框架"""

    def __init__(self,
                 model_type: str = 'lightgbm',
                 model_params: Dict = None,
                 train_window: int = 252 * 3,  # 3年
                 val_window: int = 63,  # 1季度
                 test_window: int = 21,  # 1月
                 step_size: int = 21):  # 滚动步长
        """
        初始化滚动训练器

        Args:
            model_type: 模型类型
            model_params: 模型参数
            train_window: 训练窗口（天数）
            val_window: 验证窗口
            test_window: 测试窗口
            step_size: 滚动步长
        """
        self.model_type = model_type
        self.model_params = model_params
        self.train_window = train_window
        self.val_window = val_window
        self.test_window = test_window
        self.step_size = step_size

        self.models: List[Dict] = []
        self.predictions: List[Dict] = []

        logger.info("滚动训练器初始化:")
        logger.info(f"  训练窗口: {train_window}天")
        logger.info(f"  验证窗口: {val_window}天")
        logger.info(f"  测试窗口: {test_window}天")
        logger.info(f"  滚动步长: {step_size}天")

    def get_time_splits(self,
                       dates: pd.Series,
                       min_train_samples: int = 1000) -> List[Tuple]:
        """
        生成时间序列分割

        Args:
            dates: 日期序列
            min_train_samples: 最小训练样本数

        Returns:
            [(train_idx, val_idx, test_idx), ...]
        """
        unique_dates = sorted(dates.unique())
        n_dates = len(unique_dates)

        splits = []

        # 计算起始位置
        start_idx = 0
        while start_idx + self.train_window + self.val_window + self.test_window <= n_dates:
            train_end = start_idx + self.train_window
            val_end = train_end + self.val_window
            test_end = val_end + self.test_window

            train_dates = set(unique_dates[start_idx:train_end])
            val_dates = set(unique_dates[train_end:val_end])
            test_dates = set(unique_dates[val_end:test_end])

            splits.append({
                'train_dates': train_dates,
                'val_dates': val_dates,
                'test_dates': test_dates,
                'train_start': unique_dates[start_idx],
                'train_end': unique_dates[train_end - 1],
                'test_start': unique_dates[val_end],
                'test_end': unique_dates[test_end - 1] if test_end <= n_dates else unique_dates[-1]
            })

            start_idx += self.step_size

        logger.info(f"生成 {len(splits)} 个时间分割")
        return splits

    def train_rolling(self,
                     df: pd.DataFrame,
                     feature_cols: List[str],
                     label_col: str,
                     date_col: str = 'date') -> pd.DataFrame:
        """
        执行滚动训练

        Args:
            df: 完整数据集
            feature_cols: 特征列
            label_col: 标签列
            date_col: 日期列

        Returns:
            预测结果DataFrame
        """
        logger.info("="*60)
        logger.info("开始滚动训练")
        logger.info("="*60)

        splits = self.get_time_splits(df[date_col])

        all_predictions = []

        for i, split in enumerate(splits):
            logger.info(f"\n--- Fold {i+1}/{len(splits)} ---")
            logger.info(f"训练: {split['train_start']} ~ {split['train_end']}")
            logger.info(f"测试: {split['test_start']} ~ {split['test_end']}")

            # 划分数据
            train_mask = df[date_col].isin(split['train_dates'])
            val_mask = df[date_col].isin(split['val_dates'])
            test_mask = df[date_col].isin(split['test_dates'])

            train_df = df[train_mask]
            val_df = df[val_mask]
            test_df = df[test_mask]

            if len(train_df) < 100:
                logger.warning("训练样本不足，跳过")
                continue

            # 准备数据
            X_train = train_df[feature_cols].values
            y_train = train_df[label_col].values
            X_val = val_df[feature_cols].values if len(val_df) > 0 else None
            y_val = val_df[label_col].values if len(val_df) > 0 else None
            X_test = test_df[feature_cols].values
            y_test = test_df[label_col].values

            # 处理缺失值
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X_train = imputer.fit_transform(X_train)
            if X_val is not None:
                X_val = imputer.transform(X_val)
            X_test = imputer.transform(X_test)

            # 训练模型
            trainer = ModelTrainer(self.model_type, self.model_params)
            trainer.train(X_train, y_train, X_val, y_val, feature_cols)

            # 预测
            y_pred = trainer.predict(X_test)

            # 评估
            metrics = trainer.evaluate(X_test, y_test)

            # 保存结果
            self.models.append({
                'fold': i,
                'trainer': trainer,
                'split': split,
                'metrics': metrics
            })

            # 收集预测
            pred_df = test_df[[date_col, 'code']].copy()
            pred_df['pred'] = y_pred
            pred_df['actual'] = y_test
            pred_df['fold'] = i
            all_predictions.append(pred_df)

        # 合并所有预测
        if all_predictions:
            self.predictions_df = pd.concat(all_predictions, ignore_index=True)
            logger.info(f"\n滚动训练完成，共 {len(self.predictions_df)} 条预测")
        else:
            self.predictions_df = pd.DataFrame()
            logger.warning("没有生成任何预测")

        return self.predictions_df

    def get_ensemble_predictions(self) -> pd.DataFrame:
        """获取集成预测（平均所有fold的预测）"""
        if self.predictions_df.empty:
            return pd.DataFrame()

        # 按股票和日期聚合
        ensemble = self.predictions_df.groupby(['date', 'code']).agg({
            'pred': 'mean',
            'actual': 'first'
        }).reset_index()

        return ensemble

    def get_performance_summary(self) -> pd.DataFrame:
        """获取各fold的性能汇总"""
        if not self.models:
            return pd.DataFrame()

        summary = []
        for m in self.models:
            summary.append({
                'fold': m['fold'],
                'test_start': m['split']['test_start'],
                'test_end': m['split']['test_end'],
                **m['metrics']
            })

        return pd.DataFrame(summary)


if __name__ == '__main__':
    # 测试代码
    np.random.seed(42)

    # 生成模拟数据
    n_samples = 5000
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] * 0.5 + X[:, 1] * 0.3 + np.random.randn(n_samples) * 0.1
    feature_names = [f'factor_{i}' for i in range(n_features)]

    # 划分训练/验证/测试
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]

    # 训练模型
    trainer = ModelTrainer(model_type='ridge')  # 使用ridge，不需要额外安装
    trainer.train(X_train, y_train, X_val, y_val, feature_names)

    # 评估
    metrics = trainer.evaluate(X_test, y_test)
    print("\n评估指标:", metrics)

    # 特征重要性
    if trainer.feature_importance is not None:
        print("\n特征重要性:")
        print(trainer.feature_importance)
