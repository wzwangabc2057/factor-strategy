"""
特征工程模块
为机器学习模型准备因子特征
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureBuilder:
    """特征工程构建器"""

    def __init__(self,
                 scaler_type: str = 'robust',
                 handle_missing: str = 'median'):
        """
        初始化特征工程器

        Args:
            scaler_type: 标准化方法 ('standard' 或 'robust')
            handle_missing: 缺失值处理方法 ('median', 'mean', 'zero')
        """
        self.scaler_type = scaler_type
        self.handle_missing = handle_missing
        self.scalers: Dict[str, object] = {}
        self.feature_stats: Dict[str, Dict] = {}

        logger.info(f"特征工程器初始化: scaler={scaler_type}, missing={handle_missing}")

    def handle_missing_values(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        处理缺失值

        Args:
            df: 数据DataFrame
            columns: 需要处理的列

        Returns:
            处理后的DataFrame
        """
        df = df.copy()

        for col in columns:
            if col not in df.columns:
                continue

            missing_count = df[col].isna().sum()
            if missing_count == 0:
                continue

            if self.handle_missing == 'median':
                fill_value = df[col].median()
            elif self.handle_missing == 'mean':
                fill_value = df[col].mean()
            else:
                fill_value = 0

            # 存储填充值
            self.feature_stats[col] = {'fill_value': fill_value}
            df[col] = df[col].fillna(fill_value)

            if missing_count > 0:
                logger.debug(f"  {col}: 填充 {missing_count} 个缺失值为 {fill_value:.4f}")

        return df

    def remove_outliers(self,
                       df: pd.DataFrame,
                       columns: List[str],
                       method: str = 'winsorize',
                       lower: float = 0.01,
                       upper: float = 0.99) -> pd.DataFrame:
        """
        处理异常值

        Args:
            df: 数据DataFrame
            columns: 需要处理的列
            method: 处理方法 ('winsorize' 或 'clip')
            lower: 下界分位数
            upper: 上界分位数

        Returns:
            处理后的DataFrame
        """
        df = df.copy()

        for col in columns:
            if col not in df.columns:
                continue

            if method == 'winsorize':
                lower_val = df[col].quantile(lower)
                upper_val = df[col].quantile(upper)
                df[col] = df[col].clip(lower_val, upper_val)

                # 存储边界值
                self.feature_stats[col] = {
                    **self.feature_stats.get(col, {}),
                    'lower_bound': lower_val,
                    'upper_bound': upper_val
                }

        return df

    def normalize_features(self,
                          df: pd.DataFrame,
                          columns: List[str],
                          fit: bool = True,
                          group_col: str = 'date') -> pd.DataFrame:
        """
        标准化特征（可选按日期分组标准化）

        Args:
            df: 数据DataFrame
            columns: 需要标准化的列
            fit: 是否拟合scaler
            group_col: 分组列（如按日期标准化）

        Returns:
            标准化后的DataFrame
        """
        df = df.copy()

        if group_col and group_col in df.columns:
            # 按日期分组标准化（行业中性化风格）
            for col in columns:
                if col not in df.columns:
                    continue

                def group_normalize(group):
                    if fit:
                        if self.scaler_type == 'robust':
                            median = group.median()
                            iqr = group.quantile(0.75) - group.quantile(0.25)
                            return (group - median) / (iqr + 1e-12)
                        else:
                            mean = group.mean()
                            std = group.std()
                            return (group - mean) / (std + 1e-12)
                    else:
                        # 使用训练时计算的统计量
                        stats = self.feature_stats.get(col, {})
                        if self.scaler_type == 'robust':
                            return (group - stats.get('median', 0)) / (stats.get('iqr', 1) + 1e-12)
                        else:
                            return (group - stats.get('mean', 0)) / (stats.get('std', 1) + 1e-12)

                # 计算并存储统计量
                if fit:
                    self.feature_stats[col] = {
                        'median': df[col].median(),
                        'mean': df[col].mean(),
                        'std': df[col].std(),
                        'iqr': df[col].quantile(0.75) - df[col].quantile(0.25)
                    }

                df[col] = df.groupby(group_col)[col].transform(group_normalize)
        else:
            # 全局标准化
            for col in columns:
                if col not in df.columns:
                    continue

                if fit:
                    if self.scaler_type == 'robust':
                        scaler = RobustScaler()
                    else:
                        scaler = StandardScaler()

                    df[col] = scaler.fit_transform(df[[col]])
                    self.scalers[col] = scaler
                else:
                    scaler = self.scalers.get(col)
                    if scaler:
                        df[col] = scaler.transform(df[[col]])

        return df

    def industry_neutralize(self,
                           df: pd.DataFrame,
                           feature_cols: List[str],
                           industry_col: str = 'industry') -> pd.DataFrame:
        """
        行业中性化

        Args:
            df: 数据DataFrame
            feature_cols: 需要中性化的特征列
            industry_col: 行业列

        Returns:
            中性化后的DataFrame
        """
        df = df.copy()

        if industry_col not in df.columns:
            logger.warning(f"找不到行业列: {industry_col}")
            return df

        for col in feature_cols:
            if col not in df.columns:
                continue

            # 减去行业均值
            industry_mean = df.groupby(industry_col)[col].transform('mean')
            df[col] = df[col] - industry_mean

        return df

    def build_features(self,
                      factor_df: pd.DataFrame,
                      feature_cols: List[str],
                      normalize: bool = True,
                      neutralize_industry: bool = False,
                      industry_col: str = 'industry',
                      date_col: str = 'date') -> pd.DataFrame:
        """
        构建特征

        Args:
            factor_df: 因子DataFrame
            feature_cols: 特征列名
            normalize: 是否标准化
            neutralize_industry: 是否行业中性化
            industry_col: 行业列名
            date_col: 日期列名

        Returns:
            处理后的特征DataFrame
        """
        logger.info(f"开始构建特征，共 {len(feature_cols)} 个特征")

        df = factor_df.copy()

        # 1. 处理缺失值
        df = self.handle_missing_values(df, feature_cols)

        # 2. 处理异常值
        df = self.remove_outliers(df, feature_cols)

        # 3. 行业中性化（可选）
        if neutralize_industry and industry_col in df.columns:
            df = self.industry_neutralize(df, feature_cols, industry_col)

        # 4. 标准化
        if normalize:
            df = self.normalize_features(df, feature_cols, group_col=date_col)

        logger.info(f"特征构建完成")

        return df

    def build_labels(self,
                    prices_df: pd.DataFrame,
                    forward_periods: List[int] = [5, 10, 20],
                    method: str = 'return') -> pd.DataFrame:
        """
        构建标签（目标变量）

        Args:
            prices_df: 价格数据 [date, code, close]
            forward_periods: 预测周期
            method: 标签方法 ('return', 'rank', 'binary')

        Returns:
            标签DataFrame
        """
        logger.info(f"构建标签，周期: {forward_periods}")

        # 转为宽表
        prices_wide = prices_df.pivot(index='date', columns='code', values='close')
        prices_wide = prices_wide.sort_index()

        labels = []

        for period in forward_periods:
            # 计算未来收益率
            forward_ret = prices_wide.pct_change(periods=period).shift(-period)

            if method == 'rank':
                # 按日期排名
                forward_ret = forward_ret.rank(axis=1, pct=True)
            elif method == 'binary':
                # 二分类：收益为正=1，负=0
                forward_ret = (forward_ret > 0).astype(int)

            # 转为长表
            forward_ret = forward_ret.stack()
            forward_ret.name = f'label_{period}d'
            labels.append(forward_ret)

        # 合并
        result = pd.concat(labels, axis=1).reset_index()
        result.columns = ['date', 'code'] + [f'label_{p}d' for p in forward_periods]

        logger.info(f"标签构建完成，共 {len(result)} 条")

        return result

    def prepare_training_data(self,
                             factor_df: pd.DataFrame,
                             prices_df: pd.DataFrame,
                             feature_cols: List[str],
                             label_period: int = 5,
                             start_date: str = None,
                             end_date: str = None) -> Tuple[pd.DataFrame, List[str], str]:
        """
        准备训练数据

        Args:
            factor_df: 因子数据
            prices_df: 价格数据
            feature_cols: 特征列
            label_period: 标签周期
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            (合并后的数据, 特征列名, 标签列名)
        """
        logger.info("准备训练数据...")

        # 1. 构建特征
        features = self.build_features(factor_df, feature_cols)

        # 2. 构建标签
        labels = self.build_labels(prices_df, forward_periods=[label_period])

        # 3. 合并
        label_col = f'label_{label_period}d'
        merged = features.merge(labels[['date', 'code', label_col]], on=['date', 'code'], how='inner')

        # 4. 过滤日期
        if start_date:
            merged = merged[merged['date'] >= start_date]
        if end_date:
            merged = merged[merged['date'] <= end_date]

        # 5. 移除缺失标签
        merged = merged.dropna(subset=[label_col])

        logger.info(f"训练数据准备完成: {len(merged)} 条, {len(feature_cols)} 个特征")

        return merged, feature_cols, label_col

    def get_feature_importance_ready(self,
                                    df: pd.DataFrame,
                                    feature_cols: List[str],
                                    label_col: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取准备好的特征和标签数组

        Args:
            df: 数据DataFrame
            feature_cols: 特征列
            label_col: 标签列

        Returns:
            (X, y)
        """
        X = df[feature_cols].values
        y = df[label_col].values

        # 移除包含NaN的行
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_mask]
        y = y[valid_mask]

        return X, y


if __name__ == '__main__':
    # 测试代码
    np.random.seed(42)

    # 生成模拟数据
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    codes = ['000001', '000002', '600000']

    factor_data = []
    price_data = []

    base_prices = {code: np.random.uniform(10, 50) for code in codes}

    for code in codes:
        for date in dates:
            # 因子数据
            factor_data.append({
                'date': date,
                'code': code,
                'momentum': np.random.randn(),
                'roe': np.random.uniform(5, 25),
                'volume_ratio': np.random.uniform(0.5, 2)
            })

            # 价格数据
            ret = np.random.randn() * 0.02
            base_prices[code] *= (1 + ret)
            price_data.append({
                'date': date,
                'code': code,
                'close': base_prices[code]
            })

    factor_df = pd.DataFrame(factor_data)
    prices_df = pd.DataFrame(price_data)

    # 构建特征
    builder = FeatureBuilder()
    feature_cols = ['momentum', 'roe', 'volume_ratio']

    training_data, features, label = builder.prepare_training_data(
        factor_df, prices_df, feature_cols, label_period=5
    )

    print(f"\n训练数据形状: {training_data.shape}")
    print(f"特征列: {features}")
    print(f"标签列: {label}")
    print(f"\n数据样例:")
    print(training_data.head())
