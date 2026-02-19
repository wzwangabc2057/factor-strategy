"""
因子IC分析模块
基于 Alphalens 风格的因子有效性验证

核心功能：
1. 计算 IC (Information Coefficient)
2. 计算 ICIR (IC稳定性)
3. 因子分层回测 (Quantile Returns)
4. 生成因子有效性报告
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
import logging
from dataclasses import dataclass
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ICMetrics:
    """IC指标数据类"""
    ic_mean: float  # IC均值
    ic_std: float   # IC标准差
    icir: float     # IC信息比率 (IC均值/IC标准差)
    ic_positive_ratio: float  # IC为正的比例
    t_stat: float   # t统计量
    p_value: float  # p值
    sample_count: int  # 样本数


class ICAnalyzer:
    """因子IC分析器"""

    def __init__(self,
                 forward_periods: List[int] = None,
                 quantiles: int = 5,
                 min_samples: int = 30):
        """
        初始化IC分析器

        Args:
            forward_periods: 未来收益计算周期列表，如 [5, 10, 20]
            quantiles: 分层数量
            min_samples: 最小样本数
        """
        self.forward_periods = forward_periods or [5, 10, 20]
        self.quantiles = quantiles
        self.min_samples = min_samples

        logger.info(f"IC分析器初始化:")
        logger.info(f"  预测周期: {self.forward_periods}")
        logger.info(f"  分层数: {self.quantiles}")
        logger.info(f"  最小样本: {self.min_samples}")

    def calculate_forward_returns(self,
                                  prices_df: pd.DataFrame,
                                  periods: List[int] = None) -> pd.DataFrame:
        """
        计算未来收益率

        Args:
            prices_df: DataFrame with columns [date, code, close]
            periods: 计算周期列表

        Returns:
            DataFrame with columns [date, code, close, ret_5d, ret_10d, ret_20d, ...]
        """
        if periods is None:
            periods = self.forward_periods

        logger.info(f"计算未来收益率，周期: {periods}")

        # 转为宽表
        prices_wide = prices_df.pivot(index='date', columns='code', values='close')
        prices_wide = prices_wide.sort_index()

        # 计算各周期收益率
        result_frames = []
        for period in periods:
            # 未来收益率 = (未来价格 - 当前价格) / 当前价格
            forward_ret = prices_wide.pct_change(periods=period).shift(-period)
            forward_ret = forward_ret.stack()
            forward_ret.name = f'ret_{period}d'
            result_frames.append(forward_ret)

        # 合并
        result = pd.concat(result_frames, axis=1).reset_index()
        result.columns = ['date', 'code'] + [f'ret_{p}d' for p in periods]

        # 添加原始价格
        result = result.merge(prices_df[['date', 'code', 'close']], on=['date', 'code'], how='left')

        logger.info(f"  生成 {len(result)} 条收益率数据")
        return result

    def calculate_ic(self,
                    factor_values: pd.DataFrame,
                    forward_returns: pd.DataFrame,
                    factor_col: str,
                    return_col: str = 'ret_5d',
                    method: str = 'spearman') -> Tuple[np.ndarray, ICMetrics]:
        """
        计算因子IC

        IC = corr(因子值, 未来收益)

        Args:
            factor_values: DataFrame with columns [date, code, factor_value]
            forward_returns: DataFrame with columns [date, code, return_col]
            factor_col: 因子值列名
            return_col: 收益率列名
            method: 相关系数计算方法 ('pearson' 或 'spearman')

        Returns:
            (IC时间序列, IC指标)
        """
        # 合并数据
        merged = factor_values.merge(
            forward_returns[['date', 'code', return_col]],
            on=['date', 'code'],
            how='inner'
        )

        # 移除缺失值
        merged = merged.dropna(subset=[factor_col, return_col])

        if len(merged) < self.min_samples:
            logger.warning(f"样本数不足: {len(merged)} < {self.min_samples}")
            return np.array([]), None

        # 按日期计算IC
        ic_series = []
        for date, group in merged.groupby('date'):
            if len(group) < self.min_samples:
                continue

            if method == 'spearman':
                ic, _ = stats.spearmanr(group[factor_col], group[return_col])
            else:
                ic, _ = stats.pearsonr(group[factor_col], group[return_col])

            if not np.isnan(ic):
                ic_series.append({'date': date, 'ic': ic})

        if not ic_series:
            return np.array([]), None

        ic_df = pd.DataFrame(ic_series)
        ic_values = ic_df['ic'].values

        # 计算IC指标
        ic_mean = np.mean(ic_values)
        ic_std = np.std(ic_values)
        icir = ic_mean / ic_std if ic_std > 0 else 0
        ic_positive_ratio = np.sum(ic_values > 0) / len(ic_values)

        # t检验
        t_stat, p_value = stats.ttest_1samp(ic_values, 0)

        metrics = ICMetrics(
            ic_mean=ic_mean,
            ic_std=ic_std,
            icir=icir,
            ic_positive_ratio=ic_positive_ratio,
            t_stat=t_stat,
            p_value=p_value,
            sample_count=len(ic_values)
        )

        return ic_values, metrics

    def calculate_quantile_returns(self,
                                   factor_values: pd.DataFrame,
                                   forward_returns: pd.DataFrame,
                                   factor_col: str,
                                   return_col: str = 'ret_5d') -> pd.DataFrame:
        """
        计算分层收益

        将股票按因子值分成N层，计算每层的平均收益

        Args:
            factor_values: DataFrame with columns [date, code, factor_col]
            forward_returns: DataFrame with columns [date, code, return_col]
            factor_col: 因子值列名
            return_col: 收益率列名

        Returns:
            DataFrame with columns [date, quantile_1, quantile_2, ..., quantile_N]
        """
        # 合并数据
        merged = factor_values.merge(
            forward_returns[['date', 'code', return_col]],
            on=['date', 'code'],
            how='inner'
        )
        merged = merged.dropna(subset=[factor_col, return_col])

        # 按日期分层
        results = []
        for date, group in merged.groupby('date'):
            if len(group) < self.quantiles * 2:
                continue

            # 计算分层
            group['quantile'] = pd.qcut(
                group[factor_col],
                q=self.quantiles,
                labels=False,
                duplicates='drop'
            )

            # 计算每层平均收益
            quantile_returns = group.groupby('quantile')[return_col].mean()

            result = {'date': date}
            for q in range(self.quantiles):
                result[f'quantile_{q+1}'] = quantile_returns.get(q, np.nan)

            results.append(result)

        return pd.DataFrame(results)

    def analyze_single_factor(self,
                             factor_values: pd.DataFrame,
                             forward_returns: pd.DataFrame,
                             factor_name: str,
                             factor_col: str) -> Dict:
        """
        分析单个因子

        Args:
            factor_values: DataFrame with columns [date, code, factor_col]
            forward_returns: DataFrame with columns [date, code, ret_5d, ret_10d, ...]
            factor_name: 因子名称
            factor_col: 因子值列名

        Returns:
            分析结果字典
        """
        logger.info(f"分析因子: {factor_name}")

        result = {
            'factor_name': factor_name,
            'periods': {}
        }

        # 对每个预测周期计算IC
        for period in self.forward_periods:
            return_col = f'ret_{period}d'

            if return_col not in forward_returns.columns:
                continue

            # 计算IC
            ic_values, metrics = self.calculate_ic(
                factor_values, forward_returns, factor_col, return_col
            )

            if metrics is None:
                continue

            # 计算分层收益
            quantile_returns = self.calculate_quantile_returns(
                factor_values, forward_returns, factor_col, return_col
            )

            # 计算多空收益 (最高层 - 最低层)
            if len(quantile_returns) > 0:
                long_col = f'quantile_{self.quantiles}'
                short_col = 'quantile_1'
                if long_col in quantile_returns.columns and short_col in quantile_returns.columns:
                    quantile_returns['long_short'] = quantile_returns[long_col] - quantile_returns[short_col]
                    long_short_mean = quantile_returns['long_short'].mean()
                else:
                    long_short_mean = np.nan
            else:
                long_short_mean = np.nan

            result['periods'][period] = {
                'ic_mean': metrics.ic_mean,
                'ic_std': metrics.ic_std,
                'icir': metrics.icir,
                'ic_positive_ratio': metrics.ic_positive_ratio,
                't_stat': metrics.t_stat,
                'p_value': metrics.p_value,
                'sample_count': metrics.sample_count,
                'long_short_return': long_short_mean,
                'ic_series': ic_values.tolist(),
                'quantile_returns': quantile_returns.to_dict('records') if len(quantile_returns) > 0 else []
            }

            logger.info(f"  {period}日: IC={metrics.ic_mean:.4f}, ICIR={metrics.icir:.4f}, "
                       f"胜率={metrics.ic_positive_ratio:.2%}")

        return result

    def analyze_multiple_factors(self,
                                factors_dict: Dict[str, pd.DataFrame],
                                forward_returns: pd.DataFrame,
                                factor_col_map: Dict[str, str] = None) -> pd.DataFrame:
        """
        分析多个因子

        Args:
            factors_dict: 字典 {因子名: DataFrame with [date, code, 因子值列]}
            forward_returns: DataFrame with [date, code, ret_5d, ret_10d, ...]
            factor_col_map: 字典 {因子名: 因子值列名}，默认使用因子名作为列名

        Returns:
            因子分析汇总表
        """
        logger.info("="*60)
        logger.info(f"开始分析 {len(factors_dict)} 个因子")
        logger.info("="*60)

        all_results = []

        for factor_name, factor_df in factors_dict.items():
            factor_col = factor_col_map.get(factor_name, factor_name) if factor_col_map else factor_name

            if factor_col not in factor_df.columns:
                logger.warning(f"因子 {factor_name} 缺少列 {factor_col}")
                continue

            result = self.analyze_single_factor(
                factor_df, forward_returns, factor_name, factor_col
            )

            # 提取主要指标
            for period, metrics in result['periods'].items():
                all_results.append({
                    'factor': factor_name,
                    'period': period,
                    'ic_mean': metrics['ic_mean'],
                    'ic_std': metrics['ic_std'],
                    'icir': metrics['icir'],
                    'ic_positive_ratio': metrics['ic_positive_ratio'],
                    'p_value': metrics['p_value'],
                    'long_short_return': metrics['long_short_return']
                })

        summary_df = pd.DataFrame(all_results)

        logger.info("="*60)
        logger.info("因子分析完成")
        logger.info("="*60)

        return summary_df

    def generate_factor_report(self,
                               summary_df: pd.DataFrame,
                               output_path: str = None) -> str:
        """
        生成因子分析报告

        Args:
            summary_df: analyze_multiple_factors 返回的汇总表
            output_path: 输出路径

        Returns:
            报告内容字符串
        """
        report = []
        report.append("=" * 80)
        report.append("因子有效性分析报告")
        report.append("=" * 80)
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # 按ICIR排序
        for period in sorted(summary_df['period'].unique()):
            report.append(f"\n{period}日预测周期:")
            report.append("-" * 60)

            period_df = summary_df[summary_df['period'] == period].copy()
            period_df = period_df.sort_values('icir', ascending=False)

            report.append(f"{'因子':<20} {'IC均值':>10} {'ICIR':>10} {'IC胜率':>10} {'多空收益':>12}")
            report.append("-" * 60)

            for _, row in period_df.iterrows():
                status = "✓" if row['icir'] > 0.5 and row['p_value'] < 0.05 else "✗"
                report.append(
                    f"{status} {row['factor']:<18} {row['ic_mean']:>10.4f} {row['icir']:>10.4f} "
                    f"{row['ic_positive_ratio']:>10.2%} {row['long_short_return']:>12.4%}"
                )

        # 筛选建议
        report.append("\n" + "=" * 80)
        report.append("因子筛选建议:")
        report.append("=" * 80)

        # 筛选条件：IC > 0.02, ICIR > 0.5, p < 0.05
        good_factors = summary_df[
            (summary_df['ic_mean'] > 0.02) &
            (summary_df['icir'] > 0.5) &
            (summary_df['p_value'] < 0.05)
        ]['factor'].unique()

        report.append(f"\n有效因子 ({len(good_factors)} 个):")
        for f in sorted(good_factors):
            report.append(f"  - {f}")

        # 无效因子
        bad_factors = set(summary_df['factor'].unique()) - set(good_factors)
        report.append(f"\n无效/弱效因子 ({len(bad_factors)} 个):")
        for f in sorted(bad_factors):
            report.append(f"  - {f}")

        report_text = "\n".join(report)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"报告已保存到: {output_path}")

        return report_text

    def get_recommended_factors(self,
                               summary_df: pd.DataFrame,
                               min_ic: float = 0.02,
                               min_icir: float = 0.5,
                               max_pvalue: float = 0.05) -> List[str]:
        """
        获取推荐的因子列表

        Args:
            summary_df: 因子分析汇总表
            min_ic: 最小IC阈值
            min_icir: 最小ICIR阈值
            max_pvalue: 最大p值阈值

        Returns:
            推荐因子列表
        """
        good_factors = summary_df[
            (summary_df['ic_mean'] > min_ic) &
            (summary_df['icir'] > min_icir) &
            (summary_df['p_value'] < max_pvalue)
        ]['factor'].unique().tolist()

        return sorted(good_factors)


def run_ic_analysis(prices_df: pd.DataFrame,
                   factors_df: pd.DataFrame,
                   factor_columns: List[str],
                   periods: List[int] = None) -> Tuple[pd.DataFrame, ICAnalyzer]:
    """
    快速运行IC分析的便捷函数

    Args:
        prices_df: DataFrame with [date, code, close]
        factors_df: DataFrame with [date, code, factor1, factor2, ...]
        factor_columns: 要分析的因子列名列表
        periods: 预测周期

    Returns:
        (分析汇总表, IC分析器实例)
    """
    analyzer = ICAnalyzer(forward_periods=periods)

    # 计算未来收益
    forward_returns = analyzer.calculate_forward_returns(prices_df, periods)

    # 构建因子字典
    factors_dict = {}
    for col in factor_columns:
        if col in factors_df.columns:
            factors_dict[col] = factors_df[['date', 'code', col]]

    # 分析因子
    summary = analyzer.analyze_multiple_factors(factors_dict, forward_returns)

    return summary, analyzer


if __name__ == '__main__':
    # 测试代码
    np.random.seed(42)

    # 生成模拟数据
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    codes = [f'00000{i}' for i in range(1, 11)]

    # 价格数据
    prices_data = []
    for code in codes:
        price = 10 + np.random.randn(len(dates)).cumsum() * 0.5
        for i, date in enumerate(dates):
            prices_data.append({'date': date, 'code': code, 'close': price[i]})

    prices_df = pd.DataFrame(prices_data)

    # 因子数据
    factors_data = []
    for code in codes:
        for date in dates:
            factors_data.append({
                'date': date,
                'code': code,
                'momentum': np.random.randn() * 0.1,
                'roe': np.random.randn() * 0.05,
                'dividend': np.random.randn() * 0.02
            })

    factors_df = pd.DataFrame(factors_data)

    # 运行IC分析
    summary, analyzer = run_ic_analysis(
        prices_df=prices_df,
        factors_df=factors_df,
        factor_columns=['momentum', 'roe', 'dividend'],
        periods=[5, 10, 20]
    )

    print("\n" + analyzer.generate_factor_report(summary))
