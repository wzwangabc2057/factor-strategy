"""
因子分析报告生成模块
生成可视化的因子分析报告
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import logging
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FactorReportGenerator:
    """因子分析报告生成器"""

    def __init__(self, output_dir: str = './reports'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_ic_time_series(self,
                           ic_series: List[float],
                           factor_name: str,
                           period: int,
                           output_path: str = None) -> str:
        """
        绘制IC时间序列图

        Args:
            ic_series: IC时间序列
            factor_name: 因子名称
            period: 预测周期
            output_path: 输出路径

        Returns:
            图片路径
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, f'ic_series_{factor_name}_{period}d.png')

        fig, ax = plt.subplots(figsize=(12, 4))

        # 绘制IC序列
        ax.bar(range(len(ic_series)), ic_series,
               color=['green' if x > 0 else 'red' for x in ic_series],
               alpha=0.7)

        # 添加均值线
        mean_ic = np.mean(ic_series)
        ax.axhline(y=mean_ic, color='blue', linestyle='--', label=f'Mean IC: {mean_ic:.4f}')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        ax.set_title(f'{factor_name} - IC Time Series ({period}d Forward Return)')
        ax.set_xlabel('Time')
        ax.set_ylabel('IC')
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        return output_path

    def plot_quantile_returns(self,
                              quantile_returns: List[Dict],
                              factor_name: str,
                              period: int,
                              output_path: str = None) -> str:
        """
        绘制分层收益图

        Args:
            quantile_returns: 分层收益数据
            factor_name: 因子名称
            period: 预测周期
            output_path: 输出路径

        Returns:
            图片路径
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, f'quantile_returns_{factor_name}_{period}d.png')

        df = pd.DataFrame(quantile_returns)
        if len(df) == 0:
            return None

        # 计算各层平均收益
        quantile_cols = [c for c in df.columns if c.startswith('quantile_')]
        avg_returns = df[quantile_cols].mean()

        fig, ax = plt.subplots(figsize=(10, 5))

        colors = plt.cm.RdYlGn(np.linspace(0, 1, len(quantile_cols)))
        bars = ax.bar(range(len(avg_returns)), avg_returns.values * 100, color=colors)

        ax.set_title(f'{factor_name} - Quantile Returns ({period}d Forward Return)')
        ax.set_xlabel('Quantile')
        ax.set_ylabel('Average Return (%)')
        ax.set_xticks(range(len(avg_returns)))
        ax.set_xticklabels([f'Q{i+1}' for i in range(len(avg_returns))])

        # 添加数值标签
        for bar, val in zip(bars, avg_returns.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{val*100:.2f}%', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        return output_path

    def plot_ic_comparison(self,
                          summary_df: pd.DataFrame,
                          output_path: str = None) -> str:
        """
        绘制因子IC对比图

        Args:
            summary_df: 因子分析汇总表
            output_path: 输出路径

        Returns:
            图片路径
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, 'ic_comparison.png')

        # 使用5日预测周期的数据
        df = summary_df[summary_df['period'] == 5].copy()
        if len(df) == 0:
            df = summary_df[summary_df['period'] == summary_df['period'].min()].copy()

        df = df.sort_values('icir', ascending=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(df) * 0.4)))

        # IC均值图
        colors = ['green' if x > 0 else 'red' for x in df['ic_mean']]
        axes[0].barh(df['factor'], df['ic_mean'], color=colors, alpha=0.7)
        axes[0].axvline(x=0, color='black', linewidth=0.5)
        axes[0].axvline(x=0.02, color='blue', linestyle='--', alpha=0.5, label='Threshold (0.02)')
        axes[0].set_xlabel('IC Mean')
        axes[0].set_title('Factor IC Comparison')
        axes[0].legend()

        # ICIR图
        colors = ['green' if x > 0.5 else 'orange' if x > 0 else 'red' for x in df['icir']]
        axes[1].barh(df['factor'], df['icir'], color=colors, alpha=0.7)
        axes[1].axvline(x=0.5, color='blue', linestyle='--', alpha=0.5, label='Threshold (0.5)')
        axes[1].set_xlabel('ICIR')
        axes[1].set_title('Factor ICIR Comparison')
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()

        return output_path

    def generate_html_report(self,
                            summary_df: pd.DataFrame,
                            factor_details: Dict = None,
                            output_path: str = None) -> str:
        """
        生成HTML格式的因子分析报告

        Args:
            summary_df: 因子分析汇总表
            factor_details: 各因子详细分析结果
            output_path: 输出路径

        Returns:
            HTML文件路径
        """
        if output_path is None:
            output_path = os.path.join(self.output_dir, 'factor_analysis_report.html')

        # 生成图表
        ic_comparison_path = self.plot_ic_comparison(summary_df)

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>因子有效性分析报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 1px solid #ccc; padding-bottom: 5px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .positive {{ color: green; }}
        .negative {{ color: red; }}
        .highlight {{ background-color: #ffffcc; }}
        img {{ max-width: 100%; margin: 20px 0; }}
    </style>
</head>
<body>
    <h1>因子有效性分析报告</h1>
    <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <h2>1. 因子IC对比</h2>
    <img src="{os.path.basename(ic_comparison_path)}" alt="IC对比图">

    <h2>2. 因子有效性汇总表</h2>
    {self._generate_summary_table(summary_df)}

    <h2>3. 因子筛选建议</h2>
    {self._generate_recommendations(summary_df)}

    <h2>4. 结论</h2>
    <p>基于IC分析，建议保留IC均值 > 0.02 且 ICIR > 0.5 的因子。</p>
</body>
</html>
"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        logger.info(f"HTML报告已生成: {output_path}")
        return output_path

    def _generate_summary_table(self, summary_df: pd.DataFrame) -> str:
        """生成汇总表格HTML"""
        html = '<table>\n'
        html += '<tr><th>因子</th><th>周期</th><th>IC均值</th><th>ICIR</th><th>IC胜率</th><th>多空收益</th><th>评估</th></tr>\n'

        for _, row in summary_df.iterrows():
            # 评估因子有效性
            if row['ic_mean'] > 0.02 and row['icir'] > 0.5:
                status = '<span class="positive">有效</span>'
            elif row['ic_mean'] > 0:
                status = '<span style="color:orange">弱效</span>'
            else:
                status = '<span class="negative">无效</span>'

            html += f"""
            <tr>
                <td>{row['factor']}</td>
                <td>{row['period']}d</td>
                <td class="{'positive' if row['ic_mean'] > 0 else 'negative'}">{row['ic_mean']:.4f}</td>
                <td>{row['icir']:.4f}</td>
                <td>{row['ic_positive_ratio']:.2%}</td>
                <td>{row['long_short_return']:.4%}</td>
                <td>{status}</td>
            </tr>
            """
        html += '</table>'
        return html

    def _generate_recommendations(self, summary_df: pd.DataFrame) -> str:
        """生成因子筛选建议"""
        good_factors = summary_df[
            (summary_df['ic_mean'] > 0.02) &
            (summary_df['icir'] > 0.5)
        ]['factor'].unique()

        bad_factors = set(summary_df['factor'].unique()) - set(good_factors)

        html = f"""
        <h3>推荐保留的因子 ({len(good_factors)} 个)</h3>
        <ul>
        """
        for f in sorted(good_factors):
            html += f'<li class="positive">{f}</li>\n'

        html += f"""
        </ul>
        <h3>建议剔除的因子 ({len(bad_factors)} 个)</h3>
        <ul>
        """
        for f in sorted(bad_factors):
            html += f'<li class="negative">{f}</li>\n'

        html += '</ul>'
        return html


if __name__ == '__main__':
    # 测试
    np.random.seed(42)

    # 模拟因子分析结果
    factors = ['momentum', 'roe', 'dividend', 'value', 'growth', 'quality']
    periods = [5, 10, 20]

    data = []
    for factor in factors:
        for period in periods:
            ic_mean = np.random.uniform(-0.02, 0.05)
            ic_std = abs(ic_mean) / max(0.1, np.random.uniform(0.3, 1.5))
            data.append({
                'factor': factor,
                'period': period,
                'ic_mean': ic_mean,
                'ic_std': ic_std,
                'icir': ic_mean / ic_std if ic_std > 0 else 0,
                'ic_positive_ratio': np.random.uniform(0.4, 0.65),
                'p_value': np.random.uniform(0, 0.1),
                'long_short_return': np.random.uniform(-0.01, 0.02)
            })

    summary_df = pd.DataFrame(data)
    generator = FactorReportGenerator()
    generator.generate_html_report(summary_df)
    print("测试报告生成完成")
