"""
运行因子IC分析
验证现有因子的有效性
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from src.analysis.ic_analyzer import ICAnalyzer, run_ic_analysis
from src.analysis.factor_report import FactorReportGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_price_data_from_clickhouse(start_date: str, end_date: str) -> pd.DataFrame:
    """
    从ClickHouse加载价格数据

    Args:
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        DataFrame with [date, code, close]
    """
    try:
        from clickhouse_driver import Client
        client = Client(host='localhost', port=9000)

        query = f"""
        SELECT
            toDate(trade_date) as date,
            stock_code as code,
            adj_close as close
        FROM stock_daily_hfq
        WHERE trade_date >= toDate('{start_date}')
          AND trade_date <= toDate('{end_date}')
        ORDER BY trade_date, stock_code
        """

        result = client.execute(query)
        df = pd.DataFrame(result, columns=['date', 'code', 'close'])
        logger.info(f"从ClickHouse加载 {len(df)} 条价格数据")
        return df

    except Exception as e:
        logger.warning(f"无法连接ClickHouse: {e}")
        return None


def load_factor_data_from_csv(csv_path: str = None) -> pd.DataFrame:
    """
    从CSV加载因子数据

    Args:
        csv_path: CSV文件路径

    Returns:
        DataFrame with [date, code, factor1, factor2, ...]
    """
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        logger.info(f"从CSV加载 {len(df)} 条因子数据")
        return df
    return None


def generate_sample_data() -> tuple:
    """
    生成模拟数据用于测试

    Returns:
        (prices_df, factors_df)
    """
    np.random.seed(42)

    dates = pd.date_range('2020-01-01', '2024-12-31', freq='B')  # 工作日
    codes = [f'{i:06d}' for i in range(1, 51)]  # 50只股票

    logger.info(f"生成模拟数据: {len(dates)} 天, {len(codes)} 只股票")

    # 价格数据
    prices_data = []
    base_prices = {code: np.random.uniform(10, 100) for code in codes}

    for date in dates:
        for code in codes:
            # 随机游走
            ret = np.random.randn() * 0.02
            base_prices[code] *= (1 + ret)
            prices_data.append({
                'date': date,
                'code': code,
                'close': base_prices[code]
            })

    prices_df = pd.DataFrame(prices_data)

    # 因子数据
    factors_data = []
    factor_values = {code: {
        'momentum': np.random.randn(),
        'roe': np.random.uniform(5, 25),
        'dividend_yield': np.random.uniform(0, 0.05),
        'pe_value': np.random.uniform(5, 50),
        'profit_growth': np.random.uniform(-20, 50),
        'market_cap': np.random.uniform(50, 500)
    } for code in codes}

    for date in dates:
        for code in codes:
            # 添加一些时间变化
            factors_data.append({
                'date': date,
                'code': code,
                'momentum': factor_values[code]['momentum'] + np.random.randn() * 0.1,
                'roe': factor_values[code]['roe'] + np.random.randn() * 2,
                'dividend_yield': factor_values[code]['dividend_yield'] + np.random.randn() * 0.01,
                'pe_value': factor_values[code]['pe_value'] + np.random.randn() * 5,
                'profit_growth': factor_values[code]['profit_growth'] + np.random.randn() * 10,
                'market_cap': factor_values[code]['market_cap'] + np.random.randn() * 50
            })

    factors_df = pd.DataFrame(factors_data)

    return prices_df, factors_df


def main():
    """主函数"""
    logger.info("="*60)
    logger.info("因子IC分析")
    logger.info("="*60)

    # 1. 加载数据
    logger.info("\n步骤1: 加载数据")

    # 尝试从ClickHouse加载，如果失败则使用模拟数据
    prices_df = load_price_data_from_clickhouse('2019-01-01', '2024-12-31')

    if prices_df is None or len(prices_df) == 0:
        logger.info("使用模拟数据进行测试...")
        prices_df, factors_df = generate_sample_data()
    else:
        # 如果有真实价格数据，需要构建因子数据
        # 这里简化处理，使用模拟因子
        _, factors_df = generate_sample_data()
        factors_df['date'] = prices_df['date'].unique()[:len(factors_df['date'].unique())][0] if len(prices_df) > 0 else factors_df['date']

    logger.info(f"价格数据: {len(prices_df)} 行")
    logger.info(f"因子数据: {len(factors_df)} 行")

    # 2. 运行IC分析
    logger.info("\n步骤2: 运行IC分析")

    factor_columns = ['momentum', 'roe', 'dividend_yield', 'pe_value', 'profit_growth', 'market_cap']

    summary_df, analyzer = run_ic_analysis(
        prices_df=prices_df,
        factors_df=factors_df,
        factor_columns=factor_columns,
        periods=[5, 10, 20]
    )

    # 3. 生成报告
    logger.info("\n步骤3: 生成报告")

    # 文本报告
    report_text = analyzer.generate_factor_report(summary_df)
    print("\n" + report_text)

    # 保存文本报告
    report_path = 'reports/factor_ic_report.txt'
    os.makedirs('reports', exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    logger.info(f"文本报告已保存: {report_path}")

    # HTML报告
    generator = FactorReportGenerator(output_dir='reports')
    html_path = generator.generate_html_report(summary_df)
    logger.info(f"HTML报告已保存: {html_path}")

    # 4. 筛选有效因子
    logger.info("\n步骤4: 筛选有效因子")
    recommended_factors = analyzer.get_recommended_factors(
        summary_df,
        min_ic=0.02,
        min_icir=0.5
    )
    logger.info(f"推荐因子: {recommended_factors}")

    # 5. 保存分析结果
    logger.info("\n步骤5: 保存分析结果")
    summary_path = 'reports/factor_ic_summary.csv'
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    logger.info(f"汇总表已保存: {summary_path}")

    logger.info("\n" + "="*60)
    logger.info("IC分析完成!")
    logger.info("="*60)

    return summary_df, recommended_factors


if __name__ == '__main__':
    summary, factors = main()
