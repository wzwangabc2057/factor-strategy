"""
参数扫描 - 寻找最优 tilt_strength 和 max_weight 组合
在 Mac Studio 上运行: python3 param_sweep.py
"""
import pandas as pd
import numpy as np
import sys
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import clickhouse_connect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from data.factor_calculator_v2 import EnhancedFactorCalculator

# 复用 run_backtest.py 中的类和函数
from run_backtest import LocalDataFetcher, build_factor_data, apply_mild_weight_tilt


class SweepBacktestEngine:
    """精简回测引擎 - 用于参数扫描"""

    def __init__(self):
        self.ch_client = clickhouse_connect.get_client(
            host='192.168.0.74', port=8123, compress=False, query_limit=0
        )
        self.data_fetcher = LocalDataFetcher()
        self._price_cache = {}
        self._factor_cache = {}

    def get_daily_prices(self, codes, start_date, end_date):
        cache_key = (tuple(sorted(codes)), start_date, end_date)
        if cache_key in self._price_cache:
            return self._price_cache[cache_key]

        codes_str = "', '".join(codes)
        query = f"""
        SELECT toString(date) as date, code, close
        FROM default.stock_data_qfq
        WHERE code IN ('{codes_str}')
          AND date >= '{start_date}'
          AND date <= '{end_date}'
        ORDER BY date, code
        """
        result = self.ch_client.query(query)
        df = pd.DataFrame(result.result_rows, columns=['date', 'code', 'close'])
        self._price_cache[cache_key] = df
        return df

    def get_factor_and_scores(self, codes, eval_date, strategy_type):
        cache_key = (tuple(sorted(codes)), eval_date, strategy_type)
        if cache_key in self._factor_cache:
            return self._factor_cache[cache_key]

        factor_data = build_factor_data(self.data_fetcher, codes, eval_date, min_market_cap=0)
        factor_calc = EnhancedFactorCalculator(strategy_type=strategy_type)
        scored_data = factor_calc.calculate_all_scores(factor_data)
        self._factor_cache[cache_key] = (factor_data, scored_data)
        return factor_data, scored_data

    def run_single(self, portfolio, start_date, end_date, strategy_type,
                   tilt_strength, max_weight):
        """运行单次回测，返回指标"""
        codes = portfolio['code'].tolist()

        factor_data, scored_data = self.get_factor_and_scores(codes, end_date, strategy_type)

        # 过滤无因子数据的股票
        filtered_codes = factor_data['code'].tolist()
        port = portfolio[portfolio['code'].isin(filtered_codes)].copy()
        port['weight'] = port['weight'] / port['weight'].sum()

        # 权重调整
        enhanced_portfolio = apply_mild_weight_tilt(
            port, scored_data,
            tilt_strength=tilt_strength,
            max_weight=max_weight
        )

        # 获取价格数据
        price_df = self.get_daily_prices(codes, start_date, end_date)
        price_df = price_df.drop_duplicates(subset=['date', 'code'], keep='last')
        price_pivot = price_df.pivot(index='date', columns='code', values='close')
        dates = sorted(price_pivot.index.tolist())

        if len(dates) < 10:
            return None

        # 股息率
        dividend_dict = dict(zip(factor_data['code'], factor_data.get('dividend_yield', 0)))
        daily_dividend = {code: div / 252 for code, div in dividend_dict.items()
                          if isinstance(div, (int, float))}

        def calc_daily_returns(weight_dict):
            daily_rets = []
            for i in range(1, len(dates)):
                prev_date = dates[i - 1]
                curr_date = dates[i]
                day_ret = 0.0
                for code, w in weight_dict.items():
                    if code in price_pivot.columns:
                        p0 = price_pivot.loc[prev_date].get(code)
                        p1 = price_pivot.loc[curr_date].get(code)
                        if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                            stock_ret = (p1 - p0) / p0
                            div_ret = daily_dividend.get(code, 0)
                            day_ret += w * (stock_ret + div_ret)
                daily_rets.append(day_ret)
            return daily_rets

        def calc_metrics(daily_rets):
            cum = np.cumprod(1 + np.array(daily_rets))
            total_return = cum[-1] - 1
            n_years = len(daily_rets) / 252
            annual_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

            peak = np.maximum.accumulate(cum)
            dd = (cum - peak) / peak
            max_drawdown = dd.min()

            avg = np.mean(daily_rets)
            std = np.std(daily_rets)
            sharpe = avg / std * np.sqrt(252) if std > 0 else 0

            return {
                'annual_return': annual_return,
                'max_drawdown': max_drawdown,
                'sharpe': sharpe,
                'total_return': total_return,
            }

        enhanced_weights = dict(zip(enhanced_portfolio['code'], enhanced_portfolio['adjusted_weight']))
        original_weights = dict(zip(port['code'], port['weight']))

        enhanced_rets = calc_daily_returns(enhanced_weights)
        original_rets = calc_daily_returns(original_weights)

        e_metrics = calc_metrics(enhanced_rets)
        o_metrics = calc_metrics(original_rets)

        return {
            'enhanced': e_metrics,
            'original': o_metrics,
            'improvement': e_metrics['annual_return'] - o_metrics['annual_return'],
        }


def main():
    print("=" * 70)
    print("参数扫描 - 寻找最优 tilt_strength 和 max_weight")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 加载持仓
    r4 = pd.read_csv(os.path.join(base_dir, 'r4_weights_local.csv'))
    r5 = pd.read_csv(os.path.join(base_dir, 'r5_weights_local.csv'))
    r4['code'] = r4['code'].astype(str).str.zfill(6)
    r5['code'] = r5['code'].astype(str).str.zfill(6)
    r4 = r4[['code', 'weight']].copy()
    r5 = r5[['code', 'weight']].copy()
    r4['weight'] = r4['weight'] / r4['weight'].sum()
    r5['weight'] = r5['weight'] / r5['weight'].sum()

    engine = SweepBacktestEngine()

    start_date = '2020-01-01'
    end_date = '2025-12-31'

    # 参数空间
    tilt_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5]
    max_weight_values = [0.05, 0.08, 0.10, 0.15]

    results = []

    print(f"\n扫描空间: tilt={tilt_values}, max_weight={max_weight_values}")
    print(f"总组合数: {len(tilt_values) * len(max_weight_values) * 2} (R4+R5)")
    print()

    for strategy_type, portfolio, label in [
        ('stable', r4, 'R4'),
        ('aggressive', r5, 'R5'),
    ]:
        print(f"\n{'='*50}")
        print(f"【{label} {strategy_type}】")
        print(f"{'='*50}")

        for max_w in max_weight_values:
            for tilt in tilt_values:
                result = engine.run_single(
                    portfolio, start_date, end_date,
                    strategy_type=strategy_type,
                    tilt_strength=tilt,
                    max_weight=max_w,
                )
                if result is None:
                    continue

                row = {
                    'strategy': label,
                    'tilt': tilt,
                    'max_weight': max_w,
                    'original_annual': result['original']['annual_return'],
                    'enhanced_annual': result['enhanced']['annual_return'],
                    'improvement': result['improvement'],
                    'enhanced_sharpe': result['enhanced']['sharpe'],
                    'enhanced_drawdown': result['enhanced']['max_drawdown'],
                    'original_sharpe': result['original']['sharpe'],
                }
                results.append(row)

                imp = result['improvement'] * 100
                marker = " <<<" if imp > 1.0 else ""
                print(f"  tilt={tilt:.1f} max_w={max_w:.2f}: "
                      f"增强={result['enhanced']['annual_return']*100:.2f}% "
                      f"提升={imp:+.2f}% "
                      f"夏普={result['enhanced']['sharpe']:.2f} "
                      f"回撤={result['enhanced']['max_drawdown']*100:.1f}%{marker}")

    # 汇总结果
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(base_dir, 'param_sweep_results.csv'), index=False)

    print("\n" + "=" * 70)
    print("【最优参数】")
    print("=" * 70)

    for label in ['R4', 'R5']:
        sub = df[df['strategy'] == label]
        # 按提升排序
        best_imp = sub.nlargest(5, 'improvement')
        print(f"\n{label} - Top 5 提升:")
        for _, row in best_imp.iterrows():
            print(f"  tilt={row['tilt']:.1f} max_w={row['max_weight']:.2f}: "
                  f"增强={row['enhanced_annual']*100:.2f}% "
                  f"提升={row['improvement']*100:+.2f}% "
                  f"夏普={row['enhanced_sharpe']:.2f} "
                  f"回撤={row['enhanced_drawdown']*100:.1f}%")

        # 按夏普排序
        best_sharpe = sub.nlargest(5, 'enhanced_sharpe')
        print(f"\n{label} - Top 5 夏普:")
        for _, row in best_sharpe.iterrows():
            print(f"  tilt={row['tilt']:.1f} max_w={row['max_weight']:.2f}: "
                  f"增强={row['enhanced_annual']*100:.2f}% "
                  f"夏普={row['enhanced_sharpe']:.2f} "
                  f"回撤={row['enhanced_drawdown']*100:.1f}%")

    print(f"\n结果已保存到 param_sweep_results.csv")
    print("扫描完成!")


if __name__ == '__main__':
    main()
