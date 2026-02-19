#!/usr/bin/env python
"""
本地数据回测脚本 - 集成 Gate-0 可靠性 + Softmax 权重分配

使用本地CSV数据进行回测，无需 ClickHouse 连接
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# 添加src路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# ============================================================================
# 配置
# ============================================================================

TRANSACTION_COSTS = {
    'buy_commission': 0.00026,
    'sell_commission': 0.00126,
}

# 温和8档权重调整
TIERS_MILD = [
    (0.95, 1.50), (0.90, 1.30), (0.80, 1.15), (0.60, 1.00),
    (0.40, 0.90), (0.20, 0.80), (0.10, 0.70), (0.00, 0.50),
]

# 因子权重 (R4稳健型)
FACTOR_WEIGHTS_R4 = {
    'dividend_yield': 0.16, 'roe': 0.14, 'roe_stability': 0.09,
    'reversal': 0.10, 'pe_value': 0.09, 'profit_growth': 0.09,
    'small_cap': 0.06, 'momentum': 0.04, 'low_volatility': 0.06,
    'financial_health': 0.05, 'gross_margin': 0.05, 'revenue_growth': 0.07,
}

# 因子权重 (R5进取型)
FACTOR_WEIGHTS_R5 = {
    'profit_growth': 0.16, 'small_cap': 0.13, 'momentum': 0.11,
    'roe': 0.11, 'revenue_growth': 0.09, 'reversal': 0.07,
    'dividend_yield': 0.07, 'pe_value': 0.04, 'low_volatility': 0.04,
    'financial_health': 0.06, 'gross_margin': 0.05, 'turnover': 0.07,
}


# ============================================================================
# Gate-0 可靠性熔断
# ============================================================================

class ReliabilityGate:
    """Gate-0: 数据/信号熔断"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.score_missing_rate_max = self.config.get('score_missing_rate_max', 0.02)
        self.factor_missing_rate_max = self.config.get('factor_missing_rate_max', 0.05)
        self.default_action = self.config.get('action', 'freeze_rebalance')
        self.reduce_only_ratio = self.config.get('reduce_only_ratio', 0.8)

        self.stats = {
            'gate0_triggered': False,
            'gate0_action': 'none',
            'gate0_trigger_count': 0,
            'freeze_rebalance_count': 0,
            'reduce_only_count': 0,
            'score_missing_rate': 0.0,
            'factor_missing_rate': 0.0,
        }
        self.breach_history = []

    def check(self, codes: List[str], scores: pd.Series = None,
              factor_data: pd.DataFrame = None) -> Tuple[bool, str, Dict]:
        """检查数据可靠性"""
        details = {
            'score_missing_rate': 0.0,
            'factor_missing_rate': 0.0,
        }

        # 检查得分缺失率
        if scores is not None and len(codes) > 0:
            missing = set(codes) - set(scores.index)
            score_missing_rate = len(missing) / len(codes)
        else:
            score_missing_rate = 1.0

        # 检查因子缺失率
        if factor_data is not None and len(codes) > 0:
            if 'code' in factor_data.columns:
                missing = set(codes) - set(factor_data['code'].unique())
            else:
                missing = set(codes) - set(factor_data.index)
            factor_missing_rate = len(missing) / len(codes)
        else:
            factor_missing_rate = 0.0

        details['score_missing_rate'] = score_missing_rate
        details['factor_missing_rate'] = factor_missing_rate

        self.stats['score_missing_rate'] = score_missing_rate
        self.stats['factor_missing_rate'] = factor_missing_rate

        # 判断是否熔断
        breach = False
        action = 'none'

        if score_missing_rate > self.score_missing_rate_max:
            breach = True
            action = self.default_action
            print(f"  [Gate-0] 得分缺失率 {score_missing_rate:.2%} > 阈值 {self.score_missing_rate_max:.0%}")

        if factor_missing_rate > self.factor_missing_rate_max:
            breach = True
            action = self.default_action
            print(f"  [Gate-0] 因子缺失率 {factor_missing_rate:.2%} > 阈值 {self.factor_missing_rate_max:.0%}")

        if breach:
            self.stats['gate0_triggered'] = True
            self.stats['gate0_action'] = action
            self.stats['gate0_trigger_count'] += 1
            if action == 'freeze_rebalance':
                self.stats['freeze_rebalance_count'] += 1
            elif action == 'reduce_only':
                self.stats['reduce_only_count'] += 1
            self.breach_history.append({
                'date': datetime.now().isoformat(),
                'action': action,
                'score_missing_rate': score_missing_rate
            })

        return breach, action, details

    def apply_action(self, action: str, current_weights: Dict[str, float],
                     target_weights: Dict[str, float] = None) -> Dict[str, float]:
        """应用熔断动作"""
        if action == 'none':
            return target_weights or current_weights

        if action == 'freeze_rebalance':
            print(f"  [Gate-0] 冻结调仓：保持当前仓位")
            return current_weights.copy()

        if action == 'reduce_only':
            print(f"  [Gate-0] 仅减仓模式：仓位降到{self.reduce_only_ratio:.0%}")
            adjusted = {}
            for code, weight in current_weights.items():
                if target_weights and code in target_weights:
                    new_weight = min(weight, target_weights[code])
                else:
                    new_weight = weight
                adjusted[code] = new_weight * self.reduce_only_ratio
            return adjusted

        return current_weights.copy()


# ============================================================================
# Softmax 权重分配
# ============================================================================

def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """计算 softmax 归一化权重"""
    exp_x = np.exp((x - np.max(x)) / temperature)
    return exp_x / np.sum(exp_x)


def allocate_weights_softmax(codes: List[str], scores: pd.Series,
                             total_weight: float = 0.6,
                             temperature: float = 1.5,
                             min_weight: float = 0.002) -> Dict[str, float]:
    """使用 softmax 分配权重"""
    if not codes:
        return {}

    code_scores = scores.loc[scores.index.isin(codes)]
    if len(code_scores) == 0:
        return {code: total_weight / len(codes) for code in codes}

    sorted_scores = code_scores.sort_values(ascending=False)
    raw_weights = softmax(sorted_scores.values, temperature=temperature)

    # 应用最小权重
    raw_weights = np.maximum(raw_weights, min_weight)
    raw_weights = raw_weights / raw_weights.sum() * total_weight

    return {code: float(w) for code, w in zip(sorted_scores.index, raw_weights)}


# ============================================================================
# 因子打分
# ============================================================================

def pct_score(s, ascending=True):
    return s.rank(pct=True) * 100 if ascending else (1 - s.rank(pct=True)) * 100


def score_factors(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """计算所有因子得分"""
    scores = {}

    if 'roe' in df.columns:
        scores['roe'] = pct_score(df['roe'].clip(0, 50))

    if 'pe_ttm' in df.columns:
        valid_pe = df['pe_ttm'].where((df['pe_ttm'] > 0) & (df['pe_ttm'] < 200), np.nan)
        scores['pe_value'] = pct_score(valid_pe, ascending=False).fillna(30)

    if 'dividend_yield' in df.columns:
        s = pct_score(df['dividend_yield'])
        s[df['dividend_yield'] > 5] = s[df['dividend_yield'] > 5].clip(lower=85)
        scores['dividend_yield'] = s

    if 'gross_margin' in df.columns:
        scores['gross_margin'] = pct_score(df['gross_margin'].clip(0, 80))

    if 'profit_growth' in df.columns or 'net_profit_yoy' in df.columns:
        col = 'profit_growth' if 'profit_growth' in df.columns else 'net_profit_yoy'
        scores['profit_growth'] = pct_score(df[col].clip(-50, 200))

    if 'revenue_yoy' in df.columns:
        scores['revenue_growth'] = pct_score(df['revenue_yoy'].clip(-50, 200))

    if 'total_mv' in df.columns:
        scores['small_cap'] = pct_score(df['total_mv'], ascending=False)

    if 'roe_std_3y' in df.columns:
        scores['roe_stability'] = 100 - df['roe_std_3y'].clip(0, 20) * 4

    if 'asset_liability_ratio' in df.columns:
        scores['financial_health'] = pct_score(df['asset_liability_ratio'], ascending=False)

    return scores


def composite_score(factor_scores: Dict[str, pd.Series],
                    weights: Dict[str, float]) -> pd.Series:
    """合成综合得分"""
    if not factor_scores:
        return pd.Series(50.0)

    keys = list(factor_scores.keys())
    idx = factor_scores[keys[0]].index
    composite = pd.Series(0.0, index=idx)
    total_w = 0

    for factor, weight in weights.items():
        if factor in factor_scores:
            composite += factor_scores[factor].fillna(50) * weight
            total_w += weight

    if total_w > 0:
        composite = composite / total_w

    return composite


# ============================================================================
# 权重调整
# ============================================================================

def get_multiplier(percentile, tiers):
    for threshold, mult in tiers:
        if percentile >= threshold:
            return mult
    return tiers[-1][1]


def adjust_weights(portfolio: pd.DataFrame, composite_scores: pd.Series,
                   tiers: List[Tuple[float, float]]) -> pd.DataFrame:
    """调整权重"""
    merged = portfolio.copy()
    score_df = pd.DataFrame({
        'code': composite_scores.index,
        'composite_score': composite_scores.values
    })
    merged = merged.merge(score_df, on='code', how='left')
    merged['composite_score'] = merged['composite_score'].fillna(50)
    merged['percentile'] = merged['composite_score'].rank(pct=True)
    merged['multiplier'] = merged['percentile'].apply(lambda p: get_multiplier(p, tiers))
    merged['original_weight'] = merged['weight']
    merged['adjusted_weight'] = merged['weight'] * merged['multiplier']
    merged['adjusted_weight'] /= merged['adjusted_weight'].sum()

    return merged


# ============================================================================
# 本地数据加载
# ============================================================================

class LocalDataLoader:
    """本地数据加载器"""

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self._valuation = None
        self._financial = None
        self._weights_r4 = None
        self._weights_r5 = None

    def load_valuation(self):
        if self._valuation is None:
            path = os.path.join(self.base_dir, 'valuation_local.csv')
            self._valuation = pd.read_csv(path)
            self._valuation['code'] = self._valuation['code'].astype(str).str.zfill(6)
        return self._valuation

    def load_financial(self):
        if self._financial is None:
            path = os.path.join(self.base_dir, 'financial_data_akshare.csv')
            self._financial = pd.read_csv(path)
            self._financial['code'] = self._financial['code'].astype(str).str.zfill(6)
        return self._financial

    def load_portfolio(self, strategy_type: str = 'stable'):
        if strategy_type == 'stable':
            if self._weights_r4 is None:
                path = os.path.join(self.base_dir, 'r4_weights_local.csv')
                self._weights_r4 = pd.read_csv(path)
                self._weights_r4['code'] = self._weights_r4['code'].astype(str).str.zfill(6)
            return self._weights_r4
        else:
            if self._weights_r5 is None:
                path = os.path.join(self.base_dir, 'r5_weights_local.csv')
                self._weights_r5 = pd.read_csv(path)
                self._weights_r5['code'] = self._weights_r5['code'].astype(str).str.zfill(6)
            return self._weights_r5

    def get_factor_data(self, codes: List[str]) -> pd.DataFrame:
        """获取因子数据"""
        val = self.load_valuation()
        fin = self.load_financial()

        # 合并数据
        val_subset = val[val['code'].isin(codes)].copy()
        fin_subset = fin[fin['code'].isin(codes)].copy()

        merged = val_subset.merge(fin_subset, on='code', how='left', suffixes=('', '_fin'))

        # 重命名列
        col_map = {
            'pe_ttm': 'pe_ttm',
            'gross_margin': 'gross_margin',
            'net_profit_yoy': 'net_profit_yoy',
            'revenue_yoy': 'revenue_yoy',
            'roe_std_3y': 'roe_std_3y',
            'asset_liability_ratio': 'asset_liability_ratio',
        }
        merged = merged.rename(columns=col_map)

        return merged


# ============================================================================
# 模拟回测（使用随机价格）
# ============================================================================

def simulate_backtest(loader: LocalDataLoader,
                      strategy_type: str = 'stable',
                      start_date: str = '2020-01-01',
                      end_date: str = '2025-12-31',
                      use_gate0: bool = True,
                      use_softmax: bool = True,
                      rebalance_months: List[int] = None):
    """模拟回测"""

    if rebalance_months is None:
        rebalance_months = [1, 4, 7, 10]  # 季度调仓

    factor_weights = FACTOR_WEIGHTS_R4 if strategy_type == 'stable' else FACTOR_WEIGHTS_R5

    # 加载组合
    portfolio = loader.load_portfolio(strategy_type)
    codes = portfolio['code'].tolist()

    print(f"\n{'='*60}")
    print(f"回测: {strategy_type.upper()} | Gate-0: {use_gate0} | Softmax: {use_softmax}")
    print(f"股票数: {len(codes)} | 调仓月份: {rebalance_months}")
    print(f"{'='*60}")

    # 获取因子数据
    factor_data = loader.get_factor_data(codes)
    factor_data = factor_data.set_index('code')

    # 计算因子得分
    factor_scores = score_factors(factor_data)
    comp_scores = composite_score(factor_scores, factor_weights)

    # Gate-0 检查
    if use_gate0:
        gate = ReliabilityGate({
            'score_missing_rate_max': 0.02,
            'action': 'freeze_rebalance'
        })
        breach, action, details = gate.check(codes, comp_scores, factor_data.reset_index())
        print(f"  Gate-0 检查: breach={breach}, action={action}")
        print(f"  得分缺失率: {details['score_missing_rate']:.2%}")

    # 权重调整
    portfolio_adj = adjust_weights(portfolio, comp_scores, TIERS_MILD)

    # 如果使用 softmax，对核心层使用 softmax 分配
    if use_softmax:
        # 选择 top 30% 作为核心层
        top_n = int(len(codes) * 0.3)
        top_codes = comp_scores.nlargest(top_n).index.tolist()

        # Softmax 分配核心层权重
        core_weights = allocate_weights_softmax(
            top_codes, comp_scores,
            total_weight=0.6, temperature=1.5, min_weight=0.005
        )

        # 其余等权分配卫星层
        sat_codes = [c for c in codes if c not in top_codes]
        sat_weight_each = 0.4 / len(sat_codes) if sat_codes else 0

        # 合并权重
        final_weights = {}
        for code in codes:
            if code in core_weights:
                final_weights[code] = core_weights[code]
            else:
                final_weights[code] = sat_weight_each

        # 归一化
        total = sum(final_weights.values())
        final_weights = {k: v/total for k, v in final_weights.items()}
    else:
        final_weights = dict(zip(portfolio_adj['code'], portfolio_adj['adjusted_weight']))

    # 模拟收益（使用随机数模拟）
    np.random.seed(42)
    n_days = 252 * 5  # 5年
    daily_returns = []

    for i in range(n_days):
        # 模拟每日收益
        stock_returns = np.random.normal(0.0004, 0.02, len(codes))  # 日均0.04%收益，2%波动

        # 计算组合收益
        port_return = sum(stock_returns[j] * final_weights.get(codes[j], 0)
                         for j in range(len(codes)))
        daily_returns.append(port_return)

    # 扣除交易成本（每季度一次）
    tx_cost_per_rebalance = 0.002  # 0.2%每次
    total_tx_cost = tx_cost_per_rebalance * len(rebalance_months) * 5  # 5年

    # 计算绩效
    cum_returns = np.cumprod([1 + r for r in daily_returns])
    total_return = cum_returns[-1] - 1 - total_tx_cost
    n_years = n_days / 252
    annual_return = (1 + total_return) ** (1 / n_years) - 1
    max_dd = np.min(cum_returns / np.maximum.accumulate(cum_returns) - 1)
    daily_std = np.std(daily_returns)
    sharpe = np.mean(daily_returns) / daily_std * np.sqrt(252) if daily_std > 0 else 0
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0

    results = {
        'annual_return': annual_return,
        'total_return': total_return,
        'max_drawdown': max_dd,
        'sharpe': sharpe,
        'calmar': calmar,
        'total_tx_cost': total_tx_cost,
        'final_nav': cum_returns[-1] * (1 - total_tx_cost),
    }

    print(f"\n  年化收益: {annual_return*100:.2f}%")
    print(f"  最大回撤: {max_dd*100:.2f}%")
    print(f"  夏普比率: {sharpe:.2f}")
    print(f"  卡尔玛: {calmar:.2f}")
    print(f"  交易成本: {total_tx_cost*100:.2f}%")

    if use_gate0:
        print(f"\n  Gate-0 统计:")
        print(f"    熔断次数: {gate.stats['gate0_trigger_count']}")
        print(f"    冻结调仓: {gate.stats['freeze_rebalance_count']}")

    return results


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 70)
    print("本地数据回测 - 集成 Gate-0 + Softmax")
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    loader = LocalDataLoader(base_dir)

    results = {}

    # 测试4种配置
    configs = [
        ('stable', False, False, 'R4-Baseline'),
        ('stable', True, False, 'R4-Gate0'),
        ('stable', False, True, 'R4-Softmax'),
        ('stable', True, True, 'R4-Gate0+Softmax'),
        ('aggressive', False, False, 'R5-Baseline'),
        ('aggressive', True, True, 'R5-Gate0+Softmax'),
    ]

    for strategy, use_gate0, use_softmax, label in configs:
        results[label] = simulate_backtest(
            loader, strategy,
            use_gate0=use_gate0,
            use_softmax=use_softmax
        )

    # 汇总对比
    print("\n\n" + "=" * 70)
    print("汇总对比")
    print("=" * 70)
    print(f"{'配置':<22} {'年化':>8} {'回撤':>8} {'夏普':>6} {'卡尔玛':>7}")
    print("-" * 55)

    for label, r in results.items():
        print(f"{label:<22} {r['annual_return']*100:>7.2f}% {r['max_drawdown']*100:>7.2f}% "
              f"{r['sharpe']:>6.2f} {r['calmar']:>7.2f}")

    # 保存结果
    output_path = os.path.join(base_dir, 'backtest_local_result.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'run_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'results': results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到: {output_path}")


if __name__ == '__main__':
    main()
