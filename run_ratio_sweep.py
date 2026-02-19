#!/usr/bin/env python
"""
CSI300配比扫描 — 测试不同大盘/中盘分配比例
自动修改 CSI300_RATIO 参数并运行回测，汇总结果
"""
import subprocess
import json
import os
import re
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PATH = os.path.join(BASE_DIR, 'run_ml_backtest_v5.py')
RESULT_PATH = os.path.join(BASE_DIR, 'factor_backtest_result_v5.json')

# 要测试的配比: 0=纯ML不分层, 0.3=30%大盘, 0.4=40%, 0.5=50%, 0.6=60%
RATIOS_TO_TEST = [0.0, 0.30, 0.50, 0.60]

def set_ratio(ratio):
    """修改脚本中的CSI300_RATIO参数"""
    with open(SCRIPT_PATH, 'r') as f:
        content = f.read()

    # 替换 CSI300_RATIO = xxx
    content = re.sub(
        r'CSI300_RATIO = [\d.]+',
        f'CSI300_RATIO = {ratio:.2f}',
        content
    )
    with open(SCRIPT_PATH, 'w') as f:
        f.write(content)
    print(f"  CSI300_RATIO 设为 {ratio:.0%}")

def run_backtest():
    """运行回测并返回结果"""
    result = subprocess.run(
        [sys.executable, SCRIPT_PATH],
        capture_output=True, text=True, timeout=1800
    )
    if result.returncode != 0:
        print(f"  回测失败: {result.stderr[-500:]}")
        return None

    with open(RESULT_PATH, 'r') as f:
        return json.load(f)

def main():
    all_results = {}

    print("=" * 70)
    print("CSI300配比扫描测试")
    print("=" * 70)

    for ratio in RATIOS_TO_TEST:
        label = f"{ratio:.0%}" if ratio > 0 else "0%(纯ML)"
        print(f"\n>>> 测试配比: {label}")
        t0 = time.time()

        set_ratio(ratio)
        result = run_backtest()

        elapsed = time.time() - t0
        print(f"  耗时: {elapsed:.0f}秒")

        if result:
            enh = result['v5_enhanced']
            all_results[label] = {
                'ratio': ratio,
                'annual_return': enh['annual_return'],
                'sharpe': enh['sharpe'],
                'max_drawdown': enh['max_drawdown'],
                'calmar': enh['calmar'],
                'yearly': result.get('yearly_returns', {}).get('v5', {}),
                'avg_ic': result.get('avg_monthly_ic', 0),
                'tx_cost': result.get('total_tx_cost', 0),
            }

    # 恢复为最优配比 (先打印结果再决定)
    print("\n" + "=" * 70)
    print("配比扫描结果汇总")
    print("=" * 70)

    # 加入已有的r4(40%)结果
    r4_data = {
        'ratio': 0.40,
        'annual_return': 0.2136,
        'sharpe': 1.19,
        'max_drawdown': -0.1760,
        'calmar': 1.21,
        'yearly': {'2022': -0.0408, '2023': 0.1416, '2024': 0.2782, '2025': 0.5028},
        'avg_ic': 0.0538,
        'tx_cost': 0.008,
    }
    all_results['40%(已测)'] = r4_data

    header = f"{'配比':<12} {'年化收益':>10} {'夏普':>8} {'最大回撤':>10} {'卡尔玛':>8} {'2022':>8} {'2023':>8} {'2024':>8} {'2025':>8}"
    print(header)
    print("-" * len(header))

    # 按ratio排序
    for label in sorted(all_results.keys(), key=lambda x: all_results[x]['ratio']):
        r = all_results[label]
        yr = r.get('yearly', {})
        print(f"{label:<12} {r['annual_return']*100:>9.2f}% {r['sharpe']:>8.2f} {r['max_drawdown']*100:>9.2f}% {r['calmar']:>8.2f} "
              f"{yr.get('2022',0)*100:>7.1f}% {yr.get('2023',0)*100:>7.1f}% {yr.get('2024',0)*100:>7.1f}% {yr.get('2025',0)*100:>7.1f}%")

    # 找最优
    best_label = max(all_results.keys(), key=lambda x: all_results[x]['sharpe'])
    best = all_results[best_label]
    print(f"\n最优配比(按夏普): {best_label} — 夏普={best['sharpe']:.2f}, 年化={best['annual_return']*100:.2f}%")

    # 保存汇总
    summary_path = os.path.join(BASE_DIR, 'ratio_sweep_results.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"结果已保存: {summary_path}")

    # 恢复最优配比
    set_ratio(best['ratio'])
    print(f"已将脚本恢复为最优配比: {best['ratio']:.0%}")


if __name__ == '__main__':
    main()
