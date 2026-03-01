# 量化因子回测 - 状态追踪文档

> 最后更新: 2026-02-18 13:45

## 一、当前运行中的任务

### MacStudio (192.168.0.88) - V3 回测
- **脚本**: `run_backtest_v3.py`
- **PID**: 30922
- **Python**: 3.12.12 (pyenv)
- **启动时间**: 2026-02-18 13:36
- **状态**: 运行中，处理第1期（2025-07-16），每期预计 8-10 分钟
- **预计总耗时**: 5-6 小时（约40期）
- **检查命令**:
  ```bash
  sshpass -p "918kangbing" ssh macstudio@192.168.0.88 "tail -30 ~/112/pythontest/factor-strategy/backtest_v3_output.log"
  sshpass -p "918kangbing" ssh macstudio@192.168.0.88 "ps aux | grep 30922 | grep -v grep"
  ```

---

## 二、已完成的回测结果

### 1. 快速回测 (run_quick_backtest_fast.py) - 已完成
| 指标 | 值 |
|------|-----|
| 回测区间 | 2025-04-17 ~ 2026-02-02 |
| 总收益 | 5.95% |
| 年化收益 | **5.13%** (目标30%，未达成) |
| 夏普比率 | 0.46 |
| 最大回撤 | -18.64% |
| 平均信号IC | 0.037 |
| 胜率 | 45% |
| 耗时 | 3.4分钟 |

**结论**: 纯技术因子IC加权全市场选股效果差，逻辑不符合原有策略设计。

### 2. LightGBM ML训练 (run_optimized_training.py) - 已完成
| Fold | Model | 训练集 | 测试集 | RMSE | IC |
|------|-------|--------|--------|------|-----|
| 0 | LightGBM | 19,532 | 19,529 | 9.54 | -0.194 |
| 1 | LightGBM | 39,061 | 19,529 | 9.07 | 0.099 |
| 2 | LightGBM | 58,590 | 19,529 | 6.98 | **0.282** |
| 3 | LightGBM | 78,119 | 19,529 | 6.55 | -0.038 |
| 4 | LightGBM | 97,648 | 19,529 | 6.69 | **0.195** |

**结论**:
- 平均IC ≈ 0.069，不稳定（2个fold为负）
- Ridge回归反而更好（平均IC ≈ 0.107）
- 模型预测能力不足以支撑30%年化目标
- Feature importance 可参考，但不能直接当因子权重

### 3. 因子IC分析 (factor_ic_scores.csv) - 已完成
Top 10 因子:
| 排名 | 因子 | IC值 | 方向 |
|------|------|------|------|
| 1 | momentum_accel | -0.250 | 反向 |
| 2 | vol_of_vol_10d | -0.139 | 反向 |
| 3 | momentum_diff | -0.127 | 反向 |
| 4 | volatility_10d | -0.124 | 反向 |
| 5 | momentum_10d | 0.121 | 正向 |
| 6 | trend_strength | -0.119 | 反向 |
| 7 | parkinson_vol | -0.110 | 反向 |
| 8 | return_skew | -0.108 | 反向 |
| 9 | price_slope | 0.108 | 正向 |
| 10 | macd_hist | 0.102 | 正向 |

---

## 三、问题诊断 - 为什么两天没出成果

### 核心问题
1. **回测逻辑不匹配**: 快速回测用的是"全市场选股"，而原有策略是"R4/R5基础持仓权重调整"，两套完全不同的逻辑
2. **MacStudio 反复崩溃**: v3/v4回测在 MacStudio 上因编码问题(Python 3.9)反复崩溃，浪费大量时间
3. **性能瓶颈**: 原始回测代码每期需要 O(N^2) 的价格查找，5500只股票导致每期耗时过长
4. **ML训练结果不可用**: LightGBM的IC不稳定，不能直接用于选股

### 资源浪费明细
| 时间段 | 任务 | 结果 | 问题 |
|--------|------|------|------|
| 2/17 下午 | v4回测 MacStudio | 崩溃 | Python 3.9 编码错误 |
| 2/17 晚 | v4回测重启 | 崩溃 | 同上 |
| 2/18 凌晨 | LightGBM训练 | 完成但IC差 | 模型不稳定 |
| 2/18 上午 | 快速回测本地 | 卡死 | 性能太差 |
| 2/18 中午 | 快速回测优化版 | 完成，年化5% | 逻辑不对 |
| 2/18 下午 | v3回测 MacStudio | 运行中 | Python 3.12修复编码 |

### 根本原因
- 没有先理清原有策略的完整逻辑就开始跑代码
- 用了错误的回测脚本（quick_backtest vs backtest_v3）
- MacStudio 的 Python 环境不匹配（3.9 vs 3.12）

---

## 四、原有策略设计要点 (run_factor_backtest_v3.py)

### 正确的回测逻辑
1. **基础持仓**: 基于 R4/R5 CSV 权重文件，不是全市场选股
2. **16个因子**: 财务(ROE, PE, 股息率, 利润增长) + 技术(动量, 波动率, 反转)
3. **权重调整**: 8/10层分层乘数 (0.5x~2.5x)
4. **5策略对比**: A(v1.7复刻) / B(新因子) / C(半年调仓) / D(IC动态) / E(激进分层)
5. **止损**: 沪深300 < 250日均线时减仓70%
6. **交易成本**: 买0.026% + 卖0.126%
7. **回测区间**: 2020-01-01 ~ 2025-12-31

### V1.7 基准线 (已验证)
- R4 年化: 14.66%
- R5 年化: 16.61%
- 目标: 30%+

---

## 五、待做事项

### 紧急
- [ ] 等待 V3 回测完成（MacStudio PID 30922）
- [ ] 分析 V3 回测结果，对比5个策略

### 优化方向
- [ ] 用 pivot 表优化 V3 回测的价格查找性能（当前每期 8-10 分钟太慢）
- [ ] 考虑是否需要把 LightGBM 集成到 V3 的 Strategy D (IC动态加权)
- [ ] 验证因子IC在不同时间窗口的稳定性

### 长期
- [ ] 如果纯因子达不到30%，考虑 ML 综合模型 (Phase 3)
- [ ] Walk-Forward 验证 (WF-1 ~ WF-6)
- [ ] 生产环境部署清单 (PRODUCTION_CHECKLIST.md)

---

## 六、关键文件索引

| 文件 | 用途 | 状态 |
|------|------|------|
| `run_factor_backtest_v3.py` | 正确的5策略对比回测 (871行) | MacStudio运行中 |
| `run_backtest_v3.py` | 简化版滚动回测 (395行) | MacStudio运行中 |
| `run_quick_backtest_fast.py` | 优化版快速回测 | 已完成，效果差 |
| `run_optimized_training.py` | LightGBM ML训练 | 已完成，IC不稳定 |
| `factor_ic_scores.csv` | 56个因子的IC评分 | 已完成 |
| `src/data/extended_factors.py` | 55个技术因子计算器 | 可用 |
| `src/ml/model_trainer.py` | ML模型训练框架 | 可用 |
| `src/analysis/ic_analyzer.py` | IC分析器 | 可用 |
| `README_original.md` | 原始策略文档 | 参考 |
| `docs/PRODUCTION_CHECKLIST.md` | 生产验收清单 | 参考 |

---

## 七、MacStudio 连接信息
- **地址**: macstudio@192.168.0.88
- **密码**: 918kangbing
- **Python**: `/Users/macstudio/.pyenv/versions/3.12.12/bin/python3`
- **项目目录**: `~/112/pythontest/factor-strategy/`
