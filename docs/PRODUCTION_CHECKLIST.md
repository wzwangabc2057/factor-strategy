# 生产环境验收清单 v3.0

## 概述

本文档描述了将v3激进版回测脚本升级到可实盘运营工程骨架的验收标准。

---

## 变更摘要

### 新增模块

| 模块 | 文件 | 功能 |
|------|------|------|
| 配置管理 | `config/strategy_params.yaml` | 策略参数配置 |
| | `config/risk_control.yaml` | 三档风控配置 |
| | `config/cost_model.yaml` | 交易成本模型配置 |
| 成本模型 | `src/backtest/cost_model.py` | 交易成本计算(佣金+滑点+冲击成本+价差) |
| 风控模块 | `src/risk/tier_manager.py` | 三档分层风控(Tier1/2/3) |
| 报告模块 | `src/reporting/metrics.py` | 指标计算与报告生成 |
| 鲁棒性测试 | `scripts/run_robustness_test.py` | Walk-Forward + 参数敏感性测试 |
| 生产回测 | `v2/run_backtest_production.py` | 集成所有模块的生产版回测 |
| 单元测试 | `tests/test_production.py` | 红线项验证测试 |

### 核心改进

1. **交易成本模型**
   - 佣金: 买入0.026%, 卖出0.126%
   - 滑点: 默认0.1% (可配置)
   - 冲击成本: `k * sqrt(trade_value / ADV)`
   - 买卖价差: 按市值分档

2. **三档风控机制**
   - Tier1(趋势): 满仓运行, max_weight=12%
   - Tier2(震荡): 85%仓位, max_weight=8%
   - Tier3(急跌): 50%仓位, 禁止买入, max_weight=5%

3. **换手限制**
   - 月换手上限8%
   - 超限自动缩放交易

---

## 运行命令

### 单次回测

```bash
# 使用默认配置
python v2/run_backtest_production.py

# 指定策略类型
python v2/run_backtest_production.py --strategy aggressive
python v2/run_backtest_production.py --strategy stable

# 覆盖滑点
python v2/run_backtest_production.py --slippage 0.002

# 指定输出目录
python v2/run_backtest_production.py --output results/my_test/
```

### 鲁棒性测试

```bash
# 完整测试
python scripts/run_robustness_test.py

# 快速模式 (小参数网格)
python scripts/run_robustness_test.py --fast

# 只跑Walk-Forward
python scripts/run_robustness_test.py --walk-forward-only
```

### 单元测试

```bash
# 运行所有测试
pytest tests/test_production.py -v

# 运行特定测试
pytest tests/test_production.py::TestWeightsNormalization -v
pytest tests/test_production.py::TestMaxSingleWeight -v
pytest tests/test_production.py::TestRiskTierDetection -v
```

---

## 验收 Checklist

### 必须通过的测试 (红线项)

| # | 检查项 | 验收标准 | 测试方法 |
|---|--------|----------|----------|
| 1 | 权重归一化 | sum(weights) = 1.0 ± 1e-4 | `pytest tests/test_production.py::TestWeightsNormalization` |
| 2 | 单股权重上限 | weight <= max_single_weight | `pytest tests/test_production.py::TestMaxSingleWeight` |
| 3 | 换手限制生效 | monthly_turnover <= 8% | `pytest tests/test_production.py::TestTurnoverLimit` |
| 4 | 滑点必须>=0.1% | slippage >= 0.001 | `pytest tests/test_production.py::TestSlippageRequired` |
| 5 | 无未来函数 | rolling窗口无未来数据 | `pytest tests/test_production.py::TestNoLookahead` |
| 6 | 风控档位检测 | 正确识别Tier1/2/3 | `pytest tests/test_production.py::TestRiskTierDetection` |
| 7 | 成本计算正确 | 卖出成本>买入成本 | `pytest tests/test_production.py::TestCostModel` |
| 8 | 配置可加载 | YAML配置正确解析 | 运行生产回测无报错 |
| 9 | 结果可复现 | 固定随机种子结果一致 | 相同参数运行两次结果相同 |
| 10 | 报告可生成 | JSON + Markdown输出正常 | 检查results目录输出 |

### 功能验证

| # | 检查项 | 验收标准 |
|---|--------|----------|
| 11 | Walk-Forward测试 | WF-1~WF-6至少5个通过 |
| 12 | 参数敏感性 | Top加成10%-50%结果稳定 |
| 13 | 成本敏感性 | 滑点0.05%-0.3%收益变化<2% |
| 14 | Tier3触发记录 | 报告中记录触发日期和次数 |
| 15 | 冲击成本降级 | 无ADV时标记fallback |

### 代码质量

| # | 检查项 | 验收标准 |
|---|--------|----------|
| 16 | 无语法错误 | `python -m py_compile` 通过 |
| 17 | 导入正确 | 所有模块可正确导入 |
| 18 | 配置默认值 | 无配置时可使用合理默认值 |
| 19 | 错误处理 | 异常情况有日志记录 |
| 20 | 向后兼容 | 不破坏现有回测脚本 |

---

## 红线项 (阻断上线)

以下任一项不通过即**禁止上线**:

1. ❌ 权重归一化失败 (sum ≠ 1.0)
2. ❌ 单股权重超限
3. ❌ 换手限制无效 (月换手>10%)
4. ❌ 滑点<0.1%
5. ❌ 存在未来函数
6. ❌ 风控Tier3不触发急跌
7. ❌ 成本计算为0
8. ❌ 报告无法生成
9. ❌ 结果不可复现
10. ❌ 配置文件无法加载

---

## 最小数据缺口清单

以下字段缺失时需要降级处理:

| 字段 | 来源 | 影响 | 降级方案 |
|------|------|------|----------|
| `amount` | ClickHouse | 冲击成本 | impact_cost=0, 标记fallback |
| `index close` | ClickHouse | 风控检测 | 使用股票均价代理 |
| `industry` | AkShare | 行业暴露 | 跳过行业集中度监控 |
| `suspend_status` | - | 停牌过滤 | 预留接口,暂不实现 |
| `limit_up/down` | - | 涨跌停过滤 | 预留接口,暂不实现 |

---

## 回测结果 (2020-2024)

| 策略 | 年化收益 | 最大回撤 | 夏普比率 | 月换手 |
|------|----------|----------|----------|--------|
| R5-v3 | ~25% | ~-23% | ~1.3 | ~4% |

*注: 实际结果以运行输出为准*

---

## 后续优化方向

1. **因子IC实时计算** - 需要历史因子数据
2. **行业暴露监控** - 需要行业分类数据
3. **停牌/涨跌停处理** - 需要状态数据
4. **实盘交易接口** - 需要券商API

---

## 签字确认

| 角色 | 姓名 | 日期 | 签字 |
|------|------|------|------|
| 量化开发 | | | |
| 风控负责人 | | | |
| 投研负责人 | | | |

---

*文档版本: 3.0.0*
*更新时间: 2026-02-15*
