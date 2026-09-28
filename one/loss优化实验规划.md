# one_v2 loss 优化实验规划

## 目标与评估指标

- **目标**：最大化 5 个 close 时间窗口（1/3/7/31/127）的**截面 IC / ICIR**。
- **截面定义**：同一交易日内的 Spearman rank correlation（预测 vs 真实标签）。
- **辅助头**（close_volume_delta 128×3）只作正则，不直接考核。
- **评估**：需要补一个 `evaluate.py`（对齐 three/seven/zero 口径）：按 date 分组算截面 IC，输出每个 horizon 的 IC/ICIR + mean IC。

## 现状（baseline）

- 目标标准化：**全局 z-score**（全训练集 μ/σ）。
- close loss：`asymmetric_loss = MSE + (1 - batch_cosine) * 31`
  - 问题①：`31` 是配合旧"标签 ×31"量纲凑的魔数，现在无依据；
  - 问题②：`batch_cosine` 是跨日期的伪 IC（混了截面 + 时序 + 市场因子）；
  - 问题③：`penalty_ratio=1.0`，实际是纯 MSE，非对称是摆设。
- aux loss：MSE（z-scored）。

## 核心洞察

> **若目标按日期截面标准化（同日去均值、除当日 std），则 `MSE = 2 - 2·Pearson`，即纯 MSE 直接等价于截面 IC。**

全局 z-score 只去全局均值，残留"当日市场因子"（大盘普涨日所有股票标签偏高），MSE 会浪费容量拟合市场涨跌而非截面排序。按日期截面标准化后，目标就是"当日相对强弱"，MSE 才真正对齐 IC。

## 设计空间（四个旋钮）

### 旋钮 1：目标标准化（数据层）
| 方案 | 说明 | 预期 |
|---|---|---|
| A1 全局 z-score | 全训练集 μ/σ（当前）| 残留市场因子 |
| A2 按日期截面 z-score | 按 date 分组 μ/σ | **MSE = 2-2·Pearson，对齐截面 IC** |

### 旋钮 2：close loss 函数
| 方案 | 说明 | 备注 |
|---|---|---|
| B1 纯 MSE | 最简 | 干净基线 |
| B2 非对称 MSE | penalty_ratio > 1（惩罚高估）| 对卖出信号更有意义 |
| B3 Huber | 抗异常值 | z-score+clip 后可能多余 |
| B4 显式截面 Pearson | 按 date 分组算 corr 平均 | 需 date 进 batch，重；A2 够好则可跳过 |

### 旋钮 3：跨 horizon 加权
| 方案 | 说明 | 参考 |
|---|---|---|
| C1 等权 | 5 个 horizon 直接求和 | 默认 |
| C2 逆方差加权 | 每个 horizon × 1/σ² | seven 结论最优 |
| C3 手动 per-horizon 权重 | 手调各 horizon | seven 结论有害，慎用 |

### 旋钮 4：aux 权重
| 方案 | 说明 |
|---|---|
| D1 等权 | aux MSE 直接加 |
| D2 下权重 λ ∈ {0.1, 0.5} | aux 只作轻正则 |

## 推荐实验顺序

1. **A1 + B1 + C1 + D1**：全局 z-score 纯 MSE，删 IC 项 —— 立干净 baseline。
2. **A2 + B1 + C1 + D1**：按日期截面 z-score 纯 MSE —— 预期 IC 提升（去市场因子）。
3. 在 2 基础上扫 **C2（逆方差加权）** vs C1。
4. 在 2 基础上扫 **D2（aux 下权重）**。
5. 可选：**B2 非对称**（关心高估惩罚时）、**B4 显式截面 corr**（A2 已够好则跳过）。

## 实现依赖

- **A2 按日期截面标准化**：`compute_target_stats.py` 从 h5 文件名解析 `date`（`{symbol}_{date}.h5`），按 date 分组统计每个 close horizon 的截面 μ/σ，存成 `date -> stats` 的映射；标准化时用样本所属 date 的统计。（aux 保持全局 z-score 即可。）
- **B4 显式截面 corr**：Dataset 需额外返回 `date`，loss 内按 date 分组；受 batch 内每 date 样本数（通常 1~3）限制，噪声大，非首选。
- **C2 逆方差**：在 `compute_target_stats.py` 里顺带存每个 horizon 的全局 σ，loss 里 × 1/σ²（注意与 z-score 已除 σ 的区别，避免重复除）。

## 记录模板

每个实验固定记录：方案标签（A/B/C/D 组合）、5 个 horizon 的 IC 与 ICIR、mean IC、训练日志文件、结论一句话。
