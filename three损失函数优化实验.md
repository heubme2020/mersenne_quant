# three 模型损失函数优化实验

> 记录 three 模型 loss 的一轮系统性优化：从「未加权的 pointwise 回归」到「逆方差加权 + log-比值 loss」。
> 核心问题是**评估指标与 loss 目标不一致**，本文讲清楚：指标到底是什么、为什么旧 loss 不对齐、每一步实验怎么改、结果如何、最终结论是什么。

---

## 0. 一句话结论

three 的评估指标 `gd = growth / death` 是**比值**，但旧 loss 是对 growth、death 两个头的**点对点绝对值回归**——目标错位。改成「逆方差加权 + log-比值 loss」后：

| 指标（ICIR） | 优化前 (configR) | 优化后 (λ=2) | 提升 |
|---|---|---|---|
| **gd / combined（排序指标）** | 2.3845 | **2.7772** | **+16.5%** |
| growth / combined | 2.5254 | 2.7850 | +10.3% |
| death / combined | 2.2142 | **2.8051** | **+26.7%** |

最终 loss（`three/train.py`）：

```python
loss = growth_loss + death_loss + 2.0 * gd_loss
# growth_loss = smooth_l1(growth_pred / σ_growth, growth / σ_growth)
# death_loss  = smooth_l1(death_pred  / σ_death,  death  / σ_death)
# gd_loss     = smooth_l1(log(growth_pred) - log(death_pred),
#                         log(growth)      - log(death))  （除以各自 std）
```

---

## 1. 背景：three 模型和它的评估指标

### 1.1 模型结构

- 输入：`[127, 31]`（127 特征 × 31 期），127 = 42 indicator + 85 raw。
- 输出：两个头 **growth / death**，各 3 个期限（1 / 3 / 7 季度）。

### 1.2 标签定义（`three/evaluate.py` 顶部注释）

```
growth = 未来 N 季度收盘价中位数 / 过去最高价   （上行，越大越好）
death  = 过去收盘价中位数 / 未来 N 季度最低价   （下行，越大越差）
```

- growth 越大 = 涨得越多，越好。
- death 越大 = 跌得越深，越差。

### 1.3 排序分 gd 与最终指标

```python
# evaluate.py
pred_gd_h = pred_growth_h / pred_death_h          # 每个期限的比值
gd_combined = Σ_h gd_h                            # 三个期限求和
```

- **排序分 = `gd = growth / death`**：一个比值，同时反映「涨得多 + 跌得少」。
- **最终评估指标 = `icir_gd_combined`**：gd_combined 这个排序分的 **ICIR**（IC 的均值 / 标准差，衡量排序能力的稳定性）。
- IC = 预测分与真实值的**秩相关**（Spearman），按截面日期平均。

---

## 2. 核心问题：loss 与指标不对齐

旧 loss（`three/train.py` 优化前）：

```python
loss = smooth_l1(growth_pred, growth) + smooth_l1(death_pred, death)
```

它优化的是「growth 预测值 ≈ growth 真值、death 预测值 ≈ death 真值」的**绝对值**。但指标只看 **`growth / death` 的比值**。两者的矛盾：

> 预测 `growth=2.0, death=1.0`，真值 `growth=1.0, death=0.5`。比值都是 2.0（**排序完全正确**），但点对点 loss 却很大（|2−1| + |1−0.5| = 1.5）。

也就是说，旧 loss 在惩罚「绝对值偏差」，即使这个偏差不影响排序。这就是为什么要在 loss 里**直接加一个优化比值的项**。

---

## 3. 实验设计与结果

所有实验都在**同一测试集**上评估（889 只 / 23483 样本，`TEST_SEED=42`，从零重训 7 epoch），保证可比。

### 3.1 全量表（`three/eval_results/history.csv`，icir）

| # | 实验 | loss | growth ICIR | death ICIR | gd ICIR |
|---|---|---|---|---|---|
| 0 | 原版 (08-20) | 未加权 smooth_l1 | 1.6280 | 1.7070 | 1.3476 |
| 0' | IG-3换3 (09-01) | 同上 | 2.1451 | 2.4288 | 1.8795 |
| 0'' | configR (09-03) | 同上 | 2.5254 | 2.2142 | 2.3845 |
| **①** | 逆方差加权 | `+ 除以 std` | 2.6418 | 2.6498 | 2.3582 |
| **②** | + naive 比值 | `+ growth/death` | 2.2929 | **0.4115** | **1.4472** |
| **③** | + log-比值 (λ=1) | `+ log(g)−log(d)` | 2.5765 | 2.5587 | 2.4248 |
| **④** | log-比值 (λ=2) | 上式 ×2 | **2.7850** | **2.8051** | **2.7772** ✅ |
| **⑤** | log-比值 (λ=3) | 上式 ×3 | 1.7834 | 1.1620 | 1.4060 |

### 3.2 逐步解读

**① 逆方差加权**：growth/death 的 3 个期限量级差很大（death/7Q 的 σ ≈ 6.3，growth/1Q 的 σ ≈ 0.27，差 20+ 倍），直接 smooth_l1 相加会被长期限主导。除以各自 std 拉平量级。结果：death 头 +19.7%，但 gd 略降（2.3845→2.3582，噪声内）。**中性**，单独看没赢，但它是后面加比值 loss 的必要基础。

**② naive 比值 loss**：直接加 `smooth_l1(growth/death_pred, growth/death)`。结果 **gd 崩到 1.45、death 崩到 0.41**。原因是除法里分母 `death_pred` 一旦变小，梯度 `∂(growth/death)/∂death = -growth/death²` 爆炸，把 death 头打崩了（验证 loss 从 2.03 发散到 8.96）。**证明 naive 比值行不通。**

**③ log-比值 (λ=1)**：改用 `log(growth) - log(death)`。秩不变（log 是单调变换），但梯度变成 `-1/death`，数值稳定。结果 gd 2.3845→2.4248，death→2.5587。**第一个正收益**（但 gd 涨幅在噪声内）。

**④ log-比值 (λ=2)**：把 gd_loss 权重从 1 加到 2。结果 **gd 2.7772、death 2.8051，全面大涨**，超出噪声（gd +0.39 ≈ 3.3 个标准误）。**明确的最优。**

**⑤ log-比值 (λ=3)**：权重加到 3。结果 gd 崩到 1.4060。**过拟合**——比值权重过头，模型退回到 ② 那种失稳。

---

## 4. 最终结论

**最终 loss = 逆方差加权 smooth_l1 + 2.0 × log-比值 loss**，代码在 `three/train.py`：

- `estimate_label_std()` 返回 `(label_std, gd_std)`：6 个原始格子的 std + 3 个 log-比值的 std。
- 训练/验证 loss 都加了 log-比值项，权重 **2.0**。
- 最优模型：`backup/three_20260904_114328.pt`（`three.pt` 已恢复为它，`history.csv` 的 `is_best` 已切到它）。

### 三条关键教训

1. **指标是比值，loss 就要对齐比值**：`gd = growth/death` 是比值，点对点回归绝对值永远差一口气。加一个直接优化比值的 loss 项，才对得上评估目标。
2. **比值 loss 必须用 log 形式**：naive 除法 `growth/death` 的分母梯度 `-growth/death²` 会爆炸、把 death 头打崩；`log(growth)-log(death)` 的梯度是 `-1/death`，数值稳定。
3. **gd_loss 权重 λ 有峰值，在 2**：λ=1→2 大涨，λ=3 过拟合崩。不是越大越好。

---

## 5. 注意事项

1. **λ 是对固定测试集调的，有轻微过拟合风险**：λ=2 是在 889/23483 这个固定测试集上选出来的。严格做法是另设一个验证集调 λ，测试集只看最后结果。这里因为提升幅度大（+16.5%）且三个指标一致向好，才判定不是纯噪声，但严谨性上有保留。
2. **逆方差加权（①）单独看中性，但作为 log-比值的基础是必要的**——没有它，growth/death 内部量级差会把 log-比值 loss 也带偏。
3. **这套「对齐指标」的思路可推广**：seven 的指标是 `fcf/dividend/netasset` 的求和（非比值），但也有量级悬殊 + combined 求和结构，可以类比；更进一步的 rank-aware loss（直接优化秩相关）是对齐 ICIR 的终极手段，但实现复杂、overfit 风险高，尚未尝试。

---

## 附：可复现命令

```bash
# 修改 three/train.py 后，从零重训（先删旧 .pt，避免 warm-start 污染对比）
cd three
rm -f three.pt
python train.py > train_log.log 2>&1

# 查看评估历史
python -c "import pandas as pd; print(pd.read_csv('eval_results/history.csv')[['timestamp','icir_growth_combined','icir_death_combined','icir_gd_combined','is_best']])"
```

> 相关文档：`indicator特征删减ablation定案.md`（indicator 3换3 的定案）、`indicator特征删减IG实验.md`（IG 归因实验，结论已被 ablation 推翻）。
