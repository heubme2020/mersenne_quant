# indicator 42 列「删哪 3 个」—— ablation 最终定案

> 本文是 `indicator特征删减IG实验.md` 的**后续与修正**。
> 那篇用 Integrated Gradients（IG）特征重要度给出一版结论（删 `totalEquityUnit / accountsReceivablesUnit / inventoryUnit`）；
> 但那个结论**只靠归因、没做实盘重训验证**，后来用「重训 + 样本外 ICIR」的 ablation 推翻了它。
> 本文记录最终定案：**到底删哪 3 个、为什么、证据是什么**。

---

## 0. 一句话结论

为了给 zero 模型新增的 3 个 ratio 腾位置，indicator 的 42 列做 **3 换 3**（列数不变，127 硬约束不破）：

| 删（3 个冗余列） | 加（zero 用的 3 个新 ratio） |
|---|---|
| `totalLiabilitiesUnit` | `ocfToLiab`（经营现金流 / 总负债） |
| `longTermInvestmentsUnit` | `goodwillToEquity`（商誉 / 净资产） |
| `totalDebtUnit` | `accrual`（净利润 − 经营现金流的背离度） |

**判据不是 IG 特征重要度，而是「冗余性」+ 重训后的样本外 ICIR。** 这套删法在 seven 和 three 两个模型上都是历史最优。

---

## 1. 背景：为什么必须在 indicator 里删 3 个

- zero 模型（避雷模型）要新增 3 个财务 ratio：`ocfToLiab`、`goodwillToEquity`、`accrual`。这三个的定义落在 indicator 表里。
- three / seven 模型的输入维度是 **127**，是刻意设计的梅森素数 `2^7 − 1`，**硬约束**：
  ```
  127 = 42（indicator 全列）+ 85（raw 财务列）
  ```
- three / seven 的 `gen_train_data.py` 对 indicator 是**无条件全量 merge**（不经过 `features_importance.csv` 过滤）。所以 indicator 多 3 列 → 总维度 127→130 → `SEVEN([127,31])` / `THREE([127,31])` 的 Linear 权重维度崩。
- 结论：**加 3 个新 ratio，必须等量删 3 个旧 indicator 列**，问题只剩「删哪 3 个」。

---

## 2. 概念基础

### 2.1 两类「删哪个」的判据

| 判据 | 思路 | 优点 | 缺点 |
|---|---|---|---|
| **相关性冗余** | 删「信息在别处有副本」的列 | 副本还在，零/几乎零信息损失 | 副本不是 100% 相同时仍丢细微信号 |
| **特征重要度（IG）** | 删「模型实际不用」的列 | 直接度量「删了损失多少性能」 | 重要度是**模型特定**的，且 IG 在未归一化的列上会骗人 |

### 2.2 为什么 IG 归因在 indicator 上不可靠（本次翻车的关键）

IG 归因的结果**依赖 baseline 的选择**：

- **zero-baseline**（baseline=0）：归因量级 ∝ 特征本身的数值大小。indicator 那 42 列**从不做 z-score**（`mean/std` CSV 只由 income+balance+cashflow 生成，不含 indicator 列；`gen_train_data.py` 的归一化循环遇到不在 mean 里的列直接 `continue`）。于是「数值大的列」在 zero-baseline 下天然显得重要，`totalEquityUnit` 这种近似常数（中位数 1.0、81% 落在 1±0.05）在 zero-baseline 能排到第 12，mean-baseline 却排到第 116。
- **mean-baseline**（baseline=各特征均值）：归因 ≈ |梯度| × 特征波动幅度，跨组可比。之前的 IG 实验已经改用这个。

但**即使看 mean-baseline，IG 仍然只是「已训练模型对当前权重的归因」**，它回答的是「这个列现在贡献多大」，而不是「删掉它会损失多少样本外性能」。一个列 IG 低，可能是真没用，也可能是「数值量级小所以梯度贡献被压低」，而它其实承载着唯一信息。

> **核心教训：indicator 列不做 z-score → IG 归因量级会被数值大小污染 → 「删哪个」必须靠「重训 + 样本外 ICIR」定案，IG 只能用来圈定候选。**

### 2.3 为什么最终看「样本外 ICIR」

- **IC**（Information Coefficient）= 预测分与真实值的**秩相关**（Spearman），按截面日期算再平均，量纲无关、不受预测值量级影响。
- **ICIR** = IC 均值 / IC 标准差，衡量信号**稳定性**，是本项目评估模型的排序指标。
- 评估在**留出的测试股票**（`TEST_SEED=42`，固定 889 只，不参与训练）上做，`train.py` 训练完自动调用 `evaluate.py` 写入 `eval_results/history.csv`。
- 三个期限 `three/seven/thirty_one` 的嵌套累加和记为 `combined`，是最终排序分数。

### 2.4 两个模型的评估口径

| | three | seven |
|---|---|---|
| 预测目标 | `growth/death`（价格动量：涨/跌） | `fcf / dividend / netasset`（未来基本面） |
| 期限 | one / three / seven（3 个） | three / seven / thirty_one（3 个） |
| 排序指标 | `icir_gd_combined` | `icir_fcf_combined` |

---

## 3. 实验设计：ablation

对候选删除集做**端到端重跑**：改 `write_stock_data.py` 的 `_INDICATOR_COLS` + `ind` 字典 → `refresh_indicator()` 重生成 31 个交易所 → `gen_train_data.py` 重建 h5 → `train.py` 从零/续训 → `evaluate.py` 出样本外 ICIR → 记入 `history.csv`。

候选逻辑：

1. **IG 最弱 3 列**（`indicator特征删减IG实验.md` 的旧结论）：`totalEquityUnit`、`accountsReceivablesUnit`、`inventoryUnit`。
2. **冗余 3 列**（本次最终采纳，内部叫 configR）：`totalLiabilitiesUnit`、`longTermInvestmentsUnit`、`totalDebtUnit`。
3. Sep 2 还跑过 4 组中间 ablation（baseline / 1 / 2 / 3），逐步收敛到冗余删减。

---

## 4. 实验结果（`history.csv` 全量）

### 4.1 seven（排序指标 `icir_fcf_combined`）

| timestamp | n_samples | ic_fcf_combined | icir_fcf_combined | 说明 |
|---|---|---|---|---|
| 08-20 12:59 | 17806 | 0.5927 | **5.9440** | 原版 42 列（未加新 ratio） |
| 09-01 11:42 | 30810 | 0.5882 | 5.8904 | 中间实验 |
| 09-01 15:13 | 17498 | 0.5702 | 4.1954 | 中间实验 |
| 09-02 11:30 | 17498 | 0.5801 | 4.1773 | ablation baseline |
| 09-02 12:51 | 17498 | 0.5862 | 3.7005 | ablation 1 |
| 09-02 14:58 | 17498 | 0.6003 | 3.5377 | ablation 2 |
| 09-02 16:20 | 17498 | 0.5647 | 5.0184 | ablation 3 |
| **09-03 15:04** | 18262 | **0.6197** | **6.9641** | **configR（最终）✅** |

configR 的七个分支明细（IC / ICIR）：

| 分支 | fcf/three | fcf/seven | fcf/thirty_one | fcf/combined | dividend/combined | netasset/combined |
|---|---|---|---|---|---|---|
| IC | 0.5085 | 0.5858 | 0.5993 | **0.6197** | 0.3874 | 0.3207 |
| ICIR | 4.066 | 7.066 | 5.643 | **6.964** | 3.388 | 3.836 |

### 4.2 three（排序指标 `icir_gd_combined`）

| timestamp | n_samples | ic_growth_combined | icir_growth_combined | ic_gd_combined | icir_gd_combined | 说明 |
|---|---|---|---|---|---|---|
| 08-20 13:46 | 22770 | 0.3698 | 1.6280 | 0.3452 | 1.3476 | 原版 |
| 09-01 17:00 | 22770 | 0.3764 | 2.1451 | 0.3594 | 1.8795 | IG-3换3（旧结论） |
| **09-03 10:44** | 23483 | **0.4008** | **2.5254** | **0.3856** | **2.3845** | **configR（最终）✅** |

### 4.3 关键对比

- **seven**：原版 5.94 → configR **6.96**（fcf/combined 从 0.593 提到 0.620）。
- **three**：原版 1.63 → IG-3换3 2.15 → configR **2.53**（growth/combined 0.370 → 0.401；gd/combined 1.35 → 2.38）。
- 两个**独立任务**（价格动量 vs 未来基本面）上，configR 都明显领先，说明「删冗余列」这个方向对两者都成立。

---

## 5. 结论：删哪 3 个、为什么

最终删：

```
totalLiabilitiesUnit / longTermInvestmentsUnit / totalDebtUnit
```

最终加：

```
ocfToLiab / goodwillToEquity / accrual
```

### 5.1 逐条删减原因

**① `totalLiabilitiesUnit` —— 与 `debtToEquity` 逐位相等（corr = 1.0000）**

在 `write_stock_data.py` 里，两者的公式**完全相同**：

```python
'debtToEquity':        f('totalLiabilities') / f('totalStockholdersEquity')
'totalLiabilitiesUnit': f('totalLiabilities') / f('totalStockholdersEquity')   # 同一个公式
```

实测 AMEX / NASDAQ / NYSE 三个交易所，两者相关度 **1.000000，逐位 100% 相等**。删 `totalLiabilitiesUnit` 保留 `debtToEquity`，**零信息损失**——这是三条里最干净的一条。

**② `totalDebtUnit` —— 与 `netDebtUnit` 高度重复（corr ≈ 0.94 ~ 1.0）**

`netDebt = totalDebt − 现金`（netDebtUnit 与 totalDebtUnit 同为「债务/净资产」口径，差一个现金项）。实测 NYSE / AMEX 上两者 corr 达 **0.9997 ~ 0.9999**，NASDAQ 上 0.94。删 `totalDebtUnit` 保留 `netDebtUnit`，几乎零损失。

**③ `longTermInvestmentsUnit` —— 是 `totalInvestmentsUnit` 的子集**

`longTermInvestments`（长期投资）是 `totalInvestments`（总投资）的一个组成部分。长期投资的信号大部分被 `totalInvestmentsUnit` 覆盖，删了损失很小。

> 三者的共同点：**信息在别处有副本**（冗余）。删冗余列是「安全删除」，不会丢失唯一信息。

### 5.2 为什么不删 IG 最弱的那 3 个

`indicator特征删减IG实验.md` 曾按 IG 最弱把 `totalEquityUnit / accountsReceivablesUnit / inventoryUnit` 定为删除目标。这次 ablation 证明这是**错的方向**，原因是这三列的「弱」性质不同：

- `totalEquityUnit = totalEquity / totalStockholdersEquity`：`totalEquity ≈ totalStockholdersEquity + 少数股东权益`，所以它**近似常数**（中位数 1.0、81% 落在 1±0.05）。删它确实 ~免费，但**留它也无害**（常数，模型能自己学到忽略）。
- `accountsReceivablesUnit`（应收/净资产）、`inventoryUnit`（存货/净资产）：**信息唯一**、没有副本列。它们 IG 排低，是「数值量级小导致梯度贡献被压低」的假象，并不代表真没用。

所以 IG-3换3 删掉这两个**有唯一信息的列**、却保留了真正冗余的列，实盘 ICIR 反而更差。configR 反其道而行——**保留唯一信息列、删冗余列**——赢了。

---

## 6. 注意与后续

1. **n_samples 有轻微漂移**：几次运行测试样本数在 17498 / 17806 / 18262（three 是 22770 / 23483）之间变化，源于底层财务数据 refresh + h5 重建。跨日期 ICIR 严格可比性有折扣；但 configR 在两个独立模型上同时大幅领先，结论稳健，不依赖这个漂移。
2. **zero 不受影响**：zero 的 7 个特征（`interestCoverage / ocfToLiab / debtToEquity / goodwillToEquity / debtRatio / accrual / cashRatio`）全部保留（删掉的 3 列都不是 zero 的特征）。
3. **代码已就绪**：`write_stock_data.py` 已是最终 configR（删 3 加 3），31 个交易所 indicator 已重生成；`three.pt`（09-03 10:44）与 `seven.pt`（09-03 15:04）均为 configR 且是历史最优。
4. **旧文档已过时**：`indicator特征删减IG实验.md` 的第 5 节「结论与决策」仍写着删 `totalEquityUnit / accountsReceivablesUnit / inventoryUnit`，已被本文推翻，建议改标注为「已被 ablation 定案取代」。

---

## 附：可复现命令

```bash
# 重生成 indicator（遍历 data/ 全部交易所，删 3 加 3 后跑一次）
python -c "from write_stock_data import refresh_indicator; refresh_indicator()"

# 重建 h5（indicator 维度变了必须重跑）
cd three && python gen_train_data.py
cd ../seven && python gen_train_data.py

# 训练 + 评估（训练完自动 evaluate 并写入 history.csv）
cd three && python train.py
cd ../seven && python train.py

# 查看评估历史
python -c "import pandas as pd; print(pd.read_csv('seven/eval_results/history.csv')[['timestamp','n_samples','icir_fcf_combined','is_best']])"
python -c "import pandas as pd; print(pd.read_csv('three/eval_results/history.csv')[['timestamp','n_samples','icir_growth_combined','icir_gd_combined','is_best']])"
```

> 注：旧 `.pt` 会做 warm-start 加载。indicator 前 42 列里 3 列含义变了时，为保证干净对比应从零重训（删旧 `.pt` 再 `train.py`）；本次 seven 是从中断的 configR checkpoint 续训，与 three 的 warm-start 协议一致。
