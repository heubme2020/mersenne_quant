# indicator 42 列「删哪 3 个」决策 —— IG 特征重要度实验

> 记录 zero 改造要往 indicator 里加 3 个新 ratio,需要从原 42 列里删 3 个腾位置。
> 本文讲清楚:① 两种「删哪个」的判据;② IG(Integrated Gradients)是什么、为什么看 mean-baseline;
> ③ three 和 seven 两个模型的 IG 实验与交叉验证;④ 最终删哪 3 个的结论。

---

## 0. 一句话结论

用 IG 对 three / seven 两个模型分别算 127 特征的归因,发现「最弱的 indicator 列」在**两个模型里高度一致**,都是:

```
totalEquityUnit / accountsReceivablesUnit / inventoryUnit
```

所以删这 3 个(而不是一开始按「相关性冗余」删的 `quickRatio / totalAssetsUnit / totalLiabilitiesUnit`——那几个在 three 里其实排中游,模型在用)。

---

## 1. 背景:为什么必须在 indicator 里删 3 个

- zero 改造要新增 3 个 ratio:`ocfToLiab`(现金流/总负债)、`goodwillToEquity`(商誉/净资产)、`accrual`(净利−现金流背离)。
- indicator 表是 three / seven 模型的 127 维输入的一部分:
  - **127 = 42(indicator 全列)+ 85(raw 财务列)**,是刻意设计的 2^7−1 梅森素数,**硬约束**。
  - three / seven 的 `gen_train_data.py` 是**无条件全量 merge indicator 的 42 列**,不经过 `features_importance.csv` 过滤。
- 所以要「加 3 个新列」必须「等量删 3 个旧列」,否则 indicator 变 45 列 → 127 变 130 → `SEVEN([127,31])` 的 Linear 维度崩。

---

## 2. 概念:两种「删哪个」的判据

| 判据 | 思路 | 优点 | 缺点 |
|---|---|---|---|
| **相关性冗余** | 删「信息在别处有副本」的列(如 `quickRatio`↔`cashRatio` corr 0.99) | 副本还在,零信息损失 | 副本关系不是 100% 时,仍会丢掉细微信号 |
| **特征重要度(IG)** | 删「模型实际不用」的列 | 直接度量「删了损失多少性能」 | 重要度是**模型特定**的,three 和 seven 可能不一样 |

**关键分歧**:一个「冗余但被模型在用」的列(quickRatio 相关性 0.99 但 three IG 排第 10),到底该删还是该留?

- 冗余视角:删了没关系,信号由副本(cashRatio)继续承载。
- 重要度视角:它其实被模型用着(虽然冗余),不如删「真正的、模型完全不用的」列。

本文用实验回答:到底哪 3 个是「模型真正不用的」,且在 three / seven 两个模型间一致。

---

## 3. 概念:Integrated Gradients(IG)

**归因方法(attribution)**:给一个已训练模型和一个输入样本,算出「每个输入特征对最终输出贡献了多少」。

IG 的公式(对特征 i):

```
IG_i(x) = (x_i − baseline_i) × ∫₀¹  ∂F(baseline + α·(x − baseline)) / ∂x_i  dα
```

直觉:从 **baseline** 走到输入 x 的路上,对梯度做积分,把输出的「变化」按路径分配给每个特征。绝对值越大 = 该特征越重要。

### 3.1 为什么必须看 mean-baseline(不看 zero-baseline)

IG 的 baseline 选什么,会直接影响归因量级:

- **zero-baseline**(baseline=0):归因量级 ∝ 特征本身的数值大小。
  - 前 42 个 indicator 列**没做 z-score**(是原始比值,如 debtRatio≈0.5、netAssetValuePerShare≈几百)。
  - 后 85 个 raw 列**做了 z-score**(≈N(0,1))。
  - 两组量级天差地别,zero-baseline 下**跨组比较不公平**。
- **mean-baseline**(baseline=各特征在本样本的均值):归因 ≈ |梯度| × 特征自身波动幅度,**跨组可比**。

实测验证(three 排名里):`totalEquityUnit` 在 zero-baseline 排第 12,在 mean-baseline 排第 **116**——一个近似常数的列,zero-baseline 会严重高估它。所以**跨组比较(哪个 indicator 列弱)必须看 `imp_mean` 列**。

### 3.2 three 和 seven 的任务不同,IG 不能通用

| | three | seven |
|---|---|---|
| 预测目标 | `growth/death`(**价格动量**) | `fcf/dividend/netasset`(**未来基本面**) |
| 期限 | 3 季度 | 3/7/31 季度 |
| 输出 | 2 个头 × 3 期限 | 3 个分支 × 3 期限 |

所以「某特征对 three 没用」**不代表**「对 seven 也没用」。比如 `accountsReceivablesUnit`(应收/净资产)对价格涨跌可能没信号,但应收账款变动直接进经营现金流 → fcf,理论上 seven 可能用。这正是本次要做**双模型交叉验证**的原因。

---

## 4. 实验方法

### 4.1 three(2026-08-28 已跑,本次复用)

- 模型:`three.pt`;脚本:`three/get_features_importance.py`(已修 stale bug)。
- 输入:`train/` 的 h5,形状 (31, 133)= 127 特征 + 6 label。
- 归因:`growth`、`death` 两个头分别归因(取 3Q 期限),零/均值两种 baseline,127 轮 × 127 样本,IG n_steps=32。
- 输出:`three/features_ranking_127.csv`。

### 4.2 seven(2026-09-01 本次跑)

- 模型:`seven.pt`;脚本:`seven/get_features_importance.py`(本次从 three 版适配,修掉了旧脚本三处 stale:写死 THREE/[118,31]/6 输出)。
- 输入:`train/` 的 h5,形状 (31, 136)= 127 特征 + 9 label。
- 归因:`fcf`、`dividend`、`netasset` 三个分支**分别归因**(取 7Q 期限),零/均值两种 baseline,127 轮 × 127 样本,IG n_steps=32,最后三分支平均。
- 输出:`seven/features_ranking_127.csv`。

---

## 5. 实验结果

### 5.1 three 的 indicator 42 列中最弱的 4 个(imp_mean)

| 排名(组内) | 特征 | imp_mean |
|---|---|---|
| 39 | netReceivablesUnit | 0.00019 |
| 40 | inventoryUnit | 0.00018 |
| 41 | accountsReceivablesUnit | 0.00015 |
| **42** | **totalEquityUnit** | **0.00008** |

(对照组:被最初按「冗余」删掉的 `quickRatio` 排第 10、`totalLiabilitiesUnit` 排 14、`totalAssetsUnit` 排 20——**都是中游,three 在用**。)

### 5.2 seven 的 indicator 42 列中最弱的 10 个(imp_mean)

| 排名(组内) | 特征 | imp_mean |
|---|---|---|
| 33 | goodwillToEquity ★新 | 0.000390 |
| 34 | debtRatio | 0.000367 |
| 35 | accountPayablesUnit | 0.000364 |
| 36 | propertyPlantEquipmentNetUnit | 0.000278 |
| 37 | netReceivablesUnit | 0.000250 |
| 38 | inventoryUnit | 0.000248 |
| 39 | shortTermDebtUnit | 0.000241 |
| 40 | accrual ★新 | 0.000198 |
| 41 | accountsReceivablesUnit | 0.000119 |
| **42** | **totalEquityUnit** | **0.000053** |

### 5.3 交叉验证:两个模型「最弱列」高度一致

| 特征 | three 组内排名 | seven 组内排名 | 结论 |
|---|---|---|---|
| **totalEquityUnit** | 42/42 | **42/42** | 双垫底,且近似常数(81% 落在 1±0.05) |
| **accountsReceivablesUnit** | 41/42 | **41/42** | 双双倒数第二 |
| **inventoryUnit** | 40/42 | **38/42** | 双双垫底区 |

> 之前担心的「seven 会用 AR/存货」被数据否定:它们在 seven 里同样垫底。

### 5.4 新加的 3 个 ratio 在 seven 里的表现(分化明显)

| 新特征 | seven 组内排名 | imp_mean | 评价 |
|---|---|---|---|
| ocfToLiab | 26/42 | 0.000667 | 对 seven 有用(中上) |
| goodwillToEquity | 33/42 | 0.000390 | 中等 |
| **accrual** | **40/42** | 0.000198 | **seven 几乎不用** |

`accrual`(净利−现金流背离)是 **zero 专属**的避雷信号(对 zero 的 negMdd ICIR 2.8),对 seven 预测基本面基本无用;`ocfToLiab`、`goodwillToEquity` 则对整个链都有用。

---

## 6. 结论与决策

1. **删 `totalEquityUnit` + `accountsReceivablesUnit` + `inventoryUnit`**(IG 最弱 3 个),在 three 和 seven 之间交叉验证成立,删了对两个模型都几乎零损失。
2. **恢复** `quickRatio / totalAssetsUnit / totalLiabilitiesUnit`(它们虽是冗余列,但 three 在用,删了会丢细微信号;留副本不如留原列)。
3. `ocfToLiab / goodwillToEquity / accrual` 三个新 ratio 保留(对 zero 都是强避雷信号;其中 accrual 是 zero 专属)。

---

## 附:可复现命令

```bash
# three(已跑过,结果在 three/features_ranking_127.csv)
cd three && python get_features_importance.py --model three.pt

# seven(本次跑)
cd seven && python get_features_importance.py --model seven.pt --out features_ranking_127.csv

# 重生成 indicator(遍历 data/ 全部交易所)
python -c "from write_stock_data import refresh_indicator; refresh_indicator()"
```
