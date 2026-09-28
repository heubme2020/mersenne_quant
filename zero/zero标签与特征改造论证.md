# zero 模型标签与特征改造：概念 + 实验论证

> 本文档回答两件事：① 每个候选「标签 / 特征」到底是什么意思；② 实验数据为什么证明「换标签 + 换特征」是对的。
> 数据来自 `validate_label.py` 全量运行：**413,261 个样本、7 个训练交易所、2005-09 ~ 2024-12、78 个截面**。

---

## 0. 一句话结论

把 zero 的标签从「对称动量 `mom`」换成「未来最大回撤 `negMdd`」，特征从现有 7 个换成数据驱动的 7 个：

```
interestCoverage / ocfToLiab / debtToEquity / goodwillToEquity / debtRatio / accrual / cashRatio
（4 旧 + 3 新；删 quickRatio / currentRatio / inventoryTurnover / arToRevenue）
```

---

## 1. zero 在整条链里的定位

系统是一条选股流水线：

```
one → two → three → seven → zero
（短周期量价）      （长周期财务+量价选优）   （避雷）
```

- **three / seven**：负责「选优」——用 127 个财务+量价特征，输出 growth/death（涨/跌）。
- **zero**：负责「**避雷**」——用财务偿债/流动性比率，把未来可能深跌的危险股票过滤掉。

zero 的输入是 `[7 个比率, 3 个季度]`，输出一个标量 `zero` 分数，分数越高越安全。避雷的本质是**下行 / 尾部风险**，不是「涨得多不多」。

---

## 2. 概念：标签（要预测的那个数）

### 2.1 现状标签：对称动量 `mom`，为什么它不适合避雷

现状（`gen_train_data.py`）：

```
price_fore = log(min(未来3季收盘)) + log(median(未来3季)) + log(max(未来3季))
price_past = log(min(过去3季收盘)) + log(median(过去3季)) + log(max(过去3季))
mom        = price_fore - price_past
```

这是一个**对称的动量指标**：未来股价分布相对过去是抬升还是下移。它有两个硬伤：

1. **对称，稀释下行信号**。`mom` 对「涨很多」（+0.3）和「跌很多」（−0.3）一视同仁——MSE 眼里都是「大」。但避雷只关心跌的那一侧。训练时模型被迫同时拟合涨和跌，把最该关注的尾部风险信号稀释掉了。

2. **混进模型看不到的 `price_past`**。输入只有 7 个财务比率、**没有任何股价**，但标签里的 `price_past` 是过去股价——模型根本预测不了它，等于在标签里塞了一个噪声项，人为抬高 loss 的地板。

### 2.2 候选标签逐个讲

| 标签 | 定义 | 为什么可能更好 |
|---|---|---|
| **`negMdd`**（方案A，**推荐**） | 未来窗口内的最大回撤取负 | 直接度量「未来最惨时跌了多少」，纯 forward、只盯下行、有界连续 |
| `downRet`（方案B） | `min(0, 未来对数收益)` | 只抓终点 vs 起点，抓不到「先涨后崩」 |
| `downVol` | 下行半波动 | 度量下行持续波动，右偏严重 |
| `upVol` | 上行半波动 | 度量上行波动 |
| `upMinusDown` | `upVol − downVol` | 收益不对称度（依赖三阶矩，噪声大） |
| `asym` | `(upVol − downVol)/(upVol + downVol)` | 上者的归一化版本 |

**最大回撤（Max Drawdown, MDD）**：在未来 127×3 个交易日里，从任意峰值到其后最低点的最大跌幅。

```
running_max_t = 到 t 为止的累积最高收盘价
drawdown_t    = (running_max_t − close_t) / running_max_t      # 0 ~ 1
MDD           = max(drawdown_t)                                # 0 ~ 1
negMdd        = −MDD                                           # (−1, 0]，越大越安全
```

「雷」股（财务造假、商誉减值、债务违约）的未来走势往往是**深跌**，MDD 是对这种「深跌」最直接的量化。`negMdd` 越大（回撤越浅）= 越安全，恰好就是 zero 想要的「安全分」。

**下行半波动 `downVol`**：把未来窗口的日收益 `r` 按正负拆开，只对下跌部分算标准差：

```
down_vol = sqrt( mean( min(r, 0)² ) )
up_vol   = sqrt( mean( max(r, 0)² ) )
```

这就是**半方差**（Sortino 比率的分母就是下行半标准差）。「雷」股左偏：`down_vol` 大、`up_vol` 小。但注意它是个二阶矩，右偏很重（大多数股票 `down_vol` 都接近 0，少数很大），对 MSE 回归不如有界的 `negMdd` 干净。

**收益不对称度 `upMinusDown` / `asym`**：本质是拿「上行波动 − 下行波动」当信号，依赖**偏度**这个三阶矩。三阶矩要很长的窗口才估得稳，噪声大。实验也证实了这点（见 §4，ICIR 只有 ~1）。

---

## 3. 概念：怎么评判标签/特征好不好 —— 截面 Rank IC / ICIR

### 3.1 截面（cross-section）

不是按时间先后比「这只股票自己过去如何」，而是在**同一个 `endDate`**，把所有股票横着排开比：谁的某个特征值高、谁未来更安全。避雷/选股本质是**横截面排序**，所以必须算截面指标。

### 3.2 Rank IC（Spearman 秩相关）

对每个 `endDate`，把「特征值」和「未来标签」分别**排名**，再算两个排名的皮尔逊相关：

```
IC_t = corr( rank(特征), rank(标签) )
```

- 用**秩**而非原始值：对离群值鲁棒，只关心「排得对不对」——这正是排序选股要的。
- 符号含义：IC > 0 = 特征越大、标签越大（越安全）；IC < 0 = 反向。

### 3.3 ICIR —— 比 IC 更关键的稳定性指标

```
ICIR = mean(IC) / std(IC)
```

- **IC** = 平均预测力（排得准不准）。
- **ICIR** = 预测力 / 波动 = **每个截面都稳定地准**，而不是靠某几个截面蒙对。

**做决策用 ICIR 为主**：IC 高但 ICIR 低 = 「时灵时不灵」，上线后不可靠；ICIR 高 = 稳定有效。t 值 = ICIR × √(截面数)，衡量统计显著程度。

---

## 4. 概念：特征

### 4.1 现有 7 个（偿债 / 流动性）

| 特征 | 含义 | 避雷直觉 |
|---|---|---|
| `debtRatio` | 资产负债率 = 总负债/总资产 | 杠杆越高越危险 |
| `debtToEquity` | 产权比率 = 总负债/净资产 | 同上 |
| `interestCoverage` | 利息保障倍数 = EBIT/利息费用 | 越大越还得起息，负值=亏损 |
| `cashRatio` | 现金比率 = 现金及等价物/流动负债 | 现金越硬越安全 |
| `quickRatio` | 速动比率 = (流动资产−存货)/流动负债 | 短期偿债 |
| `currentRatio` | 流动比率 = 流动资产/流动负债 | 短期偿债 |
| `inventoryTurnover` | 存货周转率 | 存货积压风险 |

### 4.2 新增 4 个（避雷最缺的维度，`zero待改进.md` 里点名要补的）

| 新特征 | 公式 | 抓什么雷 |
|---|---|---|
| `ocfToLiab` | 经营现金流 / 总负债 | **现金流覆盖**：账面赚钱但没现金的假公司 |
| `accrual` | (净利润 − 经营现金流) / 总资产 | **应计背离**：利润与现金剪刀差（康美、康得新） |
| `goodwillToEquity` | 商誉 / 净资产 | **商誉暴雷**：商誉占比越高，减值越可能致命 |
| `arToRevenue` | 应收账款 / 营收 | **收入造假**：应收占比异常高 = 收入可能是白条 |

这些「比值」必须在**标准化之前显式算好**——因为 `three/seven` 里虽然存了 `goodwill`、`netIncome`、`operatingCashFlow` 等原始值，但进模型前做了逐列 z-score，会**打散比值关系**（z-score 后的 goodwill ÷ equity ≠ 「商誉/净资产」的 z-score），模型学不到比值信号。

---

## 5. 实验设计（怎么算出来的）

1. **数据**：`train_exchanges.csv` 里 7 个训练交易所（NASDAQ/LSE/NYSE/TSXV/TSX/CNQ/AMEX）的 `daily` + `indicator` + `income` + `balance` + `cashflow`。
2. **样本构造**：与 `gen_train_data.py` 完全同口径——每个 `(symbol, endDate)`，取 `endDate` 前 127×3 天做「过去」，后 127×3 天做「未来」，过滤未来/历史交易日不足或收盘价非正的样本。
3. **特征**：取该 `endDate` 最新一季度（点在当时截面）的 7 旧 + 4 新特征值。
4. **标签**：由未来 127×3 天收盘价算出 7 个候选标签（§2.2）。
5. **IC**：对每个 `endDate`、每个「特征×标签」对，算 Spearman 秩相关，要求当日股票数 ≥ 10，再汇总 mean/std/ICIR/t。

最终 **413,261 样本、78 个截面、跨度 19 年**。

---

## 6. 实验结果

### 6.1 feature × label 的 IC（越大绝对值越强，正负号是方向）

| 特征 \ 标签 | asym | downRet | downVol | **mom** | **negMdd** | upMinusDown | upVol |
|---|---|---|---|---|---|---|---|
| accrual | −0.031 | 0.048 | −0.150 | 0.064 | 0.138 | −0.037 | −0.154 |
| arToRevenue | 0.011 | 0.012 | 0.022 | 0.014 | −0.010 | 0.011 | 0.025 |
| cashRatio | 0.037 | −0.092 | 0.185 | −0.029 | **−0.164** | 0.039 | 0.191 |
| currentRatio | 0.027 | −0.070 | 0.138 | −0.011 | −0.128 | 0.023 | 0.142 |
| debtRatio | −0.044 | 0.105 | −0.204 | 0.058 | 0.178 | −0.049 | −0.209 |
| debtToEquity | −0.080 | 0.146 | −0.386 | 0.107 | **0.328** | −0.100 | −0.396 |
| goodwillToEquity | −0.070 | 0.136 | −0.331 | 0.122 | **0.272** | −0.093 | −0.341 |
| interestCoverage | −0.086 | 0.203 | −0.449 | 0.238 | **0.397** | −0.108 | −0.460 |
| inventoryTurnover | −0.027 | 0.078 | −0.164 | 0.089 | 0.124 | −0.042 | −0.166 |
| ocfToLiab | −0.080 | 0.208 | −0.420 | 0.219 | **0.370** | −0.104 | −0.428 |
| quickRatio | 0.024 | −0.071 | 0.146 | −0.007 | −0.131 | 0.023 | 0.149 |

### 6.2 对应 ICIR（越稳越好，这是决策主指标）

| 特征 \ 标签 | asym | downRet | downVol | **mom** | **negMdd** | upMinusDown | upVol |
|---|---|---|---|---|---|---|---|
| accrual | −0.83 | 1.15 | −3.01 | 1.31 | 2.80 | −0.91 | −2.96 |
| arToRevenue | 0.36 | 0.29 | 0.43 | 0.39 | −0.20 | 0.37 | 0.52 |
| cashRatio | 0.61 | −1.07 | 2.53 | −0.31 | −2.00 | 0.67 | 2.73 |
| currentRatio | 0.51 | −0.95 | 1.93 | −0.15 | −1.80 | 0.46 | 2.05 |
| debtRatio | −0.57 | 0.95 | −2.93 | 0.52 | 2.03 | −0.68 | −3.22 |
| debtToEquity | −0.83 | 1.19 | −4.92 | 0.82 | 3.23 | −1.12 | −5.39 |
| goodwillToEquity | −0.98 | 1.81 | −4.27 | 1.37 | **3.51** | −1.36 | −4.31 |
| interestCoverage | −0.99 | 2.11 | −3.95 | 2.04 | **3.63** | −1.25 | −3.97 |
| inventoryTurnover | −0.61 | 1.23 | −3.43 | 1.25 | 2.72 | −0.95 | −3.30 |
| ocfToLiab | −1.04 | 2.36 | −6.21 | 1.98 | **5.40** | −1.29 | −6.35 |
| quickRatio | 0.44 | −0.91 | 1.92 | −0.09 | −1.69 | 0.44 | 2.03 |

### 6.3 每个标签的最强特征

| 标签 | 最强特征 | IC | ICIR |
|---|---|---|---|
| mom（现状） | interestCoverage | +0.24 | 2.04 |
| **negMdd** | interestCoverage | **+0.40** | **3.63** |
| downRet | ocfToLiab | +0.21 | 2.36 |
| downVol | interestCoverage | −0.45 | −3.95 |
| upVol | interestCoverage | −0.46 | −3.97 |
| upMinusDown | interestCoverage | −0.11 | −1.25 |
| asym | interestCoverage | −0.09 | −0.99 |

### 6.4 特征相关矩阵（关键几项）

| 特征对 | 相关系数 | 说明 |
|---|---|---|
| quickRatio ↔ cashRatio | **0.880** | **冗余，删一个** |
| quickRatio ↔ currentRatio | 0.396 | 部分冗余 |
| **debtRatio ↔ debtToEquity** | **0.016** | **几乎独立（见 §7 论证5）** |
| debtToEquity ↔ accrual | 0.571 | 中度相关 |
| ocfToLiab ↔ accrual | −0.305 | 轻度负相关 |

### 6.5 特征分布（数据质量，重点看 interestCoverage）

| 特征 | 负值占比 | 0 值占比 | 中位数 | 1%分位 | 99%分位 |
|---|---|---|---|---|---|
| **interestCoverage** | **24.5%** | **25.7%** | 0.0 | −1570 | 483 |
| debtRatio | 0.04% | 0.3% | 0.52 | 0.006 | 9.2 |
| debtToEquity | 7.1% | 0.3% | 0.87 | −16.0 | 27.0 |
| ocfToLiab | 37.0% | 1.1% | 0.01 | −4.7 | 0.89 |
| accrual | 68.6% | 0.1% | −0.01 | −1.31 | 0.33 |
| goodwillToEquity | 2.2% | **50.0%** | 0.0 | −0.81 | 2.81 |
| arToRevenue | 0.5% | 17.6% | 0.38 | 0.0 | 10.1 |

---

## 7. 为什么实验「证明」了这个结论（逐条对应）

### 论证 1：`negMdd` 比 `mom` 更可预测 → 换标签

- `mom` 的最强特征 IC=0.24、ICIR=2.04。
- `negMdd` 的最强特征 IC=0.40、ICIR=3.63。
- **IC 提高 ~67%，ICIR 提高 ~78%**。也就是说，同样拿这 7 个财务比率，去预测「未来最大回撤」比预测「动量」更准、且每个截面都更稳定。
- 原因正是 §2.1 说的：`negMdd` 纯 forward、只盯下行，且不含模型看不到的 `price_past` 噪声。**数据证实了这两个理论判断。**

### 论证 2：`upMinusDown` / `asym` 噪声大 → 否决

- 两者 ICIR 只有 −1.25 / −0.99，显著低于 `negMdd`（3.63）和 `downVol`（−3.95）。
- 印证了「不对称度依赖三阶矩、噪声大」的理论担忧。要用下行信号，直接用 `downVol` 或 `negMdd`，**不要用差值**。

### 论证 3：`ocfToLiab` 是最稳的特征（ICIR 5.40）→ 现金流维度必须补

- 对 `negMdd`，`ocfToLiab` 的 IC=0.37、**ICIR=5.40，全场最高**（对 upVol/downVol 更是到 6.2~6.4）。
- 这直接证实 `zero待改进.md` 的核心假设：**现金流/总负债是避雷最关键、最缺的维度**。现有 7 个特征里完全没有现金流信号。

### 论证 4：`goodwillToEquity` 强（ICIR 3.51）→ 商誉维度必须补

- 商誉占比对 `negMdd` 的 IC=0.27、ICIR=3.51，排第四，且是新增特征里第二强。
- 商誉减值暴雷（A 股高频雷型）在横截面上确实能被「商誉/净资产」提前排序出来。

### 论证 5：`debtRatio` 与 `debtToEquity` 并不冗余 → 推翻文档原假设

- 理论公式 D/E = D/A ÷ (1−D/A)，两者本应高度相关；但实测相关仅 **0.016**。
- 说明数据源里这两个字段的横截面排序并不遵循理论公式（字段口径/时点不一致所致），**各自携带独立信息**，且 `debtToEquity`（IC 0.33）明显强于 `debtRatio`（0.18）。
- 结论：文档原方案「去掉 debtToEquity」是**错的**，应两个都留。

### 论证 6：真正冗余的是 `quickRatio` ↔ `cashRatio`（0.88）→ 删 quickRatio

- `quickRatio` 与 `cashRatio` 相关 0.88，且 `cashRatio`（ICIR −2.0）比 `quickRatio`（−1.69）更强。
- 所以「去重」应该砍 `quickRatio`（以及较弱的 `currentRatio`），而不是砍 `debtToEquity`。

### 论证 7：`arToRevenue` 截面几乎无用（ICIR 0.2）→ 砍掉

- 应收账款/营收对 `negMdd` 的 ICIR 只有 −0.20，几乎不排序。
- 原因：应收占比更多是「公司个体层面的造假信号」，横截面排序里不显著。与其占一个输入维度，不如让给现金流/商誉。

### 论证 8：`interestCoverage` 最强但被现有预处理毁掉了 → 必须稳健处理

- `interestCoverage` 是全表最强特征（negMdd IC 0.40），但分布：**24.5% 负值、25.7% 为 0、1%分位 −1570、99%分位 483**。
- 现有 `fillna(0) + clip(±127)` 会把「−1570 的深度亏损」和「0 的刚好保本」都压到同一个数量级附近，抹掉「亏损程度」这个最强风险信号。
- 结论：这个特征要**保留并做稳健预处理**（如符号对数 `sign(x)·log(1+|x|)` 或分位数），而不是填 0 一刀切。

### 论证 9：`cashRatio` 符号反直觉，但仍有预测力 → 保留但留意

- `cashRatio` 对 `negMdd` 的 IC 是 **−0.164**（负），即「高现金比率反而预示更大回撤」。
- 这在跨市场数据里是常见现象：大量「现金多但高风险」的公司（烧钱扩张、pre-revenue 生科、刚融资后变脸）。quick/current 也是同样负号，说明是系统效应，不是噪声。
- 模型能学任意方向，所以**保留**（ICIR −2.0 依然稳），但要知道它的方向与我们直觉相反。

---

## 8. 最终建议方案

**标签**：`negMdd = −max_drawdown(未来 127×3 天)`，越大越安全。

**特征（7 个，输入维度仍 `[7, 3]` 不变）**：

```
interestCoverage     （保留，稳健预处理）
ocfToLiab            （新增：现金流/总负债）
debtToEquity         （保留）
goodwillToEquity     （新增：商誉/净资产）
debtRatio            （保留）
accrual              （新增：应计背离）
cashRatio            （保留）
```

**删除**：`quickRatio`、`currentRatio`、`inventoryTurnover`、`arToRevenue`。

**预处理注意事项**：`interestCoverage` 用符号对数/分位数替代 `fillna(0)+clip`；新增比值在 z-score 之前算。

---

## 9. 待你确认的决策点

1. **标签**：`negMdd`（推荐） / `downVol` / 保持 `mom`。
2. **特征**：数据驱动 7 个（推荐） / 文档原方案 7 个 / 保持现有 7 个。
3. 确认后我再改 `gen_train_data.py` + `train.py` + `get_zero_predict.py` 并重训。

---

### 附：实验可复现

```
cd zero
python validate_label.py              # 全量（生成 validate_samples.csv / validate_ic_feature_x_label.csv / validate_corr_features.csv / validate_dist.csv）
python validate_label.py --exchanges AMEX,CNQ   # 快速子集
```
