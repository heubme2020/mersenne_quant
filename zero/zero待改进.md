# zero 模型待改进清单

> 记录 zero 模型当前存在的问题与改进方向，供后续迭代参考。

## 一、现状

- **输入**：7 个偿债能力 / 流动性 ratio，取最近 3 个季度（`[7, 3]`）
- **用途**：避雷——用财务风险过滤危险股票
- **当前标签**：`zero = price_fore - price_past`（未来股价 vs 过去股价的对数动量）

**7 个特征：**

| 特征 | 度量 |
|---|---|
| debtRatio | 资产负债率 |
| debtToEquity | 产权比率 |
| interestCoverage | 利息保障倍数 |
| cashRatio | 现金比率 |
| quickRatio | 速动比率 |
| currentRatio | 流动比率 |
| inventoryTurnover | 存货周转率 |

---

## 二、待改进：标签（优先级最高）

### 当前标签的两个问题

1. **对称，但「避雷」关心的是下行**：`price_fore - price_past` 对涨跌一视同仁，而避雷本质是下行 / 尾部风险。对称标签把「涨很多」和「跌很多」混在一起，稀释了最该关注的信号。

2. **混进了模型看不到的 `price_past`**：输入只有 7 个财务 ratio、没有股价，但标签里的 `price_past` 是过去股价——模型根本预测不了它，等于标签里塞了个噪声项。

### 改进方向：标签从「动量」改成「未来下行风险」

标签应纯 forward-looking、只盯下行：

- **方案 A（推荐）：未来最大回撤**
  ```
  label = -max_drawdown(未来 127*3 天)     # 越大越安全
  ```
  直接对应「避雷」：财务有雷的票未来大概率深跌。纯 forward、天然不对称、连续可排序。

- **方案 B：下行收益**
  ```
  label = min(0, 未来对数收益)             # 涨了记 0，跌了记跌幅
  ```
  更简单，但抓不到「先涨后崩」的雷。

- **方案 C：二元暴雷**
  ```
  label = 1 if 未来最大回撤 < -30% else 0
  ```
  直观，但丢失「雷有多严重」的粒度；可做成软标签。

- **方案 D：向上 / 向下波动性（收益不对称度）**
  ```python
  r = close.pct_change()                          # 未来窗口日收益
  down_vol = sqrt(mean(min(r, 0)**2))             # 向下波动（只统计下跌）
  up_vol   = sqrt(mean(max(r, 0)**2))             # 向上波动（只统计上涨）
  label = up_vol - down_vol                       # 向上越大越好、向下越大越差
  # 或归一化（更稳）：
  label = (up_vol - down_vol) / (up_vol + down_vol + eps)
  ```
  本质是半方差分解 / 收益不对称性（Sortino 的下行偏差就是下行半方差）。「雷」股左偏（down_vol 大），好股右偏（up_vol 大）。
  - **权衡**：zero 在链里的定位是「避雷」，而 seven/three 已负责「选优」。纯用 `down_vol` 更聚焦避雷；用 `up_vol − down_vol` 会把 zero 变成「收益不对称 / 质量」信号，与 seven/three 的选优部分重叠。
  - **数据提醒**：上行 / 下行波动对多数股票差不多大，差值主要靠「偏度」这个高阶矩驱动，噪声大、需较长窗口才估得稳；归一化比值比裸差值更稳。

- **更硬核（需额外数据）**：直接拿真实暴雷事件当标签——ST、退市、债务违约、财务造假、巨额商誉减值。A 股语境里「雷」多指这些。

---

## 三、待改进：特征

### 当前特征的两个问题

1. **冗余重**：7 个特征实际只有约 4 类独立信号
   - 杠杆类：`debtRatio` 与 `debtToEquity` 是同一指标的不同形式（D/E = D/A ÷ (1−D/A)），留一个即可
   - 流动性类：`currentRatio` / `quickRatio` / `cashRatio` 层层嵌套，留 `cashRatio`（最硬）+ 一个即可
   - 实际上：{杠杆, 利息覆盖, 流动性, 存货周转} 4 个独立信号，另外 3 个浪费输入维度

2. **缺了避雷最关键的维度**：
   - **现金流**（最重要）：经营现金流 / 总负债，或「净利润 − 经营现金流」背离度（抓利润造假，如康美、康得新）
   - **商誉**：`goodwill / 净资产`（商誉减值暴雷）
   - **应收账款**：`应收账款 / 营收`（抓收入造假）

### 建议：保持 7 个，做「去重 + 补强」

```
cashRatio                  （最硬的流动性）
interestCoverage           （利息覆盖，违约核心）
debtRatio                  （杠杆，去掉 debtToEquity）
经营现金流 / 总负债          （现金流覆盖，抓造假）★ 新增
净利润 − 经营现金流 背离度    （利润与现金剪刀差）★ 新增
goodwill / 净资产          （商誉暴雷）★ 新增
应收账款 / 营收             （收入造假）★ 新增
```

---

## 四、数据现状（与 127 特征的关系）

- **原始成分都在 127 里**：`goodwill`、`operatingCashFlow`、`netCashProvidedByOperatingActivities`、`totalLiabilities`、`accountsReceivables`、`revenue`、`netIncome`、`totalStockholdersEquity` 等全部存在于 seven/three 的 127 个特征中。
- **但 127 里是「原始值」而非「比值」**，且进模型前做了逐列 z-score 标准化，会**打散比值关系**（z-score 后的 goodwill / equity ≠ 「商誉/净资产」比值的 z-score），所以模型不一定能学到比值信号。
- **zero 的 7 个特征里，现金流 / 商誉 / 应收三个维度完全缺失**（连原始值都没有）。

**结论**：要真正用上「现金流覆盖 / 商誉占比 / 应收占比」这些避雷信号，最好在标准化**之前**显式计算成比值列加进特征，而不是指望模型从原始 z-score 里自己学。

---

## 五、建议的下一步验证（动手前先确认）

1. 拿现有 7 个 ratio，分别对「未来最大回撤」和「现有 price_fore − price_past」算横截面 Rank IC，看哪个标签 IC 更高、ICIR 更稳——确认换标签是否值得。
2. 拉 7 个 ratio 的分布和两两相关性，量化「冗余」和 `interestCoverage` 的负值/极端值到底多严重。
3. 算新增比值（现金流/总负债、goodwill/净资产、应收/营收）对「未来最大回撤」的 Rank IC，用数据确认它们是否真的比现有 ratio 更避雷。
4. 对 `interestCoverage` 等有负值/极端值的比率，改用稳健预处理（秩 / 分位数），避免 `fillna(0) + clip(±127)` 抹掉「亏损→负利息保障」这个强风险信号。
5. 把候选标签都算出来做 IC 对比：`down_vol`、`up_vol − down_vol`、`(up−down)/(up+down)`、`up/down`、`−max_drawdown`——对「未来回撤」（避雷）和「未来收益」（选优）各算一次，用数据决定 zero 该用哪个、该不该带上行。
