# 原文精读：Improving Factor-Based Quantitative Investing by Forecasting Company Fundamentals

> Alberg & Lipton, NIPS 2017（arXiv:1711.04837）。Euclidean Technologies 的方法论地基。
> 本文 = 原文逐段精读 + 16 个基本面字段到本项目 `write_stock_data.py` 的精确映射。

---

## 0. 一句话

**不预测股价，预测「未来 12 个月的基本面」，再用预测的基本面做价值因子（EBIT/EV）选股。** 用 oracle 证明天花板（44% 年化），用深度网络逼近（17.1% 年化 vs 传统因子 14.4%）。

---

## 1. 动机：oracle 实验（图 1）

作者先做一个「预言机」仿真：如果因子能用到**未来 12 个月**的财报（完美预测），EBIT/EV 因子年化能到 **44%**。

这个 44% 是**天花板**，也是全文的动机：既然「未来基本面」这么值钱，那就训练网络去预测它。

---

## 2. 数据

- **11,815 只美股**（NYSE/NASDAQ/AMEX，1970–2017），剔除非美股、金融股、市值 <$100M。
- Compustat North America + Snapshot。
- 离散化到**月度**，输入帧间隔 1 年，预测 **12 个月后**。

---

## 3. 16 个基本面字段 + 4 个动量特征（精确列表）

**利润表用 TTM（trailing twelve months，过去 12 个月累计）**——5 个：

| # | 论文字段 | 本项目列名（write_stock_data.py） |
|---|---|---|
| 1 | Revenue | `revenue` |
| 2 | Cost of Goods Sold | `costOfRevenue` |
| 3 | SG&A Expense | `sellingGeneralAndAdministrativeExpenses` |
| 4 | **EBIT** | `ebit` |
| 5 | Net Income | `netIncome` |

**资产负债表用 MRQ（most recent quarter，最近一季度）**——11 个：

| # | 论文字段 | 本项目列名 |
|---|---|---|
| 6 | Cash & Cash Equivalents | `cashAndCashEquivalents` |
| 7 | Receivables | `netReceivables`（或 `accountsReceivables`） |
| 8 | Inventories | `inventory` |
| 9 | Other Current Assets | `otherCurrentAssets` |
| 10 | Property Plant & Equipment | `propertyPlantEquipmentNet` |
| 11 | Other Assets | `otherAssets` |
| 12 | Debt in Current Liabilities | `shortTermDebt` |
| 13 | Accounts Payable | `accountPayables` |
| 14 | Taxes Payable | `taxPayables` |
| 15 | Other Current Liabilities | `otherCurrentLiabilities` |
| 16 | Total Liabilities | `totalLiabilities` |

**4 个动量特征**（只当输入、不预测）：过去 1/3/6/9 个月的股价动量，用**全市场百分位**表示。

**总输入 = 20 特征（16 基本面 + 4 动量）。**

---

## 4. 预处理（3 处关键）

1. **市值归一化**：所有基本面特征除以「最后一个输入时间步的市值」。例：苹果营收 $215B vs National Presto $340M，不归一化网络会被大公司主导。**特意不用 EV/账面权益归一化**（可能为负）。
2. **动量用百分位而非绝对值**：让模型关注相对强弱。
3. 零均值单位方差标准化；缺失值**前向填充**。

> 落到本项目：我们的 raw 列（营收、总资产等）量级差巨大，之前 zero 标准化时已发现。市值归一化这步我们还没做。

---

## 5. 模型

- **MLP**：每月 t，取 5 个间隔 1 年的快照 `t−48,t−36,t−24,t−12,t` → 预测 `t+12`。
- **RNN（GRU/LSTM）**：同样序列，但每个时间步都预测「对应的 12 个月后」。
- **多任务**：同时预测全部 16 个基本面（共享编码器 + 16 个输出）。
- **EBIT 加权**（α1，因为 EBIT 对因子最有价值）；RNN 额外加权最后时间步（α2）。

---

## 6. 训练与防过拟合（最值得学的部分）

原文直接对比了两种做法的过拟合差异：

> 「只用收益率做目标，RNN 容易过拟合训练集、验证集毫无提升。而通过**多任务学习（同时预测 16 个基本面）**，给模型大量训练信号，因此更不易过拟合。」

- 样本内 1970–1999 / 样本外 2000–2016；30% 验证集；25 epoch 无改善早停。
- 超参：MLP 1024 单元/2 层/dropout 0.5；RNN 64 单元/2 层/循环 dropout 0.7；AdaDelta；L2 梯度裁剪。

---

## 7. 结果（核心证据）

| 策略 | 年化 CAR | Sharpe |
|---|---|---|
| S&P 500 | 4.5% | 0.19 |
| 市场平均 | 7.7% | 0.29 |
| Price-LSTM（直接预测收益） | 11.3% | 0.60 |
| QFM（传统因子，当前基本面） | 14.4% | 0.55 |
| LFM-Linear（线性预测基本面） | 15.9% | 0.63 |
| **LFM-MLP** | **17.1%** | **0.68** |
| LFM-LSTM | 16.7% | 0.67 |

三个读数：
1. **Price-LSTM（11.3%）< 所有 LFM** —— 直接预测收益不如预测基本面。
2. **LFM-Linear（15.9%）已超 QFM（14.4%）** —— 大部分收益来自「预测基本面」这个方向本身，网络深度只再加 ~1 个点。
3. **44% oracle 上界 vs 17.1%** —— 巨大改进空间。

---

## 8. 回测模拟器（严谨，可抄）

每月按 EBIT/EV 排序，买**前 50 只**、等权、持有一年；跌出前 50 就卖、买新进者。交易成本 $0.01/股 + 滑点（参与度平方增长，最高 1%）；成交量加权价成交；股息计入。

---

## 9. 对本项目 Phase 0 的直接落地建议

1. **16 字段全部能映射**（上表），我们的数据几乎不用 Compustat。
2. **要补「TTM vs MRQ」的区别**：论文利润表用 TTM（过去 4 季度累计）、资产负债表用 MRQ（单季快照）。我们的 `income_{ex}.csv` 存的是单季值，**做 label 时要自己算 TTM**（利润表项 rolling 4 季度求和）。
3. **市值归一化 + 百分位动量**：Phase 0 预处理要加。
4. **多任务预测全部 16 个字段**：别只预测一个，共享编码器一起预测，EBIT 加权。
5. **label 是「未来 12 个月的基本面」**：对应我们「未来 1 季度」的财务字段（注意他们用月度、12 个月 = 我们 1 季度近似，但频率不同，我们按季度走更合适）。

---

*关联：`rd_plan_daily2fin.md`（研发计划）、`euclidean_technologies.md`（公司调研）。*
