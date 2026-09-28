# 研发计划：日线数据 → 财务状态变化预测

> 目标：构建一个「输入日线（价格/成交量）+ 当前财务特征 → 输出未来财务状态变化」的模型，
> 用它「刷新」three 模型的财务输入，桥接财报披露滞后。
> 本文 = 调研结论 + 方法论提炼 + 落到本项目的分阶段研发计划。

---

## 0. 一句话结论

**这个方向是成熟且被验证过的**：量化公司 Euclidean Technologies 就是靠「用价格+历史财务预测未来基本面」跑出了 17.1% 年化（vs 传统因子 14.4%）。关键教训是——**预测「基本面」而不是「价格」或「模型输出」**，而我们之前测「日线→Δgd（three 输出差值）」信号弱，恰恰是因为 Δgd 是二阶量，方向错了。

---

## 1. 调研结论：有没有人搞过？

**有，而且是最主流的「基本面量化」路线之一。** 三个层次：

### 1.1 直接对标：Euclidean Technologies 的 Look-Ahead Factor Model (LFM)

- 论文：
  - [Improving Factor-Based Quantitative Investing by Forecasting Company Fundamentals](https://ar5iv.labs.arxiv.org/html/1711.04837)（Alberg & Lipton, NeurIPS 2017，arXiv:1711.04837）
  - [Uncertainty-Aware Lookahead Factor Models for Quantitative Investing](https://dl.acm.org/doi/10.5555/3524938.3525077)（ICML 2020）
- 开源代码：[deep-quant](https://github.com/gmackall/deep-quant)、[lfm_quant](https://github.com/lakshaykc/lfm_quant)
- **做法**（两步）：
  1. 用「16 个财务科目 + 4 个价格动量指标」预测**未来 12 个月的基本面**（EBIT、市值等）。
  2. 用预测出的基本面算**前瞻估值倍数**（预测 EBIT / 当前市值），买最便宜的 50 只。
- **结果**：年化 17.1%（LFM-MLP）vs 14.4%（传统因子）；Sharpe 0.55 → 0.67~0.68。
- **模型**：MLP / RNN（GRU/LSTM），**多任务学习**同时预测多个基本面字段。

### 1.2 方法论对标：Panel Data Nowcasting

- 论文：[Panel Data Nowcasting: The Case of Price–Earnings Ratios](https://onlinelibrary.wiley.com/doi/abs/10.1002/jae.3028)（Babii, Ball, Ghysels & Striaukas, JAE 2024，arXiv:2307.02673）
- **做法**：sparse-group LASSO（sg-LASSO）做混合频率面板数据 nowcasting，把 P/E 分解成「公司收益 + 分析师预测误差」两个分量分别预测。
- **结果**：超过分析师一致预期、forecast combination、elastic net。
- **关键**：利用「时间序列 + 面板」结构做正则化，比把特征当 i.i.d. 的标准 ML 好。

### 1.3 学术基础：价格领先盈利（Prices Lead Earnings / FERC）

- 会计学几十年成熟文献，价格领先盈利 1~8 个季度（FERC 模型）。
- 代表：[Lee (2018) A model of stock prices leading earnings](https://www.emerald.com/mf/article-abstract/44/7/935/289072/A-model-of-stock-prices-leading-earnings)、[Jiambalvo et al. (2002)](https://onlinelibrary.wiley.com/doi/abs/10.1506/EQUA-NVJ9-E712-UKBJ)。
- **A股 相关**：[FinReport](https://dl.acm.org/doi/10.1016/j.procs.2026.06.419)（75 只 A 股，股价+新闻情绪预测盈利，RMSE 比 LSTM 低 ~15%）。

### 1.4 反方论文（必须读的 caveat）

- [The return of return dominance (JFE 2025)](https://www.sciencedirect.com/science/article/pii/S0304405X25000674)：
  > 市盈率横截面离散度里，**75% 来自未来收益率、只有 25% 来自未来盈利增长**。
- 含义：价格更多在「预测未来涨跌」而非「预测未来盈利」。**信号天花板就在那，别期望太高**——这和我们之前测「日线→Δgd」IC 只有 0.18 一致。

---

## 2. 关键方法论提炼（5 条）

1. **预测基本面，不预测价格，也不预测模型输出**。LFM 论文的核心发现：「预测基本面比预测收益率提供更多训练信号、减少过拟合」。我们之前测 Δgd（three 输出差值）弱，就是因为那是二阶量；**应回归预测财务字段本身的变化**。

2. **多任务学习**：LFM 同时预测多个基本面字段，共享编码器、互相正则化。我们也可以同时预测多个财务指标的变化，而不是单个。

3. **两阶段解耦**：预测（基本面）和决策（估值/排序）分离。对应我们：先预测 Δ财务 → 再喂给 three 做排序。

4. **利用时间序列 + 面板结构做正则化**（sg-LASSO / 混合频率 MIDAS），而不是把特征当 i.i.d.。

5. **预测「方向」比预测「幅度」可行**：文献普遍结论，盈利变化方向的二分类 AUC 能做到 67~69%。先做「财务走强/走弱」分类，再考虑回归。

---

## 3. 映射到本项目

| 概念 | 本项目对应 |
|---|---|
| LFM 的「价格动量指标」 | 我们的日线技术因子（close/volume 衍生） |
| LFM 的「16 个财务科目」 | 我们的 127 财务特征（42 indicator + 85 raw） |
| LFM 的「未来 12 个月基本面」 | 未来 1/3/7 个季度的财务特征变化 Δ |
| LFM 的「前瞻估值倍数 + 选股」 | three 模型的 growth/death 排序 |
| Panel nowcasting 的「P/E 分解」 | 把 Δ 分解成可预测的分量 |

**核心调整（相对之前的 Δgd 方案）**：

- ❌ 之前：label = `three(窗口挪 h) − three(当前)`（模型输出差值，二阶、噪声大、耦合 three）。
- ✅ 现在：label = **财务特征本身的变化** `feature_{t+h} − feature_t`（一阶、直接对应财报变化）。
- 预测维度：不是 127 全部，而是**按 IG 重要性 + 可预测性筛出的 top-N 财务指标**（比如 20~40 个）。
- 输入：日线技术因子（拉长版 7/31/127 更匹配季度尺度）+ 当前财务特征。

---

## 4. 分阶段研发计划

### Phase 0：验证「日线 → 财务变化」的信号（1~2 天）

- **目标**：确认这个方向值得投入，别重蹈 Δgd 弱信号的覆辙。
- **做法**：
  1. 选 top-N 财务指标（用 three 的 IG 重要性排名，取前 20~40 个）。
  2. 算 label：`Δfeature = feature_{t+1Q} − feature_t`（1Q 变化，最关键的 nowcast）。
  3. 算「日线技术因子（拉长版）」对每个 Δfeature 的**逐截面 IC/ICIR**（用我们已有的 `validate_factors_vs_price.py` 改一下 label）。
  4. 判定标准：如果有若干财务指标的 |ICIR| > 0.2，方向成立，进入 Phase 1。
- **产出**：一张「日线因子 → 各财务指标 Δ」的 IC/ICIR 表。

### Phase 1：建「日线 → Δ财务」模型（3~5 天）

- **模型**：复用 TWO 的 Transformer 架构（889 天 × 31 日线因子），输出改成「top-N 财务指标的 Δ」，多任务学习（一个共享编码器 + N 个头）。
- **label**：Δfeature（1Q 为主，可加 3Q/7Q 做多任务）。
- **训练**：A股 训练 + held-out 测试（沿用 two 实验的股票隔离框架）。
- **评估**：预测 Δ 对真实 Δ 的逐截面 IC/ICIR，以及「预测方向」的 AUC。

### Phase 2：接入 three，验证端到端价值（3~5 天）

- **做法**：`three(财务特征 + 预测的 Δ) − three(财务特征)`，看排序指标 `icir_gd_combined` 是否提升。
- **对照**：直接用「当前（滞后）财务特征」的 three 基线。
- **判定**：如果 nowcast 后 three 的 gd ICIR 明显提升，方向彻底打通。

### Phase 3：回测与调参（1~2 周）

- 引入「不确定性」输出（ICML 2020 的 uncertainty-aware 版），过滤低置信度预测。
- 扩大财务指标集合、调窗口、试 sg-LASSO/混合频率结构。
- 对比分析师预期（如果数据可得）。

---

## 5. 风险与注意事项

1. **信号天花板低**（return dominance 论文）：价格领先「收益」远强于领先「盈利」。别期望像 three 的 gd ICIR 0.42 那么强，1Q 财务变化能被日线预测的 ICIR 能到 0.2 就不错。
2. **报告滞后 vs 财务季度的对齐**：日线对齐到「披露日」而不是「财报期末」，否则有 look-ahead 或信号衰减（我们之前用的是期末对齐，偏保守）。
3. **幸存者偏差 + look-ahead**：deep-quant 开源数据作者自己标了有这两个偏差。我们用自己的数据（write_stock_data.py 全量重建），但要时刻检查。
4. **A股 vs 全球**：two 实验已证明「加全球数据」对 A股任务没帮助，这个方向大概率也是 A股 专用，先别铺全球。
5. **财务指标的可预测性差异大**：有些财务指标（如利润率）比另一些（如现金）更容易被价格领先，Phase 0 就是要筛出可预测的那批。

---

## 6. 参考资源

- [deep-quant（Euclidean Technologies）](https://github.com/gmackall/deep-quant)
- [lfm_quant（Euclidean Technologies 更新版）](https://github.com/lakshaykc/lfm_quant)
- [Improving Factor-Based Quantitative Investing by Forecasting Company Fundamentals（arXiv:1711.04837）](https://ar5iv.labs.arxiv.org/html/1711.04837)
- [Uncertainty-Aware Lookahead Factor Models（ICML 2020）](https://dl.acm.org/doi/10.5555/3524938.3525077)
- [Panel Data Nowcasting: The Case of Price–Earnings Ratios（JAE 2024, arXiv:2307.02673）](https://onlinelibrary.wiley.com/doi/abs/10.1002/jae.3028)
- [Lee (2018) A model of stock prices leading earnings](https://www.emerald.com/mf/article-abstract/44/7/935/289072/A-model-of-stock-prices-leading-earnings)
- [The return of return dominance（JFE 2025）](https://www.sciencedirect.com/science/article/pii/S0304405X25000674)
- [FinReport（A股 盈利预测）](https://dl.acm.org/doi/10.1016/j.procs.2026.06.419)

---

*本文档对应项目内的实验产物：`two_2x2_experiment.md`（two 任务 2×2 证伪）、`validate_daily_to_dgd.py`（Δgd 信号验证，IC 0.18，弱）。*
