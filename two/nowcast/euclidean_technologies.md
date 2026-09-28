# Euclidean Technologies 公司调研

> 「用机器学习做长期价值投资」这条路上最纯粹的代表，也是本项目「日线 → 财务状态变化」方向的最佳参照系。

---

## 1. 公司概况

**Euclidean Technologies**，美国西雅图的一家小型量化投资管理公司，2008 年由 **John Alberg** 和 **Michael Seckler** 联合创立。

- 规模不大，但影响力远超体量——极少数把「深度价值投资哲学」用「深度学习」系统化落地、并且**开源**的机构。
- 2023 年把旗下私募基金转成 **Euclidean Fundamental Value ETF（代码 ECML）**，在 NYSE Arca 上市，向散户开放。

来源：[官方 ETF 页](https://euclideanetf.com/)、[GeekWire 报道](https://www.geekwire.com/2023/seattle-firm-debuts-etf-that-uses-machine-learning-for-long-term-equity-investing/)、[公司档案](https://www.tradermath.org/firms/euclidean-technologies)。

---

## 2. 创始人 John Alberg（非金融出身，工程师创业）

- 创业老兵：和 Seckler 一起做了十年 HR 软件 **Employease**，2006 年卖给 ADP（财富 500 强），Alberg 当到 CTO。
- 此前是 Booz Allen Hamilton 的咨询顾问，本科 Williams College。
- 家学渊源：父亲是 **Tom Alberg**（Madrona Venture Group 联合创始人，西雅图科技圈领军人物）。
- 现在是 Finpilot（金融文档管理软件）的联合创始人/董事。

**典型故事**：两个工程师卖公司拿到钱后，不想把钱交给「短期博弈、加杠杆、靠定性拍脑袋」的传统资管，于是自己下场，用最擅长的机器学习去验证一个朴素问题——**「价值投资到底是不是真的有效」**。

来源：[访谈](https://hedgefundalpha.com/news/john-alberg-applying-machine-learning-long-term-investing/)、[播客 The Man in the Machine Learning](https://podcasts.apple.com/us/podcast/john-alberg-the-man-in-the-machine-learning-s1e7/id1402620531)。

---

## 3. 投资哲学：ML + 深度价值 + 长期

核心逻辑（Alberg 原话）：

> 如果价值投资是有效的，那机器学习就应该能在历史数据里找到它的证据。

他们研究了格雷厄姆、巴菲特、席勒的历史数据，得出两条朴素结论：

1. **公司长期价值由「现金流的持续性、成长性、量级」决定**；
2. **短期股价波动远大于基本面波动**（所以短期价格是噪声，别去预测它）。

策略：**买便宜的好公司，长期持有**——纯粹的反向、深度价值、低换手。

来源：[策略介绍](https://www.tradermath.org/firms/euclidean-technologies)、[价值投资心法](https://hedgefundalpha.com/strategies/value-investing-through-market-cycles/)。

---

## 4. 核心方法论：预测基本面，不预测价格

这是他们最独特、也最值得学习的地方。深度学习模型只干两件事：

1. **估算内在价值** → 找出被低估的便宜股票；
2. **识别价值陷阱** → 找出「看起来便宜、但会继续跌」的股票，提前避开。

**关键设计**：模型**不直接预测股价**（短期价格噪声会误导神经网络），而是**预测未来的基本面**（EBIT、市值等），再用预测出的基本面算「前瞻估值倍数」来选股——这就是论文里的 **Look-Ahead Factor Model (LFM)**。

> 与我们项目的直接对照：我们测「日线→Δgd（three 模型输出差值）」信号弱（IC 0.18），就是因为那是在预测「二级的模型输出」；Euclidean 证明了对的做法是「预测一级的基本面字段」。

---

## 5. 技术演进（一条清晰的路径）

| 时间 | 里程碑 |
|---|---|
| 2017 | [Improving Factor-Based Quantitative Investing by Forecasting Company Fundamentals](https://ar5iv.labs.arxiv.org/html/1711.04837)（NeurIPS，Alberg & 亚马逊 Zachary Lipton）——MLP/RNN 预测基本面，**年化 17.1% vs 传统因子 14.4%，Sharpe 0.55→0.68** |
| 2020 | [Uncertainty-Aware Lookahead Factor Models](https://dl.acm.org/doi/10.5555/3524938.3525077)（ICML）——加不确定性估计，过滤低置信度预测 |
| 2020 起 | **提前用 seq2seq 语言模型**（即后来 ChatGPT 的架构）读 SEC 文件、财报电话会、分析师报告——比这波生成式 AI 热早了两年 |
| 开源 | [deep-quant](https://github.com/gmackall/deep-quant)、[lfm_quant](https://github.com/lakshaykc/lfm_quant) |

他们很早就把 NLP/LLM 引入基本面分析，这点很前瞻。

---

## 6. 业绩与 caveat

- 论文披露的是回测：**年化 17.1%、Sharpe 0.68**（vs 传统因子 14.4%、Sharpe 0.55）。
- ETF 上市后的真实业绩需看 ECML 官方业绩页（搜索未拿到确切数字）。
- **注意**：这类长线深度价值策略，在「成长股占优的十年」里跑赢大盘有难度，真实业绩要打折看。另外开源数据的作者自己标了有 **look-ahead 偏差 + 幸存者偏差**，不能直接用于实盘。

---

## 7. 对本项目的直接启示（硬映射）

| Euclidean | 本项目 |
|---|---|
| 预测「基本面」而非价格 | ✅ 即「日线 → 财务状态变化」 |
| 两任务：找低估 + 避价值陷阱 | ≈ three（涨/跌）+ zero（避雷） |
| 长期、季度级、低换手 | ≈ 1/3/7 季度标签 |
| 价格动量作为辅助特征 | ≈ 日线技术因子 |
| 多任务共享编码器 | ≈ 可复用的多任务训练 |
| 用预测基本面算前瞻估值再选股 | ≈ nowcast 财务 → 喂 three 排序 |

---

## 8. 一句话总结

Euclidean 用一家小公司 + 十几年时间，证明了你现在想走的这条路（预测基本面、长期价值、ML 系统化）是走得通的，而且把核心代码开源了。这是本项目「日线 → 财务变化」方向**最好的参照系和信心来源**。

---

*关联文档：`rd_plan_daily2fin.md`（研发计划，含本方向的完整调研与分阶段计划）。*
