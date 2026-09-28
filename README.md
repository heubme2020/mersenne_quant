# 量化选股系统

个人量化研究的完整工程：**财报 + 日线数据 → 五个神经网络模型 → 每日选票 → 邮件推送**，
连同两年多来的因子筛选、标签设计、估值方法的实验记录 —— **包括负面结果**。

> ⚠️ **免责声明**
> 这是研究/工程代码，**不构成投资建议**，也不是可直接实盘的系统。
> 第三方数据（MySQL 本地库、Financial Modeling Prep、BaoStock）**不随仓库分发**，
> 凭据也不在仓库里（见[快速开始](#三快速开始)）。
> 所有回测均为**毛收益**口径：等权、无交易成本/冲击成本、未剔除 ST 与退市停牌、无涨跌停限制、
> 只做多按分数排序的前 50% —— 所以**绝对值不可信，只有同一口径下的相对比较可信**。

---

## 目录

- [一、系统结构](#一系统结构)
- [二、五个模型](#二五个模型)
- [三、快速开始](#三快速开始)
- [四、每日选票链](#四每日选票链)
- [五、一键重训](#五一键重训)
- [六、研究结论（含负面结果）](#六研究结论含负面结果)
- [七、五个模型共用的约定](#七五个模型共用的约定)
- [八、目录速查](#八目录速查)
- [九、文档索引](#九文档索引)
- [十、已知问题与未完成](#十已知问题与未完成)

---

## 一、系统结构

```
                    ┌─────────────────────────────────────────────────┐
   MySQL (localhost)│  get_stock_data.py / write_stock_data.py        │
   31 个交易所       │  → data/{EX}/{income,balance,cashflow,          │
                    │       indicator,mean,std,daily}_*.csv           │
                    └───────────────────────┬─────────────────────────┘
                                            │
              ┌─────────────────────────────┴──────────────────────────┐
              │  各模型 gen_train_data.py → {模型}/train/*.h5 → train.py │
              │  → {模型}/{模型}.pt                                      │
              └─────────────────────────────┬──────────────────────────┘
                                            │
   每日选票链（refresh_candidates.py）：                                      │
     seven → three → zero → schloss → two → one → buy_predict.csv → 邮件 ◄──┘
```

## 二、五个模型

每个模型目录都是**自足**的：`*_features.py`（特征）、`*_model.py`（网络）、
`gen_train_data.py`（造 h5）、`train.py`（训练，warm start）、`get_*_predict.py`（推理）、
`evaluate.py`（评估）。

| 目录 | 预测目标 | 在链里的角色 | 实现要点 |
|---|---|---|---|
| `seven/` | 自由现金流 / 分红 / 净资产，三分支 × 1/3/7 季及合并分 | **估值主体**（产出 `dcf`） | 3×3 头；损失按各头标准差归一 |
| `three/` | growth（上行）/ death（下行）双头 × 1/3/7 季 | 成长与风险 | 逆方差加权 + 比值项 |
| `zero/` | 未来 **381 个交易日最大回撤**（取负） | **避雷** | 7 个财务比率 × 3 季，输出 `[-1,0]` |
| `two/` | nowcast 三头：毛利 / 营收 / 总资产的「中位变化」 | 候选池**砍半**（`up_down`=毛利头） | 889 天日线窗口（1017×31）+ 128 天辅助重建头 |
| `one/` | 5 个期限的 close 形态标签 | **最终排序**（`up_down`） | 24 个技术因子（`new24`，见 `one/factor_screen/`） |

## 三、快速开始

```bash
git clone <repo> && cd quant
pip install -r requirements.txt          # 版本见 requirements.txt（作者环境 Python 3.11 + CUDA 12.1）

# 凭据不在仓库里，照模板写一份（MySQL DSN / FMP key / SMTP 授权码与收件人）
cp config_local.example.py config_local.py     # 然后填值

# 没有数据？两条路：
#   ① 自己准备  data/{EX}/{income,balance,cashflow,indicator,mean,std,daily}_*.csv
#   ② 有 MySQL 库的话：python get_stock_data.py     # 导出到 data/
```

**要跑起来还需要**：每个模型的 `train/*.h5`（由各自的 `gen_train_data.py` 生成，
全量约 327 GB / 数百万个文件）与 `{模型}.pt`（五个生产权重**已随仓库提供**）。
只想看代码和实验记录的话，什么都不用装。

## 四、每日选票链

`refresh_candidates.py` 是唯一的生产入口：

* **星期路由**：周五跑全链（含数据更新）；周六/周日空转；**周一~周四只跑 schloss/two/one**
  （财务数据没更新，只需重算与日线相关的那几环）。
* **选票规则**：把 `one` 的股票池按 `buffett` 降序取五档 —— 全池 / 127 / 31 / 7 / 3 ——
  每档内再按 `up_down` 降序取第一名，共 **5 只票**（邮件按 `1: 3: 7: 31: 127:` 五行发出）。
* **打分公式**：`buffett = dcf × 每股净资产 / 收盘价 + schloss`
  （`dcf` 来自 seven；实现见 `two/get_two_predict.py`）。

## 五、一键重训

```bash
python refresh_models.py --dry-run    # 先看计划
python refresh_models.py              # 正式跑
```

| 阶段 | 做什么 |
|---|---|
| 1 | MySQL → `data/{EX}/*.csv`。**导出前后比对每个 CSV 的体积，任何文件缩小 >5% 就停下报警**（防"库里少了数据、模型跟着退化"） |
| 2 | 重生五个模型的 h5（每个模型先清空自己的 `train/`，免得新旧采样口径混在一起） |
| 3 | 训练段：依次训练五个模型（**warm start**：接着已有 `.pt` 训）→ 部署到推理路径 → 把验证/测试指标追加到 `retrain_results.csv` |

整链**以十小时计**（`two` 的 h5 单这一项就 ≈5~6 小时）。
细粒度开关（`--stage data|train`、`--data-only <模型>`、`--only <模型>`、`--skip-export`…）
见 `refresh_models.py` 的 docstring。

## 六、研究结论（含负面结果）

这是本仓库里最有价值的部分 —— 记录了什么**没**work、以及为什么。
以下引自仓库内的文档（括号里是出处）：

**1. 朴素持续性基准碾压所有价格因子**（`two/nowcast/phase1_label_v2.md` §2）

预测财务量的变化时，"上一轮重演"这个**完全不用价格信息**的基准强得离谱：

| 标签（h=3） | 价格因子 IC | 朴素基准 IC | 增量 IC（残差） |
|---|---|---|---|
| Δ净资产/净资产 | +0.342 | **+0.783** | +0.279（ICIR +4.31） |

**2. 日线模型预测财务变化：三种标签设计全部在收益端失败**（`估值方法与指标总结.md` §1）

* 把模型预测当权重去放大基线评分（λ 扫描），**λ=0（完全不用模型）结果最好**；
* 把模型学到的"增量"做**动量中性化**后，Q1−Q5 从 −5.23% 变成 **+0.09%**（几乎归零）
  → **模型学到的实质是动量代理（与 mom889 相关 +0.447），不是新信息**。

**3. 标签设计的其他结论**（同上文档 §8）

* `u9`（过去窗口 7 季 → 3 季）**更差**：验证 IC 0.2634/0.2746 vs `u6` 的 0.3394/0.3526。
  机制：过去窗口只覆盖 3 个日历季（u6 是完整的 4 季）→ **季节性偏差抵不掉**；
* `dcf` 当乘数会把 B/P 的 IC 从 0.081 削弱到 0.022；股息率指标全表最差；
* `up_down` 的量纲换算（改做"预测毛利改善额/市值"）IC 略升，但把排序头部推向大资产/低市值的
  银行保险股 → **已回退**。

**4. "可预测"与"能赚钱"是两件事** —— 这是这个仓库最大的教训：
财务量本身可以有很高的 IC（见第 1 条），但那不构成可交易的优势；
反过来，收益端的失败也不代表预测财务量没意义（两者要分开评估）。
仓库里的 `three损失函数优化实验.md`、`indicator特征删减*.md` 等
记录了对应模块的完整实验（含被否决的方案与原因）。

## 七、五个模型共用的约定

* **训练 7 个 epoch**（五个模型一致，便于横向比较）；
* `gen_train_data.py` 的**终端输出同一套措辞**（`Loading X data...` / `Started N processes...` /
  tqdm / `Finished X. Total files created: N` / `--- Summary ---`），便于编排脚本统一解析；
* 每个 `gen_train_data.py` **每次运行先无条件清空**自己的 `train/` ——
  免得改了采样参数或标签门之后，旧行留在原地与新行混成两套口径；
* h5 里**不放常量列**；模型 `.pt` 的**输出单位是真实值**
  （反标准化 affine 焼在模型 buffer 里，调用方不需要再换算 —— 漏掉会静默错一个量级）；
* 测试股票**固定**（各模型目录的 `test_symbols.txt`，各 889 只）：新数据只更新测试的**行**、
  不换测试的**股票**，否则新旧指标不可比。

## 八、目录速查

```
refresh_candidates.py   每日链入口（星期路由 + 选票 + 邮件）
refresh_models.py        一键重训（数据 → 训练 → 部署 → 记指标）。**2026-09-28 三合一**：
                        原 full_retrain.py（数据编排）+ retrain_all.py（训练编排）
                        + 同名的旧脚本（已失效）合并成这一个入口
get_stock_data.py       MySQL → CSV 导出
write_stock_data.py     数据修补与指标表刷新（重试 / 多数据源兜底）
schloss/                每股运营资本/股价（格雷厄姆 NCAV），独立一项
one/ … zero/            五个模型（见上表）
├─ one/factor_screen/   new24 因子集的筛选存档（脚本 + 结果表）
├─ one/实验结果_5win.md  因子集怎么选出来的
└─ two/nowcast/         nowcast 实验线（two 的标签来源；u1~u9 变体、收益端结论）
config_local.example.py 凭据模板（真值那份 config_local.py **不入库**）
data/  */train/  *.pt   数据与权重（*.pt 只入库 5 个生产模型 + legacy）
```

## 九、文档索引

| 文档 | 内容 |
|---|---|
| `估值方法与指标总结.md` | 估值公式与指标的完整清单（最全的一份，含"哪些没 work"） |
| `格雷厄姆投资思想与系统优化.md` | 投资框架与系统设计的来龙去脉 |
| `one/实验结果_5win.md` | 现役因子集 `new24` 的筛选过程 |
| `three损失函数优化实验.md`、`indicator特征删减*.md` | 各模块的实验记录 |
| `zero/zero标签与特征改造论证.md` | 避雷模型的标签/特征论证 |
| `two/nowcast/*.md` | nowcast 实验线：`phase1_label_v2.md`（标签定案）、`phase0_label_anchor_experiment.md`、`two_2x2_experiment.md`、估值分析等 |
| `M06-06-gtja191-formula-reference.md` | 用到的 GTJA191 因子公式参考 |

## 十、已知问题与未完成

* **`one` 的 warm start 会把 affine 带进训练**：`one/train.py` 载入旧 `.pt` 后没有把
  反标准化 buffer 还原成恒等，而训练标签是 z 空间的 —— 等于输出层从一个量级错的起点出发。
  （`two/train.py` 已经修了这一点。）**未修**，因为它会改变 one 下一次重训的起点。
* **`two/nowcast/` 里两个脚本的裸导入已经坏了**：`test_two_model.py`、
  `validate_factors_vs_price.py` 的 `from gen_train_data import add_technical_factor`
  命中了 `two/gen_train_data.py`（该名字在 `one/gen_train_data.py` 里）。见该目录的 README。
* **`seven` 的数据不含 KLS/KOE/SAU/TLV 四个交易所**：它们的季度历史不足 62 季，
  装不下"31 季过去 + 31 季未来"的标签窗口（`three` 只要 31 季、`zero` 只要 3 季，所以它们照常出文件）。
  这是过滤器的必然结果，不是故障。
* **训练段（`refresh_models.py` 的 `--stage train`）还没有端到端跑过一次**：命令拼装、五个模型的 CLI、
  指标解析都已逐条验证（用真实日志），但整段串起来尚未实跑。
* 代码里有几处**本机绝对路径**（如 `refresh_models.py` 的 `C:/quant_data/nowcast4`、
  `one/train.py` 的 `D:/quant_data/train_global`）—— 换机器要自己改。
* 研究侧未完成项见 `估值方法与指标总结.md` §8.3。
