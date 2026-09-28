# 量化选股系统（A 股为主 + 31 个海外市场）

一套个人量化研究的完整工程：**财报/日线数据 → 五个神经网络模型 → 每日选票 → 邮件推送**，
外加两年多来的因子筛选与标签设计实验记录。

> ⚠️ **免责声明**：这是研究/工程代码，**不构成投资建议**。数据来自第三方（MySQL 本地库、
> Financial Modeling Prep），不随仓库分发；复现需要自备数据源（见下面的「配置」）。

---

## 一、每日选票链（生产入口：`refresh_candidates.py`）

```
seven → three → zero → schloss → two → one → buy_predict.csv → 邮件
```

* **星期路由**：周五跑全链（含数据更新）；周六/周日空转；**周一~周四只跑 schloss/two/one**
  （财务数据没更新，只需重算与日线相关的那几环）。
* **选票规则**：把 `one` 的股票池按 `buffett` 降序取五档 —— 全池 / 127 / 31 / 7 / 3 ——
  每档内再按 `up_down` 降序取第一名，共 **5 只票**（邮件里按 `1: 3: 7: 31: 127:` 五行发出）。
* **打分公式**：`buffett = dcf × 每股净资产 / 收盘价 + schloss`
  （`dcf` 来自 seven 模型；实现见 `two/get_two_predict.py`）。

## 二、五个模型各自预测什么

| 目录 | 预测目标 | 在链里的角色 |
|---|---|---|
| `seven/` | 自由现金流 / 分红 / 净资产 三分支 × 1/3/7 季（及合并分） | 估值主体（产出 `dcf`） |
| `three/` | growth（上行）/ death（下行）双头 × 1/3/7 季 | 成长与风险 |
| `zero/` | 未来 381 个交易日的**最大回撤**（取负） | **避雷** |
| `two/` | nowcast 三头：毛利 / 营收 / 总资产的"中位变化"（7 季前向、3 季过去） | 候选池**砍半**（`up_down` = 毛利头） |
| `one/` | 5 个期限的 close 形态标签（max/median/min 的对数组合） | **最终排序**（`up_down`） |

每个模型目录是自足的：`*_features.py`（特征）、`*_model.py`（网络）、`gen_train_data.py`
（造 h5）、`train.py`（训练）、`get_*_predict.py`（推理）、`evaluate.py`（评估）。

## 三、数据线

```
MySQL(localhost) ──get_stock_data.py/write_stock_data.py──> data/{EX}/*.csv
        （income / balance / cashflow / indicator / mean / std / daily 七类）
                                   │
   各模型 gen_train_data.py ──> {模型}/train/*.h5 ──> train.py ──> {模型}/{模型}.pt
```

* **31 个交易所**（AMEX ASX BSE CNQ HKSE JKT JPX KLS KOE KSC LSE NASDAQ NSE NYSE OSL
  PAR SAO SAU SES SET SHH SHZ SIX STO TAI TLV TSX TSXV TWO WSE XETRA）。
* **测试股票固定**：每个模型目录下的 `test_symbols.txt`（各 889 只），新数据只更新测试的
  **行**、不换测试的**股票** —— 否则新旧指标不可比。
* `two` 的测试股票统一在**全局同步网格**上（每 31 个交易日一个锚点），才算得出逐日截面 IC。

## 四、一键重训

```bash
python full_retrain.py                 # 正式跑（先 --dry-run 看计划）
```

* **阶段 1**：MySQL → `data/{EX}/*.csv`（导出前后比对体积，**有任何 CSV 缩小 >5% 就停下报警**）
* **阶段 2**：重生五个模型的 h5（每个模型先清空自己的 `train/`）
* **训练段**：`retrain_all.py` 依次训练五个模型（**warm start**：接着已有 `.pt` 训）、
  部署到推理路径、把验证/测试指标追加到 `retrain_results.csv`

细粒度开关（`--stage data|train`、`--data-only <模型>`、`--only <模型>`、`--skip-export` …）
见 `full_retrain.py` 的 docstring。整链耗时以十小时计（`two` 的 h5 单这一项就 ≈5~6 小时）。

## 五、配置（**跑之前必读**）

凭据**不入库**（见 `.gitignore`）：

```bash
cp config_local.example.py config_local.py     # 然后填上你自己的值
```

填的是：MySQL DSN、FMP API key、SMTP 发件人/授权码/收件人名单。
不填的话 `get_stock_data.py` / `write_stock_data.py` / `refresh_candidates.py`
会在启动时报一句清楚的错（不会静默用错凭据）。

依赖版本见 `requirements.txt`（作者的运行环境：Windows + Python 3.11 + CUDA 12.1 的 torch）。

## 六、五个模型共用的约定

* **训练 7 个 epoch**（`one/three/seven/two/zero` 一致，便于横向比较）。
* `gen_train_data.py` 的**终端输出同一套措辞**（`Loading X data...` / `Started N processes...` /
  tqdm / `Finished X. Total files created: N` / `--- Summary ---`），编排脚本可以统一解析。
* 每个 `gen_train_data.py` **每次运行先无条件清空**自己的 `train/` ——
  免得改了采样参数或标签门之后，旧行留在原地与新行混成两套口径。
* h5 里**不放常量列**；模型 `.pt` 的**输出单位是真实值**（反标准化 affine 焼在 buffer 里，
  调用方不需要再换算）。

## 七、文档索引

| 文档 | 内容 |
|---|---|
| `估值方法与指标总结.md` | 估值公式与指标的完整清单（最全的一份） |
| `格雷厄姆投资思想与系统优化.md` | 投资框架与系统设计的来龙去脉 |
| `one/实验结果_5win.md` | 现役因子集 `new24` 的筛选过程（配套脚本与结果在 `one/factor_screen/`） |
| `three损失函数优化实验.md`、`indicator特征删减*.md` | 各模块的实验记录 |
| `zero/zero标签与特征改造论证.md`、`zero/zero待改进.md` | 避雷模型的标签/特征论证 |
| `two/nowcast/*.md` | nowcast 实验线（u1~u9 标签变体、收益端结论、估值分析） |

## 八、目录速查

```
refresh_candidates.py   每日链入口（含星期路由与选票）
full_retrain.py         一键重训（数据 + 训练 + 部署 + 记指标）
retrain_all.py          五个模型的训练编排（warm start / 部署 / 指标解析）
get_stock_data.py       MySQL -> CSV 导出
write_stock_data.py     数据修补与指标表刷新（重试 / 多数据源兜底）
schloss/                每股运营资本/股价（格雷厄姆 NCAV）一项独立计算
one/ two/ three/ seven/ zero/   五个模型
├─ one/factor_screen/   new24 因子集的筛选存档
└─ two/nowcast/         nowcast 实验线（two 的标签来源）
data/  */train/  *.pt   本地数据与权重，**不入库**
```
