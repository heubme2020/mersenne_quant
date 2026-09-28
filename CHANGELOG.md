# 更新记录

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 的结构，
版本号用 [语义化版本](https://semver.org/lang/zh-CN/)。

---

## [v0.1.0] — 2026-09-28

**首次公开**。把陆续写成的代码与实验记录整理成一个可读、可复现、可安全分享的版本。

### 包含

* **五个模型**（`one` / `two` / `three` / `seven` / `zero`）：各自的
  特征构造、网络定义、训练器（warm start）、推理与评估脚本。
  每个模型目录自足 —— 不依赖其它模型目录、也不依赖实验目录。
* **每日选票链**（`refresh_candidates.py`）：`seven → three → zero → schloss → two → one →
  buy_predict.csv → 邮件`，含星期路由（周五全链、周末空转、周一~周四只跑 schloss/two/one）
  与选票规则（`buffett` 五档 × 各档 `up_down` 头名 = 5 只票）。
* **一键重训**（`full_retrain.py` + `retrain_all.py`）：数据导出（含"CSV 缩小 >5% 就报警"）→
  五个模型的 h5 重生 → 训练（warm start）→ 部署 → 指标落盘。
* **5 个生产模型权重**（`one/two/three/seven/zero` 的 `.pt`，共 16.3 MB）+ `two/two_legacy.pt`
  （旧架构推理的回退路径在用）。
* **实验记录**（这次整理的重点）：
  * `估值方法与指标总结.md`：估值公式与指标的完整清单，含**哪些没 work 及原因**；
  * `two/nowcast/`：nowcast 实验线 —— 标签变体（u1~u9）的全部结论，其中包括
    **"日线模型预测财务变化在收益端全线失败、模型实质是动量代理"**这个关键负面结果；
  * `one/factor_screen/` + `one/实验结果_5win.md`：现役因子集 `new24` 的筛选存档；
  * 各模块的实验记录（`three损失函数优化实验.md`、`indicator特征删减*.md`、`zero/*.md` 等）；
  * 两份参考论文与 `M06-06-gtja191-formula-reference.md`。

### 工程约定（这一年里逐步统一）

* 五个模型**都是 7 个 epoch**，便于横向比较；
* `gen_train_data.py` **终端输出同一套措辞**，编排脚本可统一解析；
* 每个 `gen_train_data.py` **每次运行先无条件清空**自己的 `train/`
  —— 避免改了采样参数/标签门后新旧口径混在一起；
* h5 **不放常量列**；模型 `.pt` **直接输出真实单位**（反标准化 affine 焼进 buffer）；
* 测试股票**固定**（各 889 只），新数据只更新测试的"行"、不换"股票"。

### 安全 / 隐私（首次公开前的处理）

* **凭据全部外置**：MySQL DSN、FMP API key、SMTP 授权码与收件人名单
  → `config_local.py`（**不入库**，另给 `config_local.example.py` 模板）；
  三个脚本缺配置时**报错而不是静默用错凭据**。
* **历史重建**：上述凭据最初是明文硬编码的，本仓库的历史是**重建过的**——
  全部提交里搜这些串（以及邮箱、密钥）均为空。
* **内部系统文档不外传**（`系统说明.pdf`、`系统优化.txt`）。
* **数据不入库**：`data/`、`*/train/`、回退快照、清理暂存目录都在 `.gitignore` 里。

### 已知问题

见 [README「已知问题与未完成」](README.md#十已知问题与未完成)，摘要：

* `one` 的 warm start 未把 affine 还原成恒等（训练起点量级错位）；
* `two/nowcast/` 两个脚本的裸导入已失效（命中了 `two/gen_train_data.py`）；
* `seven` 的数据不含 KLS/KOE/SAU/TLV（历史不足 62 季，是过滤器的必然结果）；
* `full_retrain.py` 的**训练段（阶段 3）尚未端到端实跑过**；
* 几处本机绝对路径需要按环境修改。

---

## 未发布

### 移除

* **计划/待改进类文档 3 份**（内容要么已实现、要么已被后续实验取代）：
  `one/loss优化实验规划.md`、`zero/zero待改进.md`、`two/nowcast/rd_plan_daily2fin.md`。
* **论文与摘录 6 份**（PDF 3.3 MB + 4 份读论文的摘录笔记）：
  两篇 LFM/基本面预测论文、`_paper_improving.txt`、`paper_lfm_1711.04837.md`、
  `euclidean_technologies.md`、`shiller.md`。
* **两处目录说明**（`one/factor_screen/README.md`、`two/nowcast/README.md`）与
  `one/实验结果_5win.md` 顶部的迁移说明块。
* 以上都移入 `_trash_20260928/docs/`（不是从历史里删除）；实验记录类文档全部保留。

### 变更

* **三个重训脚本合并成一个入口 `refresh_models.py`**（2026-09-28）：

  | 原来的 | 职责 |
  |---|---|
  | `full_retrain.py` | 数据编排：MySQL 导出 → 重生五个模型的 h5 →（可选）nowcast 分块 |
  | `retrain_all.py` | 训练编排：五模型 warm start / 部署 / 解析验证与测试指标 |
  | `refresh_model.py`（旧名，单数） | 更早的"数据+训练一把梭" —— 2026-09 已失效（模块级裸导入直接 ImportError、缺 two、训练在同进程里跑） |

  合并为 `refresh_models.py`，**语义不变**：`--stage data|train|all`、`--only`、`--data-only`、
  `--nowcast` / `--variants` / `--nowcast-root`、`--skip-export` / `--skip-h5`、`--dry-run` 全部保留。
  合并前逐项比对：五个模型的配方（10 个字段 × 5 个）、三处测试指标正则、`H5_JOBS`、
  nowcast 相关常量 —— **全部逐字段一致**。

* `retrain_all.py` 的**转发垫片已删除**（它只为"07:57 那次进行中的重训会在阶段 3 调用它"
  而存在；那次重训已终止，垫片随之移入 `_trash_20260928/`）。等价入口：
  `refresh_models.py --stage train`。
* 入口文件名定为 **`refresh_models.py`（复数）**；文中出现的 `refresh_model.py`（单数）
  一律指"被它取代的那个旧脚本"。
* `full_retrain.py` 已移入 `_trash_20260928/`（正在跑的进程已把它读进内存，不受影响）。
* 文档里的脚本名引用同步更新：`README.md`、`two/train.py`、`two/get_two_predict.py`、
  `two/nowcast/variants.py`。
