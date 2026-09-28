# one/factor_screen —— `NEW_24` 因子集的筛选存档

> 2026-09-28 从 `<repo>/factor_screen/` 搬到这里（随 `one_5win/` 的整理一起归到 `one/` 名下）。

## 这是什么

`one` 现在用的因子集 **NEW_24**（`one/factor_config.py:17`）是用**海外市场数据贪心选出**的
24 个因子（对 A 股纯样本外）。本目录是那次筛选的脚本与原始产物：

| 文件 | 是什么 |
|---|---|
| `ic_results.csv` | 阶段 1：所有候选因子的截面 ICIR |
| `top31.csv` | 阶段 2：去相关后选出的 31 个（后来定成 24 个） |
| `ts_top24.csv` | 纯时序因子的去相关结果（另一次筛选） |
| `current_ic.csv` | 现有 24 个技术因子的 ICIR（用来找最差的 7 个） |
| `screen.py` `ts_decorr.py` `precompute_factors.py` `current_ic.py` | 产出上表的四个脚本 |

**结论与解读在 `one/实验结果_5win.md`**，本目录只是它引用的原始产物。

## ⚠️ 这四个脚本现在跑不起来（存档性质，故意保留）

它们筛的因子库 `<repo>/zoo/`（`alpha101` / `gtja191` / `qlib158` / `academic`）与
`<repo>/one_v2/` **都已不存在** —— `screen.py` 的候选因子列表因此是空的。
保留而没删的理由：它们是 `new24` 的**可审计来源**（结果表 + 当时的筛选逻辑）。

搬进 `one/` 时只改了路径、没动逻辑：

* 四个脚本的 `ROOT` 多上一层（`dirname(dirname(dirname(__file__)))`）
* `precompute_factors.py` / `ts_decorr.py` 里 `sys.path.insert(0, ROOT/'one_v2')`
  → `ROOT/'one'`（`one_v2` 早已不存在；`one/` 才是 `factor_config` 现在的家）
* `one/gen_train_data.py` 与 `one/get_one_predict.py`（**生产推理**）里的
  `'../factor_screen/new_factors.h5'` → `'factor_screen/new_factors.h5'`
  （那个 `.h5` 本身不存在，只在 `FACTOR_MODE=plan1/plan2` 时用，是惰性常量）
* `one/实验结果_5win.md` 里 `factor_screen/xxx` 的引用 → `one/factor_screen/xxx`
