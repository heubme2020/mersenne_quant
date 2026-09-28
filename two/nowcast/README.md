# two/nowcast —— nowcast 实验线（two 的 u9 标签与 u1~u8 变体研究都出自这里）

> 2026-09-28 从 `<repo>/nowcast/` 搬到这里（连同 `one/factor_screen/`，把"某模型的实验过程"
> 都归到该模型目录名下）。**搬的时候改了脚本里的路径算式**：原来 `HERE/'..'` 就是仓库根，
> 现在深了一层，全部改成 `HERE/'..'/'..'`（`variants.py` 那种多行字面量除外 —— 它不是路径）。

## 这里有什么

**能复现的（全部保留）**

| | 是什么 |
|---|---|
| 37 个 `.py` | `gen_data2.py`（造 X）、`add_u6_labels.py`（relabel 成 u6/u8/u9…）、`labels.py`（全部标签形态）、`variants.py`（u1~u9 的定义）、`model.py`（Nowcast）、`data.py`（memmap 装载）、`train.py`（arm/suffix 训练器）、`eval_v2.py` / `returns_eval.py`（评估）、`phase0_*`（标签/字段筛选）、`valuation_*` / `fund_momentum` / `ebit_ev_test`（估值与因子归因）… |
| 9 个 `.md` | 实验记录与结论：`phase1_label_v2.md`（标签设计 v2）、`phase0_label_anchor_experiment.md`、`rd_plan_daily2fin.md`、`shiller.md`、`two_2x2_experiment.md`、`生产估值公式诊断与改进.md` 等 |
| `phase0_results/` `phase0b_results/`（13 个 csv） | 标签/字段的可预测性筛选结果 |
| `two_exp/` | 2×2 实验的脚本 + 符号表 + 隔夜结果（8 个 checkpoint 已清） |
| `test_symbols_all31_889.txt` | 全局对齐的 889 只测试股票（`refresh_model.py` 与 `two/gen_train_data.py` 的兜底名单） |
| 2 份论文 PDF | 这条线的理论依据（LFM / 基本面预测） |

**已清掉的（可再生产物，见 `_trash_20260928/nowcast/`）**：`returns_eval*`（1.44 GB 逐样本 dump）、
`results/`（140 MB）、36 个实验 `.pt`（119 MB，含 `nowcast_both_u9.pt`）、`phase0b_factors_63.pkl`、
75 个日志。用对应的生成脚本重跑即可复现。

## 怎么跑（从仓库根）

```bash
python two/nowcast/gen_data2.py --exchanges SHZ --out C:/quant_data/nowcast4 --variant u1
python two/nowcast/add_u6_labels.py --root C:/quant_data/nowcast4 --suffix u9 --horizons 7 --past-win 3
python two/nowcast/train.py --arm both --suffix u9 --root C:/quant_data/nowcast4
```
`sys.path[0]` 会是 `two/nowcast/`，所以脚本之间的裸导入（`from data import Store`、
`import labels as L`、`from model import Nowcast`）照旧成立 ✓。旧的 memmap 分块仍在 `C:/quant_data/nowcast3`。

## 已知问题（2026-09-28 搬迁后逐个 import 测过：30/34 通过）

* `test_two_model.py`、`validate_factors_vs_price.py`：`from gen_train_data import
  add_technical_factor` **报 ImportError** —— 这是**既有问题**（搬迁前后解析到同一个文件
  `two/gen_train_data.py`，而那个名字在 `one/gen_train_data.py` 里）。正是 `two/two_features.py`
  docstring 里记的"坑 1"：裸导入命中了 two 自己那份。要用就把那行改成显式路径导入。
* `_smoke_mem.py`、`_smoke_rss.py`：开发用的小工具，import 时就干活、报
  `need at least one array to concatenate`（它们要的 chunk 不在默认 root 下）。搬迁反而把它们的
  `sys.path` 修对了（原来指向 `<root>/../one`）。
* 其余 30 个脚本 import 全部 OK，且 `two/_verify_vs_nowcast.py` 的**逐位校验 12/12 通过**
  （`two_features.build_x` vs `two/nowcast/gen_data2.build_x` 完全一致）—— 搬迁没有破坏任何路径。

## 与生产的关系

**two 的生产线不再依赖本目录**：`two/gen_train_data.py`（h5）与 `two/train.py` 是自足的
（特征/模型/标签三块当年都逐字移植到了 `two/two_features.py` / `two/two_nowcast_model.py` /
`two/two_labels.py`）。本目录现在的用途是**做变体实验**（u1~u8）与**查证当年的结论**，
入口是 `refresh_model.py --nowcast`（默认不跑）。
