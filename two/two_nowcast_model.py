"""`two` 的模型：`Nowcast`（3 头）—— 逐字复制自 `two/nowcast/model.py`（+ `one/one_model.py`）。

## 为什么要复制一份

用户 2026-09-26 的要求：**two 不再依赖 `nowcast/` 目录**（它是实验目录，要清理冗余）。
推理端 `two/get_two_predict.py` 已经改成从 `two_features` 取特征了，训练端的模型定义同理。

## 与原文件的逐字对应

    two/nowcast/model.py                     本文件
    ---------------------------------    ---------------------------------------------
    from one_model import ONE, Regr...   同名类，改用【显式路径导入】（见下面的坑）
    from labels import FIELDS, ...       from two_labels（头布局：u9 = 3 字段 × 1 期限）
    class Nowcast(_ONE): ...             类体、__init__ 默认值、forward 全部逐字保留

唯一的语义差别：头布局常量取的模块不同（原文件取 `two/nowcast/labels.py` 的 3×3 基线，
本文件取 `two_labels` 的 u9 = gpMedP3F/revMedP3F/taMedP3F，即 3 字段 × 1 期限）。
`Nowcast(n_fields=3, n_horizons=1)` 与现有 `two/two.pt`（3 头、1,028,486 参数）同构。

## ⚠️ 坑（与 two_features.py 记的是同一个）

`two/` 在 `sys.path` 最前，任何**裸导入** `one/` 下同名模块（尤其是 `gen_train_data`）
都会命中 `two/` 自己那份。所以 `one_model` 必须按【绝对路径】加载。
"""
import importlib.util
import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ONE = os.path.abspath(os.path.join(HERE, '..', 'one'))
sys.path.insert(0, ONE)


def _load_one_module(name):
    """从 one/ 按【绝对路径】加载模块 —— 不受 sys.path 顺序影响。

    ⚠️ 模块名必须保持 `name` 本身（不是 `_one_{name}`）并注册进 sys.modules：
    本文件里的 `Nowcast` **继承** `one_model.ONE`，它的子模块（Encoder/AttentionPool/
    RegressionHead/...）都是 `one_model` 里的类。`torch.save(model)` 会把那些子模块的
    **类**按 `模块名 类名` 存进 pickle，所以模块名一旦改成私有的 `_one_one_model`，
    存盘当场就崩：`PicklingError: Can't pickle <class '_one_one_model.Encoder'>`
    （2026-09-27 实测踩到）。注册成 `one_model` 还顺带让 .pt 里的引用与
    two/nowcast/model.py 存出来的完全一致（同一个 one/one_model.py）。
    """
    path = os.path.join(ONE, f'{name}.py')
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)      # 载入端 unpickle 时也靠这一步解析
    spec.loader.exec_module(mod)
    return mod


_one_model = _load_one_module('one_model')
_ONE = _one_model.ONE
RegressionHead = _one_model.RegressionHead
set_output_scale = _one_model.set_output_scale          # 存盘前把反标准化焼进 buffer（必须有）
AUX_OUTPUT_DAYS = _one_model.AUX_OUTPUT_DAYS

import two_labels as L                                  # noqa: E402  头布局（u9）

FIELDS = list(L.HEAD_NAMES)
HORIZONS = list(L.HORIZONS)
N_FIELDS, N_HORIZONS = len(FIELDS), len(HORIZONS)
N_HEADS = N_FIELDS * N_HORIZONS
# 展平顺序与标签数组一致（字段优先），供评估/日志用。
# u9 只有一个期限，且头名本身已经含了「P3F」（过去 3 季 / 前向 7 季），所以不再拼期限后缀
# —— 与 h5 里 `key='label'` 的列名一字不差。
FLAT_NAMES = list(FIELDS)
HEAD_NAMES = FLAT_NAMES          # 与 nowcast 的叫法对齐（那边 train.py 用的就是 HEAD_NAMES）


class Nowcast(_ONE):
    """与 one.ONE 完全同构，只把标量头换成 (B, 字段, 期限)。"""

    def __init__(self, input_shape=(31, 127 * 7), d_model=128, num_heads=8,
                 num_layers=4, dropout=0.1, aux_out_len=128, aux_out_dim=3,
                 n_fields=None, n_horizons=None):
        # n_fields/n_horizons 默认取 two_labels 的 u9 布局（3 字段 × 1 期限）；
        # 其它变体（如基线 3×3）可显式覆盖。
        nf = N_FIELDS if n_fields is None else n_fields
        nh = N_HORIZONS if n_horizons is None else n_horizons
        self.n_fields, self.n_horizons, self.n_heads = nf, nh, nf * nh
        super().__init__(input_shape, d_model=d_model, num_heads=num_heads,
                         num_layers=num_layers, dropout=dropout,
                         aux_out_len=aux_out_len, aux_out_dim=aux_out_dim,
                         n_horizons=self.n_heads)
        # RegressionHead(d_model, [out_features, out_seq]) -> view(B, out_seq, out_features)
        # 想要 (B, 字段, 期限)：out_seq=字段, out_features=期限
        self.close_head = RegressionHead(d_model, [nh, nf], dropout)

    def forward(self, x):
        aux, scalar = super().forward(x)
        return aux, scalar.view(scalar.size(0), self.n_fields, self.n_horizons)


if __name__ == '__main__':
    m = Nowcast([31, 127 * 7])
    x = torch.randn(2, 127 * 7, 31)
    aux, pred = m(x)
    print('aux', tuple(aux.shape), 'pred', tuple(pred.shape))
    print('params', sum(p.numel() for p in m.parameters()))
