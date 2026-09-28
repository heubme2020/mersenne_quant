"""`two` 自己的特征构造 —— 从 `two/nowcast/gen_data2.py` **逐字抽取**，让 two 不再依赖 nowcast。

## 为什么要有这个文件（2026-09-26）

`two/get_two_predict.py` 原先写的是 `from gen_data2 import build_x, ...` ——
也就是**生产推理依赖 `nowcast/` 目录**。而 nowcast 是实验目录、用户要清理冗余，一删，
每日选票就全断。抽出来之后 `two/` 自给自足。

## 为什么是"逐字复制"而不是重写

`build_x` 的原 docstring 明写：

> 31 维的列集合与列序、归一化基准、idx 分母三者任何一处漂移都会**静默出错**。

这类"看起来一样但不一样"的漂移不会报错，只会让模型输入悄悄变形、日更选票慢慢走偏。
所以本文件是逐字搬运，并且有 `_verify_vs_nowcast.py` 用**真实日线窗口**做**逐位相等**校验
（不是肉眼比对、不是近似相等）。**改这里的任何一行，都要重跑那个校验。**

## 依赖（都在 one/ 下，没有一个在 nowcast/ 里）

    one/factor_config.py    get_technical_factors('new24')
    one/gen_train_data.py   add_technical_factor
    one/factor_pool.py      add_pool_factors

## ⚠️ 两个必须知道的坑

1. **不能写 `from gen_train_data import add_technical_factor`（裸导入）** ——
   一旦 `two/gen_train_data.py` 存在（格式对齐后就会有），而 `two/` 在 sys.path 最前，
   裸导入就会命中 **two 自己那份**，而不是 `one/gen_train_data.py`。所以这里用
   **显式路径导入**，从根上免疫命名冲突。
2. **不要和 two/ 的旧实现混** —— `get_two_predict_legacy.py` 用的是 CURRENT_24 因子集、
   idx 分母误用 380。本文件与 **nowcast 训练端一致**：raw 块序
   `open,high,low,close,volume,delta`、技术因子 NEW_24、idx 分母 `DAYS_INPUT-1 = 888`。
"""
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ONE = os.path.abspath(os.path.join(HERE, '..', 'one'))
sys.path.insert(0, ONE)


def _load_one_module(name):
    """从 one/ 按【绝对路径】加载模块 —— 不受 sys.path 顺序影响（见模块 docstring 的坑 1）。"""
    path = os.path.join(ONE, f'{name}.py')
    spec = importlib.util.spec_from_file_location(f'_one_{name}', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_otd = _load_one_module('gen_train_data')      # add_technical_factor
_ocfg = _load_one_module('factor_config')      # get_technical_factors
_opool = _load_one_module('factor_pool')       # add_pool_factors

add_technical_factor = _otd.add_technical_factor
get_technical_factors = _ocfg.get_technical_factors
add_pool_factors = _opool.add_pool_factors

DAYS_INPUT = 127 * 7          # 889
AUX_DAYS = 128
WINDOW = DAYS_INPUT + AUX_DAYS
REF_IDX = DAYS_INPUT - 1
RAW = ['open', 'high', 'low', 'close', 'volume', 'delta']
FACTORS = get_technical_factors('new24')
COLS = RAW + FACTORS + ['idx']

# A 股财报披露截止日：(季度末 -> 最晚披露月日)；Q4 顺延到次年
DEADLINE = {'0331': (4, 30), '0630': (8, 31), '0930': (10, 31), '1231': (4, 30)}


def available_date(end_dates):
    """季度末 -> 该财报的【最晚披露日】（YYYYMMDD）。用它做 asof 才是无前视的。"""
    out = np.empty(len(end_dates), dtype='int64')
    for i, ed in enumerate(np.asarray(end_dates, dtype='int64')):
        y, md = int(ed) // 10000, f'{int(ed) % 10000:04d}'
        m, d = DEADLINE[md]
        out[i] = (y + (1 if md == '1231' else 0)) * 10000 + m * 100 + d
    return out


def build_x(w):
    """日线窗口 w（WINDOW 行）-> 模型输入 X (WINDOW, 31)，float32。

    （逐字复制自 `two/nowcast/gen_data2.py`；改动必须重跑 `_verify_vs_nowcast.py`。）

    训练与推理必须共用这一段：31 维的列集合与列序、归一化基准、idx 分母三者任何一处
    漂移都会静默出错。按【名字】选取 f[COLS]，所以 w 的物理列序无所谓。
    """
    rc, rv = w['close'].iloc[REF_IDX], w['volume'].iloc[REF_IDX]
    if not (rc > 0 and rv > 0):
        return None
    w = w.copy()
    w['open'] = w['open'] / rc
    w['high'] = w['high'] / rc
    w['low'] = w['low'] / rc
    w['close'] = w['close'] / rc
    w['volume'] = w['volume'] / rv
    w['delta'] = w['high'] - w['low']
    f = add_technical_factor(add_pool_factors(w))
    f['idx'] = f.index / (DAYS_INPUT - 1.0)
    return f[COLS].replace([np.inf, -np.inf], 0).fillna(0).clip(-127, 127).values.astype('float32')
