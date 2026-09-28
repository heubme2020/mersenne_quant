"""把 zoo 里的 alpha 因子适配成单股时序因子（截面 rank/scale/ind_neutralize → 单股 rolling 代理）。

用法：compute_per_stock(factor_id, df) -> Series（df 需含 open/high/low/close/volume，index=date）。
"""
import os
import sys
import glob
import importlib.util

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RANK_WINDOW = 10  # 单股 rolling rank 窗口

_module_cache = {}
_id_to_path = {}


def _load(path):
    if path not in _module_cache:
        spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _module_cache[path] = mod
    return _module_cache[path]


def _build_id_to_path():
    if _id_to_path:
        return
    for d in ['zoo/gtja191', 'zoo/alpha101', 'zoo/qlib158', 'zoo/academic']:
        dpath = os.path.join(ROOT, d)
        if not os.path.isdir(dpath):
            continue
        for p in glob.glob(os.path.join(dpath, '*.py')):
            if os.path.basename(p).startswith('__'):
                continue
            try:
                mod = _load(p)
                _id_to_path[mod.__alpha_meta__['id']] = p
            except Exception:
                pass


def _rolling_rank(x):
    return x.rolling(RANK_WINDOW, min_periods=RANK_WINDOW).rank(pct=True)


def compute_per_stock(factor_id, df):
    """对单只股票的 DataFrame 计算因子，返回 (date,) Series；失败返回 None。"""
    _build_id_to_path()
    path = _id_to_path.get(factor_id)
    if path is None:
        return None
    mod = _load(path)

    df = df.copy()
    if 'amount' not in df.columns:
        df['amount'] = df['close'] * df['volume']
    if 'vwap' not in df.columns:
        df['vwap'] = (df['high'] + df['low'] + df['close']) / 3.0

    cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'vwap']
    # 单股时列名必须统一成 's'，否则 high-low 这类按列对齐会变 NaN
    panel = {c: df[c].rename('s').to_frame() for c in cols if c in df.columns}
    amt = df['amount']
    for w in [5, 10, 20, 60, 120, 150]:
        panel[f'adv{w}'] = amt.rolling(w, min_periods=w).mean().rename('s').to_frame()

    # 截面算子 → 单股代理
    mod.rank = _rolling_rank
    mod.scale = lambda x, a=1.0: x
    mod.ind_neutralize = lambda x, g: x

    try:
        res = mod.compute(panel)
    except Exception:
        return None

    if isinstance(res, pd.DataFrame):
        s = res.iloc[:, 0] if res.shape[1] == 1 else res.mean(axis=1)
    elif isinstance(res, pd.Series):
        s = res
    else:
        return None
    # 压住爆炸：保序变换 sign(x)*log1p(|x|)（对正常值几乎无影响，对爆炸值压到 ~±20）
    return np.sign(s) * np.log1p(np.abs(s))
