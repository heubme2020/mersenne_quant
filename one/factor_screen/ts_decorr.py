"""纯时序因子去相关：top 40 时序因子 → 贪心去相关 → 24 个，存 ts_top24.csv。"""
import os
import sys
import re
import glob
import importlib.util
import inspect

import numpy as np
import pandas as pd

# 2026-09-28：本目录从 `<root>/factor_screen/` 搬到 `<root>/one/factor_screen/`，多上一层。
# 那句 `one_v2` 换成 `one/`：one_v2/ 早已不存在，而搬过去之后 `factor_screen` 这个包
# （namespace package，没有 __init__.py）就在 `one/` 下面 —— 下面那行 import 靠它解析。
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 输出/读回都在本目录内（原来写 ROOT+'factor_screen'，搬进来后要指向自己）
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'one'))

from factor_screen.precompute_factors import load_panel, load_module  # noqa: E402

H5 = os.path.join(HERE, 'new_factors.h5')
CORR_THRESH = 0.85
TOP_N = 40
KEEP = 24


def is_time_series(mod):
    try:
        src = inspect.getsource(mod.compute)
    except Exception:
        return False
    return not (re.search(r'\brank\(', src) or re.search(r'\bscale\(', src) or re.search(r'\bind_neutralize\(', src))


def main():
    ic = pd.read_csv(os.path.join(HERE, 'ic_results.csv'))
    ids = ic['id'].tolist()
    ts_ids = [i for i in ids if is_time_series(load_module(i))]
    top = ic[ic['id'].isin(ts_ids)].sort_values('mean_icir', key=abs, ascending=False).head(TOP_N)
    top_ids = top['id'].tolist()
    print(f'时序因子 {len(ts_ids)} 个，取 top {TOP_N} 个去相关', flush=True)

    # 预计算缺失的
    panel = load_panel()
    store = pd.HDFStore(H5, mode='a')
    try:
        existing = set(k.strip('/')[2:] for k in store.keys())
        for fid in top_ids:
            if fid in existing:
                continue
            mod = load_module(fid)
            res = mod.compute(panel)  # 纯时序，不 monkeypatch
            s = res if isinstance(res, pd.DataFrame) else res.to_frame()
            s = np.sign(s) * np.log1p(np.abs(s))
            store[f'f_{fid}'] = s.astype(np.float32)
    finally:
        store.close()

    # 日期子采样，算相关矩阵
    step = 3
    mats = {}
    store = pd.HDFStore(H5, mode='r')
    try:
        for fid in top_ids:
            df = store[f'f_{fid}'].iloc[::step]
            z = (df - df.mean()) / df.std()
            mats[fid] = z.fillna(0.0).values.ravel()
    finally:
        store.close()

    ids = list(mats.keys())
    X = np.vstack([mats[i] for i in ids]).T
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    corr = pd.DataFrame(X.T @ X / X.shape[0], index=ids, columns=ids)

    order = top.set_index('id').loc[ids, 'mean_icir'].abs().sort_values(ascending=False).index.tolist()
    selected = []
    for fid in order:
        if all(abs(corr.loc[fid, s]) < CORR_THRESH for s in selected):
            selected.append(fid)
        if len(selected) >= KEEP:
            break

    out = top[top['id'].isin(selected)].set_index('id').reindex(selected)
    out.to_csv(os.path.join(HERE, 'ts_top24.csv'))
    print(f'去相关选出 {len(selected)} 个：', flush=True)
    print(', '.join(selected), flush=True)


if __name__ == '__main__':
    main()
