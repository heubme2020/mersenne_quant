"""面板级预计算 22 个新因子（截面 rank → 单股 rolling rank 代理），存 HDF5 供查表。"""
import os
import sys
import glob
import importlib.util

import numpy as np
import pandas as pd
from tqdm import tqdm

# 2026-09-28：本目录从 `<root>/factor_screen/` 搬到 `<root>/one/factor_screen/`，多上一层。
# 同一行还把原来插 `<root>/one_v2` 的那句换成了 `<root>/one` —— one_v2/ 早已不存在，
# 而 `factor_config`（下面要 import 的 NEW_22）现在的家就在 `one/`。
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 输出/读回都在本目录内（原来写 ROOT+'factor_screen'，搬进来后要指向自己）
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'one'))

from factor_config import NEW_22  # noqa: E402

RANK_WINDOW = 10
OUT_H5 = os.path.join(HERE, 'new_factors.h5')


def load_panel():
    shz = pd.read_csv(os.path.join(ROOT, 'data/SHZ/daily_shz.csv'))
    shh = pd.read_csv(os.path.join(ROOT, 'data/SHH/daily_shh.csv'))
    df = pd.concat([shz, shh], ignore_index=True)
    df['amount'] = df['close'] * df['volume']
    df['vwap'] = (df['high'] + df['low'] + df['close']) / 3.0
    panel = {}
    for col in ['open', 'high', 'low', 'close', 'volume', 'amount', 'vwap']:
        panel[col] = df.pivot(index='date', columns='symbol', values=col).sort_index().astype(np.float32)
    amt = panel['amount']
    for w in [5, 10, 20, 60, 120, 150]:
        panel[f'adv{w}'] = amt.rolling(w, min_periods=w).mean().astype(np.float32)
    return panel


def load_module(factor_id):
    for d in ['zoo/gtja191', 'zoo/alpha101', 'zoo/qlib158', 'zoo/academic']:
        dpath = os.path.join(ROOT, d)
        if not os.path.isdir(dpath):
            continue
        for p in glob.glob(os.path.join(dpath, '*.py')):
            if os.path.basename(p).startswith('__'):
                continue
            spec = importlib.util.spec_from_file_location(os.path.basename(p)[:-3], p)
            m = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(m)
            except Exception:
                continue
            if m.__alpha_meta__['id'] == factor_id:
                return m
    return None


def _rolling_rank(x):
    return x.rolling(RANK_WINDOW, min_periods=RANK_WINDOW).rank(pct=True)


def main():
    panel = load_panel()
    print('panel loaded:', panel['close'].shape, flush=True)

    store = pd.HDFStore(OUT_H5, mode='w')
    try:
        for fid in tqdm(NEW_22, desc='预计算因子'):
            mod = load_module(fid)
            if mod is None:
                print(f'{fid}: 无模块', flush=True)
                continue
            mod.rank = _rolling_rank
            mod.scale = lambda x, a=1.0: x
            mod.ind_neutralize = lambda x, g: x
            try:
                res = mod.compute(panel)
            except Exception as e:
                print(f'{fid}: 计算报错 {e}', flush=True)
                continue
            if isinstance(res, pd.DataFrame):
                s = res
            elif isinstance(res, pd.Series):
                s = res.to_frame()
            else:
                continue
            s = np.sign(s) * np.log1p(np.abs(s))  # 压爆炸
            store[f'f_{fid}'] = s.astype(np.float32)
        print('saved to', OUT_H5, flush=True)
    finally:
        store.close()


if __name__ == '__main__':
    main()
