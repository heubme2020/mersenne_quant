"""因子筛选：Alpha101 + GTJA191 + Qlib158 + academic → 截面 ICIR 筛选 + 去相关 → top 31。

用法：
  python factor_screen/screen.py            # 阶段1：算所有候选因子的 ICIR，存 ic_results.csv
  python factor_screen/screen.py --decorr   # 阶段2：重算 top N 因子，去相关选出 31 个
"""
import os
import sys
import importlib.util
import math
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

# 2026-09-28：本目录从 `<root>/factor_screen/` 搬到 `<root>/one/factor_screen/`，
# 所以定位仓库根要**多上一层**（原来是 dirname(dirname(__file__))）。
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 输出/读回都在本目录内（原来写 ROOT+'factor_screen'，搬进来后要指向自己）
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)  # 使 src.factors.base 可导入

HORIZONS = [1, 3, 7, 31, 127]
TOP_N = 100          # 阶段2 重算的 top 因子数
CORR_THRESH = 0.85   # 去相关阈值
ZOO_DIRS = ['zoo/alpha101', 'zoo/gtja191', 'zoo/qlib158', 'zoo/academic']


def load_panel():
    """加载 A 股 SHZ+SHH，算 amount，pivot 成 (date × symbol) 面板。"""
    shz = pd.read_csv(os.path.join(ROOT, 'data/SHZ/daily_shz.csv'))
    shh = pd.read_csv(os.path.join(ROOT, 'data/SHH/daily_shh.csv'))
    df = pd.concat([shz, shh], ignore_index=True)
    df['amount'] = df['close'] * df['volume']
    df['vwap'] = (df['high'] + df['low'] + df['close']) / 3.0  # 无日内数据，用近似
    panel = {}
    for col in ['open', 'high', 'low', 'close', 'volume', 'amount', 'vwap']:
        p = df.pivot(index='date', columns='symbol', values=col)
        panel[col] = p.sort_index().astype(np.float32)
    return panel


def compute_labels(close_panel):
    """5 个 horizon 的 close 标签（面板级，向量化）。"""
    labels = {}
    for h in HORIZONS:
        base = close_panel.shift(-1)  # 明天收盘
        wmax = close_panel.rolling(h, min_periods=h).max().shift(-(h + 1))
        wmed = close_panel.rolling(h, min_periods=h).median().shift(-(h + 1))
        wmin = close_panel.rolling(h, min_periods=h).min().shift(-(h + 1))
        labels[h] = np.log(wmax) + np.log(wmed) + np.log(wmin) - 3.0 * np.log(base)
    return labels


def per_date_spearman(factor, label):
    """逐日截面 Spearman IC，返回 Series（NaN 已去除）。"""
    fr = factor.rank(axis=1, pct=True)
    lr = label.rank(axis=1, pct=True)
    fc = fr.sub(fr.mean(axis=1), axis=0)
    lc = lr.sub(lr.mean(axis=1), axis=0)
    denom = np.sqrt(fc.pow(2).sum(axis=1)) * np.sqrt(lc.pow(2).sum(axis=1))
    ic = (fc * lc).sum(axis=1) / denom
    return ic.dropna()


def icir(factor, label):
    s = per_date_spearman(factor, label)
    if len(s) < 20:
        return np.nan
    std = s.std()
    return float(s.mean() / std) if std > 1e-8 else np.nan


def load_alpha(path):
    spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def list_alpha_files():
    files = []
    for d in ZOO_DIRS:
        dpath = os.path.join(ROOT, d)
        if not os.path.isdir(dpath):
            continue
        for f in sorted(os.listdir(dpath)):
            if f.endswith('.py') and not f.startswith('__'):
                files.append(os.path.join(dpath, f))
    return files


def compute_factor(mod, panel):
    """读 columns_required，跳过 fundamental/缺列，传全面板调用 compute。"""
    cols = mod.__alpha_meta__['columns_required']
    if any(c.startswith('fund:') for c in cols):
        return None
    if any(c not in panel for c in cols):
        return None  # 缺列，跳过
    return mod.compute(panel)  # 传全面板，内部可能访问 vwap 等


def phase1():
    panel = load_panel()
    labels = compute_labels(panel['close'])
    files = list_alpha_files()
    print(f'候选因子文件：{len(files)}', flush=True)

    rows = []
    for path in tqdm(files, desc='ICIR 筛选'):
        try:
            mod = load_alpha(path)
            factor = compute_factor(mod, panel)
            if factor is None:
                continue
            factor = factor.reindex_like(panel['close'])
            icirs = [icir(factor, labels[h]) for h in HORIZONS]
            mean_icir = float(np.nanmean(icirs)) if any(~np.isnan(icirs)) else np.nan
            if np.isnan(mean_icir):
                continue
            row = {'id': mod.__alpha_meta__.get('id', os.path.basename(path)[:-3]), 'mean_icir': mean_icir}
            for h, v in zip(HORIZONS, icirs):
                row[f'icir_{h}'] = v
            rows.append(row)
        except Exception as e:
            pass  # 跳过报错的因子

    df = pd.DataFrame(rows).sort_values('mean_icir', key=abs, ascending=False)
    out = os.path.join(HERE, 'ic_results.csv')
    df.to_csv(out, index=False)
    print(f'完成 {len(df)} 个因子，保存 {out}', flush=True)
    print(df.head(50).to_string(index=False), flush=True)


def phase2():
    csv = os.path.join(HERE, 'ic_results.csv')
    if not os.path.exists(csv):
        print('先跑阶段1 生成 ic_results.csv')
        return
    top = pd.read_csv(csv).head(TOP_N)
    ids = top['id'].tolist()

    panel = load_panel()
    # 阶段2 用日期子采样，省内存
    step = 3
    panel = {c: p.iloc[::step] for c, p in panel.items()}

    files = list_alpha_files()
    file_by_id = {}
    for path in files:
        try:
            m = load_alpha(path)
            file_by_id[m.__alpha_meta__.get('id', os.path.basename(path)[:-3])] = path
        except Exception:
            pass

    mats = {}
    for fid in tqdm(ids, desc='重算 top 因子'):
        path = file_by_id.get(fid)
        if path is None:
            continue
        try:
            mod = load_alpha(path)
            factor = compute_factor(mod, panel)
            if factor is None:
                continue
            factor = factor.reindex_like(panel['close'])
            z = (factor - factor.mean()) / factor.std()
            mats[fid] = z.fillna(0.0).values.ravel()
        except Exception:
            continue

    ids = list(mats.keys())
    X = np.vstack([mats[i] for i in ids]).T  # (n, K)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    corr = X.T @ X / X.shape[0]
    corr = pd.DataFrame(corr, index=ids, columns=ids)

    order = top.set_index('id').loc[ids, 'mean_icir'].abs().sort_values(ascending=False).index.tolist()
    selected = []
    for fid in order:
        if all(abs(corr.loc[fid, s]) < CORR_THRESH for s in selected):
            selected.append(fid)
        if len(selected) >= 31:
            break

    out = top[top['id'].isin(selected)].set_index('id').reindex(selected)
    out.to_csv(os.path.join(HERE, 'top31.csv'))
    print(f'去相关后选出 {len(selected)} 个因子：', flush=True)
    print(out.to_string(), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--decorr', action='store_true')
    args = p.parse_args()
    if args.decorr:
        phase2()
    else:
        phase1()
