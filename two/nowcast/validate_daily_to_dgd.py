"""
validate_daily_to_dgd.py — 验证「日线数据（two 的 31 特征）→ three 输出变化(Δgd)」的信号强度。

label_h = gd_combined( 31 季度窗口往前挪 h 个季度 ) − gd_combined( 当前窗口 )，h ∈ {1,3,7}
日线特征用 two 模型那套 add_technical_factor（24 个技术因子），取「期末」最新值。
多交易所验证。
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'three')))
from three_model import THREE  # noqa: F401
from factors_long import add_technical_factor_long, FACTORS, SCALE_FREE  # noqa: F401  拉长版技术因子

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

EXCHANGES = ['NASDAQ', 'NYSE', 'AMEX', 'LSE', 'HKSE', 'JPX']
N_SYMBOLS_PER_EX = 150
SHIFTS = [1, 3, 7]



def load_three_model(device):
    path = os.path.join(BASE, 'three', 'three.pt')
    return torch.load(path, map_location=device, weights_only=False).to(device).eval()


def list_symbols(exchange, n):
    ind = pd.read_csv(os.path.join(BASE, 'data', exchange, f'indicator_{exchange.lower()}.csv'),
                      usecols=['symbol'])
    symbols = sorted(s for s in ind['symbol'].dropna().unique() if isinstance(s, str))
    rng = np.random.RandomState(42)
    return list(rng.choice(symbols, min(n, len(symbols)), replace=False))


def collect_h5_files(symbols):
    train_dir = os.path.join(BASE, 'three', 'train')
    sym_set = set(symbols)
    by_symbol = defaultdict(list)
    for fn in os.listdir(train_dir):
        if not fn.endswith('.h5'):
            continue
        sym, end_str = fn[:-3].rsplit('_', 1)
        if sym in sym_set:
            by_symbol[sym].append((int(end_str), os.path.join(train_dir, fn)))
    for sym in by_symbol:
        by_symbol[sym].sort()
    return by_symbol


@torch.no_grad()
def compute_gd(model, features, device):
    x = torch.tensor(features, dtype=torch.float32, device=device)
    growth, death = model(x)
    gd = growth / (death + 1e-6)
    return gd.sum(dim=-1).squeeze(-1).cpu().numpy()


def load_features(paths):
    arrs = [pd.read_hdf(p).iloc[:, :-6].values.astype('float32') for p in paths]
    return np.stack(arrs)


def compute_factors_map(exchange, symbols):
    """对每个 symbol 计算技术因子，返回 {symbol: df(按 date 排序, 含 FACTORS 列)}。"""
    daily = pd.read_csv(os.path.join(BASE, 'data', exchange, f'daily_{exchange.lower()}.csv'))
    daily = daily.sort_values(['symbol', 'date']).reset_index(drop=True)
    out = {}
    for sym, g in daily.groupby('symbol'):
        if sym not in set(symbols):
            continue
        g = g.copy()
        g['delta'] = g['high'] - g['low']  # add_technical_factor 里要用 delta 列
        g = add_technical_factor_long(g)
        out[sym] = g[['date'] + FACTORS].reset_index(drop=True)
    return out


def latest_factor_values(factor_df, endDate):
    """取 date <= endDate 的最后一行因子的值（None 若无数据）。"""
    m = factor_df[factor_df['date'] <= endDate]
    if len(m) == 0:
        return None
    return m.iloc[-1][FACTORS].values.astype(float)


def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 30:
        return np.nan, len(a)
    return pd.Series(a).rank().corr(pd.Series(b).rank()), len(a)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_three_model(device)
    print(f'设备: {device}\n')

    # 累积所有交易所的样本：每行 = [dgd_1, dgd_3, dgd_7, *24个因子]
    all_rows = []
    for ex in EXCHANGES:
        symbols = list_symbols(ex, N_SYMBOLS_PER_EX)
        by_symbol = collect_h5_files(symbols)
        if not by_symbol:
            print(f'{ex}: 无命中，跳过')
            continue

        # 计算 gd_combined（three）
        gd = {}
        batch, batch_keys = [], []
        for sym, files in by_symbol.items():
            for end, path in files:
                batch.append(path)
                batch_keys.append((sym, end))
                if len(batch) >= 256:
                    vals = compute_gd(model, load_features(batch), device)
                    for k, v in zip(batch_keys, vals):
                        gd[k] = v
                    batch, batch_keys = [], []
        if batch:
            vals = compute_gd(model, load_features(batch), device)
            for k, v in zip(batch_keys, vals):
                gd[k] = v

        # 计算技术因子
        factor_map = compute_factors_map(ex, list(by_symbol.keys()))

        # 构造样本
        n_ex = 0
        for sym, files in by_symbol.items():
            fdf = factor_map.get(sym)
            if fdf is None:
                continue
            ends = [e for e, _ in files]
            for i, (end, _) in enumerate(files):
                if i + max(SHIFTS) >= len(files):
                    continue
                fv = latest_factor_values(fdf, end)
                if fv is None:
                    continue
                cur = gd.get((sym, end))
                if cur is None:
                    continue
                dgds = [gd.get((sym, ends[i + h]), np.nan) - cur for h in SHIFTS]
                all_rows.append([*dgds, *fv])
                n_ex += 1
        print(f'{ex}: 命中 {len(by_symbol)} 只，样本 {n_ex}')

    if not all_rows:
        print('无样本。')
        return

    A = np.array(all_rows, dtype=float)  # [N, 3 + 24]
    dgd = A[:, :3]
    fac = A[:, 3:]
    print(f'\n总样本: {len(A)}')
    print('=' * 70)
    print('技术因子（最新值） vs Δgd 的 Spearman IC：')
    print(f"{'因子':<18}{'量纲':<8}" + ''.join(f'{s:>9}' for s in ['dgd_1Q', 'dgd_3Q', 'dgd_7Q']))
    for fi, fn in enumerate(FACTORS):
        tag = '自由' if fn in SCALE_FREE else '依赖'
        line = f'{fn:<18}{tag:<8}'
        for hi in range(3):
            r, _ = spearman(fac[:, fi], dgd[:, hi])
            line += f'{r:>9.4f}'
        print(line)

    print('\nΔgd 分布：')
    for hi, sn in enumerate(['dgd_1Q', 'dgd_3Q', 'dgd_7Q']):
        v = dgd[:, hi]
        print(f'  {sn}: mean={np.nanmean(v):+.4f} std={np.nanstd(v):.4f}')


if __name__ == '__main__':
    main()
