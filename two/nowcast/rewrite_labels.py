"""只重写已生成数据里的标签数组（_y / _b / _r），不动 _X。

两个用途：

1. **标签口径改动**（如 ±127 截断）只影响 9 个 float 的标签数组，而 `_X` 是
   1017×31 的特征张量、A 股两所共 19 GB。为了一个 clip 重跑 25 分钟的特征计算是浪费。
2. **头 1 / 头 2 的对照实验**（`variants.py`）：三套标签共用同一份 X，写成不同后缀
   `_y_v1.npy` / `_y_v2.npy` 共存于同一目录，由 `data.Store(suffix=...)` 选择。

对齐方式：每个 chunk 里 `_d.npy` 是锚点交易日、`_s.npy` 是符号索引。
`gen_data.sym_samples` 里样本是从标签季度 j 出发、取「≤ j_endDate 的最后一个交易日」
作为锚点日，所以反查 `j = 最后一个 endDate ≤ 锚点日` 是精确的往返映射
（j_endDate ≤ dates[a] < 下一个季度，中间不可能有别的季度）。
实测 19 个块 0 行未匹配、`y − b = r` 在 140 万个元素上完全精确。

用法：
    python two/nowcast/rewrite_labels.py --root Z:/quant_data/nowcast --exchanges SHZ SHH
    python two/nowcast/rewrite_labels.py --root Z:/quant_data/nowcast --variant v1
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import labels as L                       # noqa: E402
import variants as V                     # noqa: E402


def rewrite_exchange(root, ex, fields, suffix):
    ex_dir = os.path.join(root, ex)
    if not os.path.isdir(ex_dir):
        print(f'[{ex}] 目录不存在，跳过')
        return
    syms = json.load(open(os.path.join(ex_dir, 'symbols.json')))
    ycols = [f'y_{n}{h}' for n in fields for h in L.HORIZONS]
    bcols = [f'b_{n}{h}' for n in fields for h in L.HORIZONS]
    rcols = [f'r_{n}{h}' for n in fields for h in L.HORIZONS]

    lab = L.build_labels(L.load_panel(ex), fields)
    lab = lab.sort_values(['symbol', 'j_endDate']).reset_index(drop=True)
    Y, B, R = lab[ycols].values, lab[bcols].values, lab[rcols].values
    by = {s: (g['j_endDate'].values, g.index.values)
          for s, g in lab.groupby('symbol', sort=False)}

    for xf in sorted(glob.glob(os.path.join(ex_dir, 'chunk_*_X.npy'))):
        base = xf[:-6]
        d = np.load(base + '_d.npy')
        s_idx = np.load(base + '_s.npy')
        n = len(d)
        yy = np.full((n, len(ycols)), np.nan, dtype='float32')
        bb = np.full((n, len(bcols)), np.nan, dtype='float32')
        rr = np.full((n, len(rcols)), np.nan, dtype='float32')
        for si in np.unique(s_idx):
            m = s_idx == si
            packed = by.get(syms[si]) if si < len(syms) else None
            if packed is None:
                continue
            dates, rows = packed
            pos = np.searchsorted(dates, d[m], side='right') - 1
            okp = pos >= 0
            idx = np.where(m)[0][okp]
            r_ = rows[pos[okp]]
            yy[idx], bb[idx], rr[idx] = Y[r_], B[r_], R[r_]
        np.save(base + f'_y{suffix}.npy', yy)
        np.save(base + f'_b{suffix}.npy', bb)
        np.save(base + f'_r{suffix}.npy', rr)
        print(f'[{ex}] {os.path.basename(base)}{suffix}  n={n}  '
              f'未匹配={(~np.isfinite(yy).all(1)).sum()}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='Z:/quant_data/nowcast')
    ap.add_argument('--exchanges', nargs='+', default=['SHZ', 'SHH'])
    ap.add_argument('--variant', default='', help='""=基线 | v1 | v2（见 variants.py）')
    args = ap.parse_args()
    fields = V.fields_for(args.variant)
    print(f'变体 {args.variant!r}: {V.label_desc(args.variant)}')
    for ex in args.exchanges:
        rewrite_exchange(args.root, ex, fields, args.variant)


if __name__ == '__main__':
    main()
