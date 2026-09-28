"""模型能不能让「毛利/市值」这类排序变好？—— 扫最优混合权重。

用法决定判据：如果最终评分是 `预测水平 / 市值`，那"模型的水平预测"必须比
**免费的「上一期水平 / 市值」**更好，否则直接用滞后一期财报就行。

等权混合（50/50）差，不代表模型没用——只说明**最优权重不是 0.5**。
所以这里扫 w：  IC( w·z(模型) + (1−w)·z(基准), 实际 )

  w* = 0        -> 模型在这个头上没有可用增量（最优就是完全不用它）
  w* > 0 且 IC 提升 -> 模型有增量，提升的幅度 = 它对这个用途的价值
  w* 很小（<0.2）    -> 增量存在但很弱，谨慎

用法：python two/nowcast/blend_weight.py --tag _v2            # 基线三头
      python two/nowcast/blend_weight.py --suffix v5 --tag _v5 --labels level
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'one'))
sys.path.insert(0, HERE)

from data import Store, ASHARE_EX            # noqa: E402
from train import load_split, ARMS           # noqa: E402
import labels as _L                          # noqa: E402
import variants as _V                        # noqa: E402

MIN_STOCKS = 20
GRID = np.linspace(0, 1, 21)


def _rank(x):
    return np.argsort(np.argsort(x))


def _z(x):
    s = x.std()
    return (x - x.mean()) / s if s > 0 else x * 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', default='level')
    ap.add_argument('--suffix', default='')
    ap.add_argument('--tag', default='_v2')
    ap.add_argument('--arm', default='ashare')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--root', default='Z:/quant_data/nowcast')
    args = ap.parse_args()

    heads = _L.flat_names(_V.fields_for(args.suffix))
    store = Store(args.root, sorted(set(ARMS[args.arm]) | set(ASHARE_EX)),
                  min_per_date=64, label=args.labels, suffix=args.suffix)
    tr, va, te = load_split(store, args.arm)
    pred = np.load(os.path.join(HERE, 'results',
                                f'pred_{args.arm}{args.tag}_s{args.seed}.npz'))['pred']
    assert len(pred) == len(te)
    y = store.take_level(te)
    b = store.take_bench(te)
    d = store.dates[te]

    ud, inv = np.unique(d, return_inverse=True)
    groups = [np.where(inv == i)[0] for i in range(len(ud))]
    groups = [g for g in groups if len(g) >= MIN_STOCKS]

    print(f'suffix={args.suffix!r} tag={args.tag} 测试样本 {len(te):,}\n')
    print(f'{"头":16s}{"IC(w=0)=基准":>14s}{"最优w":>9s}{"IC(最优)":>11s}'
          f'{"提升":>9s}{"IC(w=0.5)":>11s}')
    for k, name in enumerate(heads):
        out = []
        for w in GRID:
            ics = []
            for g in groups:
                s = w * _z(pred[g, k]) + (1 - w) * _z(b[g, k])
                a = y[g, k]
                ok = np.isfinite(s) & np.isfinite(a)
                if ok.sum() < MIN_STOCKS:
                    continue
                rs, ra = _rank(s[ok]), _rank(a[ok])
                if rs.std() and ra.std():
                    ics.append(np.corrcoef(rs, ra)[0, 1])
            out.append(np.mean(ics) if ics else np.nan)
        out = np.array(out)
        base_ic = out[0]
        best_i = np.nanargmax(out)
        print(f'{name:16s}{base_ic:>+14.4f}{GRID[best_i]:>9.2f}{out[best_i]:>+11.4f}'
              f'{out[best_i]-base_ic:>+9.4f}{out[10]:>+11.4f}')
    # ---- 更严谨：逐截面把 y 对 (基准, 模型) 做二元回归，看拟合值排序 ----
    # 等权/一维权重扫是"一种特定组合形式"，可能低估模型；二元回归给出
    # "在基准已知的前提下，模型有没有可用的偏增量"的正确上限。
    _hdr = (f'{"头":16s}{"IC(仅基准)":>12s}{"IC(仅模型)":>12s}'
            f'{"IC(二元拟合)":>14s}{"拟合提升":>10s}{"模型系数":>10s}')
    print('\n' + _hdr)
    for k, name in enumerate(heads):
        only_b, only_m, both, coefs = [], [], [], []
        for g in groups:
            a = y[g, k]; bb = b[g, k]; mm = pred[g, k]
            ok = np.isfinite(a) & np.isfinite(bb) & np.isfinite(mm)
            if ok.sum() < MIN_STOCKS:
                continue
            ra, rb, rm = _rank(a[ok]).astype(float), _rank(bb[ok]).astype(float), _rank(mm[ok]).astype(float)
            if ra.std() == 0 or rb.std() == 0 or rm.std() == 0:
                continue
            only_b.append(np.corrcoef(rb, ra)[0, 1])
            only_m.append(np.corrcoef(rm, ra)[0, 1])
            X = np.c_[np.ones_like(rb), rb, rm]
            XtXi = np.linalg.pinv(X.T @ X)
            c = XtXi @ (X.T @ ra)
            fit = X @ c
            both.append(np.corrcoef(_rank(fit), ra)[0, 1])
            coefs.append(c[2])
        print(f'{name:16s}{np.mean(only_b):>+12.4f}{np.mean(only_m):>+12.4f}'
              f'{np.mean(both):>+14.4f}{np.mean(both)-np.mean(only_b):>+10.4f}'
              f'{np.mean(coefs):>+10.3f}')
    print()
    print('读法：最优 w=0 => 模型在该头上没有可用增量；提升的幅度 = 它对"水平排序"这个用途的价值。')


if __name__ == '__main__':
    main()
