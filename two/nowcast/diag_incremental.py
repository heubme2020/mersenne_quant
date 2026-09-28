"""增量信号诊断：模型到底有没有超出「朴素持续性基准」的信息？

`train.py` 报的 `IC(增量,残差) = IC(模型−基准, 实际−基准)` 有个隐患：
两边都含 `−基准` 项，当基准与实际的 |IC| 很大时（如 grossProfit 的 0.77~0.83），
这个相关系数会被 `Var(基准)` 机械地抬高。所以这里补三个更干净的口径：

  1. IC(模型, 实际) vs IC(基准, 实际)   —— 直接比，谁高谁强
  2. 正交化 IC：逐截面把「模型的 rank」对「基准的 rank」做线性回归取残差，
     再与实际算 IC。这才是**基准之外**的净信息。
  3. 混合 IC：IC(z(模型) + z(基准), 实际)，并与 IC(基准, 实际) 比。
     混合能提升 => 模型确实加了东西（且能算出最优权重）。

用法：python two/nowcast/diag_incremental.py --labels level --tag _v2
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', '..', 'one'))

from data import Store, ASHARE_EX                      # noqa: E402
from model import FLAT_NAMES                           # noqa: E402
from train import load_split, ARMS                     # noqa: E402
import labels as _L                                    # noqa: E402
import variants as _V                                  # noqa: E402

MIN_STOCKS = 20
HEAD_NAMES = list(FLAT_NAMES)


def _rank(x):
    return np.argsort(np.argsort(x))


def _ic(a, b):
    ra, rb = _rank(a), _rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def _z(x):
    s = x.std()
    return (x - x.mean()) / s if s > 0 else x * 0


def per_head(pred, y, bench, dates, fn):
    """逐 (锚点日) 截面调用 fn(model, bench, actual) -> float，返回 mean/std/n。"""
    ud, inv = np.unique(dates, return_inverse=True)
    groups = [np.where(inv == i)[0] for i in range(len(ud))]
    groups = [g for g in groups if len(g) >= MIN_STOCKS]
    out = {}
    for k, name in enumerate(HEAD_NAMES):
        vals = [fn(pred[g, k], bench[g, k], y[g, k]) for g in groups]
        vals = np.array([v for v in vals if np.isfinite(v)])
        out[name] = (vals.mean(), vals.mean() / vals.std() if len(vals) > 1 else np.nan,
                     len(vals))
    return out


def ic_plain(m, b, a):
    return _ic(m, a)


def ic_bench(m, b, a):
    return _ic(b, a)


def ic_orth(m, b, a):
    """模型 rank 对基准 rank 正交化后的净 IC。"""
    mb, bb = _rank(m), _rank(b)
    beta = np.polyfit(bb, mb, 1)
    resid = mb - (beta[0] * bb + beta[1])
    if resid.std() == 0:
        return np.nan
    return _ic(resid, _rank(a))


def ic_blend(m, b, a):
    return _ic(_z(m) + _z(b), a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', default='level')
    ap.add_argument('--tag', default='_v2')
    ap.add_argument('--arm', default='ashare')
    ap.add_argument('--root', default='Z:/quant_data/nowcast')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--suffix', default='', help='标签变体后缀（见 variants.py）')
    args = ap.parse_args()

    global HEAD_NAMES
    HEAD_NAMES = _L.flat_names(_V.fields_for(args.suffix))

    store = Store(args.root, sorted(set(ARMS[args.arm]) | set(ASHARE_EX)),
                  min_per_date=64, label=args.labels, suffix=args.suffix)
    tr, va, te = load_split(store, args.arm)
    print(f'测试样本 {len(te):,}')

    npz = np.load(os.path.join(HERE, 'results',
                               f'pred_{args.arm}{args.tag}_s{args.seed}.npz'),
                  allow_pickle=True)
    pred = npz['pred']
    assert len(pred) == len(te), f'预测行数 {len(pred)} != 测试行数 {len(te)}'

    y = store.take_level(te)
    bench = store.take_bench(te)
    dates = store.dates[te]

    # resid 版的 pred 是「增量」，水平预测 = 基准 + 增量（基准在锚点日已知）。
    # 两种口径都化成同一个「水平预测」，后面所有指标才可比。
    #
    # 关键：train.py 的 predict() 返回的是**标准化空间**的输出（loss 是在 z-score 后的
    # 标签上算的），IC 是 rank 口径所以对 level 模式无所谓；但 resid 模式要做
    # `基准 + 增量` 的加法，量纲必须还原，否则 1 个单位方差的预测加到标签量纲的
    # 基准上等于什么都没加。这里按 train.py 同样的方式估 y_mean/y_std。
    if args.labels == 'resid':
        # npz 里的 pred 是【训练过程中】存的 -> 那时模型还是 z 空间（affine 只在存盘时写进 .pt），
        # 所以这里要还原量纲。若改用 predict_only.py 重新生成的 npz（读的是带 buffer 的 .pt，
        # 已是真实单位），则不能再换算。
        rng = np.random.RandomState(args.seed + 1)
        samp = np.sort(tr[rng.choice(len(tr), min(20000, len(tr)), replace=False)])
        Ys = store.take_y(samp)
        y_mean, y_std = Ys.mean(0), Ys.std(0) + 1e-6
        print(f'resid 版：水平预测 = 基准 + 模型（已还原量纲，y_std={np.round(y_std,3).tolist()}）')
        pred = bench + pred * y_std + y_mean

    r_model = per_head(pred, y, bench, dates, ic_plain)
    r_bench = per_head(pred, y, bench, dates, ic_bench)
    r_orth = per_head(pred, y, bench, dates, ic_orth)
    r_blend = per_head(pred, y, bench, dates, ic_blend)

    print()
    print(f'{"头":16s}{"IC(模型)":>11s}{"IC(基准)":>11s}'
          f'{"IC(正交化)":>12s}{"ICIR(正交)":>12s}{"IC(混合)":>11s}')
    for n in HEAD_NAMES:
        print(f'{n:16s}{r_model[n][0]:>+11.4f}{r_bench[n][0]:>+11.4f}'
              f'{r_orth[n][0]:>+12.4f}{r_orth[n][1]:>+12.2f}{r_blend[n][0]:>+11.4f}')

    def avg(r):
        return np.mean([v[0] for v in r.values()])

    print(f'{"平均":16s}{avg(r_model):>+11.4f}{avg(r_bench):>+11.4f}'
          f'{avg(r_orth):>+12.4f}{"":>12s}{avg(r_blend):>+11.4f}')
    print()
    print('读法：')
    print('  IC(正交化) 才是「基准之外」的净信息；它显著为正 = 模型真的多了东西。')
    print('  IC(混合) > IC(基准) = 把模型和基准混起来比单用基准更好。')


if __name__ == '__main__':
    main()
