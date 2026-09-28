"""从已存的 .pt 重新出预测并存成 results/pred_*.npz —— 不重训。

用途：`train.py` 在打印基准对照表时崩过（变体的头名和硬编码的 FLAT_NAMES 对不上），
而崩溃点在保存结果**之前**，所以 json/npz 没落盘、但 .pt（每个 epoch 存的最好验证权重）
已经存了。重训要 1.5 小时，重新前向只要几分钟。

`train.predict()` 里 y_mean/y_std 其实没被用到（IC 是 rank 口径、与尺度无关），
所以这里传 None 即可；但 resid 模式若要算「基准 + 增量」就得还原量纲，那种情况
请直接用 `train.py`（它已经带 --suffix）。

用法：python two/nowcast/predict_only.py --suffix v1 --tag _v1
"""

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
# 顺序要紧：one/ 也有 train.py，若它排在 nowcast/ 前面，`import train` 会拿到 one/train.py
sys.path.insert(0, os.path.join(HERE, '..', '..', 'one'))
sys.path.insert(0, HERE)

from data import Store, ASHARE_EX            # noqa: E402
from train import load_split, predict, ARMS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suffix', default='')
    ap.add_argument('--tag', default='_v1')
    ap.add_argument('--arm', default='ashare')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--root', default='Z:/quant_data/nowcast')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    store = Store(args.root, sorted(set(ARMS[args.arm]) | set(ASHARE_EX)),
                  min_per_date=64, label='level', suffix=args.suffix)
    tr, va, te = load_split(store, args.arm)
    print(f'suffix={args.suffix!r} 测试样本 {len(te):,}')

    pt = os.path.join(HERE, f'nowcast_{args.arm}{args.tag}.pt')
    model = torch.load(pt, map_location=device, weights_only=False)
    print(f'载入 {os.path.basename(pt)}')

    tp = predict(model, store, te, None, None, device)
    ty = store.take_y(te)
    tsyms = store.symbol_of(te)
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    out = os.path.join(HERE, 'results', f'pred_{args.arm}{args.tag}_s{args.seed}.npz')
    np.savez(out, pred=tp, y=ty, date=store.dates[te], sym=np.array(tsyms))
    print(f'写入 {out}  pred{tp.shape}')


if __name__ == '__main__':
    main()
