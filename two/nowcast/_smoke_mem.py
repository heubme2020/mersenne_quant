"""临时冒烟测试：验证 train.py 改过的路径（分块统计 / take_y / predict）峰值内存。"""

import gc
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'one'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np            # noqa: E402
import psutil                 # noqa: E402
import torch                  # noqa: E402

from data import Store, ASHARE_EX                     # noqa: E402
from train import (load_split, BATCH, ARMS, predict,  # noqa: E402
                   eval_icir, DAYS_INPUT)
from model import Nowcast                             # noqa: E402

proc = psutil.Process()
peak = [proc.memory_info().rss]


def mark(label):
    gc.collect()
    cur = proc.memory_info().rss
    peak[0] = max(peak[0], cur)
    print(f'  {label:34s} RSS={cur/1e9:5.2f} GB  峰值={peak[0]/1e9:5.2f} GB',
          flush=True)


store = Store('D:/quant_data/nowcast', sorted(set(ARMS['global']) | set(ASHARE_EX)),
              min_per_date=BATCH)
mark('Store 建好')

tr, va, te = load_split(store, 'global')
mark(f'split 完成 训练{len(tr):,}/验证{len(va):,}/测试{len(te):,}')

rng = np.random.RandomState(2)
samp = np.sort(tr[rng.choice(len(tr), min(20000, len(tr)), replace=False)])
Ys = store.take_y(samp)
mark('take_y(标签统计 2 万)')

aux_acc = []
for i in range(0, len(samp), 1024):
    Xc = store.take_x(samp[i:i + 1024])
    aux_acc.append(Xc[:, DAYS_INPUT:DAYS_INPUT + 128][:, :, [3, 4, 5]]
                   .reshape(-1, 3).copy())
    del Xc
aux_s = np.concatenate(aux_acc)
del aux_acc
mark('分块 aux 统计完成')

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Nowcast().to(dev)
vp = predict(model, store, va, None, None, dev)
mark(f'predict(验证 {len(va):,})')

vy = store.take_y(va)
mark('take_y(验证)')

r = eval_icir(vp, vy, store.dates[va])
mark('eval_icir 完成')
print('  随机权重下的 IC 均值（应当≈0）:',
      round(float(np.mean([v[0] for v in r.values()])), 4))
