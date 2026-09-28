"""临时测试：验证 release_mmaps() 能否把 memmap 读出来的文件页还给系统。

对比两种模式各读 ~40,000 行（≈5 GB），看 RSS 是否线性上涨。
"""

import gc
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'one'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np            # noqa: E402
import psutil                 # noqa: E402

from data import Store, ASHARE_EX                     # noqa: E402
from train import load_split, BATCH, ARMS             # noqa: E402

proc = psutil.Process()
N = 40_000
B = 64


def rss():
    return proc.memory_info().rss / 1e9


def run(store, pos, release_every):
    rng = np.random.RandomState(0)
    for i in range(N // B):
        idx = rng.choice(pos, B, replace=False)
        X = store.take_x(idx)
        del X
        if release_every and (i + 1) % release_every == 0:
            store.release_mmaps()
            gc.collect()
        if (i + 1) % 156 == 0:      # 每 10,000 行报一次
            print(f'    读了 {(i+1)*B:>6,} 行  RSS={rss():5.2f} GB', flush=True)


store = Store('D:/quant_data/nowcast', sorted(set(ARMS['global']) | set(ASHARE_EX)),
              min_per_date=BATCH)
tr, _, _ = load_split(store, 'global')
print(f'训练集 {len(tr):,} 行，全部读一遍约 {len(tr)*126108/1e9:.0f} GB\n')

print('--- 不释放 memmap（现状）---')
print(f'    起始 RSS={rss():5.2f} GB', flush=True)
run(store, tr, release_every=0)

store.release_mmaps(); gc.collect()
print(f'    收尾后 RSS={rss():5.2f} GB  （release 之后）\n')

print('--- 每 500 步 release_mmaps() ---')
print(f'    起始 RSS={rss():5.2f} GB', flush=True)
run(store, tr, release_every=500)
store.release_mmaps(); gc.collect()
print(f'    收尾后 RSS={rss():5.2f} GB', flush=True)
