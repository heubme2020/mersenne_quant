"""逐位校验：`two/two_features.build_x` 与 `nowcast/gen_data2.build_x` 必须**完全相同**。

## 为什么用逐位而不是 allclose

`build_x` 的漂移是"静默"的 —— 因子集换了、列序变了、idx 分母写错，都不会报错，
只会让模型输入悄悄变形。漂移的典型量级是 1e-6 这种"看起来没问题"的差，
`np.allclose` 会放过去。所以这里用 `np.array_equal`，并要求**逐位相同**；
一旦不等，报出首个不同的位置和两边取值，便于直接定位。

## 校验数据

取**真实日线**（`data/SHZ/daily_shz.csv`）里若干只股票的**最后 WINDOW 行**窗口 ——
这是最接近生产推理的输入形状。合成数据测不出因子集/列序的问题。

## 用法

    python two/_verify_vs_nowcast.py                    # 默认抽 3 只
    python two/_verify_vs_nowcast.py --symbols 000001.SZ 600519.SS
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

# Windows 控制台默认 GBK，而本脚本会打印 `✓`（U+2713，GBK 编不出来）—— 不加这行会在
# **校验全部通过之后**崩在成功提示上（2026-09-28 实测：前两行比对结果都打出来了，然后
# UnicodeEncodeError）。与其它脚本同一套修法。
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))

# ⚠️ 必须先把 `two/` 从 sys.path 里去掉，再把 `one/` 放最前。
#    否则（一旦 two/gen_train_data.py 存在）`two/nowcast/gen_data2.py` 内部那句裸导入
#    `from gen_train_data import add_technical_factor` 会命中 **two 自己那份**，
#    于是"两边"其实都在用同一个实现，校验会**假通过**。
sys.path = [p for p in sys.path if os.path.abspath(p or '.') != os.path.abspath(HERE)]
sys.path.insert(0, os.path.join(ROOT, 'one'))
sys.path.insert(0, HERE)

import two_features                                        # noqa: E402
sys.path.insert(0, os.path.join(ROOT, 'two', 'nowcast'))
import gen_data2                                           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols', nargs='+', default=['000001.SZ', '000002.SZ', '000063.SZ', '002415.SZ'])
    ap.add_argument('--n', type=int, default=3, help='每只股票抽几个窗口（在最后 N*50 天内滑）')
    a = ap.parse_args()

    print(f'two_features : WINDOW={two_features.WINDOW}  COLS={len(two_features.COLS)} 维'
          f'  FACTORS={len(two_features.FACTORS)} 个')
    print(f'gen_data2    : WINDOW={gen_data2.WINDOW}  COLS={len(gen_data2.COLS)} 维'
          f'  FACTORS={len(gen_data2.FACTORS)} 个')

    # 常量先对齐（列集合/列序不一致的话，下面逐位比较必然失败，先报清楚）
    assert two_features.COLS == gen_data2.COLS, (
        f'❌ 31 维列集合/列序不一致:\n  two_features={two_features.COLS}\n  gen_data2   ={gen_data2.COLS}')
    assert two_features.RAW == gen_data2.RAW, '❌ RAW 块顺序不一致'
    assert (two_features.DAYS_INPUT, two_features.AUX_DAYS, two_features.REF_IDX) == \
           (gen_data2.DAYS_INPUT, gen_data2.AUX_DAYS, gen_data2.REF_IDX), '❌ DAYS_INPUT/AUX_DAYS/REF_IDX 不一致'
    print('✓ 常量一致（COLS / RAW / DAYS_INPUT / AUX_DAYS / REF_IDX）')

    W = two_features.WINDOW
    print(f'\n读 data/SHZ/daily_shz.csv（大文件，慢一点）...')
    df = pd.read_csv(os.path.join(ROOT, 'data', 'SHZ', 'daily_shz.csv'),
                     usecols=['symbol', 'date', 'open', 'high', 'low', 'close', 'volume'])
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

    n_ok = n_bad = n_skip = 0
    for sym in a.symbols:
        g = df[df['symbol'] == sym]
        if len(g) < W + a.n * 50:
            print(f'  {sym}: 历史只有 {len(g)} 行，不足 {W}+ 窗口，跳过')
            n_skip += 1
            continue
        for k in range(a.n):
            end = len(g) - k * 50
            w = g.iloc[end - W:end].reset_index(drop=True)
            x_new = two_features.build_x(w)
            x_old = gen_data2.build_x(w)
            if (x_new is None) != (x_old is None):
                print(f'  ❌ {sym} 窗口{k}: 一边返回 None、一边不是 —— 归一化基准判断不同')
                n_bad += 1
                continue
            if x_new is None:
                print(f'  {sym} 窗口{k}: 两边都是 None（ref 处 close/volume 非正），跳过')
                n_skip += 1
                continue
            if np.array_equal(x_new, x_old):
                print(f'  ✓ {sym} 窗口{k}: shape={x_new.shape} 逐位相同')
                n_ok += 1
            else:
                d = np.argwhere(x_new != x_old)
                i, j = d[0]
                print(f'  ❌ {sym} 窗口{k}: {len(d)} 个元素不同！首个在 [{i},{j}]：'
                      f'new={x_new[i, j]!r} old={x_old[i, j]!r} '
                      f'diff={abs(float(x_new[i,j]) - float(x_old[i,j])):.3g}')
                n_bad += 1

    print(f'\n结论：逐位相同 {n_ok} 个 / 不同 {n_bad} 个 / 跳过 {n_skip} 个')
    if n_bad:
        print('⛔ 两条路不等价 —— two_features.py 不能替换 gen_data2.build_x，先查差异来源。')
        sys.exit(1)
    if n_ok == 0:
        print('⚠️ 一个有效窗口都没测到（全部跳过）—— 这不构成"通过"，请换有足够历史的股票重跑。')
        sys.exit(2)
    print('✅ 等价，可以安全替换。')


if __name__ == '__main__':
    main()
