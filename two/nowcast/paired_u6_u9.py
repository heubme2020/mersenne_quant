"""在同一批测试样本上【配对】比较 u6 与 u9 的终点指标（逐日截面 IC(评分, 未来收益)）。

## 为什么需要这个脚本

2026-09-26 分别跑 `eval_v2.py` 得到的是**不同测试集**上的结果：
  u6 = 44,357 行 / 655 只 / 386 个锚点日
  u9 = 46,605 行 / 666 只 / 396 个锚点日
（同 root、不同 suffix -> 标签可用行数不同。）

而两者的「超出」（模型IC − 基线IC）差距只有 |Δ| < 0.01，这个量级下
**测试集不同就足以造出或抹掉差异**，所以单独跑的结果不能用来选模型。

本脚本取两者的 `(symbol, anchor)` **交集**，只在同一批行上逐锚点日算 IC，
再做**配对**比较（同一个锚点日上 u9 的超出 减 u6 的超出），这样市场因子被配对消掉。

## 输入

`returns_eval_u6_pair/eval_v2_detail.csv`、`returns_eval_u9/eval_v2_detail.csv`
（由 `eval_v2.py --root C:/quant_data/nowcast3 --tag _u6 / --tag _u9 --out ...` 生成）

⚠️ u6 那份**故意输出到 `returns_eval_u6_pair/`**，不是 `returns_eval_u6/` ——
后者里已经有一个 **09-25 06:22、49,237 行**的明细，行数既不等于 u6 的 44,357
也不等于 u9 的 46,605（是另一次跑批的产物），别覆盖它。

## 注意

* 两侧的**基线也不同**（u6 的基线用 7 季毛利中位，u9 用 3 季），这是对的 ——
  每个变体都该跟**自己那个免费基线**比（这才是"超出"的含义）。
* 只做配对比较，**不改任何生产文件、不写任何模型文件**。
"""
import os
import sys

import numpy as np
import pandas as pd

# Windows 控制台默认 GBK，而本文件会打印 `−`（U+2212）等非 GBK 字符 —— 从 GBK 控制台直接跑
# 会 UnicodeEncodeError。与其它脚本同一套修法（2026-09-28 扫描后补齐）。
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass


# 明细路径相对【本脚本所在目录】解析 —— eval_v2 的 --out 也是相对它自己（two/nowcast/）解析的，
# 用相对 cwd 的写法会从 quant/ 根目录下找不到文件。
HERE = os.path.dirname(os.path.abspath(__file__))
U6_CSV = os.path.join(HERE, 'returns_eval_u6_pair', 'eval_v2_detail.csv')
U9_CSV = os.path.join(HERE, 'returns_eval_u9', 'eval_v2_detail.csv')

HORIZONS = [1, 3, 7]
MIN_N = 5          # 与 eval_v2.ic_by_date 一致
COLS = [f'{c}{h}' for h in HORIZONS for c in ('ret', 'score', 'base')]


def ic_by_anchor(df, x, y):
    """逐锚点日截面 rank-IC。与 eval_v2.ic_by_date 同口径（截面 < MIN_N 跳过）。"""
    out = {}
    for a, g in df.groupby('anchor'):
        s = g[[x, y]].dropna()
        if len(s) < MIN_N:
            continue
        ra, rb = s[x].rank().values, s[y].rank().values
        if ra.std() and rb.std():
            out[a] = float(np.corrcoef(ra, rb)[0, 1])
    return pd.Series(out).sort_index()


def paired_t(d):
    """配对 t：d 是逐锚点日的差值序列。"""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 3 or d.std() == 0:
        return np.nan, len(d)
    return float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))), len(d)


def main():
    u6 = pd.read_csv(U6_CSV)
    u9 = pd.read_csv(U9_CSV)
    print(f'u6  {len(u6):>7,} 行 / {u6.symbol.nunique():>4,} 只 / {u6.anchor.nunique():>4} 锚点日')
    print(f'u9  {len(u9):>7,} 行 / {u9.symbol.nunique():>4,} 只 / {u9.anchor.nunique():>4} 锚点日')

    key = ['symbol', 'anchor']
    m = (u6[key + COLS]
         .merge(u9[key + COLS], on=key, suffixes=('_6', '_9'), how='inner'))
    print(f'交集 {len(m):>7,} 行 / {m.symbol.nunique():>4,} 只 / {m.anchor.nunique():>4} 锚点日\n')

    # ---- 自检：同一个数据集里，同一 (symbol, anchor) 的未来收益必须逐位相同 ----
    bad = False
    for h in HORIZONS:
        d = (m[f'ret{h}_6'] - m[f'ret{h}_9']).abs().max()
        flag = 'OK' if d < 1e-9 else '⚠️ 不一致'
        print(f'自检 ret{h}: 两侧 max|Δ| = {d:.3g}   {flag}')
        bad |= d > 1e-9
    if bad:
        print('⚠️ 收益对不齐 —— 说明两侧测试集不是同一批行，下面的结论不可用。')
        return
    print()

    print('=' * 96)
    print('  逐锚点日截面 IC(评分, 未来收益)，在【同一批行】上配对比较')
    print('=' * 96)
    print(f'{"持有":>5s}{"u6模型":>10s}{"u6基线":>10s}{"u6超出":>10s}'
          f'{"u9模型":>10s}{"u9基线":>10s}{"u9超出":>10s}'
          f'{"配对差(u9-u6)":>16s}{"t":>8s}{"n":>6s}')

    for h in HORIZONS:
        r6 = ic_by_anchor(m, f'score{h}_6', f'ret{h}_6')
        b6 = ic_by_anchor(m, f'base{h}_6', f'ret{h}_6')
        r9 = ic_by_anchor(m, f'score{h}_9', f'ret{h}_9')
        b9 = ic_by_anchor(m, f'base{h}_9', f'ret{h}_9')

        idx = r6.index.intersection(b6.index).intersection(r9.index).intersection(b9.index)
        sp6 = (r6[idx] - b6[idx])          # u6 的"超出"序列
        sp9 = (r9[idx] - b9[idx])          # u9 的"超出"序列
        d = sp9 - sp6                      # 配对差
        t, n = paired_t(d)

        print(f'{h:>4d}季{r6[idx].mean():>+10.4f}{b6[idx].mean():>+10.4f}{sp6.mean():>+10.4f}'
              f'{r9[idx].mean():>+10.4f}{b9[idx].mean():>+10.4f}{sp9.mean():>+10.4f}'
              f'{d.mean():>+16.4f}{t:>8.2f}{n:>6d}')
        win = float((d > 0).mean())
        print(f'{"":5s}（配对差 >0 的锚点日占比：{win:.1%}）')

    print('\n读法：')
    print('  * "超出" = 模型IC − 自己那个免费基线的IC。>0 才说明模型加了这个基线之上的信息。')
    print('  * 配对差 = u9超出 − u6超出，在【同一个锚点日】上算，所以市场因子被消掉。')
    print('  * t > 2 才算"u9 显著更好"；t < -2 是显著更差；中间就是分不出来。')


if __name__ == '__main__':
    main()
