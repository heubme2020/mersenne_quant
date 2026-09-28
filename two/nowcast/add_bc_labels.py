"""把 B(/市值) 和 C(/总资产) 两套标签写到已有 chunks 上 —— 不动 _X，不重跑特征。

背景（2026-09-24 用户要求）：对比三个标签方案的**可预测性**和**前 50% 组合收益**。

    分子统一 = ΣX[j+1..j+h] − h·X[j-3]
    A(u1) = 分子 / (h·X[j-3])     自相对
    B(u2) = 分子 / 市值            恒正且大 -> 无分母爆炸；但把价格带进标签
    C(u3) = 分子 / 总资产[j-3]     恒正稳定 -> 无分母爆炸，且不带价格

**样本行集完全相同**（沿用生成时 u1 的 ok 掩码），所以三者对比是干净的。
三套标签的朴素预测都是"不变"= 0 -> 基准 `b` 恒为 0，`r = y`。

为什么不在 labels.py 里算：B 的分母是**锚点日的市值**，而 labels.py 只吃季度面板
（没有日线收盘价）。所以只能在实际生成/回填时按锚点日算。

对齐方式与 rewrite_labels.py 一致：chunk 里的 `_d` 是锚点交易日、`_s` 是符号索引；
反查 `j = 最后一个【已披露】季度`（按 A 股披露截止日）——与生成端同一条规则。

用法：
    python two/nowcast/add_bc_labels.py --root C:/quant_data/nowcast3 \
        --exchanges SHZ SHH AMEX ... XETRA
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
import labels as L                                       # noqa: E402
from gen_data2 import available_date                     # noqa: E402

HORIZONS = L.HORIZONS
BACK = 3
FIELDS = ['grossProfit', 'revenue', 'totalAssets']       # 三个头
TAG = {'grossProfit': 'gpD', 'revenue': 'revD', 'totalAssets': 'taD'}


def load_panel(exchange):
    """季度面板：毛利、营收、总资产、股本。"""
    ex = exchange.lower()
    inc = pd.read_csv(os.path.join(ROOT, 'data', exchange.upper(), f'income_{ex}.csv'),
                      usecols=['symbol', 'endDate', 'grossProfit', 'revenue',
                               'weightedAverageShsOut'])
    bal = pd.read_csv(os.path.join(ROOT, 'data', exchange.upper(), f'balance_{ex}.csv'),
                      usecols=['symbol', 'endDate', 'totalAssets', 'totalStockholdersEquity'])
    # 必须和 labels.load_panel 一样用 outer —— 用 inner 会漏掉只在一张表里出现的季度，
    # 使"最后一个已披露季度"的对齐与生成端不一致（A/B/C 的样本行就对不上了）。
    p = inc.merge(bal, on=['symbol', 'endDate'], how='outer')
    p['symbol'] = p.symbol.astype(str)
    p = p[p.endDate.astype(str).str[4:].isin(['0331', '0630', '0930', '1231'])]
    p = p.drop_duplicates(subset=['symbol', 'endDate'], keep='last')
    p = p.sort_values(['symbol', 'endDate']).reset_index(drop=True)
    p['avail'] = available_date(p.endDate.values)
    g = p.groupby('symbol', sort=False)
    for c in FIELDS:
        for h in HORIZONS:
            p[f'fwd_{c}_{h}'] = g[c].transform(
                lambda x, h=h: x.shift(-1).rolling(h, min_periods=h).sum().shift(-(h - 1)))
        p[f'base_{c}'] = g[c].shift(BACK)                # X[j-3]
    p['ta_m3'] = g['totalAssets'].shift(BACK)            # C 的分母
    # D：基准 = X[jc]（jc = 最后已结束季度）本身，用 shift(0) 拿本行值
    for c in FIELDS:
        p[f'base0_{c}'] = p[c]
    p['ta_m0'] = p['totalAssets']
    # ⚠️ 2026-09-25 修 bug：这里**曾经**用 `ok = L.build_labels(p, u1).ok` 过滤面板，
    # 理由是"和生成端 gen_data2.get_labels 的 lab[lab.ok] 对齐"。那个理由本身是错的 ——
    # u1 的 ok 要求「未来 7 季完整」，于是每只股票**最后 7 个季度被滤掉**，
    # 锚点落在最后 7 季时 asof（searchsorted）就退回更早的季度：
    #   本该 NaN 的最近期锚点被「凭空造出」标签，且其未来窗口有一大截落在锚点日**之前**
    #   （部分/全部已实现）→ 系统性抬高 IC。SHZ 实测这类行占 10.6%。
    # 根因在 gen_data2.get_labels（同样在过滤后的行集上算 avail），见那边的注释。
    # 现在直接用【完整面板】做 asof：asof = 真正的「最后一个已披露季度」；
    # 窗口不完整的行自然产出 NaN，由 data.Store 的 labok 丢掉（样本少约 10%，这是对的）。
    return p


def relabel_exchange(exchange, root):
    ex_dir = os.path.join(root, exchange)
    if not os.path.isdir(ex_dir):
        print(f'[{exchange}] 目录不存在，跳过')
        return
    syms = json.load(open(os.path.join(ex_dir, 'symbols.json')))
    panel = load_panel(exchange)

    # 锚点日的收盘价（B 要用；按 (symbol, anchor) 查）
    d = pd.read_csv(os.path.join(ROOT, 'data', exchange.upper(), f'daily_{exchange.lower()}.csv'),
                    usecols=['symbol', 'date', 'close'], engine='pyarrow')
    d['symbol'] = d.symbol.astype(str)
    d = d[d.close > 0]
    close_by = {s: (g['date'].values, g['close'].values.astype('float64'))
                for s, g in d.groupby('symbol', sort=False)}
    del d

    # (已披露日, 季度末, 行号) —— D 要用"季度末"做 asof（不管有没有披露）
    # (已披露日, 季度末, 行号)。endDates 要按符号整体排好，E 还要用它做 "锚点在哪个季度"。
    by = {s: (g['avail'].values, g['endDate'].values, g.index.values)
          for s, g in panel.groupby('symbol', sort=False)}

    names = [f'{TAG[c]}{suf}' for c in FIELDS for suf in HORIZONS]
    nd = len(names)
    for xf in sorted(glob.glob(os.path.join(ex_dir, 'chunk_*_X.npy'))):
        base = xf[:-6]
        dd = np.load(base + '_d.npy')
        ss = np.load(base + '_s.npy')
        n = len(dd)
        Y2 = np.full((n, nd), np.nan, 'float32')     # B: /市值
        Y3 = np.full((n, nd), np.nan, 'float32')     # C: /总资产[j-3]
        Y4 = np.full((n, nd), np.nan, 'float32')     # D: /总资产[jc]
        Y5 = np.full((n, nd), np.nan, 'float32')     # E: /总资产[j-3]，j=锚点所在季
        for si in np.unique(ss):
            sym = syms[si] if si < len(syms) else None
            packed = by.get(sym)
            if packed is None or sym not in close_by:
                continue
            av, ed, rows = packed
            dt, cl = close_by[sym]
            m = ss == si
            idxs = np.where(m)[0]
            anchors = dd[idxs]
            # asof：最后一个已披露季度（A/B/C 用）
            pos = np.searchsorted(av, anchors, side='right') - 1
            # asof：最后一个【已结束】季度（D 用）
            posc = np.searchsorted(ed, anchors, side='right') - 1
            # E：锚点日【所在】的那个季度 = 第一个 endDate >= 锚点日
            posE = np.searchsorted(ed, anchors, side='left')
            # 锚点日收盘
            cpos = np.searchsorted(dt, anchors, side='right') - 1
            for k, (ii, pj, cp, pc, pE) in enumerate(zip(idxs, pos, cpos, posc, posE)):
                if pj < 0 or cp < 0:
                    continue
                row = panel.iloc[rows[pj]]
                # D 用"最后已结束季度"那一行
                rowc = panel.iloc[rows[pc]] if pc >= 0 else None
                # E：j = 锚点所在季度；基准 X[j-3]/TA[j-3] 取 j 行，窗口 ΣX[j..j+h-1] 取 j-1 行
                # 校验：搜到的季度必须真的"包含"锚点日（相差 ≤ 3 个月）。
                # panel 已按 u1 掩码过滤，若"所在季度"恰好被滤掉，searchsorted 会返回
                # 一个【更晚】的季度 -> 静默错位。用月份差挡住。
                _ymE = (int(ed[pE]) // 10000) * 12 + (int(ed[pE]) // 100 % 100) if 0 <= pE < len(ed) else -99
                _ymA = (int(anchors[k]) // 10000) * 12 + (int(anchors[k]) // 100 % 100)
                if not (0 <= _ymE - _ymA <= 3):
                    rowE = rowEm1 = None
                else:
                    rowE = panel.iloc[rows[pE]]
                    rowEm1 = panel.iloc[rows[pE - 1]] if pE >= 1 else None
                mcap = row['weightedAverageShsOut'] * cl[cp]
                ta = row['ta_m3']

                for fi, c in enumerate(FIELDS):
                    for hi, h in enumerate(HORIZONS):
                        fwd = row[f'fwd_{c}_{h}']
                        if not np.isfinite(fwd):
                            continue
                        col = fi * len(HORIZONS) + hi
                        b3 = row[f'base_{c}']            # X[j-3]
                        if np.isfinite(b3):
                            num = fwd - h * b3
                            if np.isfinite(mcap) and mcap > 0:
                                Y2[ii, col] = np.clip(num / mcap, -127, 127)
                            if np.isfinite(ta) and ta > 0:
                                Y3[ii, col] = np.clip(num / ta, -127, 127)
                        # D：分子分母都用最后【已结束】季度 jc
                        # E：基准 = X[j-3] / TA[j-3]（j=锚点所在季），窗口 = ΣX[j..j+h-1]
                        if rowE is not None and rowEm1 is not None:
                            fwdE = rowEm1[f'fwd_{c}_{h}']
                            bE = rowE[f'base_{c}']            # X[j-3]
                            taE = rowE['ta_m3']               # TA[j-3]
                            if (np.isfinite(fwdE) and np.isfinite(bE)
                                    and np.isfinite(taE) and taE > 0):
                                Y5[ii, col] = np.clip((fwdE - h * bE) / taE, -127, 127)
                        if rowc is not None:
                            fwdc = rowc[f'fwd_{c}_{h}']      # ΣX[jc+1..jc+h]
                            b0 = rowc[f'base0_{c}']           # X[jc]
                            ta0 = rowc['ta_m0']               # TA[jc]
                            if (np.isfinite(fwdc) and np.isfinite(b0)
                                    and np.isfinite(ta0) and ta0 > 0):
                                Y4[ii, col] = np.clip((fwdc - h * b0) / ta0, -127, 127)
        for suf, Y in [('u2', Y2), ('u3', Y3), ('u4', Y4), ('u5', Y5)]:
            np.save(base + f'_y{suf}.npy', Y)
            np.save(base + f'_b{suf}.npy', np.zeros_like(Y))     # 朴素预测 = 不变 = 0
            np.save(base + f'_r{suf}.npy', Y)                    # r = y − b = y
        print(f'[{exchange}] {os.path.basename(base)}  n={n}  '
              f'B {int(np.isfinite(Y2).all(1).sum()):,}  '
              f'C {int(np.isfinite(Y3).all(1).sum()):,}  '
              f'D {int(np.isfinite(Y4).all(1).sum()):,}  '
              f'E {int(np.isfinite(Y5).all(1).sum()):,}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='C:/quant_data/nowcast3')
    ap.add_argument('--exchanges', nargs='+', required=True)
    args = ap.parse_args()
    for ex in args.exchanges:
        relabel_exchange(ex, args.root)


if __name__ == '__main__':
    main()
