"""nowcast 训练数据生成（按符号分块存 .npy，供 memmap 随机取行）。

每个样本 = (symbol, 季度 j)：
    输入 : 截至「≤ j 末日的最后一个交易日」往前 889 天，按该日 close/volume 归一化，
           再算 one 的新 24 因子 → (1017, 31)（后 128 天供辅助头）
    标签 : labels.build_labels 的 9 个（EBIT/Revenue/OCF × 1/3/7 季）

**为什么分块而不是按股票分片**：损失与评估都是「同一锚点日、不同股票」的横截面，
batch 必须由同一天的多个股票构成。按股票分片时一个 batch 全是同一只股票的不同季度，
横截面损失直接失效；而且全球约 120 万样本做独立小文件在 Windows 上枚举会严重退化。
分块（每块 K 只股票的全部样本）后：全球约 110 个文件，每块内每个锚点日有 ~K 个样本，
天然构成横截面 batch。

输出：{out}/{EXCHANGE}/chunk_{i}_X.npy / _y.npy / _b.npy / _r.npy / _d.npy / _s.npy + symbols.json
    X (n, 1017, 31) float32 | y (n, 9) level 标签 | b (n, 9) 朴素持续性基准
    r (n, 9) 残差 = y − b | d (n,) int64 锚点日 | s (n,) int32 符号索引
（b / r 只有 9 列，体积可忽略，但让「模型 vs 基准 vs 增量」能在同一批测试样本上一次算完。）

用法：
    python two/nowcast/gen_data.py --exchanges SHZ SHH --out D:/quant_data/nowcast --chunk 250
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'one'))
sys.path.insert(0, HERE)

from factor_config import get_technical_factors               # noqa: E402
from gen_train_data import add_technical_factor               # noqa: E402
from factor_pool import add_pool_factors                      # noqa: E402
from labels import load_panel, build_labels, HORIZONS        # noqa: E402
import variants as V                                         # noqa: E402

DAYS_INPUT = 127 * 7
AUX_DAYS = 128
WINDOW = DAYS_INPUT + AUX_DAYS
REF_IDX = DAYS_INPUT - 1
RAW = ['open', 'high', 'low', 'close', 'volume', 'delta']
FACTORS = get_technical_factors('new24')
COLS = RAW + FACTORS + ['idx']
# 由 --variant 决定（build_labels 的 fields 与这三组列名必须同源）
_FIELDS = V.fields_for('')
SUFFIX = ''
LABEL_COLS = [f'y_{n}{h}' for n in _FIELDS for h in HORIZONS]
BENCH_COLS = [f'b_{n}{h}' for n in _FIELDS for h in HORIZONS]
RESID_COLS = [f'r_{n}{h}' for n in _FIELDS for h in HORIZONS]


def set_variant(variant):
    """按 variant 重设字段集、列名、输出后缀（'' -> 不写后缀，保持向后兼容）。"""
    global _FIELDS, LABEL_COLS, BENCH_COLS, RESID_COLS, SUFFIX
    _FIELDS = V.fields_for(variant)
    SUFFIX = variant
    LABEL_COLS = [f'y_{n}{h}' for n in _FIELDS for h in HORIZONS]
    BENCH_COLS = [f'b_{n}{h}' for n in _FIELDS for h in HORIZONS]
    RESID_COLS = [f'r_{n}{h}' for n in _FIELDS for h in HORIZONS]
    print(f'变体 {variant!r}: {V.label_desc(variant)}  -> 后缀 {SUFFIX!r}', flush=True)

_lab_cache = None


def sym_samples(sym, daily, lab, sidx, cols):
    """该股票全部可用样本 → (X, y, b, r, d, s) 或 None。

    cols = (LEVEL, BENCH, RESID) 列名 —— **必须显式传**：Windows 下 multiprocessing
    用 spawn，子进程会重新 import 本模块，模块级的 LABEL_COLS 会退化成基线值。
    """
    LEVEL_COLS, BENCH_COLS_, RESID_COLS_ = cols
    dates = daily['date'].values
    if len(daily) < WINDOW:
        return None
    L = lab[(lab.symbol == sym) & lab.ok]
    if L.empty:
        return None
    X, Y, B, R, D, S = [], [], [], [], [], []
    for row in L.itertuples():
        a = np.searchsorted(dates, row.j_endDate, side='right') - 1
        if a < REF_IDX or a + AUX_DAYS >= len(daily):
            continue
        w = daily.iloc[a - REF_IDX:a + AUX_DAYS + 1].copy().reset_index(drop=True)
        if len(w) != WINDOW:
            continue
        rc, rv = w['close'].iloc[REF_IDX], w['volume'].iloc[REF_IDX]
        if not (rc > 0 and rv > 0):
            continue
        w['open'] = w['open'] / rc
        w['high'] = w['high'] / rc
        w['low'] = w['low'] / rc
        w['close'] = w['close'] / rc
        w['volume'] = w['volume'] / rv
        w['delta'] = w['high'] - w['low']
        f = add_technical_factor(add_pool_factors(w))
        f['idx'] = f.index / (DAYS_INPUT - 1.0)
        X.append(f[COLS].replace([np.inf, -np.inf], 0).fillna(0)
                 .clip(-127, 127).values.astype('float32'))
        Y.append([getattr(row, c) for c in LEVEL_COLS])
        B.append([getattr(row, c) for c in BENCH_COLS_])
        R.append([getattr(row, c) for c in RESID_COLS_])
        D.append(int(dates[a]))
        S.append(sidx[sym])
    if not X:
        return None
    return (np.stack(X), np.array(Y, dtype='float32'), np.array(B, dtype='float32'),
            np.array(R, dtype='float32'),
            np.array(D, dtype='int64'), np.array(S, dtype='int32'))


def chunk_worker(task):
    """一个 worker 处理一个符号块，输出一个 .npy 组。"""
    ci, syms, daily_map, lab, ex_dir, sidx, cols, suffix = task
    Xs, Ys, Bs, Rs, Ds, Ss = [], [], [], [], [], []
    for s in syms:
        r = sym_samples(s, daily_map[s], lab, sidx, cols)
        if r is not None:
            Xs.append(r[0]); Ys.append(r[1]); Bs.append(r[2])
            Rs.append(r[3]); Ds.append(r[4]); Ss.append(r[5])
    if not Xs:
        return ci, 0
    X = np.concatenate(Xs); Y = np.concatenate(Ys); B = np.concatenate(Bs)
    R = np.concatenate(Rs); D = np.concatenate(Ds); S = np.concatenate(Ss)
    order = np.argsort(D, kind='stable')          # 按日期排序，batch 取行更友好
    np.save(os.path.join(ex_dir, f'chunk_{ci}_X.npy'), X[order])
    np.save(os.path.join(ex_dir, f'chunk_{ci}_y{suffix}.npy'), Y[order])
    np.save(os.path.join(ex_dir, f'chunk_{ci}_b{suffix}.npy'), B[order])
    np.save(os.path.join(ex_dir, f'chunk_{ci}_r{suffix}.npy'), R[order])
    np.save(os.path.join(ex_dir, f'chunk_{ci}_d.npy'), D[order])
    np.save(os.path.join(ex_dir, f'chunk_{ci}_s.npy'), S[order])
    return ci, len(X)


def gen_exchange(exchange, out_dir, chunk, workers):
    print(f'[{exchange}] 读财务面板...', flush=True)
    lab = build_labels(load_panel(exchange), _FIELDS)
    print(f'[{exchange}] 标签可用 (symbol,季度) = {int(lab.ok.sum()):,}', flush=True)
    d = pd.read_csv(os.path.join(ROOT, 'data', exchange.upper(),
                                 f'daily_{exchange.lower()}.csv'),
                    usecols=['symbol', 'date', 'open', 'high', 'low', 'close', 'volume'])
    # NASDAQ 日线有约 1,041 行 symbol 为空 —— 不剔掉会让下面的 sorted() 在
    # str/float(NaN) 比较时抛 TypeError；显式转 str 防止数字型 ticker 被读成 int。
    d = d[d.symbol.notna() & (d.date >= 20000101)].copy()
    d['symbol'] = d.symbol.astype(str)
    d = d.sort_values(['symbol', 'date'])
    syms = sorted(d.symbol.unique())
    print(f'[{exchange}] 日线 {len(d):,} 行 / {len(syms)} 只', flush=True)
    by_sym = dict(list(d.groupby('symbol', sort=False)))
    daily_map = {s: g.reset_index(drop=True) for s, g in by_sym.items()}
    del d, by_sym

    ex_dir = os.path.join(out_dir, exchange)
    os.makedirs(ex_dir, exist_ok=True)
    # 符号表：全局索引 -> symbol 名（供训练时按股票做训练/测试隔离）
    with open(os.path.join(ex_dir, 'symbols.json'), 'w') as fp:
        json.dump(syms, fp)
    sidx = {s: i for i, s in enumerate(syms)}   # 必须显式传给 worker（Windows spawn 不共享全局）
    chunks = [syms[i:i + chunk] for i in range(0, len(syms), chunk)]
    # 只把该块自己的日线传进去 —— 传整个 daily_map 会让每个任务 pickle 几百 MB
    cols = (LABEL_COLS, BENCH_COLS, RESID_COLS)
    tasks = [(i, c, {x: daily_map[x] for x in c}, lab, ex_dir, sidx, cols, SUFFIX)
             for i, c in enumerate(chunks)]

    import multiprocessing as mp
    total = 0
    with mp.Pool(workers) as pool:
        for i, (ci, n) in enumerate(pool.imap_unordered(chunk_worker, tasks)):
            total += n
            if (i + 1) % 5 == 0:
                print(f'  块 {i+1}/{len(tasks)}，累计样本 {total:,}', flush=True)
    print(f'[{exchange}] 完成：{len(chunks)} 块 / {total:,} 个样本', flush=True)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exchanges', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--chunk', type=int, default=250, help='每块的股票数')
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--variant', default='', help='标签变体（见 variants.py）：""=基线 | u1 | ...')
    args = ap.parse_args()
    set_variant(args.variant)
    workers = args.workers or max(1, min(12, (os.cpu_count() or 4) - 4))
    os.makedirs(args.out, exist_ok=True)
    meta = {}
    for ex in args.exchanges:
        meta[ex] = gen_exchange(ex, args.out, args.chunk, workers)
    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print(f'\n全部完成：{meta}  -> {args.out}')


if __name__ == '__main__':
    main()
