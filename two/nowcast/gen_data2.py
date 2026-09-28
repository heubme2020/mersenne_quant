"""nowcast 数据生成 v2：锚点 = 任意交易日，j = 最后【已披露】季度，采样按股票分两种模式。

## 与 gen_data.py（v1）的三处差别

1. **锚点不再是季度末**，而是任意交易日 —— 匹配生产"每天都预测"的事实。
2. **j（标签基准）= 最后一个【已披露】的季度**，按 A 股披露截止日判定：
   Q1→4/30、Q2→8/31、Q3→10/31、Q4→次年4/30。
   v1 用"最后一个已结束的季度"→ 锚点在季度末时当季财报还没披露 → **前视**。
3. **采样按股票分两种模式**（损失改逐样本后，两种可以共存）：
   * **测试股票**（`--aligned-symbols` 指定）→ **全局同步网格**，每 `--key-step` 交易日一个锚点。
     目的：测试集要算**逐日截面 IC**，需要每个锚点有完整截面。
   * **其余股票（训练/验证）** → **per-stock 随机采样**，概率 = `--train-rate`（默认 1/31）。
     目的：覆盖日级相位（生产每天跑），但数据量只 ×2 而不是 ×63。

## 输出

{out}/{EXCHANGE}/chunk_{i}_X.npy / _y{suffix}.npy / _b{suffix}.npy / _r{suffix}.npy / _d.npy / _s.npy
  d = 锚点交易日（`--aligned-symbols` 的股票落在全局网格上，其余是随机日）
  s = 符号索引
另写 `grid.json`（对齐用的全局网格）和 `symbols.json`。

用法：
    python two/nowcast/gen_data2.py --exchanges SHZ SHH --out Z:/quant_data/nowcast2 \
        --variant u1 --aligned-symbols two/nowcast/test_symbols_global889.txt
"""

import argparse
import json
import os
import sys
import zlib

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'one'))
sys.path.insert(0, HERE)

from factor_config import get_technical_factors               # noqa: E402
from gen_train_data import add_technical_factor               # noqa: E402
from factor_pool import add_pool_factors                      # noqa: E402
from labels import load_panel, build_labels, HORIZONS         # noqa: E402
import variants as V                                          # noqa: E402

DAYS_INPUT = 127 * 7          # 889
AUX_DAYS = 128
WINDOW = DAYS_INPUT + AUX_DAYS
REF_IDX = DAYS_INPUT - 1
RAW = ['open', 'high', 'low', 'close', 'volume', 'delta']
FACTORS = get_technical_factors('new24')
COLS = RAW + FACTORS + ['idx']

# A 股财报披露截止日：(季度末 -> 最晚披露月日)；Q4 顺延到次年
DEADLINE = {'0331': (4, 30), '0630': (8, 31), '0930': (10, 31), '1231': (4, 30)}


def available_date(end_dates):
    """季度末 -> 该财报的【最晚披露日】（YYYYMMDD）。用它做 asof 才是无前视的。"""
    out = np.empty(len(end_dates), dtype='int64')
    for i, ed in enumerate(np.asarray(end_dates, dtype='int64')):
        y, md = int(ed) // 10000, f'{int(ed) % 10000:04d}'
        m, d = DEADLINE[md]
        out[i] = (y + (1 if md == '1231' else 0)) * 10000 + m * 100 + d
    return out


_lab_cache = {}


def get_labels(exchange, fields):
    key = (exchange, tuple(fields))
    if key not in _lab_cache:
        lab = build_labels(load_panel(exchange), fields)
        # ⚠️ 2026-09-25 修 bug：这里**曾经**先 `lab = lab[lab.ok]` 再算 avail。那是错的 ——
        # ok 要求「未来 max(HORIZONS) 季完整」，于是每只股票**最后 7 个季度被滤掉**；
        # 锚点日落在最后 7 季时，searchsorted 找「最后一个已披露季度」就退回更早的季度：
        #   (a) 本该无标签的最近期锚点被「凭空造出」标签；
        #   (b) 造出来的标签，其未来窗口有一大截落在锚点日**之前**（部分/全部已实现）
        #       → 系统性抬高 IC。SHZ 实测这类行占 10.6%。
        # 现在 avail 在【完整行集】上算；ok 只作为一个列保留，由 sym_samples 在取到 asof
        # 行之后检查（ok=False 就跳过该锚点 —— 那个锚点的标签本来就不存在）。
        lab['avail'] = available_date(lab.j_endDate.values)
        _lab_cache[key] = lab.sort_values(['symbol', 'avail']).reset_index(drop=True)
    return _lab_cache[key]


def build_x(w):
    """日线窗口 w（WINDOW 行）-> 模型输入 X (WINDOW, 31)，float32。

    **训练与推理必须共用这一段**（`two/get_two_predict.py` 也调它）：
    31 维的列集合与列序、归一化基准、idx 分母三者任何一处漂移都会静默出错。
    注意两点都与 `two/` 的旧实现不同：
      * raw 块顺序是 open,high,low,close,volume,delta（two 是 open,low,high,...）
      * 技术因子用 one/factor_config.py 的 NEW_24（two 用 CURRENT_24）
      * idx 分母是 DAYS_INPUT-1 = 888（two 的生产脚本误用 380）
    按【名字】选取 f[COLS]，所以 w 的物理列序无所谓。
    """
    rc, rv = w['close'].iloc[REF_IDX], w['volume'].iloc[REF_IDX]
    if not (rc > 0 and rv > 0):
        return None
    w = w.copy()
    w['open'] = w['open'] / rc
    w['high'] = w['high'] / rc
    w['low'] = w['low'] / rc
    w['close'] = w['close'] / rc
    w['volume'] = w['volume'] / rv
    w['delta'] = w['high'] - w['low']
    f = add_technical_factor(add_pool_factors(w))
    f['idx'] = f.index / (DAYS_INPUT - 1.0)
    return f[COLS].replace([np.inf, -np.inf], 0).fillna(0).clip(-127, 127).values.astype('float32')


def make_sample(w, row, cols):
    """从日线窗口 w 与标签行 row 造一个样本 -> (X, y, b, r) 或 None。"""
    X = build_x(w)
    if X is None:
        return None
    lc, bc, rc_ = cols
    return (X, [getattr(row, c) for c in lc], [getattr(row, c) for c in bc],
            [getattr(row, c) for c in rc_])


def sym_samples(sym, daily, lab_sym, sidx, cols, aligned, grid=None, rate=None, seed=0):
    """该股票的样本。aligned=True -> 对齐全局网格；否则按 rate 逐股随机采。

    随机采样用 `np.random.RandomState(seed)`，seed 由 `crc32(symbol)` 决定 —— **不用
    内置 hash()**：Python 的字符串 hash 每进程随机化（PYTHONHASHSEED），会让子进程
    采到不同的日，结果不可复现。
    """
    dates = daily['date'].values
    if len(daily) < WINDOW or lab_sym is None or lab_sym.empty:
        return None
    avail = lab_sym['avail'].values
    labok = lab_sym['ok'].values          # 该季度的标签是否真的存在（见 get_labels 的修 bug 说明）
    lo, hi = REF_IDX, len(dates) - AUX_DAYS - 1
    if hi < lo:
        return None

    if aligned:
        pos = np.searchsorted(dates, grid, side='right') - 1     # 每格 -> 该股 ≤ 格的最后交易日
        cand = np.unique(pos[(pos >= lo) & (pos <= hi)])
    else:
        rs = np.random.RandomState(seed)
        n = hi - lo + 1
        cand = (lo + np.where(rs.random_sample(n) < rate)[0]).astype(np.int64)
    if len(cand) == 0:
        return None

    X, Y, B, R, D, S = [], [], [], [], [], []
    for a in cand:
        t = int(dates[a])
        k = np.searchsorted(avail, t, side='right') - 1      # 最后一个【已披露】季度
        if k < 0 or not labok[k]:
            # ok=False：该季度的标签不存在（未来窗口超出数据）→ 这个锚点没有标签，
            # 直接跳过。**不能**退回更早的季度 —— 那会造出「未来窗口已实现」的假标签。
            continue
        row = lab_sym.iloc[k]
        w = daily.iloc[a - REF_IDX:a + AUX_DAYS + 1]
        if len(w) != WINDOW:
            continue
        got = make_sample(w.reset_index(drop=True), row, cols)
        if got is None:
            continue
        X.append(got[0]); Y.append(got[1]); B.append(got[2]); R.append(got[3])
        D.append(t); S.append(sidx[sym])
    if not X:
        return None
    return (np.stack(X), np.array(Y, 'float32'), np.array(B, 'float32'),
            np.array(R, 'float32'), np.array(D, 'int64'), np.array(S, 'int32'))


def chunk_worker(task):
    ci, syms, daily_map, labs, ex_dir, sidx, cols, suffix, grid, rate, seed, al = task
    Xs, Ys, Bs, Rs, Ds, Ss = [], [], [], [], [], []
    for j, s in enumerate(syms):
        r = sym_samples(s, daily_map[s], labs.get(s), sidx, cols,
                        aligned=(s in al), grid=grid, rate=rate,
                        seed=seed + zlib.crc32(s.encode()))
        if r is not None:
            Xs.append(r[0]); Ys.append(r[1]); Bs.append(r[2])
            Rs.append(r[3]); Ds.append(r[4]); Ss.append(r[5])
    if not Xs:
        return ci, 0
    X = np.concatenate(Xs); Y = np.concatenate(Ys); B = np.concatenate(Bs)
    R = np.concatenate(Rs); D = np.concatenate(Ds); S = np.concatenate(Ss)
    order = np.argsort(D, kind='stable')
    np.save(os.path.join(ex_dir, f'chunk_{ci}_X.npy'), X[order])
    np.save(os.path.join(ex_dir, f'chunk_{ci}_y{suffix}.npy'), Y[order])
    np.save(os.path.join(ex_dir, f'chunk_{ci}_b{suffix}.npy'), B[order])
    np.save(os.path.join(ex_dir, f'chunk_{ci}_r{suffix}.npy'), R[order])
    np.save(os.path.join(ex_dir, f'chunk_{ci}_d.npy'), D[order])
    np.save(os.path.join(ex_dir, f'chunk_{ci}_s.npy'), S[order])
    return ci, len(X)


def load_daily(exchange):
    d = pd.read_csv(os.path.join(ROOT, 'data', exchange.upper(),
                                 f'daily_{exchange.lower()}.csv'),
                    usecols=['symbol', 'date'] + RAW[:5])
    d = d[d.symbol.notna() & (d.date >= 20000101)].copy()
    d['symbol'] = d.symbol.astype(str)
    d = d.sort_values(['symbol', 'date'])
    return d


def gen_exchange(exchange, out_dir, chunk, workers, fields, suffix, aligned, rate, grid, seed):
    print(f'[{exchange}] 读财务面板...', flush=True)
    lab = get_labels(exchange, fields)
    print(f'[{exchange}] 标签可用 (symbol,季度) = {len(lab):,}', flush=True)
    d = load_daily(exchange)
    syms = sorted(d.symbol.unique())
    print(f'[{exchange}] 日线 {len(d):,} 行 / {len(syms)} 只  '
          f'对齐股 {sum(1 for s in syms if f"{exchange}:{s}" in aligned)} 只', flush=True)
    daily_map = {s: g.reset_index(drop=True) for s, g in d.groupby('symbol', sort=False)}
    del d


    labs = {s: g for s, g in lab.groupby('symbol', sort=False)}
    al = {s for s in syms if f'{exchange}:{s}' in aligned}

    ex_dir = os.path.join(out_dir, exchange)
    os.makedirs(ex_dir, exist_ok=True)
    with open(os.path.join(ex_dir, 'symbols.json'), 'w') as fp:
        json.dump(syms, fp)
    sidx = {s: i for i, s in enumerate(syms)}
    cols = ([f'y_{n}{h}' for n in fields for h in HORIZONS],
            [f'b_{n}{h}' for n in fields for h in HORIZONS],
            [f'r_{n}{h}' for n in fields for h in HORIZONS])
    chunks = [syms[i:i + chunk] for i in range(0, len(syms), chunk)]
    tasks = [(i, c, {x: daily_map[x] for x in c}, {x: labs.get(x) for x in c},
              ex_dir, sidx, cols, suffix, grid, rate, seed, {x for x in c if x in al})
             for i, c in enumerate(chunks)]

    import multiprocessing as mp
    total = 0
    with mp.Pool(workers) as pool:
        for i, (ci, n) in enumerate(pool.imap_unordered(chunk_worker, tasks)):
            total += n
            if (i + 1) % 5 == 0:
                print(f'  块 {i+1}/{len(tasks)}，累计样本 {total:,}', flush=True)
    print(f'[{exchange}] 完成：{len(chunks)} 块 / {total:,} 个样本', flush=True)
    return total, grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exchanges', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--chunk', type=int, default=250)
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--variant', default='u1')
    ap.add_argument('--aligned-symbols',
                    default=os.path.join(HERE, 'test_symbols_all31_889.txt'),
                    help="这些股票（'EX:SYM'）用日期对齐采样（测试集）")
    ap.add_argument('--key-step', type=int, default=31, help='对齐网格步长（交易日）')
    ap.add_argument('--train-rate', type=float, default=1.0 / 31.0, help='其余股票的随机采样率')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--grid-exchange', default='SHZ', help='全局网格的交易日历基准')
    args = ap.parse_args()

    fields = V.fields_for(args.variant)
    print(f'变体 {args.variant!r}: {V.label_desc(args.variant)}', flush=True)

    # 对齐名单【只】来自 --aligned-symbols（'EX:SYM' 格式）—— 用户要求测试集
    # 从 31 个交易所均匀随机，不要偏袒 A 股，所以不再叠加 one/test_symbols.txt。
    aligned = set()
    if os.path.exists(args.aligned_symbols):
        for line in open(args.aligned_symbols, encoding='utf-8'):
            s = line.strip()
            if s:
                aligned.add(s)
    print(f'对齐采样的股票 {len(aligned):,} 只（测试集；其余 1/{round(1/args.train_rate)} 随机采）',
          flush=True)

    workers = args.workers or max(1, min(12, (os.cpu_count() or 4) - 4))
    os.makedirs(args.out, exist_ok=True)
    # 全局网格：以参考交易所（默认 SHZ）的交易日历为准，每 key_step 取一格。
    # 用 A 股日历做基准 -> A 股测试股天然对齐；其它交易所取「≤ 格日的最后交易日」。
    gd = pd.read_csv(os.path.join(ROOT, 'data', args.grid_exchange.upper(),
                                  f'daily_{args.grid_exchange.lower()}.csv'),
                     usecols=['date'])
    gdates = np.sort(gd['date'].unique())
    grid = gdates[gdates >= 20040101][::args.key_step]
    print(f'全局网格：{len(grid)} 个锚点（{grid[0]} → {grid[-1]}，步长 {args.key_step} 交易日）',
          flush=True)
    meta = {}
    for ex in args.exchanges:
        n, grid = gen_exchange(ex, args.out, args.chunk, workers, fields,
                               args.variant, aligned, args.train_rate,
                               grid, args.seed)
        meta[ex] = n
        with open(os.path.join(args.out, ex, 'grid.json'), 'w') as fp:
            json.dump(grid.tolist(), fp)
    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print(f'\n全部完成：{meta}  -> {args.out}')


if __name__ == '__main__':
    main()
