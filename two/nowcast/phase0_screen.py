"""Phase 0：纯日线技术因子 → 前瞻财务标签的逐截面 IC / ICIR 筛选。

对比两组分母方案（字段 X ∈ {ebit, revenue, operatingCashFlow}，h ∈ {1,3,7} 季度）：

  A 组（统一净资产锚）   label = ΣX[j+1 : j+h] / 净资产[j-3]
  B 组（自相对锚）       label = ΣX[j+1 : j+h] / X[j-3]

其中 j = 锚点日 t 之前最后一个已结束的季度（endDate ≤ t）。
分子从 j+1 起 => 第一个被预测的季度就是「当前正在进行、尚未披露」的那一季。

因子：完全复刻 one/gen_train_data.py 的 baseline 24 个技术因子 ——
      取锚点日前 889 个交易日，按窗口最后一天的 close / volume 归一化，再算因子。
      即模型在真实训练里能看到的那 24 个数。

指标：IC = 逐锚点横截面 Spearman 相关；ICIR = mean(IC)/std(IC)（与 one/evaluate.py 同口径）。

用法：
    python phase0_screen.py --exchanges SHZ SHH --step 63 --cache
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(BASE, 'one'))

from gen_train_data import add_technical_factor  # noqa: E402
from factor_config import get_technical_factors  # noqa: E402
from factors_long import add_technical_factor_long, FACTORS as LONG_FACTORS  # noqa: E402

DAYS_INPUT = 127 * 7           # 889：模型输入窗口

# one 的 24 个因子最长窗口 31 日，覆盖不到「季度级」尺度；longplus 再补长周期动量/回撤
EXTRA_LONG = ['mom127', 'mom381', 'mom889', 'hi889', 'lo889']
FACTOR_SETS = {
    'one': (add_technical_factor, get_technical_factors('baseline'), []),
    'long': (add_technical_factor_long, LONG_FACTORS, []),
    'longplus': (add_technical_factor_long, LONG_FACTORS, EXTRA_LONG),
    'onelong': (add_technical_factor, get_technical_factors('baseline'), EXTRA_LONG),
}
FIELDS = {'ebit': ('income', 'ebit'),
          'revenue': ('income', 'revenue'),
          'ocf': ('cashflow', 'operatingCashFlow')}
HORIZONS = [1, 3, 7]
EQ = 'totalStockholdersEquity'


# ---------------------------------------------------------------- 标签
def load_panel(exchange):
    """公司 × 季度的财务面板（含 ebit / revenue / ocf / 净资产）。"""
    ex = exchange.lower()
    inc = pd.read_csv(os.path.join(BASE, 'data', exchange.upper(), f'income_{ex}.csv'),
                      usecols=['symbol', 'endDate', 'ebit', 'revenue'])
    cf = pd.read_csv(os.path.join(BASE, 'data', exchange.upper(), f'cashflow_{ex}.csv'),
                     usecols=['symbol', 'endDate', 'operatingCashFlow'])
    bal = pd.read_csv(os.path.join(BASE, 'data', exchange.upper(), f'balance_{ex}.csv'),
                      usecols=['symbol', 'endDate', EQ])
    p = inc.merge(cf, on=['symbol', 'endDate'], how='inner')
    p = p.merge(bal, on=['symbol', 'endDate'], how='inner')
    p = p[p.endDate.astype(str).str[4:].isin(['0331', '0630', '0930', '1231'])]
    p = p.drop_duplicates(subset=['symbol', 'endDate'], keep='last')
    return p.sort_values(['symbol', 'endDate']).reset_index(drop=True)


def build_labels(panel):
    """对每个 (symbol, j) 生成 A/B 两组标签。j = 该季度自身。"""
    g = panel.groupby('symbol', sort=False)
    out = panel[['symbol', 'endDate']].copy()

    for name, (_, col) in FIELDS.items():
        s = panel[col]
        for h in HORIZONS:
            # 未来 h 季之和：j+1 .. j+h
            fwd = g[col].transform(
                lambda x: x.shift(-1).rolling(h, min_periods=h).sum().shift(-(h - 1)))
            out[f'{name}_fwd{h}'] = fwd
        out[f'{name}_m3'] = g[col].shift(3)          # B 组分母
    out['eq_m3'] = g[EQ].shift(3)                    # A 组分母
    return out.replace([np.inf, -np.inf], np.nan)


def attach_labels(panel, labels):
    """给定 (symbol, endDate=j) 的标签表，按 j 分配；同时给出锚点日 t 的映射。"""
    lab = labels.copy()
    lab['j_endDate'] = lab['endDate']
    keep = ['symbol', 'j_endDate', 'eq_m3'] + \
           [f'{n}_fwd{h}' for n in FIELDS for h in HORIZONS] + \
           [f'{n}_m3' for n in FIELDS]
    return lab[keep]


# ---------------------------------------------------------------- 因子
def load_daily(exchanges):
    frames = []
    for ex in exchanges:
        d = pd.read_csv(os.path.join(BASE, 'data', ex.upper(), f'daily_{ex.lower()}.csv'),
                        usecols=['symbol', 'date', 'open', 'high', 'low', 'close', 'volume'])
        d = d[d.date >= 20050101]
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    return d.sort_values(['symbol', 'date']).reset_index(drop=True)


def add_long_horizon(w):
    """补长周期动量/回撤（都在 889 日窗口内，尺度无关）。"""
    c = w['close']
    w['mom127'] = c / c.shift(127) - 1
    w['mom381'] = c / c.shift(381) - 1
    w['mom889'] = c / c.shift(889) - 1
    w['hi889'] = c.rolling(DAYS_INPUT).max() / c
    w['lo889'] = c.rolling(DAYS_INPUT).min() / c
    return w


def factors_for_symbol(sym, g, anchors, mode='one'):
    """单只股票：在全局锚点日网格上算因子（复刻 one 的归一化 + 因子流程）。

    锚点日取「该股票 ≤ 全局锚点日的最后一个交易日」，并保证前面有满 889 天的历史。
    返回的 date 是**全局锚点日**（不是实际交易日），这样各股票的截面能对齐。
    """
    fn, names, extra = FACTOR_SETS[mode]
    all_names = list(names) + list(extra)
    g = g.sort_values('date').reset_index(drop=True)
    dates = g['date'].values
    close = g['close'].values
    vol = g['volume'].values
    pos = np.searchsorted(dates, anchors, side='right') - 1
    rows = []
    for k in range(len(anchors)):
        a = pos[k]
        if a < DAYS_INPUT - 1:
            continue
        rc, rv = close[a], vol[a]
        if not (rc > 0 and rv > 0):
            continue
        w = g.iloc[a - DAYS_INPUT + 1:a + 1].copy().reset_index(drop=True)
        if (w['close'] <= 0).any() or (w['volume'] <= 0).any():
            continue
        w['open'] = w['open'] / rc
        w['high'] = w['high'] / rc
        w['low'] = w['low'] / rc
        w['close'] = w['close'] / rc
        w['volume'] = w['volume'] / rv
        w['delta'] = w['high'] - w['low']
        w = fn(w)
        if extra:
            w = add_long_horizon(w)
        last = w.iloc[-1]
        rec = {'symbol': sym, 'date': int(anchors[k])}
        for c in all_names:
            rec[c] = float(last[c])
        rows.append(rec)
    return rows


def factors_at(daily, symbols, step, workers=None, mode='one'):
    """全局锚点日网格 × 全部股票，多进程算因子。"""
    daily = daily[daily.symbol.isin(symbols)]
    all_dates = np.sort(daily['date'].unique())
    anchors = all_dates[DAYS_INPUT - 1::step]
    print(f'因子集 {mode}（{len(get_set_names(mode))} 个）  全局锚点日 {len(anchors)} 个：'
          f'{anchors[0]} → {anchors[-1]}', flush=True)

    groups = [(s, g) for s, g in daily.groupby('symbol', sort=False) if len(g) > DAYS_INPUT]
    print(f'参与计算的股票 {len(groups)} 只', flush=True)

    tasks = [(s, g, anchors, mode) for s, g in groups]
    workers = workers or max(1, (os.cpu_count() or 4) - 1)
    rows = []
    if workers > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            for i, r in enumerate(pool.starmap(factors_for_symbol, tasks, chunksize=8)):
                rows.extend(r)
                if (i + 1) % 500 == 0:
                    print(f'  {i+1}/{len(tasks)} 只，累计 {len(rows)} 行', flush=True)
    else:
        for s, g, a, m in tasks:
            rows.extend(factors_for_symbol(s, g, a, m))
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)


def get_set_names(mode):
    fn, names, extra = FACTOR_SETS[mode]
    return list(names) + list(extra)


# ---------------------------------------------------------------- IC
def ic_stats(df, factor, label, min_stocks=20):
    """逐日横截面 Spearman IC → (IC, ICIR, t, n)。与 one/evaluate.py 口径一致。"""
    ics = []
    for _, x in df[['date', factor, label]].dropna().groupby('date'):
        if len(x) < min_stocks:
            continue
        ics.append(x[factor].rank().corr(x[label].rank()))
    s = pd.Series(ics).dropna()
    if len(s) < 2:
        return np.nan, np.nan, np.nan, 0
    m, sd = s.mean(), s.std()
    return m, (m / sd if sd > 0 else np.nan), (m / (sd / np.sqrt(len(s))) if sd > 0 else np.nan), len(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exchanges', nargs='+', default=['SHZ', 'SHH'])
    ap.add_argument('--step', type=int, default=63, help='锚点间隔（交易日）')
    ap.add_argument('--symbols-per-ex', type=int, default=0, help='0=全部')
    ap.add_argument('--cache', action='store_true', help='缓存因子表')
    ap.add_argument('--workers', type=int, default=0, help='0=自动')
    ap.add_argument('--factor-set', default='one', choices=list(FACTOR_SETS))
    ap.add_argument('--out', default='phase0_results')
    args = ap.parse_args()
    factors = get_set_names(args.factor_set)

    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         f'{args.out}_factors_{args.factor_set}.pkl')

    panels, labels = [], []
    for ex in args.exchanges:
        p = load_panel(ex)
        panels.append(p)
        labels.append(build_labels(p))
        print(f'{ex}: {p.symbol.nunique()} 只 × {p.endDate.nunique()} 季', flush=True)
    panel = pd.concat(panels, ignore_index=True)
    lab = attach_labels(panel, pd.concat(labels, ignore_index=True))

    if args.cache and os.path.exists(cache):
        fac = pd.read_pickle(cache)
        print(f'读取因子缓存 {cache}（{len(fac)} 行）', flush=True)
    else:
        daily = load_daily(args.exchanges)
        syms = sorted(daily.symbol.unique())
        if args.symbols_per_ex:
            rng = np.random.RandomState(42)
            ex_of = {s: s.split('.')[-1] for s in syms}
            keep = []
            for tag in sorted(set(ex_of.values())):
                ss = [s for s in syms if ex_of[s] == tag]
                keep += list(rng.choice(ss, min(args.symbols_per_ex, len(ss)), replace=False))
            syms = sorted(keep)
        print(f'日线 {len(daily)} 行，{len(syms)} 只股票，锚点间隔 {args.step} 日', flush=True)
        fac = factors_at(daily, set(syms), args.step, args.workers or None, args.factor_set)
        print(f'因子表 {len(fac)} 行', flush=True)
        if args.cache:
            fac.to_pickle(cache)

    # ---- 把锚点日 t 映射到「最后一个已结束的季度 j」 ----
    fac = fac.sort_values(['symbol', 'date']).reset_index(drop=True)
    lab = lab.sort_values(['symbol', 'j_endDate']).reset_index(drop=True)

    # 手写 asof：锚点日 t → 最后一个 endDate ≤ t 的季度 j
    lab_cols = [c for c in lab.columns if c != 'symbol']
    lab_by = {s: (g['j_endDate'].values, g.index.values) for s, g in lab.groupby('symbol')}
    take = np.full(len(fac), -1, dtype=np.int64)
    fac_sym = fac['symbol'].values
    fac_date = fac['date'].values
    unseen = 0
    for s, idx in fac.groupby('symbol', sort=False).indices.items():
        packed = lab_by.get(s)
        if packed is None:
            unseen += len(idx)
            continue
        dates, rows = packed
        pos = np.searchsorted(dates, fac_date[idx], side='right') - 1
        ok = pos >= 0
        take[idx[ok]] = rows[pos[ok]]
        unseen += int((~ok).sum())
    ok = take >= 0
    df = fac[ok].reset_index(drop=True)
    for c in lab_cols:
        df[c] = lab[c].values[take[ok]]
    print(f'合并后 {len(df)} 行（未匹配 {unseen} 行）', flush=True)

    # ---- 构造标签 ----
    for n in FIELDS:
        for h in HORIZONS:
            df[f'A_{n}{h}'] = df[f'{n}_fwd{h}'] / df['eq_m3']
            df[f'B_{n}{h}'] = df[f'{n}_fwd{h}'] / df[f'{n}_m3']

    # ---- 逐 (组, 字段, horizon) × 因子 算 IC/ICIR ----
    # common：A/B 两个分母都 > 0 且两个标签都有限 —— 同一批股票日上比，隔离「分母选择」本身
    # own   ：只用该组自己的分母 > 0 —— 看每组在最大可用样本上的表现（含各自的覆盖损失）
    for n in FIELDS:
        print(f'  分母 ≤0 占比 [{n}]  净资产_m3 {(df.eq_m3<=0).mean()*100:.1f}%  '
              f'{n}_m3 {(df[f"{n}_m3"]<=0).mean()*100:.1f}%', flush=True)

    recs = []
    for scope in ['common', 'own']:
        for grp in ['A', 'B']:
            for n in FIELDS:
                den = 'eq_m3' if grp == 'A' else f'{n}_m3'
                paired = 'B' if grp == 'A' else 'A'
                for h in HORIZONS:
                    label, other = f'{grp}_{n}{h}', f'{paired}_{n}{h}'
                    mask = (df[den] > 0) & np.isfinite(df[label])
                    if scope == 'common':
                        mask &= (df['eq_m3'] > 0) & (df[f'{n}_m3'] > 0) & np.isfinite(df[other])
                    sub = df[mask]
                    for f in factors:
                        ic, icir, t, nn = ic_stats(sub, f, label)
                        recs.append(dict(scope=scope, group=grp, field=n, h=h, factor=f,
                                         ic=ic, icir=icir, t=t, n_dates=nn,
                                         n_obs=int(mask.sum())))
    res = pd.DataFrame(recs)
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out)
    print(f'因子 {len(factors)} 个: {factors}', flush=True)
    os.makedirs(outdir, exist_ok=True)
    res.to_csv(os.path.join(outdir, f'ic_by_factor_{args.factor_set}.csv'), index=False)

    # ---- 汇总：每个标签上「最好的因子」与「因子平均 |IC|」 ----
    def summarize(scope):
        r = res[res.scope == scope]
        rows = []
        for (g, n, h), x in r.groupby(['group', 'field', 'h']):
            x = x.dropna(subset=['ic'])
            best = x.loc[x.ic.abs().idxmax()] if len(x) else None
            rows.append(dict(scope=scope, group=g, field=n, h=h,
                             mean_abs_ic=x.ic.abs().mean(),
                             best_factor=best['factor'] if best is not None else None,
                             best_ic=best['ic'] if best is not None else np.nan,
                             best_icir=best['icir'] if best is not None else np.nan,
                             best_t=best['t'] if best is not None else np.nan,
                             best_ndates=int(best['n_dates']) if best is not None else 0,
                             n_obs=int(best['n_obs']) if best is not None else 0))
        return pd.DataFrame(rows)

    s = pd.concat([summarize('common'), summarize('own')], ignore_index=True)
    s.to_csv(os.path.join(outdir, f'label_summary_{args.factor_set}.csv'), index=False)

    pd.set_option('display.width', 200)
    for scope in ['common', 'own']:
        print(f'\n{"="*96}\n  样本范围：{scope}\n{"="*96}')
        t = s[s.scope == scope].copy()
        t['标签'] = t.group + '_' + t.field + '_' + t.h.astype(str) + 'Q'
        t = t.sort_values(['field', 'h', 'group'])
        print(t[['标签', 'mean_abs_ic', 'best_factor', 'best_ic', 'best_icir', 'best_t',
                 'best_ndates', 'n_obs']]
              .to_string(index=False, float_format=lambda v: f'{v:+.3f}'))
    print(f'\n明细写入 {outdir}/')


if __name__ == '__main__':
    main()
