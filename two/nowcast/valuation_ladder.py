"""估值指标梯子：把「每股X ÷ 股价」（= X ÷ 市值，量纲直接对齐股价）的候选都量一遍收益 IC。

用途：回答「还有哪些能直接和股价对齐的估值指标可以拿来用」，以及检验
「越靠利润表上方越有效」这条规律。所有候选都是同一量纲（某物 ÷ 市值），可直接相加。

方法与 `valuation_compare.py` 保持一致，便于交叉验证：
  * 锚点 = nowcast 的对齐网格（`<root>/<EX>/chunk_*_d.npy`），与旧诊断脚本同源
  * 财务数据 **lag=N 个季度**（默认 1）-> lag-safe，不使用未披露报表
  * 逐季度截面：按信号降序分 5 组，看中位数收益；Q1−Q5 + t + 前50% + 负收益频率 + 回撤
  * NCAV 用 `schloss/get_schloss.py::ncav_per_share`（**生产同一份定义**，避免回测/生产两套口径）

用法：
  python two/nowcast/valuation_ladder.py --root Z:/quant_data/nowcast --lag 1
  python two/nowcast/valuation_ladder.py --exchanges SHZ SHH --h 1
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'schloss'))
from get_schloss import ncav_per_share                      # noqa: E402

DAYS_PER_Q = 63
HORIZONS = [1, 3, 7]
GP = 'grossProfit'

# 候选：name -> (表, 列, 形式)
#   yield      = 列 / 市值
#   change     = (列[j] − 列[j−4]) / 市值      <- 同比变化额（"提升额"）
#   tangible   = (总权益 − 商誉及无形) / 市值
#   ncav       = 格雷厄姆净流动资产 / 市值（= schloss）
#   div        = |commonDividendsPaid| / 市值   <- 股息率
CAND = {
    # ---- 利润表，从上到下（检验"越靠上越有效"）----
    '营收/市值':          ('income',   'revenue',                        'yield'),
    '毛利/市值':          ('income',   'grossProfit',                    'yield'),
    # 「毛利中位/市值」的不同窗口：检验"新鲜 vs 平滑"哪边好（用户 2026-09-26 提出加 3 季版）
    '毛利2季中位/市值':   ('income',   'grossProfit',                    'med2'),
    '毛利3季中位/市值':   ('income',   'grossProfit',                    'med3'),
    '毛利4季中位/市值':   ('income',   'grossProfit',                    'med4'),
    '毛利7季中位/市值':   ('income',   'grossProfit',                    'med7'),
    '营业利润/市值':      ('income',   'operatingIncome',                'yield'),
    'EBITDA/市值':        ('income',   'ebitda',                         'yield'),
    'EBIT/市值':          ('income',   'ebit',                           'yield'),
    '税前利润/市值':      ('income',   'incomeBeforeTax',                'yield'),
    '净利/市值(E/P)':     ('income',   'netIncome',                      'yield'),
    '研发/市值':          ('income',   'researchAndDevelopmentExpenses', 'yield'),
    # ---- 利润表，变化类 ----
    '毛利提升额/市值':    ('income',   'grossProfit',                    'change'),
    '营收提升额/市值':    ('income',   'revenue',                        'change'),
    '净利提升额/市值':    ('income',   'netIncome',                      'change'),
    # ---- 资产负债表 ----
    'B/P(净资产)':        ('balance',  'totalStockholdersEquity',        'yield'),
    '有形B/P':            ('balance',  'totalStockholdersEquity',        'tangible'),
    '净流动资产/市值':    ('balance',  'totalCurrentAssets',             'ncav'),
    '现金+短投/市值':     ('balance',  'cashAndCashEquivalents',         'cash'),
    '净现金/市值':        ('balance',  'cashAndCashEquivalents',         'netcash'),
    '固定资产/市值':      ('balance',  'propertyPlantEquipmentNet',      'yield'),
    '存货/市值':          ('balance',  'inventory',                      'yield'),
    '应收/市值':          ('balance',  'netReceivables',                 'yield'),
    # ---- 现金流表 ----
    'FCF/市值':           ('cashflow', 'freeCashFlow',                   'yield'),
    'OCF/市值':           ('cashflow', 'netCashProvidedByOperatingActivities', 'yield'),
    # ---- 分红 ----
    '股息率':             ('cashflow', 'commonDividendsPaid',            'div'),
}


def load_daily_close(exchanges):
    out = {}
    for ex in exchanges:
        f = os.path.join(ROOT, 'data', ex.upper(), f'daily_{ex.lower()}.csv')
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f, usecols=['symbol', 'date', 'close'], engine='pyarrow')
        d['symbol'] = d.symbol.astype(str)
        d = d[d.close > 0]
        for s, g in d.groupby('symbol', sort=False):
            out[(ex, s)] = (g['date'].values, g['close'].values.astype('float64'))
        del d
    return out


def max_dd(x):
    eq = np.cumprod(1 + np.asarray(x))
    return float((eq / np.maximum.accumulate(eq) - 1).min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exchanges', nargs='+', default=['SHZ', 'SHH'])
    ap.add_argument('--root', default='Z:/quant_data/nowcast')
    ap.add_argument('--lag', type=int, default=1, help='财报滞后几个季度（lag-safe）')
    ap.add_argument('--h', type=int, nargs='+', default=HORIZONS)
    ap.add_argument('--out', default='returns_eval')
    args = ap.parse_args()
    os.makedirs(os.path.join(HERE, args.out), exist_ok=True)

    dd = []
    for ex in args.exchanges:
        for f in sorted(glob.glob(os.path.join(args.root, ex, 'chunk_*_d.npy'))):
            dd.append(np.load(f))
    anchors = np.unique(np.concatenate(dd))
    print(f'锚点 {len(anchors)} 个   lag={args.lag}', flush=True)

    print('读日线...', flush=True)
    daily = load_daily_close(args.exchanges)

    # ---- 面板：income + balance + cashflow ----
    frames = []
    for ex in args.exchanges:
        base = os.path.join(ROOT, 'data', ex.upper())
        inc = pd.read_csv(os.path.join(base, f'income_{ex.lower()}.csv'))
        bal = pd.read_csv(os.path.join(base, f'balance_{ex.lower()}.csv'))
        cf = pd.read_csv(os.path.join(base, f'cashflow_{ex.lower()}.csv'))
        keep_i = ['symbol', 'endDate', 'weightedAverageShsOut']
        keep_b = ['symbol', 'endDate']
        keep_c = ['symbol', 'endDate']
        for _, (tab, col, _form) in CAND.items():
            if tab == 'income' and col in inc.columns and col not in keep_i:
                keep_i.append(col)
            if tab == 'balance' and col in bal.columns and col not in keep_b:
                keep_b.append(col)
            if tab == 'cashflow' and col in cf.columns and col not in keep_c:
                keep_c.append(col)
        for c in ['goodwillAndIntangibleAssets', 'totalLiabilities', 'minorityInterest',
                  'preferredStock', 'totalDebt', 'shortTermInvestments']:
            if c in bal.columns and c not in keep_b:
                keep_b.append(c)
        f = (inc[keep_i].merge(bal[keep_b], on=['symbol', 'endDate'], how='inner')
             .merge(cf[keep_c], on=['symbol', 'endDate'], how='inner'))
        f['ex'] = ex
        frames.append(f)
    p = pd.concat(frames, ignore_index=True)
    p['symbol'] = p.symbol.astype(str)
    p = p[p.endDate.astype(str).str[4:].isin(['0331', '0630', '0930', '1231'])]
    p = p.drop_duplicates(subset=['ex', 'symbol', 'endDate'], keep='last')
    p = p.sort_values(['ex', 'symbol', 'endDate']).reset_index(drop=True)
    print(f'  财务面板 {len(p):,} 行 / {p.symbol.nunique():,} 只', flush=True)

    # ---- lag-safe：每个字段后移 lag 个季度 ----
    g = p.groupby(['ex', 'symbol'], sort=False)
    lg = args.lag
    Q = pd.DataFrame({'ex': p.ex, 'symbol': p.symbol, 'j_endDate': p.endDate,
                      'shares': p['weightedAverageShsOut']})
    for name, (tab, col, form) in CAND.items():
        s = p[col].astype(float)
        if form == 'change':
            v = g[col].shift(lg).astype(float) - g[col].shift(lg + 4).astype(float)
        elif form.startswith('med'):
            # 「截至某季的最近 w 季中位数」，再按 lag 后移（窗口右端 = 滞后后的那一季）
            w = int(form[3:])
            win = g[col].transform(lambda x, w=w: x.rolling(w, min_periods=w).median())
            v = win.groupby([p.ex, p.symbol], sort=False).shift(lg)
        elif form == 'tangible':
            v = (p['totalStockholdersEquity'].astype(float)
                 - p['goodwillAndIntangibleAssets'].astype(float)).groupby(
                     [p.ex, p.symbol], sort=False).shift(lg)
        elif form in ('cash', 'netcash'):
            v = (p['cashAndCashEquivalents'].astype(float)
                 + p['shortTermInvestments'].astype(float))
            if form == 'netcash':
                v = v - p['totalDebt'].astype(float)
            v = v.groupby([p.ex, p.symbol], sort=False).shift(lg)
        else:
            v = g[col].shift(lg).astype(float)
        Q[name] = v.values
    # NCAV：直接用生产同一份定义（schloss/get_schloss.py），再按 lag 后移
    nc = ncav_per_share(p, p[['symbol', 'endDate', 'weightedAverageShsOut']])
    nc['endDate'] = nc.endDate.astype(int)
    # 防错：左连接必须保证「一行对一行」，否则后面的 .values 赋值会整体错位
    assert nc.groupby(['symbol', 'endDate']).size().max() == 1, 'ncav_per_share 出现重复键'
    Q = Q.merge(nc, left_on=['symbol', 'j_endDate'], right_on=['symbol', 'endDate'], how='left')
    Q['净流动资产/市值'] = Q.groupby(['ex', 'symbol'], sort=False)['ncavPerShare'].shift(lg) * Q['shares']
    Q = Q.drop(columns=['endDate', 'ncavPerShare'])
    Q = Q.dropna(subset=['shares']).reset_index(drop=True)
    print(f'  面板 {len(Q):,} 行', flush=True)

    # ---- 前瞻收益 ----
    print('算收益...', flush=True)
    rows = []
    for (ex, sym), (dt, cl) in daily.items():
        pos = np.searchsorted(dt, anchors, side='right') - 1
        v = pd.DataFrame({'ex': ex, 'symbol': sym, 'anchor': anchors})
        v['close'] = np.where(pos >= 0, cl[np.clip(pos, 0, None)], np.nan)
        for h in args.h:
            ep = pos + DAYS_PER_Q * h
            good = (pos >= 0) & (ep < len(cl))
            r = np.full(len(anchors), np.nan)
            r[good] = cl[ep[good]] / cl[pos[good]] - 1.0
            v[f'ret{h}'] = r
        rows.append(v)
    R = pd.concat(rows, ignore_index=True)
    del rows

    by = {k: (gg['j_endDate'].values, gg.index.values)
          for k, gg in Q.groupby(['ex', 'symbol'], sort=False)}
    take = np.full(len(R), -1, dtype=np.int64)
    for k, idx in R.groupby(['ex', 'symbol'], sort=False).indices.items():
        packed = by.get(k)
        if packed is None:
            continue
        ed, rr = packed
        pos = np.searchsorted(ed, R['anchor'].values[idx], side='right') - 1
        ok = pos >= 0
        take[idx[ok]] = rr[pos[ok]]
    ok = take >= 0
    D = R[ok].reset_index(drop=True).copy()
    for c in Q.columns:
        if c not in ('ex', 'symbol', 'j_endDate'):
            D[c] = Q[c].values[take[ok]]
    ym = D['anchor'].values // 100
    D['qkey'] = (ym // 100) * 10 + ((ym % 100 - 1) // 3 + 1)
    D['mcap'] = D['shares'] * D['close']
    D = D[(D.mcap > 0) & D['shares'].notna()].reset_index(drop=True)
    print(f'  合并后 {len(D):,} 行 / {D.qkey.nunique()} 个季度截面，'
          f'平均 {len(D)/D.qkey.nunique():.0f} 只\n', flush=True)

    for name in CAND:
        D[name] = D[name].astype(float) / D['mcap']
        D.loc[~np.isfinite(D[name]), name] = np.nan

    def ev(sig, h):
        ret = f'ret{h}'
        per = []
        for _, gg in D[['qkey', sig, ret]].dropna().groupby('qkey'):
            if len(gg) < 50:
                continue
            r = gg[sig].rank(ascending=False, method='first')
            q = np.ceil(r / len(gg) * 5).astype(int).clip(1, 5)
            rec = {f'Q{k}': gg.loc[q == k, ret].median() for k in range(1, 6)}
            hs = gg.loc[r <= len(gg) / 2.0, ret]
            rec['top50'] = hs.median()
            per.append(rec)
        a = pd.DataFrame(per)
        sp = a['Q1'] - a['Q5']
        t = sp.mean() / (sp.std() / np.sqrt(len(sp))) if len(sp) > 1 and sp.std() > 0 else np.nan
        return sp.mean() * 100, t, a['top50'].mean() * 100, len(sp)

    # ---- 分布体检（确认量纲/单位没写错）----
    print('  中位数与 5%~95% 区间（确认单位合理）:')
    for name in ['B/P(净资产)', '毛利/市值', '净利/市值(E/P)', '净流动资产/市值', '股息率', 'FCF/市值']:
        v = D[name].dropna()
        print(f'    {name:16s} 中位 {v.median():>+8.4f}   [{v.quantile(.05):>+8.4f}, '
              f'{v.quantile(.95):>+8.4f}]   非空 {len(v):>7,}')

    # 落盘明细：后续做「候选之间的相关性 / 增量 IC」分析不必重跑
    out = os.path.join(HERE, args.out, 'valuation_ladder_detail.csv')
    keep = (['ex', 'symbol', 'anchor', 'qkey', 'close', 'shares', 'mcap']
            + [f'ret{h}' for h in args.h] + list(CAND))
    D[keep].to_csv(out, index=False)
    print(f'\n  明细 -> {out}  ({len(D):,} 行)')

    for h in args.h:
        print(f'\n{"="*104}\n  持有 {h} 季：按「Q1−Q5」排序（Q1=信号最强组）\n{"="*104}')
        print(f'  {"指标":20s}{"Q1−Q5":>10s}{"t":>7s}{"前50%":>10s}{"n季度":>8s}')
        res = [(nm,) + ev(nm, h) for nm in CAND]
        res = [r for r in res if np.isfinite(r[1])]
        for nm, sp, t, top, nq in sorted(res, key=lambda r: -r[1]):
            print(f'  {nm:20s}{sp:>+9.2f}%{t:>7.2f}{top:>+9.2f}%{nq:>8d}')


if __name__ == '__main__':
    main()
