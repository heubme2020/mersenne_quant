"""生产估值公式的成分体检 + 优化方案对比。

生产端（`two/get_two_predict.py:205`）：
    buffett = dcf × (每股净资产 / 收盘价)  +  schloss
              ↑seven输出   ↑ B/P              ↑ 营运资本/收盘价（格雷厄姆）
    ↓ 按 buffett 取前 N，再按 up_down 取第一

本脚本回答三个问题：
  1. 现有两项（schloss、B/P）**各自**有没有用？（从来没人测过）
  2. 我们的「毛利/市值」「毛利提升」能不能补进去？
  3. 「直接相加」vs「先标准化再加」差多少？

注意：`dcf` 是 seven 模型对当前季度的输出，历史序列拿不到 -> 回测里
用 `B/P` 单独代表那一项（dcf 是乘数，不改变 B/P 的排序方向，但会改变加权，
所以这里测的是"没有 dcf 加权"的版本，结论对"该不该加新维度"仍然有效）。

全部 lag 安全（锚点在季度末时当季财报尚未披露，回测里统一往后推 lag 个季度）。
用法：python two/nowcast/valuation_compare.py --exchanges SHZ SHH --lag 1
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import labels as L                                   # noqa: E402

# Windows 控制台默认 GBK，而本文件会打印 `−`（U+2212）等非 GBK 字符 —— 从 GBK 控制台直接跑
# 会 UnicodeEncodeError。与其它脚本同一套修法（2026-09-28 扫描后补齐）。
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass


HORIZONS = [1, 3, 7]
DAYS_PER_Q = 63
GP = 'grossProfit'


def load_daily_close(exchanges):
    frames = []
    for ex in exchanges:
        frames.append(pd.read_csv(
            os.path.join(L.ROOT, 'data', ex.upper(), f'daily_{ex.lower()}.csv'),
            usecols=['symbol', 'date', 'close'], engine='pyarrow'))
    d = pd.concat(frames, ignore_index=True)
    d['symbol'] = d.symbol.astype(str)
    d = d[d.close > 0].sort_values(['symbol', 'date'])
    out = {s: (g['date'].values, g['close'].values.astype('float64'))
           for s, g in d.groupby('symbol', sort=False)}
    del d
    return out


def max_dd(x):
    eq = np.cumprod(1 + np.asarray(x))
    return float((eq / np.maximum.accumulate(eq) - 1).min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exchanges', nargs='+', default=['SHZ', 'SHH'])
    ap.add_argument('--root', default='Z:/quant_data/nowcast')
    ap.add_argument('--out', default='returns_eval')
    ap.add_argument('--lag', type=int, default=1)
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

    # ---- 面板：毛利/股本（income） + 每股营运资本/每股净资产（indicator）----
    frames = []
    for ex in args.exchanges:
        inc = pd.read_csv(os.path.join(L.ROOT, 'data', ex.upper(), f'income_{ex.lower()}.csv'),
                          usecols=['symbol', 'endDate', GP, 'weightedAverageShsOut'])
        ind = pd.read_csv(os.path.join(L.ROOT, 'data', ex.upper(), f'indicator_{ex.lower()}.csv'),
                          usecols=['symbol', 'endDate', 'operatingCapitalPerShare',
                                   'netAssetValuePerShare'])
        frames.append(inc.merge(ind, on=['symbol', 'endDate'], how='inner'))
    p = pd.concat(frames, ignore_index=True)
    p['symbol'] = p.symbol.astype(str)
    p = p[p.endDate.astype(str).str[4:].isin(['0331', '0630', '0930', '1231'])]
    p = p.drop_duplicates(subset=['symbol', 'endDate'], keep='last')
    p = p.sort_values(['symbol', 'endDate']).reset_index(drop=True)
    g = p.groupby('symbol', sort=False)

    lg = args.lag
    P = pd.DataFrame({'symbol': p.symbol, 'j_endDate': p.endDate,
                      'shares': p['weightedAverageShsOut']})
    P['opcap_ps'] = g['operatingCapitalPerShare'].shift(lg)
    P['nav_ps'] = g['netAssetValuePerShare'].shift(lg)
    for h in HORIZONS:
        P[f'trail{h}'] = g[GP].transform(
            lambda x, h=h: x.rolling(h, min_periods=h).sum()).shift(lg)
    # 毛利提升额：最近一季 vs 去年同季（lag 安全）
    P['dgp'] = g[GP].shift(lg) - g[GP].shift(lg + 4)
    P = P.dropna(subset=['shares']).sort_values(['symbol', 'j_endDate']).reset_index(drop=True)
    print(f'  面板 {len(P):,} 行', flush=True)

    # ---- 前瞻收益 ----
    print('算收益...', flush=True)
    rows = []
    for sym, (dt, cl) in daily.items():
        pos = np.searchsorted(dt, anchors, side='right') - 1
        v = pd.DataFrame({'symbol': sym, 'anchor': anchors})
        v['close'] = np.where(pos >= 0, cl[np.clip(pos, 0, None)], np.nan)
        for h in HORIZONS:
            ep = pos + DAYS_PER_Q * h
            good = (pos >= 0) & (ep < len(cl))
            r = np.full(len(anchors), np.nan)
            r[good] = cl[ep[good]] / cl[pos[good]] - 1.0
            v[f'ret{h}'] = r
        rows.append(v)
    R = pd.concat(rows, ignore_index=True)
    del rows

    by = {s: (gg['j_endDate'].values, gg.index.values)
          for s, gg in P.groupby('symbol', sort=False)}
    take = np.full(len(R), -1, dtype=np.int64)
    ad = R['anchor'].values
    for s, idx in R.groupby('symbol', sort=False).indices.items():
        packed = by.get(s)
        if packed is None:
            continue
        ed, rr = packed
        pos = np.searchsorted(ed, ad[idx], side='right') - 1
        okk = pos >= 0
        take[idx[okk]] = rr[pos[okk]]
    ok = take >= 0
    D = R[ok].reset_index(drop=True).copy()
    for c in P.columns:
        if c not in ('symbol', 'j_endDate'):
            D[c] = P[c].values[take[ok]]
    D = D.copy()
    ym = (D['anchor'].values // 100)
    D['qkey'] = (ym // 100) * 10 + ((ym % 100 - 1) // 3 + 1)
    D['mcap'] = D['shares'] * D['close']
    D = D[D.mcap > 0]
    # 现有两项
    D['schloss'] = D['opcap_ps'] / D['close']
    D['bp'] = D['nav_ps'] / D['close']
    # 我们验证过的
    D['gp_yield'] = D['trail1'] / D['mcap']          # 单季毛利/市值（保守）
    D['gp_chg'] = D['dgp'] / D['mcap']               # 毛利提升额/市值
    print(f'  合并后 {len(D):,} 行 / {D.qkey.nunique()} 季度截面，'
          f'平均 {len(D)/D.qkey.nunique():.0f} 只', flush=True)

    # ---- 分布体检（看单位是否可疑）----
    print('\n  成分分布体检（中位数 / 5% / 95%）:')
    for c in ['schloss', 'bp', 'gp_yield', 'gp_chg']:
        v = D[c].dropna()
        print(f'    {c:10s} {v.median():>+10.3f}  [{v.quantile(.05):>+9.3f}, '
              f'{v.quantile(.95):>+9.3f}]   非空 {len(v):,}')

    # ---- 分组评估 ----
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
            rec['neg'] = (hs < 0).mean()
            per.append(rec)
        a = pd.DataFrame(per)
        sp = a['Q1'] - a['Q5']
        return (sp.mean(), sp.mean() / (sp.std() / np.sqrt(len(sp))),
                a['top50'].mean(), a['neg'].mean(), max_dd(a['top50'].values))

    # ---- 合成方案（逐截面 rank 标准化后加权）----
    def combo(cols, weights=None):
        """逐截面把各列 rank 到 [0,1]，加权求和成一个分数列。"""
        w = weights or [1.0] * len(cols)
        s = pd.Series(0.0, index=D.index)
        cnt = pd.Series(0.0, index=D.index)
        for c, wi in zip(cols, w):
            rk = D.groupby('qkey')[c].rank(pct=True)
            s = s.add(rk * wi, fill_value=0)
            cnt = cnt.add(rk.notna() * wi, fill_value=0)
        return s / cnt.replace(0, np.nan)

    print()
    for h in HORIZONS:
        print(f'{"="*100}\n  持有 {h} 季（中位数收益）\n{"="*100}')
        print(f'  {"方案":26s}{"Q1":>9s}{"Q2":>9s}{"Q3":>9s}{"Q4":>9s}{"Q5":>9s}'
              f'{"Q1−Q5":>10s}{"t":>7s}{"前50%":>9s}{"负收益":>8s}{"回撤":>9s}')
        plans = [
            ('schloss(现有)', ['schloss']),
            ('B/P(现有)', ['bp']),
            ('现有 = bp + schloss（直接加）',
             None),                       # 特殊：直接相加
            ('毛利/市值', ['gp_yield']),
            ('毛利提升/市值', ['gp_chg']),
            ('现有 + 毛利', ['bp', 'schloss', 'gp_yield']),
            ('现有 + 毛利提升', ['bp', 'schloss', 'gp_chg']),
            ('现有 + 两者（标准化）', ['bp', 'schloss', 'gp_yield', 'gp_chg']),
        ]
        for name, cols in plans:
            if cols is None:
                D['_raw'] = D['bp'] + D['schloss']
                col = '_raw'
            else:
                col = '_c'
                D[col] = combo(cols)
            m, t, top, neg, mdd = ev(col, h)
            q = []
            for _, gg in D[['qkey', col, f'ret{h}']].dropna().groupby('qkey'):
                if len(gg) < 50:
                    continue
                r = gg[col].rank(ascending=False, method='first')
                qq = np.ceil(r / len(gg) * 5).astype(int).clip(1, 5)
                q.append([gg.loc[qq == k, f'ret{h}'].median() for k in range(1, 6)])
            qm = np.nanmean(np.array(q), axis=0)
            print(f'  {name:26s}' + ''.join(f'{x*100:>+8.2f}%' for x in qm)
                  + f'{m*100:>+9.2f}%{t:>+7.2f}{top*100:>+8.2f}%'
                  f'{neg*100:>7.0f}%{mdd*100:>+8.1f}%')
        print()
    D.to_csv(os.path.join(HERE, args.out, 'valuation_detail.csv'), index=False)


if __name__ == '__main__':
    main()
