"""基本面动量：用**财报实际数据**构造"毛利提升"信号，测收益与风险。

为什么这个和之前所有实验不同：
  * 信息源是**财报**，不是价量 —— 所以"已被价格反映"这道墙不必然成立
  * 机制是**市场对财报反应不足**（公告后漂移 PEAD）—— 行为偏差，不是信息优势
  * 不需要模型 → 可以用**全 A 股全样本**（约 70 万观测），功效远高于模型测试集的 3 万

候选信号（全部只用锚点日已知数据，无前视）：
  gp_yoy1   = (GP[j] − GP[j-4]) / |GP[j-4]|             单季毛利同比提升率
  gp_yoy3   = (ΣGP[j-2..j] − ΣGP[j-6..j-4]) / |...|     3 季毛利同比提升率
  gp_yield  = (GP[j] − GP[j-4]) / 市值                  毛利提升额 ÷ 市值
  gp3_yield = (ΣGP[j-2..j] − ΣGP[j-6..j-4]) / 市值
对照：
  value     = ΣGP[j-h+1..j] / 市值                      （已证明有效的估值因子）
组合：
  value ∧ gp_yoy1 都取前 50%                            （便宜 + 在改善）

输出：五分组单调性 / 多空差 t / 中位数收益 / 负收益频率 / 最大回撤 / 最差5%
另外：各信号与长期动量的相关性（判断是不是又一个动量代理）

用法：python two/nowcast/fund_momentum.py --exchanges SHZ SHH
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


def max_dd(series):
    eq = np.cumprod(1 + np.asarray(series))
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1).min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exchanges', nargs='+', default=['SHZ', 'SHH'])
    ap.add_argument('--root', default='Z:/quant_data/nowcast')
    ap.add_argument('--out', default='returns_eval')
    ap.add_argument('--lag', type=int, default=1,
                    help='保守滞后：锚点日只用**已披露**的季度。'
                         '0=含未披露的当季(有前视) 1=用 j-1(保守，匹配 A 股披露截止)')
    args = ap.parse_args()
    os.makedirs(os.path.join(HERE, args.out), exist_ok=True)

    dd = []
    for ex in args.exchanges:
        for f in sorted(glob.glob(os.path.join(args.root, ex, 'chunk_*_d.npy'))):
            dd.append(np.load(f))
    anchors = np.unique(np.concatenate(dd))
    print(f'锚点 {len(anchors)} 个', flush=True)

    print('读日线...', flush=True)
    daily = load_daily_close(args.exchanges)

    # ---- 财报面板：毛利 + 股本，构造同比信号 ----
    frames = []
    for ex in args.exchanges:
        p = pd.read_csv(os.path.join(L.ROOT, 'data', ex.upper(), f'income_{ex.lower()}.csv'),
                        usecols=['symbol', 'endDate', GP, 'weightedAverageShsOut'])
        frames.append(p)
    p = pd.concat(frames, ignore_index=True)
    p['symbol'] = p.symbol.astype(str)
    p = p[p.endDate.astype(str).str[4:].isin(['0331', '0630', '0930', '1231'])]
    p = p.drop_duplicates(subset=['symbol', 'endDate'], keep='last')
    p = p.sort_values(['symbol', 'endDate']).reset_index(drop=True)
    g = p.groupby('symbol', sort=False)

    lg = args.lag
    print(f'保守滞后 lag={lg}（{"含未披露当季 -> 有前视" if lg == 0 else "只用已披露季度"}）', flush=True)
    P = pd.DataFrame({'symbol': p.symbol, 'j_endDate': p.endDate,
                      'shares': p['weightedAverageShsOut']})
    # 所有"最新"项都往后推 lg 个季度：锚点日在季度末时，当季财报尚未披露
    P['gp_latest'] = g[GP].shift(lg)
    P['gp_yoy1'] = (g[GP].shift(lg) - g[GP].shift(lg + 4)) / g[GP].shift(lg + 4).abs().clip(lower=1)
    s3 = g[GP].transform(lambda x: x.rolling(3, min_periods=3).sum()).shift(lg)
    s3b = g[GP].transform(lambda x: x.rolling(3, min_periods=3).sum()).shift(lg + 4)
    P['gp3'] = s3
    P['gp_yoy3'] = (s3 - s3b) / s3b.abs().clip(lower=1)
    P['dgp1'] = g[GP].shift(lg) - g[GP].shift(lg + 4)
    P['dgp3'] = s3 - s3b
    for h in HORIZONS:
        P[f'trail{h}'] = g[GP].transform(
            lambda x, h=h: x.rolling(h, min_periods=h).sum()).shift(lg)
    P = P.dropna(subset=['shares']).sort_values(['symbol', 'j_endDate']).reset_index(drop=True)
    print(f'  面板 {len(P):,} 行', flush=True)

    # ---- 前瞻收益 ----
    print('算收益...', flush=True)
    rows = []
    for sym, (dt, cl) in daily.items():
        pos = np.searchsorted(dt, anchors, side='right') - 1
        v = pd.DataFrame({'symbol': sym, 'anchor': anchors})
        v['entry'] = np.where(pos >= 0, cl[np.clip(pos, 0, None)], np.nan)
        for h in HORIZONS:
            ep = pos + DAYS_PER_Q * h
            good = (pos >= 0) & (ep < len(cl))
            r = np.full(len(anchors), np.nan)
            r[good] = cl[ep[good]] / cl[pos[good]] - 1.0
            v[f'ret{h}'] = r
        rows.append(v)
    R = pd.concat(rows, ignore_index=True)
    del rows

    # ---- asof 合并 ----
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
    D['mcap'] = D['shares'] * D['entry']
    D = D[D.mcap > 0]
    D['gp_yield'] = D['dgp1'] / D['mcap']
    D['gp3_yield'] = D['dgp3'] / D['mcap']
    D['mom889'] = np.nan
    for s, idx in D.groupby('symbol', sort=False).indices.items():
        if s not in daily:
            continue
        dt, cl = daily[s]
        pos = np.searchsorted(dt, D['anchor'].values[idx], side='right') - 1
        good = pos >= 889
        m = np.full(len(idx), np.nan)
        m[good] = cl[pos[good]] / cl[pos[good] - 889] - 1.0
        D.iloc[idx, D.columns.get_loc('mom889')] = m
    print(f'  合并后 {len(D):,} 行 / {D.qkey.nunique()} 个季度截面，'
          f'平均每季度 {len(D)/D.qkey.nunique():.0f} 只', flush=True)

    SIGNALS = ['gp_yoy1', 'gp_yoy3', 'gp_yield', 'gp3_yield']

    def quintile(sig, ret):
        per = []
        for _, gg in D[['qkey', sig, ret]].dropna().groupby('qkey'):
            if len(gg) < 50:
                continue
            r = gg[sig].rank(ascending=False, method='first')
            q = np.ceil(r / len(gg) * 5).astype(int).clip(1, 5)
            rec = {'all': gg[ret].median()}
            for k in range(1, 6):
                rec[f'Q{k}'] = gg.loc[q == k, ret].median()
            hs = gg.loc[r <= len(gg) / 2.0, ret]
            rec['top50'] = hs.median()
            rec['negfrac'] = (hs < 0).mean()
            per.append(rec)
        return pd.DataFrame(per)

    print()
    for h in HORIZONS:
        ret = f'ret{h}'
        D['_value'] = D[f'trail{h}'] / D['mcap']     # 正确的估值分数 = 毛利和/市值
        print(f'{"="*104}\n  持有 {h} 季（中位数收益）\n{"="*104}')
        print(f'  {"信号":12s}{"Q1":>9s}{"Q2":>9s}{"Q3":>9s}{"Q4":>9s}{"Q5":>9s}'
              f'{"Q1−Q5":>10s}{"t":>7s}{"前50%":>9s}{"负收益":>8s}{"回撤":>9s}')
        for sig in SIGNALS + ['_value']:
            a = quintile(sig, ret)
            if a.empty:
                continue
            sp = a['Q1'] - a['Q5']
            cum = a['top50'].values
            lab = 'value(估值)' if sig == '_value' else sig
            print(f'  {lab:12s}' + ''.join(f'{a[f"Q{k}"].mean()*100:>+8.2f}%' for k in range(1, 6))
                  + f'{sp.mean()*100:>+9.2f}%'
                  + f'{sp.mean()/(sp.std()/np.sqrt(len(sp))):>+7.2f}'
                  + f'{a["top50"].mean()*100:>+8.2f}%'
                  + f'{a["negfrac"].mean()*100:>7.0f}%'
                  + f'{max_dd(cum)*100:>+8.1f}%')
        # 便宜 ∧ 毛利改善
        sub = D[['qkey', '_value', 'gp_yoy1', ret]].dropna()
        keep = []
        for _, gg in sub.groupby('qkey'):
            if len(gg) < 50:
                continue
            cheap = gg['_value'] >= gg['_value'].quantile(0.5)
            grow = gg['gp_yoy1'] >= gg['gp_yoy1'].quantile(0.5)
            v = gg.loc[cheap & grow, ret]
            if len(v) >= 20:
                keep.append(v.median())
        if keep:
            k2 = np.array(keep)
            print(f'  {"便宜∧改善":12s}' + ' ' * 44 +
                  f'{"":>10s}{"":>7s}{k2.mean()*100:>+8.2f}%'
                  f'{(k2<0).mean()*100:>7.0f}%{max_dd(k2)*100:>+8.1f}%')
        print()
    # 与动量的相关性
    print('  各信号与长期动量 mom889 的截面相关：')
    for sig in SIGNALS + ['_value']:
        cs = []
        for _, gg in D[['qkey', sig, 'mom889']].dropna().groupby('qkey'):
            if len(gg) < 50:
                continue
            ra = np.argsort(np.argsort(gg[sig].values))
            rb = np.argsort(np.argsort(gg['mom889'].values))
            if ra.std() and rb.std():
                cs.append(np.corrcoef(ra, rb)[0, 1])
        print(f'    {sig:12s} {np.mean(cs):+.3f}')
    D.to_csv(os.path.join(HERE, args.out, 'fund_mom_detail.csv'), index=False)


if __name__ == '__main__':
    main()
