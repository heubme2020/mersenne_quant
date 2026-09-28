"""生产估值公式：schloss 归因 + dcf 补进回测。

## 任务 1：schloss 该不该留？
`buffett = dcf × B/P + schloss`。单独看 schloss 在短中期不显著（t=1.7/1.8）
且五分组非单调，但和 B/P 相加后**组合的 t 反而升高** —— 像是"降波动的分散项"。
这里扫权重：score = z(B/P) + λ·z(schloss)，看**夏普**（不是幅度）怎么随 λ 变。
  λ*=0 -> 去掉它；λ*>0 且夏普改善 -> 留（但要按最优权重重加权）

## 任务 2：dcf 补进回测
七个模型的历史预测只有 **889 只股票、≤2018Q3、且是 after_train 回填（有事后拟合）**，
所以用两个版本交叉验证：
  dcf_seven : seven/eval_results/after_train_predictions.csv 的 pred_fcf 三月期和
  dcf_proxy : lag 安全的**实际**现金流代理 (近4季 FCF + 近4季分红) / 净资产
两者作为"质量乘数"分别测，结论一致才可信。

## 夏普口径
h=1 时锚点间隔 63 交易日 = 持有期 -> **非重叠**，夏普干净（年化 ×√4）。
h=3/7 重叠，夏普偏乐观，只作参考。
分位收益用中位数（抗极值）；夏普用 **winsor 均值**（抗极值但要均值）。

用法：python two/nowcast/valuation_attrib.py --exchanges SHZ SHH --lag 1
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
SEVEN_PRED = os.path.join(L.ROOT, 'seven', 'eval_results', 'after_train_predictions.csv')


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


def wmean(x, lo=0.01, hi=0.99):
    if len(x) < 20:
        return np.nan
    a, b = np.quantile(x, [lo, hi])
    return float(np.clip(x, a, b).mean())


def sharpe(series, per_year=4):
    s = np.asarray(series, dtype=float)
    s = s[np.isfinite(s)]
    return float(s.mean() / s.std() * np.sqrt(per_year)) if len(s) > 5 and s.std() > 0 else np.nan


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
    daily = load_daily_close(args.exchanges)

    # ---- 面板 ----
    frames = []
    for ex in args.exchanges:
        inc = pd.read_csv(os.path.join(L.ROOT, 'data', ex.upper(), f'income_{ex.lower()}.csv'),
                          usecols=['symbol', 'endDate', GP, 'weightedAverageShsOut'])
        cf = pd.read_csv(os.path.join(L.ROOT, 'data', ex.upper(), f'cashflow_{ex.lower()}.csv'),
                         usecols=['symbol', 'endDate', 'freeCashFlow', 'commonDividendsPaid'])
        bal = pd.read_csv(os.path.join(L.ROOT, 'data', ex.upper(), f'balance_{ex.lower()}.csv'),
                          usecols=['symbol', 'endDate', 'totalStockholdersEquity'])
        ind = pd.read_csv(os.path.join(L.ROOT, 'data', ex.upper(), f'indicator_{ex.lower()}.csv'),
                          usecols=['symbol', 'endDate', 'operatingCapitalPerShare',
                                   'netAssetValuePerShare'])
        m = inc.merge(cf, on=['symbol', 'endDate'], how='inner') \
               .merge(bal, on=['symbol', 'endDate'], how='inner') \
               .merge(ind, on=['symbol', 'endDate'], how='inner')
        frames.append(m)
    p = pd.concat(frames, ignore_index=True)
    p['symbol'] = p.symbol.astype(str)
    p = p[p.endDate.astype(str).str[4:].isin(['0331', '0630', '0930', '1231'])]
    p = p.drop_duplicates(subset=['symbol', 'endDate'], keep='last')
    p = p.sort_values(['symbol', 'endDate']).reset_index(drop=True)
    g = p.groupby('symbol', sort=False)

    lg = args.lag
    P = pd.DataFrame({'symbol': p.symbol, 'j_endDate': p.endDate, 'shares': p['weightedAverageShsOut']})
    P['opcap_ps'] = g['operatingCapitalPerShare'].shift(lg)
    P['nav_ps'] = g['netAssetValuePerShare'].shift(lg)
    P['gp_latest'] = g[GP].shift(lg)
    P['dgp'] = g[GP].shift(lg) - g[GP].shift(lg + 4)
    P['eq'] = g['totalStockholdersEquity'].shift(lg)
    # lag 安全的现金流质量代理（近 4 季）
    fcf4 = g['freeCashFlow'].transform(lambda x: x.rolling(4, min_periods=4).sum()).shift(lg)
    div4 = g['commonDividendsPaid'].transform(lambda x: x.rolling(4, min_periods=4).sum()).shift(lg)
    P['dcf_proxy'] = (fcf4 - div4) / P['eq'].abs().clip(lower=1)   # 分红字段是负数（流出）
    for h in HORIZONS:
        P[f'trail{h}'] = g[GP].transform(
            lambda x, h=h: x.rolling(h, min_periods=h).sum()).shift(lg)
    P = P.dropna(subset=['shares']).sort_values(['symbol', 'j_endDate']).reset_index(drop=True)

    # seven 的历史预测（只有 889 只、≤2018Q3、after_train 回填）
    sp = pd.read_csv(SEVEN_PRED, usecols=['symbol', 'endDate',
                                          'pred_fcf_three', 'pred_fcf_seven', 'pred_fcf_thirty_one'])
    sp['symbol'] = sp.symbol.astype(str)
    sp['dcf_seven'] = (sp['pred_fcf_three'] + sp['pred_fcf_seven'] + sp['pred_fcf_thirty_one'])
    P = P.merge(sp[['symbol', 'endDate', 'dcf_seven']].rename(columns={'endDate': 'j_endDate'}),
                on=['symbol', 'j_endDate'], how='left')
    print(f'  面板 {len(P):,} 行；dcf_seven 可得 {P.dcf_seven.notna().sum():,} 行，'
          f'dcf_proxy 可得 {P.dcf_proxy.notna().sum():,} 行', flush=True)

    # ---- 前瞻收益 ----
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
    D['schloss'] = D['opcap_ps'] / D['close']
    D['bp'] = D['nav_ps'] / D['close']
    D['gp_yield'] = D['trail1'] / D['mcap']
    D['gp_chg'] = D['dgp'] / D['mcap']
    print(f'  合并后 {len(D):,} 行 / {D.qkey.nunique()} 季度截面\n', flush=True)

    def z(col):
        return D.groupby('qkey')[col].rank(pct=True)

    def ev(col):
        """-> (Q1..Q5 中位数, Q1−Q5 均值/t/夏普, top50 中位数/winsor/夏普/负收益率/回撤)"""
        per = []
        for _, gg in D[['qkey', col, 'ret1', 'ret3', 'ret7']].dropna(subset=[col]).groupby('qkey'):
            if len(gg) < 50:
                continue
            r = gg[col].rank(ascending=False, method='first')
            q = np.ceil(r / len(gg) * 5).astype(int).clip(1, 5)
            hs = gg.loc[r <= len(gg) / 2.0]
            rec = {}
            for h in HORIZONS:
                rec[f'sp{h}'] = gg.loc[q == 1, f'ret{h}'].median() - gg.loc[q == 5, f'ret{h}'].median()
                rec[f'top{h}'] = hs[f'ret{h}'].median()
                rec[f'topw{h}'] = wmean(hs[f'ret{h}'].values)
                rec[f'neg{h}'] = (hs[f'ret{h}'] < 0).mean()
            rec['Q'] = [gg.loc[q == k, 'ret1'].median() for k in range(1, 6)]
            per.append(rec)
        return pd.DataFrame(per)

    # ================= 任务 1：schloss 归因（λ 扫描）=================
    print('=' * 104)
    print('  任务 1：schloss 归因   score = z(B/P) + λ·z(schloss)   [h=1，非重叠，夏普干净]')
    print('=' * 104)
    cs = []
    for _, gg in D[['qkey', 'bp', 'schloss']].dropna().groupby('qkey'):
        if len(gg) < 50:
            continue
        ra = np.argsort(np.argsort(gg['bp'].values))
        rb = np.argsort(np.argsort(gg['schloss'].values))
        if ra.std() and rb.std():
            cs.append(np.corrcoef(ra, rb)[0, 1])
    print(f'  B/P 与 schloss 的截面相关（均值）：{np.mean(cs):+.3f}'
          f'   （越低越分散）\n')
    print(f'  {"λ":>6s}{"Q1−Q5":>10s}{"多空t":>8s}{"多空夏普":>10s}'
          f'{"前50%":>9s}{"前50%夏普":>11s}{"负收益":>8s}{"回撤":>9s}')
    zb, zs = z('bp'), z('schloss')
    for lam in [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]:
        D['_s'] = zb + lam * zs
        a = ev('_s')
        sp = a['sp1']
        tw = a['topw1']
        print(f'  {lam:>6.2f}{sp.mean()*100:>+9.2f}%'
              f'{sp.mean()/(sp.std()/np.sqrt(len(sp))):>+8.2f}'
              f'{sharpe(sp):>+10.2f}{a["top1"].mean()*100:>+8.2f}%'
              f'{sharpe(tw):>+11.2f}{a["neg1"].mean()*100:>7.0f}%'
              f'{max_dd(tw):>+8.1f}%')

    # ================= 任务 2：dcf 补进回测 =================
    for dname in ['dcf_seven', 'dcf_proxy']:
        print()
        print('=' * 104)
        print(f'  任务 2：dcf 补进回测  （dcf 版本 = {dname}）  [h=1]')
        print('=' * 104)
        sub = D[D[dname].notna()].copy()
        if len(sub) < 10000:
            print('  样本不足，跳过')
            continue
        print(f'  样本 {len(sub):,} 行 / {sub.qkey.nunique()} 截面')
        zb2 = sub.groupby('qkey')['bp'].rank(pct=True)
        zs2 = sub.groupby('qkey')['schloss'].rank(pct=True)
        zg2 = sub.groupby('qkey')['gp_yield'].rank(pct=True)
        zc2 = sub.groupby('qkey')['gp_chg'].rank(pct=True)
        zd2 = sub.groupby('qkey')[dname].rank(pct=True)
        plans = {
            'F1 现状: dcf×B/P + schloss': zd2 * zb2 + zs2,
            'F2 现状 + 毛利 + 提升': zd2 * zb2 + zs2 + zg2 + zc2,
            'F3 全标准化（含 dcf）': zd2 + zb2 + zs2 + zg2 + zc2,
            'F4 去掉 dcf': zb2 + zs2 + zg2 + zc2,
        }
        print(f'  {"方案":26s}{"Q1":>9s}{"Q5":>9s}{"Q1−Q5":>10s}{"多空t":>8s}'
              f'{"多空夏普":>10s}{"前50%":>9s}{"前50%夏普":>11s}')
        for nm, sc in plans.items():
            old = D
            D = sub
            D['_s'] = np.asarray(sc)
            a = ev('_s')
            sp, tw = a['sp1'], a['topw1']
            print(f'  {nm:26s}{a["Q"].apply(lambda x: x[0]).mean()*100:>+8.2f}%'
                  f'{a["Q"].apply(lambda x: x[4]).mean()*100:>+8.2f}%'
                  f'{sp.mean()*100:>+9.2f}%'
                  f'{sp.mean()/(sp.std()/np.sqrt(len(sp))):>+8.2f}'
                  f'{sharpe(sp):>+10.2f}{a["top1"].mean()*100:>+8.2f}%'
                  f'{sharpe(tw):>+11.2f}')
            D = old
    D.to_csv(os.path.join(HERE, args.out, 'valuation_attrib.csv'), index=False)


if __name__ == '__main__':
    main()
