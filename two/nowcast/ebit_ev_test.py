"""LFM 的真正构造：**预测 EBIT / 当前 EV** 的收益检验。

LFM（Alberg & Lipton 2017）验证过的是这个：
    预测未来 EBIT  ÷  当前企业价值(EV)   -> 买最便宜的
不是 FCF/市值（我们刚测出 IC 只有 0.022）。

EBIT 预测来自 **v4 模型的头 3 = `ebit_TA`**（标签 = ΣEBIT[j+1..j+h]/总资产[j]），
所以 预测ΣEBIT_fwd = 预测值 × 总资产[j]（一步乘法，精确反算）。

对照：
  基线 = 上一期 h 季 EBIT 和 ÷ EV        <- 免费，等价于"EBIT/EV 价值因子"
  全市场等权
  （另外报 剔除现金/有息负债的纯市值口径 预测EBIT/市值，看 EV 有没有加分）

前视处理：锚点在季度末时当季财报未披露 -> 基线用已披露季度（lag）；EV 用锚点日价格（已知）。

用法：python two/nowcast/ebit_ev_test.py --exchanges SHZ SHH --lag 1 --tag _v4
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'one'))
sys.path.insert(0, HERE)

from data import Store, ASHARE_EX            # noqa: E402
from train import load_split                 # noqa: E402
import labels as L                           # noqa: E402
import variants as V                         # noqa: E402

HORIZONS = [1, 3, 7]
DAYS_PER_Q = 63
TA = 'totalAssets'


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exchanges', nargs='+', default=['SHZ', 'SHH'])
    ap.add_argument('--root', default='Z:/quant_data/nowcast')
    ap.add_argument('--tag', default='_v4')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--lag', type=int, default=1)
    ap.add_argument('--out', default='returns_eval')
    args = ap.parse_args()
    os.makedirs(os.path.join(HERE, args.out), exist_ok=True)

    # ---- 模型预测（v4 头 3 = ebit_TA）----
    store = Store(args.root, sorted(set(ASHARE_EX)), min_per_date=64,
                  label='level', suffix='v4')
    tr, va, te = load_split(store, 'ashare')
    pred = np.load(os.path.join(HERE, 'results',
                                f'pred_ashare{args.tag}_s{args.seed}.npz'))['pred']
    assert len(pred) == len(te), f'{len(pred)} vs {len(te)}'
    rng = np.random.RandomState(args.seed + 1)
    samp = np.sort(tr[rng.choice(len(tr), min(20000, len(tr)), replace=False)])
    Ys = store.take_y(samp)
    y_mean, y_std = Ys.mean(0), Ys.std(0) + 1e-6
    syms = store.symbol_of(te)
    T = pd.DataFrame({'symbol': [s.split(':')[-1] for s in syms],
                      'anchor': store.dates[te]})
    for i, h in enumerate(HORIZONS):
        j = 6 + i                      # 头 3 的三个期限
        T[f'pebit{h}'] = pred[:, j] * y_std[j] + y_mean[j]
    print(f'模型预测 {len(T):,} 行（v4 的 ebit_TA 头）', flush=True)

    # ---- 面板：EBIT、总资产、股本、有息负债、现金 ----
    frames = []
    for ex in args.exchanges:
        inc = pd.read_csv(os.path.join(L.ROOT, 'data', ex.upper(), f'income_{ex.lower()}.csv'),
                          usecols=['symbol', 'endDate', 'ebit', 'weightedAverageShsOut'])
        bal = pd.read_csv(os.path.join(L.ROOT, 'data', ex.upper(), f'balance_{ex.lower()}.csv'),
                          usecols=['symbol', 'endDate', TA, 'totalDebt', 'cashAndCashEquivalents'])
        frames.append(inc.merge(bal, on=['symbol', 'endDate'], how='inner'))
    p = pd.concat(frames, ignore_index=True)
    p['symbol'] = p.symbol.astype(str)
    p = p[p.endDate.astype(str).str[4:].isin(['0331', '0630', '0930', '1231'])]
    p = p.drop_duplicates(subset=['symbol', 'endDate'], keep='last')
    p = p.sort_values(['symbol', 'endDate']).reset_index(drop=True)
    g = p.groupby('symbol', sort=False)
    lg = args.lag

    P = pd.DataFrame({'symbol': p.symbol, 'j_endDate': p.endDate,
                      'shares': p['weightedAverageShsOut'],
                      'ta_j': p[TA],                       # 总资产[j]（反算用；存量，变动小）
                      'debt': g['totalDebt'].shift(lg),
                      'cash': g['cashAndCashEquivalents'].shift(lg),
                      'ebit_latest': g['ebit'].shift(lg)})
    for h in HORIZONS:
        P[f'ebit_trail{h}'] = g['ebit'].transform(
            lambda x, h=h: x.rolling(h, min_periods=h).sum()).shift(lg)
    P = P.dropna(subset=['shares']).sort_values(['symbol', 'j_endDate']).reset_index(drop=True)

    # ---- 锚点日收盘价 ----
    anchors = np.unique(T['anchor'].values)
    dd = []
    for ex in args.exchanges:
        for f in sorted(glob.glob(os.path.join(args.root, ex, 'chunk_*_d.npy'))):
            dd.append(np.load(f))
    anchors = np.unique(np.concatenate(dd))
    daily = load_daily_close(args.exchanges)
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
    mcap = D['shares'] * D['close']
    D['mcap'] = mcap
    D['ev'] = mcap + D['debt'].fillna(0) - D['cash'].fillna(0)
    D = D[(D.mcap > 0)]
    print(f'  面板合并后 {len(D):,} 行 / {D.qkey.nunique()} 季度截面', flush=True)
    print(f'  EV<=0（净现金过多）占比 {(D.ev <= 0).mean()*100:.2f}%', flush=True)

    # ---- 模型预测合并 ----
    D = D.merge(T, on=['symbol', 'anchor'], how='left')
    D['pebit_sum1'] = D['pebit1'] * D['ta_j']
    D['pebit_sum3'] = D['pebit3'] * D['ta_j']
    D['pebit_sum7'] = D['pebit7'] * D['ta_j']
    n_m = D['pebit1'].notna().sum()
    print(f'  匹配到模型预测 {n_m:,} 行\n', flush=True)

    def ev_score(df, sig, ret):
        per = []
        for _, gg in df[['qkey', sig, ret]].dropna().groupby('qkey'):
            if len(gg) < 50:
                continue
            r = gg[sig].rank(ascending=False, method='first')
            q = np.ceil(r / len(gg) * 5).astype(int).clip(1, 5)
            hs = gg.loc[r <= len(gg) / 2.0, ret]
            per.append({'Q1': gg.loc[q == 1, ret].median(),
                        'Q2': gg.loc[q == 2, ret].median(),
                        'Q3': gg.loc[q == 3, ret].median(),
                        'Q4': gg.loc[q == 4, ret].median(),
                        'Q5': gg.loc[q == 5, ret].median(),
                        'top': hs.median(), 'all': gg[ret].median(),
                        'neg': (hs < 0).mean()})
        return pd.DataFrame(per)

    sub = D[D['pebit1'].notna()].copy()
    print('=' * 104)
    print('  LFM 构造检验：预测 EBIT / 当前 EV          （同一批测试股，78 个季度截面）')
    print('=' * 104)
    for h in HORIZONS:
        ret = f'ret{h}'
        print(f'\n  持有 {h} 季（中位数收益）')
        print(f'  {"方案":26s}{"Q1":>9s}{"Q2":>9s}{"Q3":>9s}{"Q4":>9s}{"Q5":>9s}'
              f'{"Q1−Q5":>10s}{"t":>7s}{"前50%":>9s}{"全市场":>9s}{"负收益":>8s}')
        plans = [
            (f'预测EBIT/EV（模型）', f'pebit_sum{h}', 'ev'),
            (f'上一期EBIT/EV（免费）', f'ebit_trail{h}', 'ev'),
            (f'预测EBIT/市值（模型）', f'pebit_sum{h}', 'mcap'),
            (f'上一期EBIT/市值（免费）', f'ebit_trail{h}', 'mcap'),
        ]
        for nm, num, den in plans:
            D['_s'] = D[num] / D[den]
            a = ev_score(D, '_s', ret)
            if a.empty:
                continue
            sp = a.Q1 - a.Q5
            print(f'  {nm:26s}' + ''.join(f'{a[k].mean()*100:>+8.2f}%'
                                          for k in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
                  + f'{sp.mean()*100:>+9.2f}%'
                  + f'{sp.mean()/(sp.std()/np.sqrt(len(sp))):>+7.2f}'
                  + f'{a.top.mean()*100:>+8.2f}%{a["all"].mean()*100:>+8.2f}%'
                  f'{a.neg.mean()*100:>7.0f}%')
    D.to_csv(os.path.join(HERE, args.out, 'ebit_ev_detail.csv'), index=False)


if __name__ == '__main__':
    main()
