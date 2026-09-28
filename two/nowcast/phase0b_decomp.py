"""Phase 0b 稳健性：把候选标签拆成「分子 / 分母」分别打 IC。

Phase 0 用这套办法抓出过 revenue 的问题：A 组 revenue 的比值 IC 主要来自分母
（净资产），分子单独的 IC 符号相反。本脚本对 Phase 0b 里排名靠前的标签做同样的事，
回答「这些高 IC 的『变化』标签，是真在预测财务变化，还是在预测分母（风格代理）」。

做法（每个标签 y = num / den）：
  ic_y    = IC(因子, num/den)      比值本身
  ic_num  = IC(因子, num)          分子单独
  ic_den  = IC(因子, den)          分母单独
  判定：|ic_num| 与 |ic_y| 同号同量级 => 真信号；
        分子 IC 远小于比值 IC、或分母 IC 量级相当 => 分母污染。
另给 IC(因子, y | den)：先对 den 的截面 rank 做线性回归取残差，再算 IC（控制分母）。

用法：python phase0b_decomp.py --exchanges SHZ SHH --step 63 --h 3
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from phase0b_fields import (build_features, load_raw, ic_stats,  # noqa: E402
                            EQ, TA, BACK, FACTORS)

# (标签名, 分子（expr 字符串）, 分母)  —— 全部 h=3
SPECS = [
    # 存量变化（Phase 0b 排名最高的一批）
    ('equity_d',     ('stock_d', 'totalStockholdersEquity'), 'totalStockholdersEquity'),
    ('assets_d',     ('stock_d', 'totalAssets'),             'totalAssets'),
    ('retained_d',   ('stock_d', 'retainedEarnings'),        'totalAssets'),
    ('inventory_d',  ('stock_d', 'inventory'),               'totalAssets'),
    ('receivables_d', ('stock_d', 'netReceivables'),         'totalAssets'),
    # 流量（当前三个头 + 对照）
    ('ebit_A',       ('flow_sum', 'ebit'),                   'totalStockholdersEquity'),
    ('ebit_a',       ('flow_sum', 'ebit'),                   'totalAssets'),
    ('ebit_d',       ('flow_diff', 'ebit'),                  'totalStockholdersEquity'),
    ('netIncome_a',  ('flow_sum', 'netIncome'),              'totalAssets'),
    ('netIncome_d',  ('flow_diff', 'netIncome'),             'totalAssets'),
    ('grossProfit_a', ('flow_sum', 'grossProfit'),           'totalAssets'),
    ('grossProfit_d', ('flow_diff', 'grossProfit'),          'totalAssets'),
    ('revenue_g',    ('flow_sum', 'revenue'),                'BACKSUM:revenue'),
    ('revenue_a',    ('flow_sum', 'revenue'),                'totalAssets'),
    ('ocf_A',        ('flow_sum', 'operatingCashFlow'),      'totalStockholdersEquity'),
    ('ocf_a',        ('flow_sum', 'operatingCashFlow'),      'totalAssets'),
    ('ocf_d',        ('flow_diff', 'operatingCashFlow'),     'totalAssets'),
    ('capex_d',      ('flow_diff', 'capitalExpenditure'),    'totalAssets'),
    ('accrual_a',    ('flow_sum', 'accrual'),                'totalAssets'),
]
COLS = set()


def build_num_den(panel, h):
    g = panel.groupby('symbol', sort=False)

    def flow_sum(col):
        return g[col].transform(lambda x: x.shift(-1).rolling(h, min_periods=h)
                                .sum().shift(-(h - 1)))

    def flow_back(col):
        return g[col].transform(lambda x: x.rolling(h, min_periods=h).sum().shift(BACK))

    out = panel[['symbol', 'endDate']].copy()
    for name, num_spec, den in SPECS:
        kind, col = num_spec
        if kind == 'flow_sum':
            num = flow_sum(col)
            # 朴素基准：上一轮同季 h 季之和（「去年同期重演」）—— 锚点日已知
            bench = flow_back(col)
        elif kind == 'flow_diff':
            num = flow_sum(col) - flow_back(col)
            bench = pd.Series(np.nan, index=panel.index)
        elif kind == 'stock_d':
            num = g[col].shift(-h) - g[col].shift(BACK)
            # 朴素基准：把「最近 h 季的存量变化」外推（锚点日已知）
            bench = panel[col] - g[col].shift(BACK)
        else:
            raise ValueError(kind)
        if den.startswith('BACKSUM:'):
            d = flow_back(den.split(':')[1])
        else:
            d = g[den].shift(BACK)
        out[f'{name}__num'] = num.replace([np.inf, -np.inf], np.nan)
        out[f'{name}__den'] = d.replace([np.inf, -np.inf], np.nan)
        out[f'{name}__y'] = (num / d).where(d > 0).replace([np.inf, -np.inf], np.nan)
        out[f'{name}__bench'] = (bench / d).where(d > 0).replace([np.inf, -np.inf], np.nan)
    return out


def partial_ic(df, factor, y, den, min_stocks=20):
    """控制分母后的 IC：逐日对 den 的 rank 回归 y 的 rank，取残差再与因子 rank 相关。"""
    ics = []
    for _, x in df[['date', factor, y, den]].dropna().groupby('date'):
        if len(x) < min_stocks:
            continue
        ry = x[y].rank()
        rd = x[den].rank()
        rf = x[factor].rank()
        # y 对 den 的残差（rank 空间）
        beta = np.polyfit(rd, ry, 1)
        resid = ry - (beta[0] * rd + beta[1])
        if resid.std() == 0:
            continue
        ics.append(rf.corr(resid))
    s = pd.Series(ics).dropna()
    if len(s) < 2:
        return np.nan, np.nan
    return s.mean(), s.mean() / s.std() if s.std() > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exchanges', nargs='+', default=['SHZ', 'SHH'])
    ap.add_argument('--step', type=int, default=63)
    ap.add_argument('--h', type=int, default=3)
    ap.add_argument('--out', default='phase0b_results')
    args = ap.parse_args()

    fac = build_features(args.exchanges, args.step,
                         os.path.join(HERE, f'phase0b_factors_{args.step}.pkl'))
    panel = load_raw(args.exchanges)
    nd = build_num_den(panel, args.h)
    del panel

    fac = fac.sort_values(['symbol', 'date']).reset_index(drop=True)
    nd = nd.sort_values(['symbol', 'endDate']).reset_index(drop=True)
    cols = [c for c in nd.columns if c not in ('symbol', 'endDate')]
    by = {s: (g['endDate'].values, g.index.values) for s, g in nd.groupby('symbol')}
    take = np.full(len(fac), -1, dtype=np.int64)
    fac_date = fac['date'].values
    for s, idx in fac.groupby('symbol', sort=False).indices.items():
        packed = by.get(s)
        if packed is None:
            continue
        dates, rows = packed
        pos = np.searchsorted(dates, fac_date[idx], side='right') - 1
        ok = pos >= 0
        take[idx[ok]] = rows[pos[ok]]
    ok = take >= 0
    df = fac[ok].reset_index(drop=True).copy()
    for c in cols:
        df[c] = nd[c].values[take[ok]]

    recs = []
    for f in FACTORS:
        for name, _, _ in SPECS:
            y, num, den = f'{name}__y', f'{name}__num', f'{name}__den'
            ic_y, icir_y, _ = ic_stats(df, f, y)
            ic_num, _, _ = ic_stats(df, f, num)
            ic_den, _, _ = ic_stats(df, f, den)
            pic, picir = partial_ic(df, f, y, den)
            icb, icirb, _ = ic_stats(df, f'{name}__bench', y)
            # 增量：标签减去朴素基准（「去年同期重演」）后的残差
            df['_resid'] = df[y] - df[f'{name}__bench']
            icr, icirr, _ = ic_stats(df, f, '_resid')
            recs.append(dict(factor=f, label=name, ic_y=ic_y, icir_y=icir_y,
                             ic_num=ic_num, ic_den=ic_den, ic_partial=pic, icir_partial=picir,
                             ic_resid=icr, icir_resid=icirr,
                             coverage=df[y].notna().mean()))
    # 朴素基准单独算一次（与因子无关，只跟标签有关）
    bench = []
    for name, _, _ in SPECS:
        icb, icirb, nb = ic_stats(df.assign(_bench=df[f'{name}__bench']), '_bench', f'{name}__y')
        bench.append(dict(label=name, ic_bench=icb, icir_bench=icirb))
    res = pd.DataFrame(recs)
    outdir = os.path.join(HERE, args.out)
    os.makedirs(outdir, exist_ok=True)
    res.to_csv(os.path.join(outdir, f'decomp_h{args.h}.csv'), index=False)

    # 每个标签取 |ic_y| 最大的因子那一行，并接上朴素基准
    best = (res.assign(a=res.ic_y.abs()).sort_values('a', ascending=False)
               .groupby('label').head(1).drop(columns='a')
               .sort_values('ic_y', key=lambda s: s.abs(), ascending=False))
    bdf = pd.DataFrame(bench)
    best = best.merge(bdf, on='label', how='left')
    pd.set_option('display.width', 240)
    print(f'=== h={args.h}（每个标签取 |IC| 最大的因子）===')
    print(best[['factor', 'label', 'ic_y', 'ic_bench', 'ic_resid', 'icir_resid',
                'ic_num', 'ic_partial', 'coverage']]
          .to_string(index=False, float_format=lambda v: f'{v:+.3f}'))
    print(f'\n写入 {outdir}/decomp_h{args.h}.csv')


if __name__ == '__main__':
    main()
