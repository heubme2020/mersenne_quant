"""Phase 0b：输出端「财务变化特征」的全字段筛选。

Phase 0（phase0_screen.py）只测了 3 个字段（ebit / revenue / ocf），
本脚本把候选字段扩到 income/cashflow/balance 的**全部可用科目**，
用同一套「日线长周期动量」因子（Phase 0 里胜出的 hi889 / mom381 一族）逐个打 IC/ICIR，
回答「输出这 3 个（或几个）财务变化特征，到底该选谁」。

与 phase0_screen.py 的差异（为了能扫全字段而做的简化）：
  * 因子只算 Phase 0 里真正胜出的长周期一族（hi889/lo889/mom127/mom381/mom889/rsi127），
    不再算 24 个短窗口因子 —— 因为已证「提升全来自这 5 个」。
  * 因子在**每只股票的完整日线上滚动一次**，再在锚点日采样，
    而不是每个锚点都对 889 天窗口重算 —— 数值等价（见 --check 对 phase0 的复现），但快几十倍。

标签（field spec: 表 + 列 + 归一化 + 形式）：
  form=sum   流量项：ΣX[j+1 .. j+h] / denom[j-3]      （前瞻水平/收益类）
  form=level 存量项：X[j+h] / denom[j-3]              （前瞻水平）
  form=delta 存量项：X[j+h] − X[j-3]，再 / denom      （存量变化 = 增长类）
  form=growth 流量项：ΣX[j+1..j+h] / ΣX[j-3..j+h-4]   （自相对增长，mod4 同季消季节性）
  分母 denom ∈ {equity, assets, self}；denom ≤ 0 的样本置 NaN（与 Phase 0 一致）。

用法：
    python phase0b_fields.py --exchanges SHZ SHH --step 63 --cache
    python phase0b_fields.py --check          # 只跑 3 个原字段，和 phase0 对数
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DAYS_INPUT = 127 * 7           # 889
EQ = 'totalStockholdersEquity'
TA = 'totalAssets'
HORIZONS = [1, 3, 7]
BACK = 3

# Phase 0 里胜出的长周期因子族（hi889/lo889 是「距 889 日高点/低点」，mom* 是动量）
FACTORS = ['hi889', 'lo889', 'mom127', 'mom381', 'mom889', 'rsi127']

# ---------------------------------------------------------------- 候选字段
# (name, table, column, denom, form)
#   denom: equity | assets | self
#   form : sum | level | delta | growth
FIELDS = [
    # —— 利润表（流量） ——
    ('revenue',        'income',   'revenue',                'equity', 'sum'),
    ('revenue_g',      'income',   'revenue',                'self',   'growth'),
    ('grossProfit',    'income',   'grossProfit',            'equity', 'sum'),
    ('grossProfit_a',  'income',   'grossProfit',            'assets', 'sum'),
    ('ebit',           'income',   'ebit',                   'equity', 'sum'),
    ('ebit_a',         'income',   'ebit',                   'assets', 'sum'),
    ('ebit_g',         'income',   'ebit',                   'self',   'growth'),
    ('ebitda',         'income',   'ebitda',                 'equity', 'sum'),
    ('operatingIncome', 'income',  'operatingIncome',        'equity', 'sum'),
    ('netIncome',      'income',   'netIncome',              'equity', 'sum'),
    ('netIncome_a',    'income',   'netIncome',              'assets', 'sum'),
    ('incomeBeforeTax', 'income',  'incomeBeforeTax',        'equity', 'sum'),
    ('sga',            'income',   'sellingGeneralAndAdministrativeExpenses', 'equity', 'sum'),
    ('rnd',            'income',   'researchAndDevelopmentExpenses', 'equity', 'sum'),
    ('da',             'income',   'depreciationAndAmortization', 'equity', 'sum'),
    # —— 现金流（流量） ——
    ('ocf',            'cashflow', 'operatingCashFlow',      'equity', 'sum'),
    ('ocf_a',          'cashflow', 'operatingCashFlow',      'assets', 'sum'),
    ('ocf_g',          'cashflow', 'operatingCashFlow',      'self',   'growth'),
    ('capex',          'cashflow', 'capitalExpenditure',     'assets', 'sum'),
    ('capex_g',        'cashflow', 'capitalExpenditure',     'self',   'growth'),
    ('fcf',            'cashflow', 'freeCashFlow',           'equity', 'sum'),
    ('cfi',            'cashflow', 'netCashProvidedByInvestingActivities', 'assets', 'sum'),
    ('changeWC',       'cashflow', 'changeInWorkingCapital', 'assets', 'sum'),
    # —— 资产负债表（存量 → level / delta） ——
    ('assets_l',       'balance',  'totalAssets',            'assets', 'level'),
    ('assets_d',       'balance',  'totalAssets',            'assets', 'delta'),
    ('equity_l',       'balance',  'totalStockholdersEquity', 'assets', 'level'),
    ('equity_d',       'balance',  'totalStockholdersEquity', 'equity', 'delta'),
    ('liab_d',         'balance',  'totalLiabilities',       'assets', 'delta'),
    ('inventory_d',    'balance',  'inventory',              'assets', 'delta'),
    ('receivables_d',  'balance',  'netReceivables',         'assets', 'delta'),
    ('ppe_d',          'balance',  'propertyPlantEquipmentNet', 'assets', 'delta'),
    ('cash_d',         'balance',  'cashAndCashEquivalents', 'assets', 'delta'),
    ('debt_d',         'balance',  'totalDebt',              'assets', 'delta'),
    ('goodwill_d',     'balance',  'goodwillAndIntangibleAssets', 'assets', 'delta'),
    ('retained_d',     'balance',  'retainedEarnings',       'assets', 'delta'),
    # —— 组合量 ——
    ('accrual',        'derived',  'accrual',                'assets', 'sum'),    # netIncome - ocf
    ('accrual_g',      'derived',  'accrual',                'self',   'growth'),
]

# —— 「变化」形式：D 组 = 未来 h 季之和 − 同季对比的过去 h 季之和，再除分母 ——
# 这是「财务变化」最直白的读法：同一 mod-4 季节位、水平差分，分母固定。
# 与上面 A 组（水平比 = ΣX[j+1..j+h]/denom）对照，就能拆开
# 「模型在预测水平（持续性 + 风格推断）」和「模型在预测变化（真 nowcast）」。
CHANGE_FIELDS = [
    ('ebit_d',        'income',   'ebit',                   'equity'),
    ('ebit_da',       'income',   'ebit',                   'assets'),
    ('revenue_d',     'income',   'revenue',                'assets'),
    ('grossProfit_d', 'income',   'grossProfit',            'assets'),
    ('netIncome_d',   'income',   'netIncome',              'assets'),
    ('ocf_d',         'cashflow', 'operatingCashFlow',      'assets'),
    ('ocf_de',        'cashflow', 'operatingCashFlow',      'equity'),
    ('capex_d',       'cashflow', 'capitalExpenditure',     'assets'),
    ('fcf_d',         'cashflow', 'freeCashFlow',           'assets'),
    ('accrual_d',     'derived',  'accrual',                'assets'),
]
FIELDS += [(name, tbl, col, den, 'dsum') for name, tbl, col, den in CHANGE_FIELDS]

TABLES = {
    'income':   'income_{ex}.csv',
    'cashflow': 'cashflow_{ex}.csv',
    'balance':  'balance_{ex}.csv',
}


# ---------------------------------------------------------------- 因子
def build_features(exchanges, step, cache):
    """每只股票在全局锚点日上的长周期因子。返回 DataFrame(symbol,date,6 因子)。"""
    if cache and os.path.exists(cache):
        print(f'读取因子缓存 {cache}', flush=True)
        return pd.read_pickle(cache)

    frames = []
    for ex in exchanges:
        d = pd.read_csv(os.path.join(BASE, 'data', ex.upper(), f'daily_{ex.lower()}.csv'),
                        usecols=['symbol', 'date', 'close'], engine='pyarrow')
        d = d[d.date >= 20050101]
        frames.append(d)
        print(f'{ex}: 日线 {len(d):,} 行', flush=True)
    daily = pd.concat(frames, ignore_index=True)
    daily['symbol'] = daily.symbol.astype(str)
    daily = daily.sort_values(['symbol', 'date']).reset_index(drop=True)

    all_dates = np.sort(daily['date'].unique())
    anchors = all_dates[DAYS_INPUT - 1::step]
    print(f'全局锚点日 {len(anchors)} 个：{anchors[0]} → {anchors[-1]}', flush=True)

    g = daily.groupby('symbol', sort=False)['close']
    feat = pd.DataFrame({'symbol': daily['symbol'], 'date': daily['date']})
    feat['hi889'] = g.transform(lambda s: s.rolling(DAYS_INPUT, min_periods=DAYS_INPUT).max()) / daily['close']
    feat['lo889'] = g.transform(lambda s: s.rolling(DAYS_INPUT, min_periods=DAYS_INPUT).min()) / daily['close']
    for w in (127, 381, 889):
        feat[f'mom{w}'] = daily['close'] / g.transform(lambda s, w=w: s.shift(w)) - 1
    delta = g.transform(lambda s: s.diff())
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    up = gain.groupby(daily['symbol']).transform(lambda s: s.rolling(127, min_periods=127).mean())
    dn = loss.groupby(daily['symbol']).transform(lambda s: s.rolling(127, min_periods=127).mean())
    feat['rsi127'] = 100 - 100 / (1 + up / (dn + 1e-6))
    del daily, g, delta, gain, loss, up, dn

    # 锚点采样：每只股票取 ≤ 锚点日的最后一个交易日
    out = []
    for sym, gg in feat.groupby('symbol', sort=False):
        dates = gg['date'].values
        pos = np.searchsorted(dates, anchors, side='right') - 1
        ok = pos >= 0
        if not ok.any():
            continue
        idx = gg.index.values[pos[ok]]
        blk = feat.loc[idx, FACTORS].copy()
        blk['symbol'] = sym
        blk['date'] = anchors[ok]
        out.append(blk)
    fac = pd.concat(out, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    print(f'因子表 {len(fac):,} 行 × {len(FACTORS)} 因子', flush=True)
    if cache:
        fac.to_pickle(cache)
    return fac


# ---------------------------------------------------------------- 财务面板
def load_raw(exchanges):
    """把所有候选字段按 (symbol, endDate) 合成一张宽表。"""
    cols_needed = {}
    for _, tbl, col, _, _ in FIELDS:
        if tbl == 'derived':
            continue          # accrual 等组合量在下文由原始列算出
        cols_needed.setdefault(tbl, set()).add(col)
    frames = []
    for ex in exchanges:
        parts = None
        for tbl, cols in cols_needed.items():
            path = os.path.join(BASE, 'data', ex.upper(), TABLES[tbl].format(ex=ex.lower()))
            avail = pd.read_csv(path, nrows=0).columns
            use = ['symbol', 'endDate'] + [c for c in sorted(cols) if c in avail]
            missing = sorted(set(cols) - set(avail))
            if missing:
                print(f'  [warn] {ex}/{tbl} 缺列 {missing}', flush=True)
            p = pd.read_csv(path, usecols=use)
            parts = p if parts is None else parts.merge(p, on=['symbol', 'endDate'], how='outer')
        parts['symbol'] = parts.symbol.astype(str)
        parts = parts[parts.endDate.astype(str).str[4:].isin(['0331', '0630', '0930', '1231'])]
        parts = parts.drop_duplicates(subset=['symbol', 'endDate'], keep='last')
        frames.append(parts)
    p = pd.concat(frames, ignore_index=True)
    # 组合量
    p['accrual'] = p['netIncome'] - p['operatingCashFlow']
    return p.sort_values(['symbol', 'endDate']).reset_index(drop=True)


def build_labels(panel):
    """对每个 (symbol, j) 生成所有候选字段 × horizon 的标签。"""
    g = panel.groupby('symbol', sort=False)
    out = panel[['symbol', 'endDate']].copy()

    def fwd_sum(col, h):
        # Σ X[j+1 .. j+h]
        return g[col].transform(
            lambda x: x.shift(-1).rolling(h, min_periods=h).sum().shift(-(h - 1)))

    for name, tbl, col, denom, form in FIELDS:
        if col not in panel.columns:
            continue
        if form == 'growth':
            past = g[col].transform(lambda x: x.shift(BACK).rolling(3, min_periods=3).sum())
        for h in HORIZONS:
            if form == 'sum':
                num = fwd_sum(col, h)
                den = (g[EQ].shift(BACK) if denom == 'equity'
                       else g[TA].shift(BACK) if denom == 'assets' else g[col].shift(BACK))
            elif form == 'growth':
                num = fwd_sum(col, h)
                den = past
            elif form == 'dsum':
                # 未来 h 季之和 − 上一轮同季 h 季之和（mod 4 对齐，消季节性）
                back_sum = g[col].transform(
                    lambda x: x.rolling(h, min_periods=h).sum().shift(BACK))
                num = fwd_sum(col, h) - back_sum
                den = (g[EQ].shift(BACK) if denom == 'equity' else g[TA].shift(BACK))
            elif form == 'level':
                num = g[col].shift(-h)
                den = g[EQ].shift(BACK) if denom == 'equity' else g[TA].shift(BACK)
            elif form == 'delta':
                num = g[col].shift(-h) - g[col].shift(BACK)
                den = g[EQ].shift(BACK) if denom == 'equity' else g[TA].shift(BACK)
            else:
                raise ValueError(form)
            y = (num / den).where(den > 0)
            out[f'{name}_{h}'] = y.replace([np.inf, -np.inf], np.nan).astype('float32')
    return out


# ---------------------------------------------------------------- IC
def ic_stats(df, factor, label, min_stocks=20):
    xs = df[['date', factor, label]].dropna()
    ics = []
    for _, x in xs.groupby('date'):
        if len(x) < min_stocks:
            continue
        ics.append(x[factor].rank().corr(x[label].rank()))
    s = pd.Series(ics).dropna()
    if len(s) < 2:
        return np.nan, np.nan, 0
    m, sd = s.mean(), s.std()
    return m, (m / sd if sd > 0 else np.nan), len(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exchanges', nargs='+', default=['SHZ', 'SHH'])
    ap.add_argument('--step', type=int, default=63)
    ap.add_argument('--cache', action='store_true')
    ap.add_argument('--check', action='store_true', help='只跑原 3 字段，和 phase0 对数')
    ap.add_argument('--out', default='phase0b_results')
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    fac = build_features(args.exchanges, args.step,
                         os.path.join(here, f'phase0b_factors_{args.step}.pkl')
                         if args.cache else None)
    panel = load_raw(args.exchanges)
    print(f'财务面板 {len(panel):,} 行 / {panel.symbol.nunique()} 只', flush=True)
    lab = build_labels(panel)
    del panel

    # asof：锚点日 t → 最后一个 endDate ≤ t 的季度 j
    fac = fac.sort_values(['symbol', 'date']).reset_index(drop=True)
    lab = lab.sort_values(['symbol', 'endDate']).reset_index(drop=True)
    lab_cols = [c for c in lab.columns if c not in ('symbol', 'endDate')]
    lab_by = {s: (g['endDate'].values, g.index.values) for s, g in lab.groupby('symbol')}
    take = np.full(len(fac), -1, dtype=np.int64)
    fac_date = fac['date'].values
    for s, idx in fac.groupby('symbol', sort=False).indices.items():
        packed = lab_by.get(s)
        if packed is None:
            continue
        dates, rows = packed
        pos = np.searchsorted(dates, fac_date[idx], side='right') - 1
        ok = pos >= 0
        take[idx[ok]] = rows[pos[ok]]
    ok = take >= 0
    df = fac[ok].reset_index(drop=True)
    for c in lab_cols:
        df[c] = lab[c].values[take[ok]]
    df = df.copy()          # 去碎片化
    print(f'合并后 {len(df):,} 行 / {df.date.nunique()} 个锚点日', flush=True)

    if args.check:
        keep = {'ebit_1', 'ebit_3', 'ebit_7', 'revenue_1', 'revenue_3', 'revenue_7',
                'ocf_1', 'ocf_3', 'ocf_7'}
        labels = [c for c in lab_cols if c in keep]
    else:
        labels = list(lab_cols)

    recs = []
    for lb in labels:
        cov = df[lb].notna().mean()
        for f in FACTORS:
            ic, icir, n = ic_stats(df, f, lb)
            recs.append(dict(label=lb, factor=f, ic=ic, icir=icir, n_dates=n, coverage=cov))
    res = pd.DataFrame(recs)

    outdir = os.path.join(here, args.out)
    os.makedirs(outdir, exist_ok=True)
    tag = 'check' if args.check else 'full'
    res.to_csv(os.path.join(outdir, f'ic_by_field_{tag}.csv'), index=False)

    # 汇总：每个标签上最强的因子 + 「同向强度」= 与最强因子同号因子的平均 |IC|
    rows = []
    for lb, x in res.groupby('label'):
        x = x.dropna(subset=['ic'])
        if x.empty:
            continue
        best = x.loc[x.ic.abs().idxmax()]
        same = x[np.sign(x.ic) == np.sign(best.ic)]
        rows.append(dict(label=lb, coverage=x.coverage.iloc[0],
                         best_factor=best.factor, best_ic=best.ic, best_icir=best.icir,
                         n_agree=int(((x.ic.abs() > 0.05) &
                                      (np.sign(x.ic) == np.sign(best.ic))).sum()),
                         mean_abs_ic=x.ic.abs().mean(),
                         agree_mean_abs=same.ic.abs().mean()))
    s = pd.DataFrame(rows)
    s['abs_best'] = s.best_ic.abs()
    s = s.sort_values('abs_best', ascending=False)
    s.to_csv(os.path.join(outdir, f'field_summary_{tag}.csv'), index=False)
    pd.set_option('display.width', 220)
    print(s.to_string(index=False, float_format=lambda v: f'{v:+.3f}'))
    print(f'\n明细写入 {outdir}/')


if __name__ == '__main__':
    main()
