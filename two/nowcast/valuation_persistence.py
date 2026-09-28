"""「前瞻估值倍数」相对「当期估值倍数」到底变了多少？—— 决定一个头值不值得预测。

问法：把某个财务量做成估值倍数（水平 ÷ 当期市值），
      再问「**前瞻** h 季的倍数」的横截面排序 与「**当期**倍数」的排序有多一致。

  IC(当期倍数, 前瞻倍数) 越接近 1  ->  前瞻版几乎等于当期版，
                                    而当期版在锚点日是**免费**的（不用模型）-> 预测这个头没有增量价值。
  IC 越低                          ->  排序真的会变 -> 预测它才有意义。

这正是「预测这个头值不值得做」的直接判据，比单因子 IC 干净：
它不含任何模型/因子，只问"这个量的未来排序能不能靠今天免费猜到"。

各头的「水平」定义（分母都是锚点日已知的市值，所以两侧同分母、尺度无关）：
  营收   : Σ营收[fwd h 季] / 市值      vs  Σ营收[back h 季] / 市值
  毛利   : Σ毛利[fwd h 季] / 市值      vs  Σ毛利[back h 季] / 市值
  净资产 : 净资产[j+h] / 市值          vs  净资产[j] / 市值
  总资产 : 总资产[j+h] / 市值          vs  总资产[j] / 市值
  经营现金流: ΣOCF[fwd] / 市值          vs  ΣOCF[back] / 市值
  归母净利 : Σ净利[fwd] / 市值          vs  Σ净利[back] / 市值

市值 = weightedAverageShsOut[j] × close[锚点日]（两者都锚点日已知，无前视）。

用法：python two/nowcast/valuation_persistence.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import labels as L                                   # noqa: E402

FACTOR_CACHE = os.path.join(HERE, 'phase0b_factors_63.pkl')
HORIZONS = [1, 3, 7]
TA, EQ = 'totalAssets', 'totalStockholdersEquity'

# name -> (表, 列, 类型)  type: flow（区间求和） | stock（时点存量）
HEADS = {
    '营收':    ('income',   'revenue',                 'flow'),
    '毛利':    ('income',   'grossProfit',             'flow'),
    '归母净利': ('income',   'netIncome',               'flow'),
    '经营现金流': ('cashflow', 'operatingCashFlow',      'flow'),
    '净资产':   ('balance',  'totalStockholdersEquity', 'stock'),
    '总资产':   ('balance',  'totalAssets',             'stock'),
}


def load_panel(exchange):
    ex = exchange.lower()
    inc = pd.read_csv(os.path.join(L.ROOT, 'data', exchange.upper(), f'income_{ex}.csv'),
                      usecols=['symbol', 'endDate', 'revenue', 'grossProfit',
                               'netIncome', 'weightedAverageShsOut'])
    cf = pd.read_csv(os.path.join(L.ROOT, 'data', exchange.upper(), f'cashflow_{ex}.csv'),
                     usecols=['symbol', 'endDate', 'operatingCashFlow'])
    bal = pd.read_csv(os.path.join(L.ROOT, 'data', exchange.upper(), f'balance_{ex}.csv'),
                      usecols=['symbol', 'endDate', EQ, TA])
    p = inc.merge(cf, on=['symbol', 'endDate'], how='outer') \
           .merge(bal, on=['symbol', 'endDate'], how='outer')
    p = p[p.symbol.notna() & p.endDate.notna()]
    p['symbol'] = p.symbol.astype(str)
    p = p[p.endDate.astype(str).str[4:].isin(['0331', '0630', '0930', '1231'])]
    p = p.drop_duplicates(subset=['symbol', 'endDate'], keep='last')
    return p.sort_values(['symbol', 'endDate']).reset_index(drop=True)


def _rank(x):
    return np.argsort(np.argsort(x))


def ic_pairs(df, a, b, min_stocks=20):
    ics = []
    for _, g in df[['date', a, b]].dropna().groupby('date'):
        if len(g) < min_stocks:
            continue
        ra, rb = _rank(g[a].values), _rank(g[b].values)
        if ra.std() and rb.std():
            ics.append(np.corrcoef(ra, rb)[0, 1])
    s = pd.Series(ics).dropna()
    return (float(s.mean()), len(s)) if len(s) >= 2 else (np.nan, 0)


def main():
    fac = pd.read_pickle(FACTOR_CACHE)[['symbol', 'date']]
    # 只留 SHZ（因子缓存是 SHZ+SHH，这里为省内存只看一所；结论不受影响）
    shz = set(pd.read_csv(os.path.join(L.ROOT, 'data', 'SHZ', 'income_shz.csv'),
                          usecols=['symbol']).symbol.astype(str).unique())
    fac = fac[fac.symbol.isin(shz)]
    print(f'锚点样本 {len(fac):,}')

    d = pd.read_csv(os.path.join(L.ROOT, 'data', 'SHZ', 'daily_shz.csv'),
                    usecols=['symbol', 'date', 'close'], engine='pyarrow')
    d['symbol'] = d.symbol.astype(str)
    px = fac.merge(d, on=['symbol', 'date'], how='left')
    print(f'匹配到收盘价 {px.close.notna().mean()*100:.1f}%')
    px = px[px.close.notna() & (px.close > 0)]

    panel = load_panel('SHZ')
    g = panel.groupby('symbol', sort=False)

    # 逐头算「当期水平」与「前瞻水平」（都除以市值）
    out = px.copy()
    cols = {'symbol': panel.symbol, 'endDate': panel.endDate}
    for name in HEADS:
        cols[f'cur_{name}'] = np.nan
        for h in HORIZONS:
            cols[f'fwd{h}_{name}'] = np.nan
    # 为了能用 asof 合并，先按 (symbol, j) 建一张宽表
    wide = {'symbol': panel.symbol, 'j_endDate': panel.endDate}
    wide['shares'] = panel['weightedAverageShsOut']
    for name in HEADS:
        tbl, col, kind = HEADS[name]
        s = panel[col]
        for h in HORIZONS:
            if kind == 'flow':
                fwd = g[col].transform(lambda x: x.shift(-1).rolling(h, min_periods=h)
                                       .sum().shift(-(h - 1)))
                back = g[col].transform(lambda x: x.rolling(h, min_periods=h).sum())
            else:
                fwd = g[col].shift(-h)
                back = panel[col]
            wide[f'fwd{h}_{name}'] = fwd
            wide[f'cur_{name}{h}'] = back
    W = pd.DataFrame(wide).sort_values(['symbol', 'j_endDate']).reset_index(drop=True)

    by = {s: (gg['j_endDate'].values, gg.index.values)
          for s, gg in W.groupby('symbol', sort=False)}
    take = np.full(len(out), -1, dtype=np.int64)
    od = out['date'].values
    for s, idx in out.groupby('symbol', sort=False).indices.items():
        packed = by.get(s)
        if packed is None:
            continue
        dd, rr = packed
        pos = np.searchsorted(dd, od[idx], side='right') - 1
        ok = pos >= 0
        take[idx[ok]] = rr[pos[ok]]
    ok = take >= 0
    df = out[ok].reset_index(drop=True).copy()
    for c in W.columns:
        if c not in ('symbol', 'j_endDate'):
            df[c] = W[c].values[take[ok]]
    df = df.copy()

    # 市值 = 股本 × 锚点日收盘；两侧同分母，具体尺度不影响排序
    mcap = df['shares'] * df['close']
    df = df[(mcap > 0)].copy()
    mcap = mcap[mcap > 0]

    print()
    print('%-10s %6s %14s %14s' % ('头', 'h(季)', 'IC(当期,前瞻)', 'n截面'))
    for name in HEADS:
        for h in HORIZONS:
            m = df[f'fwd{h}_{name}'] / mcap
            c = df[f'cur_{name}{h}'] / mcap
            d2 = df[['date']].copy()
            d2['a'], d2['b'] = c.values, m.values
            ic, n = ic_pairs(d2, 'a', 'b')
            print('%-10s %6d %+14.3f %14d' % (name, h, ic, n))
    print()
    print('读法：IC(当期,前瞻) 越接近 1，说明「前瞻倍数」几乎等于「当期免费倍数」，'
          '预测这个头就没有增量价值。')


if __name__ == '__main__':
    main()
