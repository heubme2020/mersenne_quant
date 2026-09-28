"""头 1 候选对比：毛利/总资产（方案 B） vs Δ净资产/净资产（方案 C）等。

背景：phase0b 的字段扫描里，`Δ净资产/净资产` 的原始 IC / 增量 IC 都比 `毛利/总资产` 高
（0.342/0.279 vs 0.287/0.197），但那个数字不能直接用来选头，原因有三：

  1. 扫描是**单因子** IC（6 个因子里挑最好的 mom889），有选择偏差；
     而实际的头是**训练出来的多任务模型**，能提取的远多于一个因子。
  2. 扫描用的是**旧的窗口约定**（以 j-3 为基点、窗口 [j-3, j+h]），
     那个约定下标签里有很大一部分"已经实现的变化"是锚点日就已知的
     （实测 corr(y,b) 高达 0.99），IC 会被这部分灌水。
  3. 最关键的**冗余问题**：Δ净资产 ≈ Δ留存收益 ≈ 净利 ≈ ROE，本质是同一个信号；
     三个头如果是一条腿，多头就退化成单头，共享编码器学不到互补表征。

本脚本用**修正后的窗口约定**、在**同一批样本**上，对每个候选头 1 算：
  覆盖率 / 与另外两个头的截面相关 / 基准 IC / 单因子 IC / 增量 IC。
增量 IC 用 mom889（扫描里最强的单因子）打，作为"这个头能不能被日线预测"的上限参考。

用法：python two/nowcast/head1_candidates.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import labels as L                      # noqa: E402

FACTOR_CACHE = os.path.join(HERE, 'phase0b_factors_63.pkl')

# 候选头 1（外加两个固定头，用来看冗余）
# 重点：**同一个分子 × 两个分母**（净资产 vs 总资产）成对放，隔离分母选择。
CANDIDATES = {
    # —— EBIT × 两个分母（本次的重点对照）——
    'ebit_EQ':         ('income',  'ebit',                  'level_flow_eq'),
    'ebit_TA':         ('income',  'ebit',                  'level_flow'),
    # —— 其他分子 × 两个分母 ——
    'netIncome_EQ':    ('income',  'netIncome',             'level_flow_eq'),
    'netIncome_TA':    ('income',  'netIncome',             'level_flow'),
    'grossProfit_EQ':  ('income',  'grossProfit',           'level_flow_eq'),
    'grossProfit_TA':  ('income',  'grossProfit',           'level_flow'),
    'revenue_EQ':      ('income',  'revenue',               'level_flow_eq'),
    'revenue_TA':      ('income',  'revenue',               'level_flow'),
    # —— 存量分母（净资产自身的变化，作参照）——
    'equityGrowth':    ('balance', 'totalStockholdersEquity', 'level_stock'),
    # —— 固定头（冗余对照）——
    'revGrowth_fixed': ('income',  'revenue',               'growth'),
    'assetGrowth_fix': ('balance', 'totalAssets',           'level_stock'),
}
H = 3          # 只看 3 季（代表期限，1/7 结论同向）
HORIZON = [H]


def load_panel_wide(exchange):
    """比 labels.load_panel 多读几列（候选头 1 需要 netIncome / ebit / retainedEarnings）。"""
    ex = exchange.lower()
    inc = pd.read_csv(os.path.join(L.ROOT, 'data', exchange.upper(), f'income_{ex}.csv'),
                      usecols=['symbol', 'endDate', 'revenue', 'grossProfit',
                               'netIncome', 'ebit'])
    bal = pd.read_csv(os.path.join(L.ROOT, 'data', exchange.upper(), f'balance_{ex}.csv'),
                      usecols=['symbol', 'endDate', L.TA, L.EQ, 'retainedEarnings'])
    p = inc.merge(bal, on=['symbol', 'endDate'], how='outer')
    p = p[p.symbol.notna() & p.endDate.notna()]
    p['symbol'] = p.symbol.astype(str)
    p = p[p.endDate.astype(str).str[4:].isin(['0331', '0630', '0930', '1231'])]
    p = p.drop_duplicates(subset=['symbol', 'endDate'], keep='last')
    return p.sort_values(['symbol', 'endDate']).reset_index(drop=True)


def _rank(x):
    return np.argsort(np.argsort(x))


def _ic(a, b):
    ra, rb = _rank(a), _rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def ic_stats(df, col_y, col_x, min_stocks=20):
    """逐 (date) 截面 Spearman IC -> (mean, ICIR, n)。"""
    ics = []
    for _, g in df[['date', col_x, col_y]].dropna().groupby('date'):
        if len(g) < min_stocks:
            continue
        ics.append(_ic(g[col_x].values, g[col_y].values))
    s = pd.Series(ics).dropna()
    if len(s) < 2:
        return np.nan, np.nan, 0
    return float(s.mean()), float(s.mean() / s.std()), len(s)


def main():
    # 用 labels.py 的窗口逻辑，只把 FIELDS 换成候选集合
    L.FIELDS = dict(CANDIDATES)
    L.load_panel = load_panel_wide          # 借用 labels.build_labels 的窗口逻辑
    panel = load_panel_wide('SHZ')
    lab = L.build_labels(panel)
    print(f'面板 {len(panel):,} 行  ok={int(lab.ok.sum()):,}')

    # asof：锚点日 -> 最后一个 endDate <= 锚点日的季度
    fac = pd.read_pickle(FACTOR_CACHE)
    fac = fac.sort_values(['symbol', 'date']).reset_index(drop=True)
    lab = lab.sort_values(['symbol', 'j_endDate']).reset_index(drop=True)
    cols = [c for c in lab.columns if c not in ('symbol', 'j_endDate')]
    by = {s: (g['j_endDate'].values, g.index.values)
          for s, g in lab.groupby('symbol', sort=False)}
    take = np.full(len(fac), -1, dtype=np.int64)
    fd = fac['date'].values
    for s, idx in fac.groupby('symbol', sort=False).indices.items():
        packed = by.get(s)
        if packed is None:
            continue
        dd, rr = packed
        pos = np.searchsorted(dd, fd[idx], side='right') - 1
        ok = pos >= 0
        take[idx[ok]] = rr[pos[ok]]
    ok = take >= 0
    df = fac[ok].reset_index(drop=True).copy()
    for c in cols:
        df[c] = lab[c].values[take[ok]]
    df = df.copy()
    print(f'合并后 {len(df):,} 行 / {df.date.nunique()} 个锚点日，h={H}\n')

    # 固定头（用于算冗余）：营收增长、资产增长
    fix = ['revGrowth_fixed', 'assetGrowth_fix']
    rows = []
    for name in CANDIDATES:
        y, b = f'y_{name}{H}', f'b_{name}{H}'
        if y not in df.columns:
            continue
        cov = df[y].notna().mean()
        bic, bicir, _ = ic_stats(df, y, b)                     # 基准 IC
        fic, ficir, _ = ic_stats(df, y, 'mom889')               # 单因子 IC
        df['_r'] = df[y] - df[b]
        ric, ricir, _ = ic_stats(df, '_r', 'mom889')            # 增量 IC
        # 与两个固定头的截面相关（逐截面 Spearman 取均值）
        cors = []
        for other in fix:
            if name == other:
                continue
            oc, _, _ = ic_stats(df, y, f'y_{other}{H}')
            cors.append(oc)
        sd = df[y].std() / (abs(df[y].median()) + 1e-9)      # 相对离散度
        rows.append(dict(head=name, cov=cov, ic_bench=bic, ic_mom889=fic,
                         ic_incr=ric, icir_incr=ricir, rel_sd=sd,
                         corr_fixed=np.nanmean(cors) if cors else np.nan))
    r = pd.DataFrame(rows).sort_values('ic_incr', ascending=False)
    pd.set_option('display.width', 200)
    print('%-18s %7s %9s %10s %10s %10s %8s %10s' %
          ('候选头1 (h=3)', '覆盖', '相对离散', 'IC(基准)', 'IC(mom889)', '增量IC',
           '增量ICIR', '与固定头'))
    for _, x in r.iterrows():
        print('%-18s %7.3f %9.2f %+10.3f %+10.3f %+10.3f %8.2f %+10.3f' %
              (x['head'], x['cov'], x['rel_sd'], x['ic_bench'], x['ic_mom889'],
               x['ic_incr'], x['icir_incr'], x['corr_fixed']))
    print('\n读法：')
    print('  增量IC = mom889 对 (实际−基准) 的 IC，即「这个头里能被日线预测的净信息」。')
    print('  与固定头相关：越高说明这个头和已有的营收增长/资产增长头越冗余。')


if __name__ == '__main__':
    main()
