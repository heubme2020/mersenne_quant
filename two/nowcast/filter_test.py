"""「排除型」用法检验：用模型预测的毛利变化做**过滤器**，值不值？

用户的用法：不是拿信号挑赢家（alpha），而是**排除毛利会下降的股票**（避雷/防套）。
判据因此不同：不是收益，而是**风险维度**（波动、回撤、尾部、负收益频率）。

对照三个组合（同一批测试股、同一批季度截面）：
  A 无过滤    : 按「上一期毛利/市值」取前 50%（已知有效的估值组合）
  B 叠加过滤  : A 里面再剔除「模型预测毛利会下降」的（incr <= 0）
  C 叠加过滤  : A 里面再剔除 incr 最低的 25%
  D 只过滤    : 全样本里只要 incr > 0 的
  E 全市场    : 全样本等权（参照）

风险指标：中位数收益 / 均值 / 波动 / 下行波动 / 最差 5% / 负收益频率 / 时序最大回撤

数据来源：returns_eval/incr_detail.csv（由 returns_eval.py --form increment 产出）
用法：python two/nowcast/filter_test.py
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'returns_eval', 'incr_detail.csv')
HORIZONS = [1, 3, 7]


def _wmean(x, lo=0.01, hi=0.99):
    if len(x) < 20:
        return np.nan
    a, b = np.quantile(x, [lo, hi])
    return float(np.clip(x, a, b).mean())


def max_dd(series):
    """时序最大回撤（按累乘净值算）。"""
    eq = np.cumprod(1 + np.asarray(series))
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1).min())


def stats(ret):
    r = np.asarray(ret, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 20:
        return None
    neg = r[r < 0]
    return dict(n=len(r), med=np.median(r), mean=r.mean(), std=r.std(),
                dstd=neg.std() if len(neg) > 5 else np.nan,
                p5=np.percentile(r, 5), negfrac=(r < 0).mean())


def main():
    d = pd.read_csv(SRC)
    d['symbol'] = d['symbol'].astype(str)
    print(f'样本 {len(d):,} 行 / {d.qkey.nunique()} 个季度截面\n')

    rows = []
    for h in HORIZONS:
        rc, bc, ic = f'ret{h}', f'base{h}', f'incr{h}'
        per = {k: [] for k in ['A', 'B', 'C', 'D', 'E']}
        for _, g in d.groupby('qkey'):
            g = g[[rc, bc, ic]].dropna()
            if len(g) < 50:
                continue
            top = g[g[bc] >= g[bc].quantile(0.5)]           # 估值前 50%（便宜）
            per['A'].append(top[rc].median())
            per['B'].append(top[top[ic] > 0][rc].median())
            per['C'].append(top[top[ic] > top[ic].quantile(0.25)][rc].median())
            per['D'].append(g[g[ic] > 0][rc].median())
            per['E'].append(g[rc].median())
        names = {'A': 'A 无过滤（估值前50%）', 'B': 'B 叠加：剔除预测下降的',
                 'C': 'C 叠加：剔除最差25%', 'D': 'D 只要预测改善的',
                 'E': 'E 全市场等权'}
        print(f'{"="*94}\n  持有 {h} 季\n{"="*94}')
        print(f'  {"组合":22s}{"期数":>5s}{"中位数":>9s}{"均值":>9s}{"波动":>9s}'
              f'{"下行波动":>10s}{"最差5%":>9s}{"负收益":>8s}{"最大回撤":>10s}')
        for k in ['A', 'B', 'C', 'D', 'E']:
            s = np.array(per[k], dtype=float)
            s = s[np.isfinite(s)]
            if len(s) < 5:
                continue
            st = stats(s) or {}
            print(f'  {names[k]:22s}{len(s):>5d}{np.median(s)*100:>+8.2f}%'
                  f'{s.mean()*100:>+8.2f}%{s.std()*100:>8.2f}%'
                  f'{st.get("dstd", np.nan)*100:>9.2f}%{st.get("p5", np.nan)*100:>+8.2f}%'
                  f'{st.get("negfrac", np.nan)*100:>7.0f}%{max_dd(s)*100:>+9.1f}%')
            rows.append(dict(h=h, port=k, **st))
        # 过滤的代价/收益
        a = np.array(per['A'], dtype=float)
        for k in ['B', 'C']:
            b = np.array(per[k], dtype=float)
            m = np.isfinite(a) & np.isfinite(b)
            dd = b[m] - a[m]
            print(f'  → {names[k][:1]} − A：收益 {dd.mean()*100:+.2f}%/期  '
                  f't={dd.mean()/(dd.std()/np.sqrt(len(dd))):+.2f}  '
                  f'胜率={(dd>0).mean()*100:.0f}%  '
                  f'保留样本 {len(b)/max(len(a),1)*100:.0f}%')
        print()
    pd.DataFrame(rows).to_csv(os.path.join(HERE, 'returns_eval', 'filter_stats.csv'),
                              index=False)


if __name__ == '__main__':
    main()
