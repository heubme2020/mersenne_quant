"""收益端评估：按「毛利/市值」排序能不能赚钱，以及**模型有没有比免费基线更好**。

## 为什么必须做这个

前面所有 IC（水平/正交化/混合）都在问"能不能把**毛利**排对"，但你要赚的是**收益**。
毛利排对了 ≠ 赚钱——因为"未来毛利高"可能**已经被价格反映**了。
而且把分子分母同除市值，在 IC 口径下是**恒等变换**（同一个分母在 Spearman 里约掉），
所以"÷市值"这个动作只有在和**收益**对照时才起作用。

## 三个组合（每个锚点日各算一次）

  基线 = 上一期 h 季毛利和 ÷ 当天市值      <- 免费，不用模型
  模型 = 预测未来 h 季毛利和 ÷ 当天市值    <- 用模型（只有 held-out 测试集有）
  全市场 = 全样本等权                      <- 判断"选股本身"有没有用

分数越高 = 毛利/市值越高 = 越便宜。取**前 50%**（用户的做法），
也输出五分组看单调性。

## 无前视

* 市值 = `weightedAverageShsOut[j] × close[锚点日]`，两者锚点日已知
* 基线分子 = 已披露的 `Σ毛利[j-h+1..j]`
* 收益 = `close[锚点日 + 63h 个交易日] / close[锚点日] − 1`

## 已知偏差（读结果时要记住）

* 锚点间隔 63 交易日、持有 h 季 -> **相邻组合重叠**，收益自相关，t 值偏乐观
* 没有剔 ST/退市/停牌、没有交易成本、没有流动性约束 -> 毛收益，会比实盘好
* 只做多前 50%，所以看**多头组收益**；空头组收益仅作参考

用法：
  python two/nowcast/returns_eval.py --mode baseline --exchanges SHZ SHH   # 全 A 股，功效最大
  python two/nowcast/returns_eval.py --mode compare  --tag _v2             # 基线 vs 模型
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'one'))
sys.path.insert(0, HERE)

from data import Store, ASHARE_EX            # noqa: E402
from train import load_split, ARMS           # noqa: E402
import labels as L                           # noqa: E402
import variants as V                         # noqa: E402

ROOT_DATA = L.ROOT
HORIZONS = [1, 3, 7]
DAYS_PER_Q = 63          # 1 季 ≈ 63 交易日（与标签窗口一致）
GP = 'grossProfit'


# ---------------------------------------------------------------- 数据
def load_daily_close(exchanges):
    """{symbol: (dates array, close array)}，只留 >0 的收盘价。"""
    frames = []
    for ex in exchanges:
        d = pd.read_csv(os.path.join(ROOT_DATA, 'data', ex.upper(), f'daily_{ex.lower()}.csv'),
                        usecols=['symbol', 'date', 'close'], engine='pyarrow')
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d['symbol'] = d.symbol.astype(str)
    d = d[d.close > 0].sort_values(['symbol', 'date'])
    out = {s: (g['date'].values, g['close'].values.astype('float64'))
           for s, g in d.groupby('symbol', sort=False)}
    del d
    return out


def load_panel(exchanges):
    """(symbol, endDate) -> grossProfit, shares，用于算基线的分子与市值。"""
    frames = []
    for ex in exchanges:
        p = pd.read_csv(os.path.join(ROOT_DATA, 'data', ex.upper(), f'income_{ex.lower()}.csv'),
                        usecols=['symbol', 'endDate', GP, 'weightedAverageShsOut'])
        frames.append(p)
    p = pd.concat(frames, ignore_index=True)
    p['symbol'] = p.symbol.astype(str)
    p = p[p.endDate.astype(str).str[4:].isin(['0331', '0630', '0930', '1231'])]
    p = p.drop_duplicates(subset=['symbol', 'endDate'], keep='last')
    return p.sort_values(['symbol', 'endDate']).reset_index(drop=True)


def trailing_sum(panel, h, lag=0):
    """每个 (symbol, j) 的 Σ毛利[j-h+1-lag .. j-lag]。

    lag=1 = 只用**已披露**的季度（锚点日在季度末时当季财报尚未公布），
    避免前视：否则"上一期"其实是"未来"，会把基准灌成近乎恒等关系。
    """
    g = panel.groupby('symbol', sort=False)
    v = g[GP].transform(lambda x: x.rolling(h, min_periods=h).sum())
    return v.shift(lag) if lag else v


def forward_ret(daily, anchors):
    """每个锚点日的 1/3/7 季前瞻收益（按交易日偏移 63h）。"""
    res = {}
    for sym, (dt, cl) in daily.items():
        pos = np.searchsorted(dt, anchors, side='right') - 1
        ok = pos >= 0
        pos_ok = pos[ok]
        e = np.full(len(anchors), np.nan)
        e[ok] = cl[pos_ok]
        row = {'symbol': sym, 'anchor': anchors, 'entry': e}
        for h in HORIZONS:
            exit_pos = pos + DAYS_PER_Q * h
            good = ok & (exit_pos < len(cl))
            r = np.full(len(anchors), np.nan)
            r[good] = cl[exit_pos[good]] / e[good] - 1.0
            row[f'ret{h}'] = r
        res[sym] = pd.DataFrame(row)
    return pd.concat(res.values(), ignore_index=True)


def trailing_mom(daily, anchors):
    """每个 (symbol, 锚点) 的简单动量（只要 shift，O(1)）。

    为什么要重算：phase0b 的因子缓存用的是**另一套锚点网格**（全局 63 日网格），
    和 nowcast 的锚点（每只股票自己的"≤ 季度末的最后一个交易日"）对不上，
    直接 merge 只剩 1 千行。这里在 nowcast 锚点上现算。
    """
    res = {}
    for sym, (dt, cl) in daily.items():
        pos = np.searchsorted(dt, anchors, side='right') - 1
        row = {'symbol': sym, 'anchor': anchors}
        for w in [127, 381, 889]:
            v = np.full(len(anchors), np.nan)
            good = pos >= w
            v[good] = cl[pos[good]] / cl[pos[good] - w] - 1.0
            row[f'mom{w}'] = v
        res[sym] = pd.DataFrame(row)
    return pd.concat(res.values(), ignore_index=True)


# ---------------------------------------------------------------- 组合
def _wmean(x, lo=0.01, hi=0.99):
    """组内 winsorize 后的均值：把最高/最低 1% 拉回分位点。

    必须做这一步：A 股等权收益的**算术均值被极值主导**——
    长期停牌复牌、重组复牌会让个别股票"收益"翻几十倍，
    实测全市场等权 1 季平均 +161%（不可能），而中位数只有个位数。
    """
    if len(x) < 20:
        return np.nan
    a, b = np.quantile(x, [lo, hi])
    return float(np.clip(x, a, b).mean())


def portfolio_table(df, score_col, ret_col, n_q=5):
    """逐 (年-季) 分组 -> 各组收益。Q1 = 分数最高（最便宜）。

    每个组合报三个数：中位数（最稳健）、winsorize 均值、原始均值（仅供参考）。
    """
    rows = []
    for d, g in df[['qkey', score_col, ret_col]].dropna().groupby('qkey'):
        if len(g) < 50:
            continue
        r = g[score_col].rank(ascending=False, method='first')
        q = np.ceil(r / len(g) * n_q).astype(int).clip(1, n_q)
        top50 = (r <= len(g) / 2.0).values   # 前 50%（用户的做法）
        masks = {'all': np.ones(len(g), bool), 'top_half': top50}
        for k in range(1, n_q + 1):
            masks[f'Q{k}'] = (q == k).values
        rec = {'qkey': d}
        for cname, m in masks.items():
            v = g[ret_col].values[m]
            rec[f'{cname}_med'] = np.median(v)
            rec[f'{cname}_w'] = _wmean(v)
            rec[f'{cname}_raw'] = np.mean(v)
        rows.append(rec)
    a = pd.DataFrame(rows)
    out = {}
    for c in a.columns:
        if c == 'qkey':
            continue
        v = a[c].dropna()
        out[c] = (v.mean(), v.mean() / (v.std() / np.sqrt(len(v))) if v.std() > 0 else np.nan,
                  len(v))
    out['_groups'] = a
    return out


def fmt(label, t):
    m, tv, n = t
    return f'{label:14s}{m*100:>+8.2f}%{tv:>+8.2f}{n:>7d}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['baseline', 'compare'], default='baseline')
    ap.add_argument('--exchanges', nargs='+', default=['SHZ', 'SHH'])
    ap.add_argument('--tag', default='_v2')
    ap.add_argument('--suffix', default='', help='标签变体（v1 = 头1 是毛利增长）')
    ap.add_argument('--sweep', action='store_true', help='growth 形式下扫 λ 收缩权重')
    ap.add_argument('--form', choices=['level', 'growth', 'composite', 'increment', 'u1'], default='level',
                    help='level: 评分=预测(毛利/总资产)×总资产/市值  |  '
                         'growth: 评分=预测毛利增长×上一期毛利和/市值')
    ap.add_argument('--lag', type=int, default=1,
                    help='财报披露滞后：1=基准只用已披露季度（无前视） 0=含未披露当季')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--root', default='Z:/quant_data/nowcast')
    ap.add_argument('--out', default='returns_eval')
    args = ap.parse_args()
    os.makedirs(os.path.join(HERE, args.out), exist_ok=True)

    # 锚点网格 = 模型评估用的那批（从分块文件的 _d 取）
    dd = []
    for ex in args.exchanges:
        for f in sorted(glob.glob(os.path.join(args.root, ex, 'chunk_*_d.npy'))):
            dd.append(np.load(f))
    anchors = np.unique(np.concatenate(dd))
    print(f'锚点 {len(anchors)} 个：{anchors[0]} → {anchors[-1]}', flush=True)

    print('读日线...', flush=True)
    daily = load_daily_close(args.exchanges)
    print(f'  {len(daily):,} 只股票', flush=True)
    print('读财报...', flush=True)
    panel = load_panel(args.exchanges)
    g = panel.groupby('symbol', sort=False)
    lab = panel[['symbol', 'endDate']].copy()
    for h in HORIZONS:
        lab[f'trail{h}'] = trailing_sum(panel, h, args.lag).values
    lab['gp_m3'] = panel.groupby('symbol')[GP].shift(3).values   # GP[j-3]，早已披露
    lab['shares'] = panel['weightedAverageShsOut'].values
    lab = lab.dropna(subset=['shares'])
    print(f'  面板 {len(lab):,} 行', flush=True)

    print('算前瞻收益...', flush=True)
    R = forward_ret(daily, anchors)
    print(f'  {len(R):,} 行', flush=True)

    # asof: 锚点 -> 最后一个 endDate <= 锚点的季度
    lab = lab.sort_values(['symbol', 'endDate']).reset_index(drop=True)
    by = {s: (gg['endDate'].values, gg.index.values)
          for s, gg in lab.groupby('symbol', sort=False)}
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
    for c in lab.columns:
        if c not in ('symbol', 'endDate'):
            D[c] = lab[c].values[take[ok]]
    D = D.copy()
    # 按「年-季」分组：锚点是逐股票的（停牌会让同季锚点差几天），
    # 用精确日期分组会把一个季度的截面打碎成几百个小组。
    ym = (D['anchor'].values // 100)
    D['qkey'] = (ym // 100) * 10 + ((ym % 100 - 1) // 3 + 1)
    # 市值 = 股本 × 锚点日收盘（单位不影响排序）
    D['mcap'] = D['shares'] * D['entry']
    D = D[(D.mcap > 0)]
    for h in HORIZONS:
        D[f'base{h}'] = D[f'trail{h}'] / D['mcap']
    print(f'  合并后 {len(D):,} 行 / {D.anchor.nunique()} 个锚点，'
          f'平均每锚点 {len(D)/D.anchor.nunique():.0f} 只', flush=True)

    # ---- 模型分数：只对 held-out 测试集 ----
    if args.mode == 'compare':
        store = Store(args.root, sorted(set(ASHARE_EX)), min_per_date=64,
                      label='level', suffix=args.suffix)
        tr, va, te = load_split(store, 'ashare')
        npz_path = os.path.join(HERE, 'results',
                                f'pred_ashare{args.tag}_s{args.seed}.npz')
        print(f'  读预测 {os.path.basename(npz_path)}', flush=True)
        pred = np.load(npz_path)['pred']
        # 测试集的 y_mean/y_std（还原量纲，与 train.py 同法）
        rng = np.random.RandomState(args.seed + 1)
        samp = np.sort(tr[rng.choice(len(tr), min(20000, len(tr)), replace=False)])
        Ys = store.take_y(samp)
        y_mean, y_std = Ys.mean(0), Ys.std(0) + 1e-6
        heads = L.flat_names(V.fields_for(args.suffix))
        # 头 1 的三个期限永远是展平后的前三个（字段优先）
        gi = [0, 1, 2]
        print(f'  头 1 = {heads[0][:-1]}（{args.form} 形式）', flush=True)

        # 测试集每行 -> (symbol, anchor)
        syms = store.symbol_of(te)
        T = pd.DataFrame({
            'symbol': [s.split(':')[-1] for s in syms],
            'anchor': store.dates[te]})
        for i, h in enumerate(HORIZONS):
            # 预测的 毛利/总资产 -> × 总资产[j] -> 预测毛利和
            pred_gp = pred[:, gi[i]] * y_std[gi[i]] + y_mean[gi[i]]
            T[f'pred_gp{h}'] = pred_gp
        # 取测试集的总资产[j]
        ta = L.load_panel('SHZ')[['symbol', 'endDate', 'totalAssets']]
        ta2 = L.load_panel('SHH')[['symbol', 'endDate', 'totalAssets']]
        ta = pd.concat([ta, ta2], ignore_index=True).sort_values(['symbol', 'endDate'])
        if args.lag:
            ta = ta.copy()
            ta['totalAssets'] = ta.groupby('symbol', sort=False)['totalAssets'].shift(args.lag)
        by2 = {s: (gg['endDate'].values, gg['totalAssets'].values)
               for s, gg in ta.groupby('symbol', sort=False)}
        TA = np.full(len(T), np.nan)
        for s, idx in T.groupby('symbol', sort=False).indices.items():
            packed = by2.get(s)
            if packed is None:
                continue
            ed, va_ = packed
            pos = np.searchsorted(ed, T['anchor'].values[idx], side='right') - 1
            okk = pos >= 0
            TA[idx[okk]] = va_[pos[okk]]
        T['ta'] = TA
        D = D.merge(T[['symbol', 'anchor', 'ta'] + [f'pred_gp{h}' for h in HORIZONS]],
                    on=['symbol', 'anchor'], how='left')
        if args.form == 'level':
            # 评分 = 预测(毛利/总资产) × 总资产[j] ÷ 市值
            for h in HORIZONS:
                D[f'model{h}'] = D[f'pred_gp{h}'] * D['ta'] / D['mcap']
        elif args.form == 'u1':
            # u1 标签 = (ΣGP[j+1..j+h] − h·GP[j-3]) / (h·GP[j-3])
            # 反算：ΣGP[j+1..j+h] = h·GP[j-3]·(1 + 预测)
            # 评分 = ΣGP_fwd / 市值 = (1 + 预测) × GP[j-3] / 市值   （h 是公因子，约掉）
            # 注意 pred=0（"不变"）时评分就是 GP[j-3]/市值 —— 一个**价值因子**，
            # 所以这里测的是"模型的同比预测有没有改善那个价值因子的排序"。
            for h in HORIZONS:
                D[f'model{h}'] = (1.0 + D[f'pred_gp{h}']) * D['gp_m3'] / D['mcap']
            D['gp_m3s'] = D['gp_m3'] / D['mcap']
        elif args.form == 'growth':
            # 评分 = 预测毛利增长 × 上一期毛利和 ÷ 市值 = 模型预测 × 基线评分
            for h in HORIZONS:
                D[f'model{h}'] = D[f'pred_gp{h}'] * D[f'base{h}']
        elif args.form == 'increment':
            # ---- 增量（超预期）分数 ----
            #   incr_h = ( 预测未来 h 季毛利和 − 上一期 h 季毛利和 ) ÷ 市值
            # 完全**不含估值水平**：朴素预测的"变化"是 0，所以这里没有任何免费基准，
            # 任何排序能力都纯粹是模型的贡献。
            for h in HORIZONS:
                gp_fwd = D[f'pred_gp{h}'] * D['ta']
                D[f'incr{h}'] = (gp_fwd - D[f'trail{h}']) / D['mcap']
                D[f'incrr{h}'] = (gp_fwd - D[f'trail{h}']) / D[f'trail{h}'].abs().clip(lower=1.0)
            gp_c = D[['pred_gp1', 'pred_gp3', 'pred_gp7']].sum(axis=1) * D['ta']
            tr_c = D['trail1'] + D['trail3'] + D['trail7']
            D['incr_comp'] = (gp_c - tr_c) / D['mcap']
        else:
            # ---- composite：多期限合成 ----
            # 基线分子 = 上季毛利 + 近 3 季毛利和 + 近 7 季毛利和（三档嵌套，等价于
            #            对近 7 季做线性递减加权）
            # 模型分子 = 预测下 1 季 + 下 3 季和 + 下 7 季和（同样三档嵌套）
            D['base_comp'] = (D['trail1'] + D['trail3'] + D['trail7']) / D['mcap']
            D['model_comp'] = (D[['pred_gp1', 'pred_gp3', 'pred_gp7']].sum(axis=1)
                               * D['ta']) / D['mcap']
        _mc = ({'composite': 'model_comp', 'increment': 'incr1'}
               .get(args.form, 'model1'))
        n_match = D[_mc].notna().sum()
        print(f'  测试集匹配上 {n_match:,} 行（模型分数可用）', flush=True)

    # ---- 出结果 ----
    for h in HORIZONS:
        print(f'\n{"="*96}\n  持有 {h} 季（{DAYS_PER_Q*h} 交易日）\n{"="*96}')
        bcol = ({'composite': 'base_comp', 'u1': 'gp_m3s'}
                .get(args.form, f'base{h}'))
        mcol = ({'composite': 'model_comp', 'increment': 'incr1'}
                .get(args.form, f'model{h}'))
        sub = D[D[mcol].notna()].copy() if args.mode == 'compare' else D
        # ---- 增量（超预期）测试：按"预测的变化"排序，终点是收益 ----
        if args.form == 'increment':
            for cname2, col2 in [(f'增量 h={h}', f'incr{h}'),
                                 ('增量(合成)', 'incr_comp'),
                                 (f'相对增量 h={h}', f'incrr{h}')]:
                ti = portfolio_table(sub, col2, f'ret{h}')
                print(f'\n  【{cname2}】按分数高低分组（Q1=最看好）')
                print(f'  {"组":14s}{"中位数":>10s}{"winsor":>10s}')
                for k in range(1, 6):
                    m2, _, _ = ti[f'Q{k}_med']
                    print(f'  Q{k:<13d}{m2*100:>+9.2f}%{ti[f"Q{k}_w"][0]*100:>+9.2f}%')
                m2, t2, _ = ti['top_half_med']
                m3, t3, _ = ti['all_med']
                print(f'  {"前50%":14s}{m2*100:>+9.2f}%   （全市场 {m3*100:+.2f}%，'
                      f'差 {(m2-m3)*100:+.2f}pp, t={t2:+.2f}）')
                g2 = ti['_groups']
                sp = (g2['Q1_med'] - g2['Q5_med'])
                print(f'  Q1−Q5 多空差：{sp.mean()*100:+.2f}%/期  '
                      f't={sp.mean()/(sp.std()/np.sqrt(len(sp))):+.2f}  n={len(sp)}')
                # 与估值排序的相关性（逐截面 rank 相关）
                cs = []
                for _, gg in sub[['qkey', col2, bcol]].dropna().groupby('qkey'):
                    if len(gg) < 50:
                        continue
                    ra = np.argsort(np.argsort(gg[col2].values))
                    rb = np.argsort(np.argsort(gg[bcol].values))
                    if ra.std() and rb.std():
                        cs.append(np.corrcoef(ra, rb)[0, 1])
                print(f'  与「估值排序」的相关性：{np.mean(cs):+.3f}'
                      f'   （越接近 0 越是正交的第二路信号）')
            # ---- 增量是不是"动量代理"？----
            if args.mode == 'compare':
                mm = trailing_mom(daily, anchors)
                sub2 = sub.merge(mm, on=['symbol', 'anchor'], how='left')
                print()
                print('  增量分数与动量的截面相关（判断它是不是"动量代理"）')
                print(f'  {"分数":<10s}{"mom127":>9s}{"mom381":>9s}{"mom889":>9s}')
                for c3 in [f'incr{h}', 'incr_comp', f'base{h}']:
                    vals = []
                    for fac in ['mom127', 'mom381', 'mom889']:
                        cs3 = []
                        for _, gg in sub2[['qkey', c3, fac]].dropna().groupby('qkey'):
                            if len(gg) < 50:
                                continue
                            ra = np.argsort(np.argsort(gg[c3].values))
                            rb = np.argsort(np.argsort(gg[fac].values))
                            if ra.std() and rb.std():
                                cs3.append(np.corrcoef(ra, rb)[0, 1])
                        vals.append(np.mean(cs3))
                    print(f'  {c3:<10s}{vals[0]:>+9.3f}{vals[1]:>+9.3f}{vals[2]:>+9.3f}')
            # ---- 动量中性化：把 incr 对 mom889 逐截面回归取残差，再测收益 ----
            # 这是"模型有没有超出动量的价值"的决定性检验。
            if args.mode == 'compare':
                mm2 = trailing_mom(daily, anchors)
                s2 = sub.merge(mm2[['symbol', 'anchor', 'mom889']],
                               on=['symbol', 'anchor'], how='left')
                print()
                print(f'  【动量中性化后的增量 h={h}】Q1=残差最正')
                for c4 in [f'incr{h}', f'incr{h}_neu']:
                    if c4.endswith('_neu'):
                        neu = np.full(len(s2), np.nan)
                        for _, gg in s2[['qkey', f'incr{h}', 'mom889']].dropna().groupby('qkey'):
                            if len(gg) < 50:
                                continue
                            ri = np.argsort(np.argsort(gg[f'incr{h}'].values)).astype(float)
                            rm = np.argsort(np.argsort(gg['mom889'].values)).astype(float)
                            if rm.std() == 0:
                                continue
                            b2 = np.polyfit(rm, ri, 1)
                            neu[gg.index.values] = ri - (b2[0] * rm + b2[1])
                        s2[c4] = neu
                    t4 = portfolio_table(s2, c4, f'ret{h}')
                    q1, _, _ = t4['Q1_med']
                    q5, _, _ = t4['Q5_med']
                    g4 = t4['_groups']
                    sp4 = g4['Q1_med'] - g4['Q5_med']
                    lab4 = '中性化后' if c4.endswith('_neu') else '原始增量'
                    print(f'  {lab4:8s} Q1={q1*100:+.2f}%  Q5={q5*100:+.2f}%  '
                          f'Q1−Q5={sp4.mean()*100:+.2f}%  '
                          f't={sp4.mean()/(sp4.std()/np.sqrt(len(sp4))):+.2f}')
            # 存明细，供"增量是不是动量代理"的验证
            keep = [c for c in ['symbol', 'anchor', 'qkey', 'incr1', 'incr3', 'incr7',
                                'incr_comp', 'incrr1', 'incrr3', 'incrr7',
                                'base1', 'base3', 'base7', 'ret1', 'ret3', 'ret7']
                    if c in sub.columns]
            sub[keep].to_csv(os.path.join(HERE, args.out, 'incr_detail.csv'), index=False)
            continue
        # ---- λ 扫描：score = 基线 × 模型预测^λ（λ=0 就是纯基线）----
        if args.form == 'growth' and args.sweep:
            print('\n  【λ 扫描】score = 基线 × 模型增长预测^λ   （λ=0 = 纯基线）')
            print(f'  {"λ":>6s}{"前50%中位数":>13s}{"t值":>8s}{"vs λ=0":>10s}{"胜率":>8s}')
            ref = None
            for lam in [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]:
                col = f'_lam{lam}'
                sub[col] = sub[f'base{h}'] * np.maximum(sub[f'pred_gp{h}'], 1e-9) ** lam
                tt = portfolio_table(sub, col, f'ret{h}')
                m, tv, n = tt['top_half_med']
                if ref is None:
                    ref = tt['_groups'].set_index('qkey')['top_half_med']
                    w = '-'
                else:
                    cur = tt['_groups'].set_index('qkey')['top_half_med']
                    j = pd.concat([ref, cur], axis=1, keys=['a', 'b']).dropna()
                    dd = j['b'] - j['a']
                    w = f'{(dd>0).mean()*100:.0f}%'
                print(f'  {lam:>6.2f}{m*100:>+12.2f}%{tv:>+8.2f}{"":>10s}{w:>8s}')
            print()

        t = portfolio_table(sub, bcol, f'ret{h}')
        print(f'\n  按「上一期毛利/市值」分组（Q1=最便宜）   [中位数收益 | winsor均值 | 原始均值]')
        print(f'  {"组合":14s}{"中位数":>10s}{"t值":>8s}{"winsor":>10s}{"原始均值":>12s}')
        for k in range(1, 6):
            nm = f'Q{k}' + ('(最便宜)' if k == 1 else '(最贵)' if k == 5 else '')
            m, tv, n = t[f'Q{k}_med']
            print(f'  {nm:14s}{m*100:>+9.2f}%{tv:>+8.2f}'
                  f'{t[f"Q{k}_w"][0]*100:>+9.2f}%{t[f"Q{k}_raw"][0]*100:>+11.2f}%')
        print('  ' + '-' * 62)
        for c, nm in [('all', '全市场等权'), ('top_half', '基线前50%')]:
            m, tv, n = t[f'{c}_med']
            print(f'  {nm:14s}{m*100:>+9.2f}%{tv:>+8.2f}'
                  f'{t[f"{c}_w"][0]*100:>+9.2f}%{t[f"{c}_raw"][0]*100:>+11.2f}%')
        if args.mode == 'compare':
            tm = portfolio_table(sub, mcol, f'ret{h}')
            m, tv, n = tm['top_half_med']
            print(f'  {"模型前50%":14s}{m*100:>+9.2f}%{tv:>+8.2f}'
                  f'{tm["top_half_w"][0]*100:>+9.2f}%{tm["top_half_raw"][0]*100:>+11.2f}%')
            a = t['_groups'].set_index('qkey')
            b = tm['_groups'].set_index('qkey')
            j = a[['top_half_med']].join(b[['top_half_med']], lsuffix='_b', rsuffix='_m').dropna()
            dd = j['top_half_med_m'] - j['top_half_med_b']
            print(f'  → 模型 − 基线（中位数口径）：{dd.mean()*100:+.2f}%/期  '
                  f't={dd.mean()/(dd.std()/np.sqrt(len(dd))):+.2f}  n={len(dd)}  '
                  f'胜率={(dd>0).mean()*100:.0f}%')
        t['_groups'].to_csv(os.path.join(HERE, args.out, f'{args.mode}_h{h}.csv'),
                            index=False)
    print(f'\n明细写入 {args.out}/')


if __name__ == '__main__':
    main()
