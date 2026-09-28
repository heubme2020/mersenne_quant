"""v2 模型的收益端评估：**用「预测的毛利变化」排序，能不能比免费基线多赚**。

v2 的三个新特征（见 gen_data2.py）：
  * 锚点 = 任意交易日（对齐网格上的测试股每 31 交易日一个）
  * j = 最后一个【已披露】的季度（按 A 股披露截止日）
  * 测试股日期对齐 -> 每格有几百只 -> **逐日截面 IC 可算**

## 评分构造（u1 标签的反算）

u1 标签 = (ΣX[j+1..j+h] − h·X[j-3]) / (h·X[j-3])
=> 预测的未来 X 和 = h·X[j-3]·(1 + 预测)
=> 评分 = 预测X和 / 市值 = (1 + 预测) × X[j-3] / 市值      （h 是公因子，约掉）

pred=0（"不变"）时评分就是 `X[j-3]/市值` —— 一个**价值因子**，也就是这里的**免费基线**。
所以这个检验问的是：**模型的同比预测有没有改善那个价值因子的排序**。

## 判据（按优先级）

1. **逐日截面 IC(评分, 未来收益)** —— 终点指标，不是 IC(评分, 财务量)
2. **动量中性化后的 IC** —— 排除"只是动量代理"
3. 组合：前 50% 的中位数收益 vs 免费基线 vs 全市场；负收益频率；最大回撤
4. **A 股子样本单列**（用户实际要投的市场）

用法：python two/nowcast/eval_v2.py --root C:/quant_data/nowcast2 --tag _u1gv2 --seed 1
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'one'))
sys.path.insert(0, HERE)

from data import Store, MARKETS, GLOBAL_EX, ASHARE_EX     # noqa: E402
from train import load_split_global                       # noqa: E402
from gen_data2 import available_date, DEADLINE            # noqa: E402
import labels as L                                        # noqa: E402

HORIZONS = [1, 3, 7]
DAYS_PER_Q = 63
GP = 'grossProfit'


def load_daily_close(exchanges):
    out = {}
    for ex in exchanges:
        f = os.path.join(ROOT, 'data', ex.upper(), f'daily_{ex.lower()}.csv')
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f, usecols=['symbol', 'date', 'close'], engine='pyarrow')
        d['symbol'] = d.symbol.astype(str)
        d = d[d.close > 0]
        for s, g in d.groupby('symbol', sort=False):
            out[(ex, s)] = (g['date'].values, g['close'].values.astype('float64'))
        del d
    return out


def build_panel(exchanges):
    """(ex, symbol, endDate) -> grossProfit, shares, avail(披露日)。"""
    frames = []
    for ex in exchanges:
        f = os.path.join(ROOT, 'data', ex.upper(), f'income_{ex.lower()}.csv')
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f, usecols=['symbol', 'endDate', GP, 'weightedAverageShsOut'])
        bal = pd.read_csv(os.path.join(ROOT, 'data', ex.upper(), f'balance_{ex.lower()}.csv'),
                          usecols=['symbol', 'endDate', 'totalAssets'])
        d = d.merge(bal, on=['symbol', 'endDate'], how='outer')
        d['ex'] = ex
        frames.append(d)
    p = pd.concat(frames, ignore_index=True)
    p['symbol'] = p.symbol.astype(str)
    p = p[p.endDate.astype(str).str[4:].isin(['0331', '0630', '0930', '1231'])]
    p = p.drop_duplicates(subset=['ex', 'symbol', 'endDate'], keep='last')
    p['avail'] = available_date(p.endDate.values)
    return p.sort_values(['ex', 'symbol', 'endDate']).reset_index(drop=True)


def max_dd(x):
    eq = np.cumprod(1 + np.asarray(x))
    return float((eq / np.maximum.accumulate(eq) - 1).min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='C:/quant_data/nowcast2')
    ap.add_argument('--tag', default='_u1gv2')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', default='returns_eval')
    ap.add_argument('--suffix', default=None,
                    help='标签后缀，必须和该方案一致（u1/u2/u3）——否则测试行与预测错位')
    ap.add_argument('--scheme', choices=['A', 'B', 'C', 'D', 'E', 'F'], default='A',
                    help='A: 分子/(h·X[j-3]) 自相对 | B: 分子/市值 | C: 分子/总资产[j-3]')
    args = ap.parse_args()
    os.makedirs(os.path.join(HERE, args.out), exist_ok=True)

    exs = sorted(set(ASHARE_EX) | set(GLOBAL_EX))
    suf = args.suffix or {'A': 'u1', 'B': 'u2', 'C': 'u3', 'D': 'u4',
                          'E': 'u5', 'F': 'u6'}[args.scheme]
    store = Store(args.root, exs, min_per_date=5, label='level', suffix=suf)
    tr, va, te = load_split_global(store)
    print(f'测试集 {len(te):,} 行 / {len(set(store.symbol_of(te))):,} 只', flush=True)

    npz = np.load(os.path.join(HERE, 'results', f'pred_both{args.tag}_s{args.seed}.npz'),
                  allow_pickle=True)
    pred = npz['pred']
    assert len(pred) == len(te), f'{len(pred)} vs {len(te)}'

    # 标签的稳健尺度（与 train.py 同法重算，用于把预测还原到标签量纲）
    rng = np.random.RandomState(args.seed + 1)
    samp = np.sort(tr[rng.choice(len(tr), min(20000, len(tr)), replace=False)])
    Ys = store.take_y(samp)
    y_med = np.median(Ys, 0)
    y_scale = (np.percentile(Ys, 75, 0) - np.percentile(Ys, 25, 0)) / 1.349 + 1e-6
    # 预测的量纲判别（2026-09-26 起反标准化焼进了模型的 buffer，npz 里可能是两种之一）：
    #   * 新 .pt（带 out_scale/out_mean buffer）-> npz 已是【真实单位】-> 不能再换算
    #   * 旧 .pt（恒等 buffer / 无 buffer）  -> npz 是 z 空间 -> 需要 ×scale+median
    # 用离散度自动判别：z 空间的 std≈0.3~1，真实单位的 std≈标签尺度(0.01 量级)。
    if float(np.nanstd(pred)) > 0.1 * float(np.max(y_scale) + 1e-12):
        pred_lab = pred * y_scale + y_med          # 旧：z 空间 -> 真实单位
        print('  预测量纲：z 空间 -> 已换算回真实单位', flush=True)
    else:
        pred_lab = pred                            # 新：模型已输出真实单位
        print('  预测量纲：真实单位（反标准化已焼在模型里）', flush=True)

    full = store.symbol_of(te)
    T = pd.DataFrame({
        'ex': [s.split(':')[0] for s in full],
        'symbol': [s.split(':')[1] for s in full],
        'anchor': store.dates[te]})

    # ---- 合并面板（asof：最后一个【已披露】季度）----
    panel = build_panel(exs)
    by = {k: (g['avail'].values, g.index.values)
          for k, g in panel.groupby(['ex', 'symbol'], sort=False)}
    take = np.full(len(T), -1, dtype=np.int64)
    takec = np.full(len(T), -1, dtype=np.int64)          # D 用：最后【已结束】季度
    takeE = np.full(len(T), -1, dtype=np.int64)          # E 用：锚点【所在】季度
    # 必须在循环【外】声明 —— 放里面每组都会被重置，只有最后一组幸存（评分会全 NaN）
    for (ex, sym), idx in T.groupby(['ex', 'symbol']).indices.items():
        packed = by.get((ex, sym))
        if packed is None:
            continue
        av, rows = packed
        pos = np.searchsorted(av, T['anchor'].values[idx], side='right') - 1
        ok = pos >= 0
        take[idx[ok]] = rows[pos[ok]]
        ed = panel['endDate'].values[rows]              # 该符号的季度末（已按 endDate 排序）
        posc = np.searchsorted(ed, T['anchor'].values[idx], side='right') - 1
        okc = posc >= 0
        takec[idx[okc]] = rows[posc[okc]]
        # E：锚点【所在】的季度（第一个 endDate >= 锚点）+ 同一条"包含"校验
        posE = np.searchsorted(ed, T['anchor'].values[idx], side='left')
        okE = np.zeros(len(idx), bool)
        for q, (pp, aa) in enumerate(zip(posE, T['anchor'].values[idx])):
            if 0 <= pp < len(ed):
                ymE = (int(ed[pp]) // 10000) * 12 + (int(ed[pp]) // 100 % 100)
                ymA = (int(aa) // 10000) * 12 + (int(aa) // 100 % 100)
                okE[q] = 0 <= ymE - ymA <= 3
        takeE[idx[okE]] = rows[posE[okE]]
    ok = take >= 0
    oki = np.flatnonzero(ok)          # T 里的行位置 —— 预测数组是按测试行排的，必须用这个索引
    D = T.iloc[oki].reset_index(drop=True).copy()
    for c in ['endDate', 'grossProfit', 'weightedAverageShsOut']:
        D[c] = panel[c].values[take[ok]]
    P = panel[['ex', 'symbol', 'endDate', GP, 'weightedAverageShsOut', 'totalAssets']].copy()
    P['gp_m3'] = P.groupby(['ex', 'symbol'], sort=False)[GP].shift(3)      # GP[j-3]
    P['ta_m3'] = P.groupby(['ex', 'symbol'], sort=False)['totalAssets'].shift(3)  # 总资产[j-3]
    P['gp_m0'] = P[GP]                                # GP[jc]（D 用）
    P['ta_m0'] = P['totalAssets']                     # 总资产[jc]（D 用）
    # F（u6）用：median(GP[jc-6..jc]) —— 过去 7 个季度的毛利中位数，与 add_u6_labels.py 同口径
    P['gp_med0'] = (P.groupby(['ex', 'symbol'], sort=False)[GP]
                    .transform(lambda x: x.rolling(7, min_periods=7).median()))
    D = D.merge(P[['ex', 'symbol', 'endDate', 'gp_m3', 'ta_m3']],
                on=['ex', 'symbol', 'endDate'], how='left')
    # D：换成"最后已结束季度"那一行的值时（gp_m0/ta_m0）
    D['gp_m0'] = P['gp_m0'].values[takec[oki]]
    D['ta_m0'] = P['ta_m0'].values[takec[oki]]
    D['gp_med0'] = P['gp_med0'].values[takec[oki]]    # F 用
    # E：X[j-3] 与 TA[j-3]（j = 锚点所在季）——直接取 P 的 gp_m3/ta_m3 在该行的值
    _tE = np.where(takeE[oki] >= 0, takeE[oki], 0)
    D['gp_m3E'] = np.where(takeE[oki] >= 0, P['gp_m3'].values[_tE], np.nan)
    D['ta_m3E'] = np.where(takeE[oki] >= 0, P['ta_m3'].values[_tE], np.nan)
    if args.scheme == 'F':
        # ⚠️ u6 的 pred 三列是【字段】（gpMed7 / revMed7 / taMed7）而**不是**头 1 的三个期限，
        # 所以不能像 A~E 那样按列取期限 —— 评分只用第 0 列（毛利）。u6 标签只有一个期限 7，
        # 该分数对 1/3/7 季持有期都评一遍，**对齐的是 7 季**。
        for h in HORIZONS:
            D[f'pred{h}'] = pred_lab[oki, 0]
        print(f'  评分头 = {suf} 第 1 列（毛利 7 季中位数变化）；持有期对齐 7 季', flush=True)
    else:
        for i, h in enumerate(HORIZONS):
            D[f'pred{h}'] = pred_lab[oki, i]           # 头 1（gpDelta）的三个期限
    print(f'  面板合并后 {len(D):,} 行', flush=True)

    # ---- 前瞻收益 ----
    daily = load_daily_close(exs)
    rets = {h: np.full(len(D), np.nan) for h in HORIZONS}
    close_t = np.full(len(D), np.nan)
    for (ex, sym), idx in D.groupby(['ex', 'symbol']).indices.items():
        if (ex, sym) not in daily:
            continue
        dt, cl = daily[(ex, sym)]
        pos = np.searchsorted(dt, D['anchor'].values[idx], side='right') - 1
        good = pos >= 0
        ii = idx[good]
        close_t[ii] = cl[pos[good]]
        for h in HORIZONS:
            ep = pos + DAYS_PER_Q * h
            g2 = (pos >= 0) & (ep < len(cl))
            rr = np.full(len(idx), np.nan)
            rr[g2] = cl[ep[g2]] / cl[pos[g2]] - 1.0
            rets[h][idx[g2]] = rr[g2]     # g2 是 good 的子集（远期可能越界），要按 g2 取行
    D['close'] = close_t
    for h in HORIZONS:
        D[f'ret{h}'] = rets[h]
    D['mcap'] = D['weightedAverageShsOut'] * D['close']
    D = D[(D.mcap > 0) & D['gp_m3'].notna()]
    if args.scheme == 'F':
        D = D[D['gp_med0'].notna() & (D['ta_m0'] > 0)]
    # ---- 评分构造（三方案不同；统一反算成「预测的未来毛利和 / 市值」）----
    #  A: label=(ΣGP_fwd−h·GP[j-3])/(h·GP[j-3]) -> 预测ΣGP_fwd = h·GP[j-3]·(1+pred)
    #  B: label=(ΣGP_fwd−h·GP[j-3])/市值        -> 预测ΣGP_fwd = pred·市值 + h·GP[j-3]
    #  C: label=(ΣGP_fwd−h·GP[j-3])/总资产[j-3] -> 预测ΣGP_fwd = pred·总资产[j-3] + h·GP[j-3]
    # 免费基线三者相同：朴素预测"不变" -> ΣGP_fwd = h·GP[j-3] -> 评分 = h·GP[j-3]/市值
    for h in HORIZONS:
        if args.scheme == 'A':
            pred_gp = h * D['gp_m3'] * (1.0 + D[f'pred{h}'])
            base_gp = h * D['gp_m3']
        elif args.scheme == 'B':
            pred_gp = D[f'pred{h}'] * D['mcap'] + h * D['gp_m3']
            base_gp = h * D['gp_m3']
        elif args.scheme == 'C':
            pred_gp = D[f'pred{h}'] * D['ta_m3'] + h * D['gp_m3']
            base_gp = h * D['gp_m3']
        elif args.scheme == 'D':   # D：分子分母都用「最后【已结束】季度 jc」
            pred_gp = D[f'pred{h}'] * D['ta_m0'] + h * D['gp_m0']
            base_gp = h * D['gp_m0']
        elif args.scheme == 'E':   # E：j = 锚点所在季，X[j-3]/TA[j-3]
            pred_gp = D[f'pred{h}'] * D['ta_m3E'] + h * D['gp_m3E']
            base_gp = h * D['gp_m3E']
        else:   # F（u6）：label=(med7_fwd − med7_past)/总资产[jc]
            #   朴素预测"中位数不变" -> med7_fwd = med7_past -> 评分 = med7_past/市值（价值因子）
            pred_gp = D[f'pred{h}'] * D['ta_m0'] + D['gp_med0']
            base_gp = D['gp_med0']
        D[f'score{h}'] = pred_gp / D['mcap']
        D[f'base{h}'] = base_gp / D['mcap']
    D['qkey'] = D['anchor']
    print(f'  可用 {len(D):,} 行 / {D.qkey.nunique()} 个锚点日\n', flush=True)

    is_cn = D['ex'].isin(ASHARE_EX).values

    def ic_by_date(df, x, y, min_n=5):
        ics = []
        for _, g in df.groupby('qkey'):
            sub = g[[x, y]].dropna()
            if len(sub) < min_n:
                continue
            ra = sub[x].rank().values
            rb = sub[y].rank().values
            if ra.std() and rb.std():
                ics.append(np.corrcoef(ra, rb)[0, 1])
        s = pd.Series(ics).dropna()
        return (s.mean(), s.mean() / s.std() if len(s) > 1 and s.std() > 0 else np.nan, len(s))

    print('=' * 100)
    print('  判据 1：逐日截面 IC(评分, 未来收益)   —— 终点指标（不是对财务量的 IC）')
    print('=' * 100)
    print(f'  {"持有":>4s}{"模型IC":>10s}{"基线IC":>10s}{"超出":>9s}'
          f'{"A股模型":>10s}{"A股基线":>10s}{"A股超出":>10s}{"n日":>6s}')
    for h in HORIZONS:
        m = ic_by_date(D, f'score{h}', f'ret{h}')
        c = ic_by_date(D[is_cn], f'score{h}', f'ret{h}')
        b = ic_by_date(D, f'base{h}', f'ret{h}')
        bc = ic_by_date(D[is_cn], f'base{h}', f'ret{h}')
        print(f'  {h:>3d}季{m[0]:>+10.4f}{b[0]:>+10.4f}{m[0]-b[0]:>+9.4f}'
              f'{c[0]:>+10.4f}{bc[0]:>+10.4f}{c[0]-bc[0]:>+10.4f}{m[2]:>6d}')

    print()
    print('=' * 100)
    print('  判据 2：动量中性化后的 IC（排除"只是动量代理"）')
    print('=' * 100)
    # 在锚点日算长期动量（每股票、每个锚点）
    D['mom'] = np.nan
    for (ex, sym), idx in D.groupby(['ex', 'symbol']).indices.items():
        if (ex, sym) not in daily:
            continue
        dt, cl = daily[(ex, sym)]
        pos = np.searchsorted(dt, D['anchor'].values[idx], side='right') - 1
        g2 = pos >= 889
        mm = np.full(len(idx), np.nan)
        mm[g2] = cl[pos[g2]] / cl[pos[g2] - 889] - 1.0
        D.iloc[idx, D.columns.get_loc('mom')] = mm
    print(f'  {"持有":>4s}{"原始IC":>10s}{"动量中性IC":>12s}{"n日":>6s}')
    for h in HORIZONS:
        raw = ic_by_date(D, f'score{h}', f'ret{h}')
        neu = []
        for _, g in D.groupby('qkey'):
            sub = g[[f'score{h}', f'ret{h}', 'mom']].dropna()
            if len(sub) < 20:
                continue
            rs = sub[f'score{h}'].rank().values.astype(float)
            rm = sub['mom'].rank().values.astype(float)
            if rm.std() == 0:
                continue
            b = np.polyfit(rm, rs, 1)
            res = rs - (b[0] * rm + b[1])
            rt = sub[f'ret{h}'].rank().values
            if res.std() and rt.std():
                neu.append(np.corrcoef(res, rt)[0, 1])
        s = pd.Series(neu).dropna()
        print(f'  {h:>3d}季{raw[0]:>+10.4f}{s.mean():>+12.4f}{len(s):>6d}')

    print()
    print('=' * 100)
    print('  判据 3：组合（前 50%，中位数收益）')
    print('=' * 100)
    for h in HORIZONS:
        rows = []
        for _, g in D.groupby('qkey'):
            if len(g) < 20:
                continue
            for nm, col in [('score', f'score{h}'), ('base', f'base{h}')]:
                r = g[col].rank(ascending=False, method='first')
                hs = g.loc[r <= len(g) / 2.0, f'ret{h}']
                if len(hs) >= 10:
                    rows.append({'qkey': _, 'nm': nm, 'med': hs.median(),
                                 'neg': (hs < 0).mean()})
            rows.append({'qkey': _, 'nm': 'all', 'med': g[f'ret{h}'].median(),
                         'neg': (g[f'ret{h}'] < 0).mean()})
        a = pd.DataFrame(rows)
        print(f'\n  持有 {h} 季：')
        for nm, lab in [('score', '模型评分前50%'), ('base', '免费基线前50%'),
                        ('all', '全市场等权')]:
            x = a[a.nm == nm]
            print(f'    {lab:16s} 中位数 {x["med"].mean()*100:+.2f}%   '
                  f'负收益 {x["neg"].mean()*100:.0f}%   n={len(x)}')
    D.to_csv(os.path.join(HERE, args.out, 'eval_v2_detail.csv'), index=False)
    print(f'\n明细 -> {args.out}/eval_v2_detail.csv')


if __name__ == '__main__':
    main()
