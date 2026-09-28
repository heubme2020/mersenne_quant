"""中位数窗口之差标签（u6 / u7）：3 个字段 × h 个期限。

    label_h = ( median(X[jc+1 .. jc+h]) − median(X[jc-h+1 .. jc]) ) / 总资产[jc]

jc = 锚点日前【最后一个已结束】的季度。锚点 7/21 -> jc = 2024Q2（6/30）。

* `u6`：只有 h=7（3 个头），用户 2026-09-25 指定。
* `u7`：h ∈ {1,3,7}（9 个头），用户 2026-09-25 指定 —— 把 u6 扩到和 u3 相同的头数，
  让 u3 / u6 / u7 三者在同等头数下可比。h=1 就是「未来一季度 − 刚过去的一季度」。

⚠️ **前视**：jc 的报告在锚点日可能尚未披露（7/21 时 Q2 要到 8/31）→ 有前视。
   对比时注意：u3 用的是最后【已披露】季度，**没有**这个问题；u6/u7 有。
   量级：h=7 时过去窗口是 7 个季度、只有 jc 一个未披露，中位数受影响小；
   但 h=1 的过去窗口就是 [jc] 本身，整个都依赖未披露数据 —— 各期限受影响程度不同。

用法：
  # u6（3 个头）
  python two/nowcast/add_u6_labels.py --root C:/quant_data/nowcast3 --exchanges SHZ ... --suffix u6 --horizons 7
  # u7（9 个头）
  python two/nowcast/add_u6_labels.py --root C:/quant_data/nowcast3 --exchanges SHZ ... --suffix u7 --horizons 1 3 7
  # 只跑一只做回归/体检
  python two/nowcast/add_u6_labels.py --root C:/quant_data/nowcast3 --exchanges SHZ --suffix u6 --horizons 7 --verify
"""
import argparse, glob, json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE)

FIELDS = ['grossProfit', 'revenue', 'totalAssets']
# 字段名 -> 头名前缀；最终头名 = 前缀 + 期限（与 labels.flat_names 的「字段优先」展平一致）
TAG = {'grossProfit': 'gpMed', 'revenue': 'revMed', 'totalAssets': 'taMed'}


def load_panel(exchange, horizons, form='meddiff', past_win=None):
    """公司 × 季度面板（**完整**，不做任何样本过滤）。

    form='meddiff'（u6/u7）：带 past/fwd 的 h 季「水平」中位数 —— 见 relabel 的取值处。
    form='yoymed' （u8）  ：带 yoymed_{c} —— 未来 h 季【同比变化】的中位数：
        毛利/营收：r_q = X_q / TA_q（当季总资产），标签 = median_q (r_q − r_{q−4})
        总资产    ：标签 = median_q (TA_q − TA_{q−4}) / TA_{q−4}  （教科书 asset growth 的中位版）
        **比值本身已无量纲，同比差分又消了季节 → 不需要再除以任何东西。**

    ⚠️ 2026-09-25 修 bug：这里**曾经**用 `L.build_labels(p, u1).ok` 过滤面板来「对齐样本行」。
    那是错的 —— u1 的 ok 要求「未来 7 季完整」，于是每只股票**最后 7 个季度被滤掉**；
    锚点日落在最后 7 季时 `searchsorted` 就退回更早的季度，于是
      (a) 本该 NaN 的最近期锚点被「凭空造出」标签（SHZ 实测 10.61% 的行），
      (b) 造出来的标签，其未来窗口有一大截落在锚点日**之前**（部分已实现）→ 系统性抬高 IC。
    现在直接用完整面板做 asof：jc = 真正的「最后一个已结束季度」；窗口不完整的行自然 NaN，
    由 `data.Store` 的 labok（要求 y/b/r 全有限）丢掉。样本行数会因此少约 10%，这是对的。
    """
    ex = exchange.lower()
    inc = pd.read_csv(os.path.join(ROOT, 'data', exchange.upper(), f'income_{ex}.csv'),
                      usecols=['symbol', 'endDate', 'grossProfit', 'revenue'])
    bal = pd.read_csv(os.path.join(ROOT, 'data', exchange.upper(), f'balance_{ex}.csv'),
                      usecols=['symbol', 'endDate', 'totalAssets', 'totalStockholdersEquity'])
    p = inc.merge(bal, on=['symbol', 'endDate'], how='outer')
    p['symbol'] = p.symbol.astype(str)
    p = p[p.endDate.astype(str).str[4:].isin(['0331', '0630', '0930', '1231'])]
    p = p.drop_duplicates(subset=['symbol', 'endDate'], keep='last')
    p = p.sort_values(['symbol', 'endDate']).reset_index(drop=True)
    g = p.groupby('symbol', sort=False)
    if form == 'yoymed':
        ta = p['totalAssets']
        for c in FIELDS:
            if c == 'totalAssets':
                lag = g[c].shift(4)                       # TA_{q-4}
                s = (p[c] - lag) / lag.where(lag > 0)
            else:
                r = (p[c] / ta).where(np.isfinite(p[c] / ta))
                s = r - r.groupby(p['symbol'], sort=False).shift(4)   # r_q − r_{q−4}
            for h in horizons:
                # 未来窗口 [jc+1 .. jc+h] 的中位数
                p[f'yoymed_{c}_{h}'] = s.groupby(p['symbol'], sort=False).transform(
                    lambda x, h=h: x.shift(-1).rolling(h, min_periods=h).median().shift(-(h - 1)))
        return p
    # meddiff：过去窗口长度 pw 默认 = 前向期限 h（u6/u7），可用 past_win 覆盖（u9：前向 7、过去 3）
    for c in FIELDS:
        for h in horizons:
            pw = past_win or h
            gg = g[c]
            p[f'past_{c}_{pw}'] = gg.transform(
                lambda x: x.rolling(pw, min_periods=pw).median())
            # 未来窗口 [jc+1 .. jc+h]：先 shift(-1) 让 rolling 从下一季开始，再 shift(-(h-1)) 把它左移回来
            p[f'fwd_{c}_{h}'] = gg.transform(
                lambda x: x.shift(-1).rolling(h, min_periods=h).median().shift(-(h - 1)))
    return p


def _q_end(yyyymmdd, k):
    """把 YYYYMMDD 的季度末往后推 k 个季度。"""
    y, m = yyyymmdd // 10000, yyyymmdd // 100 % 100
    q = (m - 1) // 3 + k
    y += q // 4
    m = q % 4 * 3 + 3
    return y * 10000 + m * 100 + {3: 31, 6: 30, 9: 30, 12: 31}[m]


def relabel(exchange, root, suffix, horizons, verify=False, form='meddiff', past_win=None):
    ex_dir = os.path.join(root, exchange)
    if not os.path.isdir(ex_dir):
        print(f'[{exchange}] 跳过'); return
    syms = json.load(open(os.path.join(ex_dir, 'symbols.json')))
    panel = load_panel(exchange, horizons, form, past_win)
    by = {s: (g['endDate'].values, g.index.values) for s, g in panel.groupby('symbol', sort=False)}
    names = [f'{TAG[c]}{h}' for c in FIELDS for h in horizons] if form == 'meddiff' \
        else [f'{TAG[c]}Yoy{h}' for c in FIELDS for h in horizons]
    nf, nh = len(FIELDS), len(horizons)
    print(f'[{exchange}] 头名 {names}  股票 {len(by):,}', flush=True)
    n_lab = n_leak = 0
    for xf in sorted(glob.glob(os.path.join(ex_dir, 'chunk_*_X.npy'))):
        base = xf[:-6]
        dd = np.load(base + '_d.npy'); ss = np.load(base + '_s.npy')
        n = len(dd)
        Y = np.full((n, nf * nh), np.nan, 'float32')
        for si in np.unique(ss):
            sym = syms[si] if si < len(syms) else None
            packed = by.get(sym)
            if packed is None: continue
            ed, rows = packed
            idxs = np.where(ss == si)[0]
            # jc = 锚点日前最后一个【已结束】的季度（在完整面板上取 —— 就是定义本身）
            pos = np.searchsorted(ed, dd[idxs], side='right') - 1
            for ii, pj in zip(idxs, pos):
                if pj < 0: continue
                jc = int(ed[pj])
                # 完整性自检：未来窗口【末行】必须晚于锚点日，否则就是泄漏。
                # 必须用【面板实际行】ed[pj+h] 而不是日历推算的季度末 —— 非 A 股市场的
                # 财报不严格按季度，面板有跳档，日历口径会大量误报（XETRA 实测误报 2120 个）。
                # pj+h 越界时 fwd 必为 NaN（rolling 要满 h 行）→ 不产生标签，无需检查。
                for h in horizons:
                    n_lab += 1
                    if pj + h < len(ed) and int(ed[pj + h]) <= int(dd[ii]):
                        n_leak += 1
                row = panel.iloc[rows[pj]]
                if form == 'yoymed':
                    # 标签 = 未来 h 季【同比变化】的中位数，本身已无量纲 → 不再除以任何东西
                    for fi, c in enumerate(FIELDS):
                        for hi, h in enumerate(horizons):
                            v = row[f'yoymed_{c}_{h}']
                            if np.isfinite(v):
                                Y[ii, fi * nh + hi] = np.clip(v, -127, 127)
                    continue
                ta = row['totalAssets']
                if not (np.isfinite(ta) and ta > 0): continue
                for fi, c in enumerate(FIELDS):
                    for hi, h in enumerate(horizons):
                        pw = past_win or h
                        fv, pv = row[f'fwd_{c}_{h}'], row[f'past_{c}_{pw}']
                        if np.isfinite(fv) and np.isfinite(pv):
                            Y[ii, fi * nh + hi] = np.clip((fv - pv) / ta, -127, 127)
        if verify:
            print(f'  {os.path.basename(base)}  n={n}  非空 {int(np.isfinite(Y).all(1).sum()):,}')
            continue
        np.save(base + f'_y{suffix}.npy', Y)
        np.save(base + f'_b{suffix}.npy', np.zeros_like(Y))
        np.save(base + f'_r{suffix}.npy', Y)
        print(f'[{exchange}] {os.path.basename(base)}  n={n}  非空 {int(np.isfinite(Y).all(1).sum()):,}', flush=True)
    print(f'[{exchange}] 泄漏自检：{n_lab:,} 个标签位，未来窗口末季落在锚点日前 = {n_leak} '
          f'{"OK" if n_leak == 0 else "FAIL"}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='C:/quant_data/nowcast3')
    ap.add_argument('--exchanges', nargs='+', required=True)
    ap.add_argument('--suffix', default='u6', help='输出后缀：u6 / u7 / u8')
    ap.add_argument('--horizons', nargs='+', type=int, default=[7])
    ap.add_argument('--past-win', type=int, default=None,
                    help='meddiff 的【过去】窗口长度；默认 = 前向期限。u9 = 前向 7 季、过去 3 季 -> 传 3')
    ap.add_argument('--form', choices=['meddiff', 'yoymed'], default='meddiff',
                    help='meddiff=水平中位数之差 ÷TA[jc]（u6/u7）| '
                         'yoymed=未来 h 季【同比变化】的中位数，无量纲（u8）')
    ap.add_argument('--verify', action='store_true',
                    help='只做体检（打印头名/非空率/泄漏自检），不写文件')
    a = ap.parse_args()
    for ex in a.exchanges:
        relabel(ex, a.root, a.suffix, a.horizons, verify=a.verify, form=a.form,
                past_win=a.past_win)


if __name__ == '__main__':
    main()
