"""`two` 的标签层：u9（meddiff）标签 + 采样用的 `ok` 门。**不依赖 nowcast/**。

## provenance（全是逐字移植；改任何一行都要重跑逐位验证）

* 面板与 meddiff 标签  ← `two/nowcast/add_u6_labels.py`
    - 常量 `FIELDS` / `TAG`                    ：第 30~32 行
    - `load_panel(..., form='meddiff')`        ：第 35~87 行（含两个 rolling 窗口）
    - `relabel()` 里「jc → row → (fwd−past)/TA[jc]」那段：第 121~151 行
    - 头名 = `TAG[字段] + 期限`（与 `labels.flat_names` 的「字段优先」展平一致）：第 106~107 行
* `ok` 门（u1 的 yoy_delta）← `two/nowcast/labels.py`
    - 行集与 `load_panel`                      ：第 70~82 行
    - `_clean`                                 ：第 88~91 行
    - `build_labels` 的 `yoy_delta` 分支       ：第 136~146 行
    - `gen_data2.get_labels` 用它当 `labok`    ：`two/nowcast/gen_data2.py` 第 69~86 行
* `available_date` ← `two_features`（与 `nowcast/gen_data2.available_date` 同一套规则）

## 为什么连 `ok` 也要搬过来（这是本文件最容易被误解的地方）

nowcast 那条线里，样本的 **(symbol, 锚点) 集合**不是 u9 决定的，而是 `gen_data2.py` 用
**u1 变体的 `ok`** 筛出来的（`get_labels` → `lab_sym['ok']` → `labok[k]`，ok=False 的锚点整段跳过）；
u9 的标签只是 `add_u6_labels.relabel()` **事后覆写** y/b/r 而已。所以：

    旧线行集 = gen_data2(u1).ok 门 ∩ (relabel 写出的 u9 标签有限)     ← 后者由 Store 的 labok 再筛一次

`two` 要一次成型、且行集与旧线**逐行可比**，这个 ok 门就必须一起搬。少搬它 -> 多出一批
u1.ok=False 的锚点，新数据与旧数据不可比、也无法做逐位验证。

## ⚠️ 两个口径**故意不同**，不要"统一"它们

* `jc`（u9 标签的基准季度）= 锚点日前最后一个**【已结束】**的季度（用 `endDate`，不看披露）。
  它的报告在锚点日可能尚未披露（A 股 Q2 → 8/31）→ **u9 标签本身含前视**，见
  `two/nowcast/variants.py` 的 u9 段。这是标签的定义，照搬。
* `ok` 门用的季度 = 最后一个**【已披露】**的季度（用 `available_date`）→ 无前视。

两者用不同的季度索引是有意的（前者是 `relabel` 的 `searchsorted(endDate)`，后者是
`gen_data2` 的 `searchsorted(avail)`）。
"""
import os

import numpy as np
import pandas as pd

from two_features import available_date       # 同一套披露日规则（DEADLINE 表）

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
DATA = os.path.join(ROOT, 'data')

# ---- u9 的参数（见 two/nowcast/variants.py 的 u9 段：前向 7 季、过去 3 季）----
FIELDS = ['grossProfit', 'revenue', 'totalAssets']
TAG = {'grossProfit': 'gpMed', 'revenue': 'revMed', 'totalAssets': 'taMed'}
HORIZONS = [7]
PAST_WIN = 3
# 头名（展平顺序 = 字段优先）。u9 的名字把两个窗口都编码进去了：
# 过去 P3 季 / 前向 F7 季 -> 'gpMedP3F'，与 two/nowcast/variants.py 的 u9 键名一字不差。
assert HORIZONS == [7] and PAST_WIN == 3, '头名硬编码了 u9 的 P3F，改窗口长度必须同步改这里'
HEAD_NAMES = [f'{TAG[c]}P{PAST_WIN}F' for c in FIELDS]
N_HEADS = len(HEAD_NAMES)

# ---- u1 的 ok 门（two/nowcast/labels.py 的模块级常量）----
OK_HORIZONS = [1, 3, 7]      # labels.py:HORIZONS —— 与 variant 无关，恒为 1/3/7
BACK = 3                     # labels.py:BACK（j-3 与 j+1 是同一个日历季）
CLIP = 127.0                 # labels.py:CLIP（与 three/seven 的标签口径一致）


def _clean(x, d):
    """x/d，要求分母为正；inf/nan 归 nan；再截断到 ±CLIP（逐字复制 labels.py:_clean）。"""
    v = (x / d).where(d > 0).replace([np.inf, -np.inf], np.nan)
    return v.clip(-CLIP, CLIP).astype('float32')


def load_panel(exchange):
    """公司 × 季度面板：meddiff 的 past/fwd 窗口 + u1 的 ok + 披露日 avail。

    （逐字移植 add_u6_labels.load_panel 的 meddiff 分支；ok 与 avail 是本文件新增的两列。）
    返回的面板是**完整**的（不做任何样本过滤）—— 过滤是采样时按 `ok` / 标签有限性做的，
    这是 2026-09-25 修过的一个 bug 的结论，见 add_u6_labels.load_panel 的 docstring。
    """
    ex = exchange.lower()
    inc = pd.read_csv(os.path.join(DATA, exchange.upper(), f'income_{ex}.csv'),
                      usecols=['symbol', 'endDate', 'grossProfit', 'revenue'])
    bal = pd.read_csv(os.path.join(DATA, exchange.upper(), f'balance_{ex}.csv'),
                      usecols=['symbol', 'endDate', 'totalAssets', 'totalStockholdersEquity'])
    p = inc.merge(bal, on=['symbol', 'endDate'], how='outer')
    p['symbol'] = p.symbol.astype(str)
    p = p[p.endDate.astype(str).str[4:].isin(['0331', '0630', '0930', '1231'])]
    p = p.drop_duplicates(subset=['symbol', 'endDate'], keep='last')
    p = p.sort_values(['symbol', 'endDate']).reset_index(drop=True)

    g = p.groupby('symbol', sort=False)
    for c in FIELDS:
        for h in HORIZONS:
            # 过去窗口 [jc-pw+1 .. jc] 的水平中位数
            p[f'past_{c}_{PAST_WIN}'] = g[c].transform(
                lambda x, pw=PAST_WIN: x.rolling(pw, min_periods=pw).median())
            # 未来窗口 [jc+1 .. jc+h]：先 shift(-1) 让 rolling 从下一季开始，再 shift(-(h-1)) 左移回来
            p[f'fwd_{c}_{h}'] = g[c].transform(
                lambda x, h=h: x.shift(-1).rolling(h, min_periods=h).median().shift(-(h - 1)))

    # 披露日（ok 门用）。avail 与 endDate 单调同序（DEADLINE 把 Q4 映射到次年 4/30，
    # 与次年 Q1 同值），所以按 endDate 排的面板可以直接用 avail 做 searchsorted。
    p['avail'] = available_date(p.endDate.values)
    p['ok'] = _ok_mask(p)
    return p


def _ok_mask(p):
    """u1（yoy_delta）的 `ok` —— 逐字移植 labels.py:build_labels 的 yoy_delta 分支。

        num = ΣX[j+1..j+h] − h·X[j−3]        den = h·X[j−3]
        b   = 0（bnum=0 / bden=1 -> 恒有限）

    fields = variants.fields_for('u1') = {gpDelta/revDelta/taDelta} -> 三个字段；
    期限恒为 labels.py 的 HORIZONS = [1,3,7]。所以 ok ⟺ 3 字段 × 3 期限的 y 全有限。
    """
    g = p.groupby('symbol', sort=False)
    ok = pd.Series(True, index=p.index)
    for c in FIELDS:
        for h in OK_HORIZONS:
            fwd = g[c].transform(
                lambda x, h=h: x.shift(-1).rolling(h, min_periods=h).sum().shift(-(h - 1)))
            hbase = g[c].shift(BACK) * h          # labels.py: hbase = g[col].shift(BACK) * h
            ok &= _clean(fwd - hbase, hbase).notna()
    return ok


def label_matrices(panel):
    """面板 -> (Y, B) 两个 (n_rows, 3) float32 —— 逐字等价于对每行跑 relabel() 的标签循环。

    Y[:, fi*nh + hi] = clamp((fwd_{c}_{h} − past_{c}_{pw}) / totalAssets[jc], ±127)
    B 与 nowcast 一致**恒为 0**（relabel 写的是 `np.zeros_like(Y)`）：
    meddiff 标签本身已经是"变化量"，它的朴素基准（"上一轮重演"）就是 0。
    """
    n = len(panel)
    Y = np.full((n, N_HEADS), np.nan, 'float32')
    ta = panel['totalAssets'].values.astype('float64')
    fwd = [panel[f'fwd_{c}_{h}'].values for c in FIELDS for h in HORIZONS]   # 字段优先
    past = [panel[f'past_{c}_{PAST_WIN}'].values for c in FIELDS for h in HORIZONS]
    good = np.isfinite(ta) & (ta > 0)                      # relabel: not (isfinite(ta) and ta > 0) -> continue
    for i in np.where(good)[0]:
        for k in range(len(FIELDS) * len(HORIZONS)):
            fv, pv = fwd[k][i], past[k][i]
            if np.isfinite(fv) and np.isfinite(pv):
                Y[i, k] = np.clip((fv - pv) / ta[i], -CLIP, CLIP)   # 赋进 float32 -> 与旧线同样的舍入
    return Y, np.zeros_like(Y)


def pack(panel):
    """面板 -> {symbol: (avail, endDate, ok, rows, Y, B)}，全是 numpy（采样时零 pandas 开销）。

    * `rows` 是该股票各季度在【面板】里的行号（endDate 升序）。
    * `Y`/`B` 是所有股票共享的整块矩阵，用 rows[pj] 索引即可。
    """
    Y, B = label_matrices(panel)
    out = {}
    for s, g in panel.groupby('symbol', sort=False):
        out[s] = (g['avail'].values.astype('int64'),
                  g['endDate'].values.astype('int64'),
                  g['ok'].values.astype(bool),
                  g.index.values.astype('int64'), Y, B)
    return out


def label_frame(y):
    """一个样本的标签 -> 1 行 3 列 DataFrame（列名与 two/nowcast/variants.py 的 u9 段一致）。

    **只写三个头**（2026-09-27 起）。以前还会写 `b_gpMedP3F/b_revMedP3F/b_taMedP3F` 和 `ok`
    两列，但这两个都是**常量列**，写进 h5 一点信息都没有：

    * `b_*` —— meddiff 口径下朴素基准恒为 0（见 `label_matrices` 的 docstring）。
    * `ok`  —— 生成端已经把 ok=False 的锚点整段丢掉了（`gen_train_data.chunk_worker`），
      所以落盘的行 `ok` 恒为 True；训练端 `two/train.py` 以前那句抽检也只是在确认这一点。

    170 万个文件 × 4 列 × 4B 的纯常量，删掉既省磁盘也省掉训练端"先把 ok 列读出来看一眼"
    的开销。其它四个模型（one/three/seven/zero）的 h5 里也没有这类常量列，格式上顺带对齐。
    """
    return pd.DataFrame([list(y)], columns=HEAD_NAMES)
