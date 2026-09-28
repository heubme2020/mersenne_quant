"""nowcast 标签（v2）：每个 (symbol, 季度 j) 产出 9 个 level 标签 + 9 个基准 + 9 个残差。

## 三个输出头（2026-09-23 定案，见 two/nowcast/phase1_label_v2.md）

    grossProfit   = Σ grossProfit[j+1 .. j+h] / totalAssets[j]        <- 前瞻毛利率（盈利性）
    revenueGrowth = Σ revenue[j+1 .. j+h] / Σ revenue[j-h+1 .. j]     <- 前瞻营收增长（规模）
    assetGrowth   = totalAssets[j+h] / totalAssets[j]                 <- 前瞻资产增长（投资）

分母统一用**总资产**：恒正，所以不再有「分母 ≤0 → 丢样本」的问题
（旧版用净资产锚，EBIT ≤0 丢 17.6%、OCF ≤0 丢 29.9%）。相比旧版 `{ebit, revenue, ocf}`，
换掉了 OCF —— 价格因子对它的**增量** IC 只有 −0.043（≈0），是三个头里唯一没有增量信息的。

头 1 用「毛利/总资产」而不是 EBIT：换算出前瞻估值倍数时用 `毛利/市值`（市值恒正，
而 EV 在净现金公司会变负、EBIT 有 17.6% 为负）。从毛利反推不出 EBIT ——
中间要减掉的期间费用（margin）恰恰是最难预测的部分。

## 窗口约定（关键：label 只用未来，benchmark 只用过去，两者不重叠）

`j` = 锚点日 t 之前最后一个已结束的季度（endDate ≤ t）。**所有分母和基准都只用 ≤ j 的数据。**

    label    y = 只用未来窗口（j+1 .. j+h）
    bench    b = 同一形状、紧邻其前的一个窗口（j-h+1 .. j）

两者窗口**没有重叠**，所以 b 是 y 的干净对照（"上一轮重演"），不会把 y 里已知的部分
偷偷塞进 b 里。**这一点上一版写错过**：当时以 j-3 为基点、窗口取 [j-3, j+h-4]，
在 h>4 时会伸进未来季度（前视）；而存量项直接用 [j-3, j+h] 会让「已经实现的变化」
占到标签的 99%（实测 corr(y,b)=0.993），标签几乎完全被预先决定，预测变得无意义。

## 为什么同时产出「朴素基准」

Phase 0b 扫描发现：**「上一轮重演」这个基准（锚点日即可算出、完全不用价格）
在所有字段上都打得过价格因子**（h=3：营收/总资产 0.200 vs 基准 0.828；EBIT 0.280 vs 0.525）。
所以「模型预测的 IC 很漂亮」根本不能说明它有价值 —— 必须和基准同口径比，
并且看 **残差**上的增量 IC。`b_*` 就是给这个对照用的，不是标签。

    y_<head><h>   level 标签（水平/前瞻值）
    b_<head><h>   朴素基准（锚点日已知，不用价格）
    r_<head><h>   = y − b，增量标签
"""

import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

# name -> (table, column, form)
#   form : level_flow  = 前瞻流量 / 总资产[j]
#          growth      = 前瞻流量 / 尾部流量
#          level_stock = 存量[j+h] / 存量[j]
FIELDS = {
    'grossProfit':   ('income',  'grossProfit', 'level_flow'),
    'revenueGrowth': ('income',  'revenue',     'growth'),
    'assetGrowth':   ('balance', 'totalAssets', 'level_stock'),
}
HORIZONS = [1, 3, 7]
EQ = 'totalStockholdersEquity'
TA = 'totalAssets'
BACK = 3          # 同比基准往回推几个季度（j-3 与 j+1 是同一个日历季度）
MIN_Q = 2 * max(HORIZONS) + 1     # 建基准所需的最少季度数（growth 要回看 2h 个季度）


def flat_names(fields=None, horizons=None):
    """展平顺序：字段优先。必须与 model.FIELDS / gen_data 的 y 数组顺序一致。"""
    return [f'{n}{h}' for n in (fields if fields is not None else FIELDS)
            for h in (horizons if horizons is not None else HORIZONS)]


def load_panel(exchange):
    """公司 × 季度的财务面板（income + balance）。"""
    ex = exchange.lower()
    inc = pd.read_csv(os.path.join(ROOT, 'data', exchange.upper(), f'income_{ex}.csv'),
                      usecols=['symbol', 'endDate', 'revenue', 'grossProfit', 'ebit'])
    bal = pd.read_csv(os.path.join(ROOT, 'data', exchange.upper(), f'balance_{ex}.csv'),
                      usecols=['symbol', 'endDate', TA, EQ])
    p = inc.merge(bal, on=['symbol', 'endDate'], how='outer')
    p = p[p.symbol.notna() & p.endDate.notna()]
    p['symbol'] = p.symbol.astype(str)
    p = p[p.endDate.astype(str).str[4:].isin(['0331', '0630', '0930', '1231'])]
    p = p.drop_duplicates(subset=['symbol', 'endDate'], keep='last')
    return p.sort_values(['symbol', 'endDate']).reset_index(drop=True)


CLIP = 127.0     # 与 three/seven 的标签口径一致（both 都按 ±127 截断）


def _clean(x, d):
    """x/d，要求分母为正；inf/nan 归 nan；再截断到 ±CLIP。"""
    v = (x / d).where(d > 0).replace([np.inf, -np.inf], np.nan)
    return v.clip(-CLIP, CLIP).astype('float32')


def build_labels(panel, fields=None):
    """对每个 (symbol, j) 算 9 个 y + 9 个 b + 9 个 r + 一个 validity 掩码。

    fields 显式传入时用它（供 `variants.py` 的头 1/头 2 对照实验共用同一份 X）。
    """
    fields = FIELDS if fields is None else fields
    g = panel.groupby('symbol', sort=False)
    out = panel[['symbol', 'endDate']].copy().rename(columns={'endDate': 'j_endDate'})
    ok = pd.Series(True, index=out.index)
    ta_now = panel[TA]                                   # 总资产[j]
    eq_now = panel[EQ]                                   # 净资产[j]（同为锚点日已知）

    def fwd_sum(col, h):
        """Σ col[j+1 .. j+h]（未来窗口）。"""
        return g[col].transform(
            lambda x: x.shift(-1).rolling(h, min_periods=h).sum().shift(-(h - 1)))

    def back_sum(col, h):
        """Σ col[j-h+1 .. j]（紧邻其前的同长度窗口）。"""
        return g[col].transform(
            lambda x: x.rolling(h, min_periods=h).sum())

    def back_sum2(col, h):
        """Σ col[j-2h+1 .. j-h]（再往前一个同长度窗口）。"""
        return g[col].transform(
            lambda x: x.rolling(h, min_periods=h).sum().shift(h))

    for name, (_, col, form) in fields.items():
        for h in HORIZONS:
            if form == 'level_flow':
                num, den = fwd_sum(col, h), ta_now
                bnum, bden = back_sum(col, h), ta_now
            elif form == 'level_flow_eq':
                # 同 level_flow，只把分母从总资产换成**净资产**（隔离分母选择）
                num, den = fwd_sum(col, h), eq_now
                bnum, bden = back_sum(col, h), eq_now
            elif form == 'growth':
                num, den = fwd_sum(col, h), back_sum(col, h)
                bnum, bden = back_sum(col, h), back_sum2(col, h)
            elif form == 'level_stock':
                num, den = g[col].shift(-h), panel[col]
                bnum, bden = panel[col], g[col].shift(h)
            elif form == 'yoy_delta':
                # 用户 2026-09-23 定的形式：
                #   label = ( Σ_{q=j+1}^{j+h} X[q] − h × X[j-3] ) / ( h × X[j-3] )
                # = 「未来 h 季的水平之和」相对「去年同期那一季水平 × h」的变化率。
                # j-3 与 j+1 相差 4 季 -> 同一个日历季度，天然消季节性。
                # 注意：总资产是**时点存量**，所以 fwd_sum 对它是「未来 h 个季末水平之和」。
                # 基准（朴素预测"不变"）恒为 0 -> 没有截面基准，增量就是全部信号。
                hbase = g[col].shift(BACK) * h
                num, den = fwd_sum(col, h) - hbase, hbase
                bnum = pd.Series(0.0, index=panel.index)
                bden = pd.Series(1.0, index=panel.index)
            elif form == 'growth_signed':
                # 自相对，但分母取 |·| —— 这样**不丢负分母样本**、且符号正确：
                #   (fwd − back)/|back|: 亏损收窄 -> 正，盈利下滑 -> 负
                # 对比 growth 形式（fwd/back）：分母 ≤0 时会被 `_clean` 的 `.where(d>0)` 丢掉，
                # 而分母为负时 fwd/back 会把「变好」和「变差」映射到同一个值。
                # 对恒正科目（毛利、营收）两者只差一个常数 1，rank 完全等价。
                bk = back_sum(col, h)
                num, den = fwd_sum(col, h) - bk, bk.abs()
                bk2 = back_sum2(col, h)
                bnum, bden = bk - bk2, bk2.abs()
            else:
                raise ValueError(form)

            y = _clean(num, den)
            b = _clean(bnum, bden)
            out[f'y_{name}{h}'] = y
            out[f'b_{name}{h}'] = b
            out[f'r_{name}{h}'] = (y - b).astype('float32')            # 两种标签口径共用同一批样本，X 才能复用
            ok &= y.notna() & b.notna()

    out['ok'] = ok
    return out
