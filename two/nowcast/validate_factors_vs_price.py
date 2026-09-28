"""
validate_factors_vs_price.py — 对比「短周期因子(3/7/31)」vs「拉长因子(7/31/127)」对
two 原 label（未来 7/31/127 天股价变化）的 IC / ICIR。

label：未来 h 天 log 收益 = log(close[j+h]) - log(close[j])，h ∈ {7, 31, 127}
（two 原公式 log(median)+log(min)+log(max)-3log(close_now) 的 rank 等价简化版）。
IC：逐日截面 Spearman；ICIR = mean(IC) / std(IC)。
"""

import os
import sys
import math
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'two')))
from factors_long import add_technical_factor_long, FACTORS as LONG_FACTORS, SCALE_FREE as LONG_SCALE_FREE  # noqa
from gen_train_data import add_technical_factor  # noqa

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# 短周期因子（原 3/7/31）
SHORT_FACTORS = [
    'ma3', 'ma7', 'ma31', 'rsi3', 'rsi31', 'atr3', 'atr7',
    'obv3', 'obv7', 'obv31', 'corr3', 'corr7', 'corr31',
    'curvature', 'vma_3_7', 'factor', 'overnight3', 'overnight7', 'overnight31',
    'alpha15_wq', 'alpha128_gtja', 'alpha101_gtja', 'aplha22_3_7', 'aplha22_7_31',
]

EXCHANGES = ['NASDAQ', 'NYSE', 'AMEX', 'LSE', 'HKSE', 'JPX']
N_SYMBOLS_PER_EX = 80
HORIZONS = [7, 31, 127]   # 未来天数
DAY_STEP = 21             # 每 21 个交易日取一个截面


def load_daily(exchange):
    return pd.read_csv(os.path.join(BASE, 'data', exchange, f'daily_{exchange.lower()}.csv'))


def per_date_ic(df, col, label, min_stocks=5):
    """逐日截面 Spearman IC 序列。"""
    def _ic(g):
        if len(g) < min_stocks:
            return np.nan
        return g[col].rank().corr(g[label].rank())
    return df.groupby('date')[[col, label]].apply(_ic).dropna()


def main():
    # 每个交易所一个 IC 表（列 = factor_label 对），最后拼起来
    ic_frames = []

    for ex in EXCHANGES:
        daily = load_daily(ex)
        daily = daily.sort_values(['symbol', 'date']).reset_index(drop=True)
        groups = list(daily.groupby('symbol'))
        rng = np.random.RandomState(42)
        idx = rng.choice(len(groups), min(N_SYMBOLS_PER_EX, len(groups)), replace=False)
        groups = [groups[i] for i in idx]

        samples = []
        for sym, g in groups:
            g = g.reset_index(drop=True)
            if len(g) < 127 + 127 + 1:
                continue
            g['delta'] = g['high'] - g['low']
            short = add_technical_factor(g.copy())
            long = add_technical_factor_long(g.copy())
            for j in range(127, len(g) - 127, DAY_STEP):
                date = g['date'].iloc[j]
                c0 = g['close'].iloc[j]
                if c0 <= 0:
                    continue
                labs = [math.log(g['close'].iloc[j + h]) - math.log(c0) for h in HORIZONS]
                sv = [short[fn].iloc[j] for fn in SHORT_FACTORS]
                lv = [long[fn].iloc[j] for fn in LONG_FACTORS]
                samples.append([date, *labs, *sv, *lv])

        if not samples:
            continue
        cols = ['date', 'l7', 'l31', 'l127'] + [f's_{f}' for f in SHORT_FACTORS] + [f'g_{f}' for f in LONG_FACTORS]
        df = pd.DataFrame(samples, columns=cols)

        # 逐日 IC
        ic_rows = {}
        for f in SHORT_FACTORS:
            for hi, h in enumerate(HORIZONS):
                ic_rows[f's_{f}|{h}d'] = per_date_ic(df, f's_{f}', f'l{h}')
        for f in LONG_FACTORS:
            for hi, h in enumerate(HORIZONS):
                ic_rows[f'g_{f}|{h}d'] = per_date_ic(df, f'g_{f}', f'l{h}')
        ic_df = pd.DataFrame(ic_rows)
        ic_frames.append(ic_df)
        print(f'{ex}: 截面数 {ic_df.shape[0]}')

    if not ic_frames:
        print('无数据')
        return
    ic_all = pd.concat(ic_frames, ignore_index=True)

    print(f'\n总截面数（逐日）: {ic_all.shape[0]}')
    print('=' * 100)
    print('因子（短 vs 拉长）对股价变化的 IC / ICIR：')
    print(f"{'因子':<16}{'短 IC':>9}{'短 ICIR':>9}  |  {'拉长因子':<16}{'长 IC':>9}{'长 ICIR':>9}")

    # 配对展示：短 ma3 对应长 ma7 等（同名去掉数字，按顺序对齐）
    for si, f_short in enumerate(SHORT_FACTORS):
        f_long = LONG_FACTORS[si]  # 顺序一一对应（短 3/7/31 -> 长 7/31/127）
        # 用 127 天 label 作为主对比
        s_ic = ic_all[f's_{f_short}|127d'].mean()
        s_icir = ic_all[f's_{f_short}|127d'].mean() / ic_all[f's_{f_short}|127d'].std()
        g_ic = ic_all[f'g_{f_long}|127d'].mean()
        g_icir = ic_all[f'g_{f_long}|127d'].mean() / ic_all[f'g_{f_long}|127d'].std()
        print(f'{f_short:<16}{s_ic:>9.4f}{s_icir:>9.3f}  |  {f_long:<16}{g_ic:>9.4f}{g_icir:>9.3f}')

    print('\n--- 拉长因子里最强的（对 127 天 label）---')
    top = []
    for f in LONG_FACTORS:
        s = ic_all[f'g_{f}|127d']
        top.append((abs(s.mean()), s.mean(), s.mean() / s.std(), f))
    for abs_ic, ic, icir, f in sorted(top, reverse=True)[:8]:
        print(f'  {f:<16} IC={ic:+.4f} ICIR={icir:+.3f}')


if __name__ == '__main__':
    main()
