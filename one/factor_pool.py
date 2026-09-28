"""one 的因子候选池：把每个因子族扩到 1/3/7/31/127 五个窗口。

与 one/gen_train_data.py 的约定一致：因子在**已归一化**的窗口上计算
（价格 ÷ 窗口末日的 close、成交量 ÷ 窗口末日的 volume），所以所有因子都是尺度无关的。

池子 = 50 个新候选；现有 24 个（one/factor_config.CURRENT_24）另行计算，作为基线对照。
"""

import numpy as np
import pandas as pd

# 项目尺度阶梯：127 日 ≈ 一个季度
LADDER = [1, 3, 7, 31, 127]
WIN = [3, 7, 31, 127]          # 需要 ≥2 点的族用这四档
WIN_BIG = [7, 31, 127]         # 需要长窗口才稳定的族
PAIRS = [(3, 7), (7, 31), (31, 127)]


def _rsi(close, period):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    return (100 - (100 / (1 + (gain / (loss + 1e-6))))) * 0.01


def add_pool_factors(d):
    """在归一化窗口 d 上加 50 个候选因子。返回新帧。"""
    c, h, l, o, v = d['close'], d['high'], d['low'], d['open'], d['volume']
    rng = h - l                       # 与 one 的 'delta' 同义
    ret = c.pct_change()
    amount = (c * v).replace(0, np.nan)

    # 1) 动量（五个窗口全给）
    for W in LADDER:
        d[f'mom{W}'] = c / c.shift(W) - 1

    # 2) 均线偏离
    for W in WIN:
        d[f'ma{W}'] = c.rolling(W).mean() / c - 1

    # 3) 波动率
    for W in WIN:
        d[f'std{W}'] = c.rolling(W).std() / c

    # 4) RSI
    for W in WIN:
        d[f'rsi{W}'] = _rsi(c, W)

    # 5) 振幅
    for W in WIN:
        d[f'atr{W}'] = rng.rolling(W).mean()

    # 6) OBV（量价方向）
    for W in WIN:
        d[f'obv{W}'] = (rng * v).rolling(W).mean()

    # 7) 量价相关
    for W in WIN:
        d[f'corr{W}'] = v.rolling(W).corr(c)

    # 8) 隔夜收益
    on = o * c / c.shift(1).replace(0, np.nan) - 1
    for W in WIN:
        d[f'overnight{W}'] = on.rolling(W).mean()

    # 9) 量比
    for A, B in PAIRS:
        d[f'vma{A}_{B}'] = v.rolling(A).mean() / v.rolling(B).mean() - 1

    # 10) 区间位置 RSV
    for W in WIN_BIG:
        lo, hi = l.rolling(W).min(), h.rolling(W).max()
        d[f'rsv{W}'] = (c - lo) / (hi - lo + 1e-9)

    # 11) 收盘价在自身窗口内的分位
    for W in WIN_BIG:
        d[f'rank{W}'] = c.rolling(W).rank(pct=True)

    # 12) 均线价差
    for A, B in PAIRS:
        d[f'maspread{A}_{B}'] = (c.rolling(A).mean() - c.rolling(B).mean()) / c

    # 13) 偏度
    for W in [31, 127]:
        d[f'skew{W}'] = ret.rolling(W).skew()

    # 14) Amihud 非流动性
    for W in WIN_BIG:
        d[f'amihud{W}'] = (ret.abs() / amount).rolling(W).mean()

    return d


def pool_names():
    """池子里 50 个新因子的名字（顺序与 add_pool_factors 一致）。"""
    n = []
    n += [f'mom{W}' for W in LADDER]
    n += [f'ma{W}' for W in WIN]
    n += [f'std{W}' for W in WIN]
    n += [f'rsi{W}' for W in WIN]
    n += [f'atr{W}' for W in WIN]
    n += [f'obv{W}' for W in WIN]
    n += [f'corr{W}' for W in WIN]
    n += [f'overnight{W}' for W in WIN]
    n += [f'vma{A}_{B}' for A, B in PAIRS]
    n += [f'rsv{W}' for W in WIN_BIG]
    n += [f'rank{W}' for W in WIN_BIG]
    n += [f'maspread{A}_{B}' for A, B in PAIRS]
    n += [f'skew{W}' for W in [31, 127]]
    n += [f'amihud{W}' for W in WIN_BIG]
    return n
