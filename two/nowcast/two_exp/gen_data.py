"""
gen_data.py — two 任务数据生成（优化版）。

一次遍历日线，同时生成短因子和长因子两套 h5（省掉重复读 CSV）。

用法：
  python gen_data.py --out-short <dir> --out-long <dir> --symbols <file>
  --symbols 每行 "EXCHANGE:SYMBOL"，只生成这些股票的数据。
"""

import os
import sys
import math
import random
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'two')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gen_train_data import add_technical_factor      # noqa: F401  短因子
from factors_long import add_technical_factor_long   # noqa: F401  长因子

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

DAYS_INPUT = 127 * 7
DAYS_OUTPUT = 127
SUBSAMPLE = 127   # 每 127 个窗口保留 1 个

SHORT_FACTORS = [
    'ma3', 'ma7', 'ma31', 'rsi3', 'rsi31', 'atr3', 'atr7',
    'obv3', 'obv7', 'obv31', 'corr3', 'corr7', 'corr31',
    'curvature', 'vma_3_7', 'factor', 'overnight3', 'overnight7', 'overnight31',
    'alpha15_wq', 'alpha128_gtja', 'alpha101_gtja', 'aplha22_3_7', 'aplha22_7_31',
]
LONG_FACTORS = [
    'ma7', 'ma31', 'ma127', 'rsi7', 'rsi127', 'atr7', 'atr31',
    'obv7', 'obv31', 'obv127', 'corr7', 'corr31', 'corr127',
    'curvature', 'vma_7_31', 'factor', 'overnight7', 'overnight31', 'overnight127',
    'alpha15_wq', 'alpha128_gtja', 'alpha101_gtja', 'aplha22_7_31', 'aplha22_31_127',
]
FEATURE_BASE = ['open', 'low', 'high', 'close', 'volume', 'delta']


def build_features(data_input, data_fore, factor_fn, factor_names):
    """对输入/输出窗口分别算因子，拼接成 1016×31，返回 float32 DataFrame。"""
    di = factor_fn(data_input.copy())
    df = factor_fn(data_fore.copy())
    data = pd.concat([di, df], axis=0).reset_index(drop=True)
    data['idx'] = data.index / (DAYS_INPUT - 1.0)
    data.drop(columns=['symbol', 'date'], inplace=True, errors='ignore')
    data = data.fillna(0).replace([np.inf, -np.inf], 0).clip(-127, 127)
    return data[FEATURE_BASE + factor_names + ['idx']].astype('float32')


def gen_symbol(g, sym, out_short, out_long):
    g = g.sort_values('date').reset_index(drop=True)
    g = g.iloc[:-DAYS_OUTPUT].reset_index(drop=True)
    if len(g) < DAYS_INPUT + DAYS_OUTPUT:
        return 0
    ref_idx = DAYS_INPUT - 1
    n = 0
    for j in range(DAYS_INPUT, len(g) - DAYS_OUTPUT):
        if random.random() >= 1.0 / SUBSAMPLE:
            continue
        data = g.iloc[j - DAYS_INPUT + 1: j + DAYS_OUTPUT + 1].copy().reset_index(drop=True)
        if len(data) != DAYS_INPUT + DAYS_OUTPUT:
            continue
        ref_close = data['close'].iloc[ref_idx]
        ref_volume = data['volume'].iloc[ref_idx]
        if ref_close <= 0 or ref_volume <= 0:
            continue
        data['open'] = data['open'] / ref_close
        data['high'] = data['high'] / ref_close
        data['low'] = data['low'] / ref_close
        data['delta'] = data['high'] - data['low']
        data['close'] = data['close'] / ref_close
        data['volume'] = data['volume'] / ref_volume

        # 127 天 gain 过滤
        close_tomorrow = data['close'].iloc[DAYS_INPUT]
        fore = data['close'].iloc[DAYS_INPUT: DAYS_INPUT + DAYS_OUTPUT]
        close_127 = (math.log(fore.median()) + math.log(fore.min()) + math.log(fore.max())
                     - 3 * math.log(close_tomorrow))
        if abs(close_127) > 127:
            continue

        data_input = data.iloc[:DAYS_INPUT]
        data_fore = data.iloc[DAYS_INPUT:]
        date = g['date'].iloc[j]

        short_df = build_features(data_input, data_fore, add_technical_factor, SHORT_FACTORS)
        long_df = build_features(data_input, data_fore, add_technical_factor_long, LONG_FACTORS)
        short_df.to_hdf(os.path.join(out_short, f'{sym}_{date}.h5'), key='data', mode='w')
        long_df.to_hdf(os.path.join(out_long, f'{sym}_{date}.h5'), key='data', mode='w')
        n += 1
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out-short', required=True)
    p.add_argument('--out-long', required=True)
    p.add_argument('--symbols', required=True)
    args = p.parse_args()

    os.makedirs(args.out_short, exist_ok=True)
    os.makedirs(args.out_long, exist_ok=True)

    sym_by_ex = {}
    with open(args.symbols) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex, sym = line.split(':', 1)
            sym_by_ex.setdefault(ex, set()).add(sym)

    total = 0
    for ex, syms in sym_by_ex.items():
        daily_path = os.path.join(BASE, 'data', ex, f'daily_{ex.lower()}.csv')
        if not os.path.exists(daily_path):
            print(f'跳过 {ex}（无日线数据）')
            continue
        daily = pd.read_csv(daily_path, usecols=['symbol', 'date', 'open', 'low', 'high', 'close', 'volume'],
                            engine='pyarrow')
        daily = daily[daily['symbol'].isin(syms)]  # 保持原有顺序（已按 symbol/date 排序）
        n_ex = 0
        for sym, g in daily.groupby('symbol', sort=False):
            n_ex += gen_symbol(g, sym, args.out_short, args.out_long)
        total += n_ex
        print(f'{ex}: {n_ex} 个样本', flush=True)

    print(f'\n总样本数: {total}，短因子目录 {args.out_short}，长因子目录 {args.out_long}', flush=True)


if __name__ == '__main__':
    main()
