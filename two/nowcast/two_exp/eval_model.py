"""
eval_model.py — 在 held-out A股测试股票上评估 two 模型的 IC/ICIR。

用法：
  python eval_model.py --model <模型路径> --test <test_symbols.txt> --factor short|long
"""

import os
import sys
import math
import argparse
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'two')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from two_model import TWO  # noqa: F401
from gen_train_data import add_technical_factor      # noqa: F401
from factors_long import add_technical_factor_long   # noqa: F401

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

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
DAYS_INPUT = 889
DAY_STEP = 63
HORIZONS = [7, 31, 127]


def build_sample(g, factor_fn, factor_names, j):
    if j - DAYS_INPUT + 1 < 0 or j + 127 >= len(g):
        return None
    data = g.iloc[j - DAYS_INPUT + 1: j + 128].copy().reset_index(drop=True)
    if len(data) != DAYS_INPUT + 127:
        return None
    ref_idx = DAYS_INPUT - 1
    ref_close = data['close'].iloc[ref_idx]
    ref_volume = data['volume'].iloc[ref_idx]
    if ref_close <= 0 or ref_volume <= 0:
        return None
    data['open'] = data['open'] / ref_close
    data['high'] = data['high'] / ref_close
    data['low'] = data['low'] / ref_close
    data['delta'] = data['high'] - data['low']
    data['close'] = data['close'] / ref_close
    data['volume'] = data['volume'] / ref_volume

    data_input = data.iloc[:DAYS_INPUT].copy()
    data_input = factor_fn(data_input)
    data_input['idx'] = data_input.index / (DAYS_INPUT - 1.0)
    data_input = data_input.drop(columns=['symbol', 'date'], errors='ignore')
    data_input = data_input.fillna(0).replace([np.inf, -np.inf], 0).clip(-127, 127)
    cols = FEATURE_BASE + factor_names + ['idx']
    x = data_input[cols].values.astype('float32')

    raw_close = g['close'].values
    close_tomorrow = raw_close[j + 1]
    if close_tomorrow <= 0:
        return None
    labels = []
    for h in HORIZONS:
        fore = raw_close[j + 1: j + 1 + h]
        gain = (math.log(fore.max()) + math.log(float(np.median(fore)))
                + math.log(fore.min()) - 3 * math.log(close_tomorrow)) * 31.0
        labels.append(gain)
    return x, labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--test', required=True)
    p.add_argument('--factor', required=True, choices=['short', 'long'])
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(args.model, map_location=device, weights_only=False).to(device).eval()

    factor_fn = add_technical_factor if args.factor == 'short' else add_technical_factor_long
    factor_names = SHORT_FACTORS if args.factor == 'short' else LONG_FACTORS

    # 读测试符号，按交易所分组
    sym_by_ex = {}
    with open(args.test) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex, sym = line.split(':', 1)
            sym_by_ex.setdefault(ex, set()).add(sym)

    rows = []
    for ex, syms in sym_by_ex.items():
        daily = pd.read_csv(os.path.join(BASE, 'data', ex, f'daily_{ex.lower()}.csv'))
        daily = daily[daily['symbol'].isin(syms)].sort_values(['symbol', 'date'])
        for sym, g in daily.groupby('symbol'):
            g = g.reset_index(drop=True)
            for j in range(DAYS_INPUT, len(g) - 127, DAY_STEP):
                s = build_sample(g, factor_fn, factor_names, j)
                if s is None:
                    continue
                x, labels = s
                rows.append([g['date'].iloc[j], *labels, x])

    if not rows:
        print('无测试样本')
        return

    dates = [r[0] for r in rows]
    labels = np.array([[r[1], r[2], r[3]] for r in rows], dtype=float)
    xs = np.stack([r[4] for r in rows])

    preds = []
    with torch.no_grad():
        for b in range(0, len(xs), 128):
            xb = torch.tensor(xs[b:b+128], dtype=torch.float32, device=device)
            _, s7, s31, s127 = model(xb)
            preds.append(torch.stack([s7.flatten(), s31.flatten(), s127.flatten()], dim=1).cpu().numpy())
    preds = np.concatenate(preds)

    df = pd.DataFrame({'date': dates, 'l7': labels[:, 0], 'l31': labels[:, 1], 'l127': labels[:, 2],
                       'p7': preds[:, 0], 'p31': preds[:, 1], 'p127': preds[:, 2]})
    print(f'测试样本: {len(df)}')
    print('=' * 60)
    print('two 模型（held-out A股测试）IC / ICIR：')
    print(f"{'期限':<8}{'IC':>10}{'ICIR':>10}{'n_dates':>10}")
    for h in HORIZONS:
        s = df.groupby('date').apply(lambda gg: gg[f'p{h}'].rank().corr(gg[f'l{h}'].rank())
                                     if len(gg) >= 5 else np.nan).dropna()
        ic = s.mean()
        icir = ic / s.std() if s.std() > 0 else np.nan
        print(f'{h:>3}天{ic:>10.4f}{icir:>10.3f}{len(s):>10}')


if __name__ == '__main__':
    main()
