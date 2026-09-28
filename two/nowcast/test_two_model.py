"""
test_two_model.py — 在非训练交易所上测试 two 模型的 IC / ICIR。

two 训练集是 A 股（SHZ/SHH），这里用其它交易所（US/欧洲/亚洲）做样本外测试。
label = gain_h = 31 * (log(max)+log(median)+log(min) - 3*log(close_tomorrow))，h ∈ {7,31,127}
（gain 对归一化不变，直接用原始 close 计算）。
IC = 逐日截面 Spearman；ICIR = mean(IC)/std(IC)。
"""

import os
import sys
import math
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'two')))
from two_model import TWO  # noqa: F401
from gen_train_data import add_technical_factor  # noqa: F401

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# 24 个技术因子（与 add_technical_factor 输出顺序一致）
SHORT_FACTORS = [
    'ma3', 'ma7', 'ma31', 'rsi3', 'rsi31', 'atr3', 'atr7',
    'obv3', 'obv7', 'obv31', 'corr3', 'corr7', 'corr31',
    'curvature', 'vma_3_7', 'factor', 'overnight3', 'overnight7', 'overnight31',
    'alpha15_wq', 'alpha128_gtja', 'alpha101_gtja', 'aplha22_3_7', 'aplha22_7_31',
]
# 31 特征列顺序（raw 归一化 6 + 技术因子 24 + idx 1）
FEATURE_COLS = ['open', 'low', 'high', 'close', 'volume', 'delta'] + SHORT_FACTORS + ['idx']

DAYS_INPUT = 889
EXCHANGES = ['SHZ', 'SHH']   # A 股（two 的训练市场）
N_SYMBOLS_PER_EX = 60
DAY_STEP = 21
HORIZONS = [7, 31, 127]


def load_two_model(device):
    # 2026-09-26：`two.pt` 已换成生产模型（nowcast 架构）；这里要的是旧 TWO 架构，
    # 权重已改名 `two_legacy.pt`。
    path = os.path.join(BASE, 'two', 'two_legacy.pt')
    return torch.load(path, map_location=device, weights_only=False).to(device).eval()


def build_sample(g, j):
    """从 symbol 日线 g 的第 j 天构造 two 输入 (889,31) 和 3 个 label。"""
    if j - DAYS_INPUT + 1 < 0 or j + 127 >= len(g):
        return None
    data = g.iloc[j - DAYS_INPUT + 1: j + 128].copy().reset_index(drop=True)  # 1016 天
    if len(data) != DAYS_INPUT + 127:
        return None
    ref_idx = DAYS_INPUT - 1
    ref_close = data['close'].iloc[ref_idx]
    ref_volume = data['volume'].iloc[ref_idx]
    if ref_close <= 0 or ref_volume <= 0:
        return None

    # 归一化（与 gen_train_data 一致）
    data['open'] = data['open'] / ref_close
    data['high'] = data['high'] / ref_close
    data['low'] = data['low'] / ref_close
    data['delta'] = data['high'] - data['low']
    data['close'] = data['close'] / ref_close
    data['volume'] = data['volume'] / ref_volume

    # 输入窗口（889 天，技术因子只在输入窗口上算）
    data_input = data.iloc[:DAYS_INPUT].copy()
    data_input = add_technical_factor(data_input)
    data_input['idx'] = data_input.index / (DAYS_INPUT - 1.0)
    data_input = data_input.drop(columns=['symbol', 'date'], errors='ignore')
    data_input = data_input.fillna(0).replace([np.inf, -np.inf], 0)
    data_input = data_input.clip(-127, 127)
    x = data_input[FEATURE_COLS].values.astype('float32')  # [889, 31]

    # label（原始 close，gain 归一化不变）
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_two_model(device)
    print(f'设备: {device}')

    frames = []
    for ex in EXCHANGES:
        daily = pd.read_csv(os.path.join(BASE, 'data', ex, f'daily_{ex.lower()}.csv'))
        daily = daily.sort_values(['symbol', 'date']).reset_index(drop=True)
        groups = list(daily.groupby('symbol'))
        rng = np.random.RandomState(42)
        idx = rng.choice(len(groups), min(N_SYMBOLS_PER_EX, len(groups)), replace=False)
        groups = [groups[i] for i in idx]

        rows = []
        for sym, g in groups:
            g = g.reset_index(drop=True)
            for j in range(DAYS_INPUT, len(g) - 127, DAY_STEP):
                s = build_sample(g, j)
                if s is None:
                    continue
                x, labels = s
                rows.append([g['date'].iloc[j], *labels, x])

        if not rows:
            print(f'{ex}: 无样本')
            continue

        dates = [r[0] for r in rows]
        labels = np.array([[r[1], r[2], r[3]] for r in rows], dtype=float)  # [N, 3]
        xs = np.stack([r[4] for r in rows])  # [N, 889, 31]

        # 跑模型（分 batch）
        preds = []
        with torch.no_grad():
            for b in range(0, len(xs), 128):
                xb = torch.tensor(xs[b:b+128], dtype=torch.float32, device=device)
                _, s7, s31, s127 = model(xb)
                preds.append(torch.stack([s7.squeeze(-1).squeeze(-1),
                                          s31.squeeze(-1).squeeze(-1),
                                          s127.squeeze(-1).squeeze(-1)], dim=1).cpu().numpy())
        preds = np.concatenate(preds)  # [N, 3]

        df = pd.DataFrame({'date': dates,
                           'l7': labels[:, 0], 'l31': labels[:, 1], 'l127': labels[:, 2],
                           'p7': preds[:, 0], 'p31': preds[:, 1], 'p127': preds[:, 2]})

        ic_rows = {}
        for h in HORIZONS:
            s = df.groupby('date').apply(lambda gg: gg[f'p{h}'].rank().corr(gg[f'l{h}'].rank())
                                         if len(gg) >= 5 else np.nan).dropna()
            ic_rows[f'{h}d'] = s
        frames.append(pd.DataFrame(ic_rows))
        print(f'{ex}: 样本 {len(df)}，截面 {frames[-1].shape[0]}')

    if not frames:
        print('无数据')
        return
    ic_all = pd.concat(frames, ignore_index=True)
    print(f'\n总截面数: {ic_all.shape[0]}')
    print('=' * 60)
    print('two 模型在非训练交易所的 IC / ICIR：')
    print(f"{'期限':<8}{'IC':>10}{'ICIR':>10}{'n_dates':>10}")
    for h in HORIZONS:
        s = ic_all[f'{h}d'].dropna()
        ic = s.mean()
        icir = ic / s.std() if s.std() > 0 else np.nan
        print(f'{h:>3}天{ic:>10.4f}{icir:>10.3f}{len(s):>10}')


if __name__ == '__main__':
    main()
