"""计算现有 24 个技术因子的截面 ICIR（面板级），找出最差 7 个。"""
import os
import sys
import numpy as np
import pandas as pd

# 2026-09-28：本目录从 `<root>/factor_screen/` 搬到 `<root>/one/factor_screen/`，多上一层。
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 输出/读回都在本目录内（原来写 ROOT+'factor_screen'，搬进来后要指向自己）
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

HORIZONS = [1, 3, 7, 31, 127]


def load_panel():
    shz = pd.read_csv(os.path.join(ROOT, 'data/SHZ/daily_shz.csv'))
    shh = pd.read_csv(os.path.join(ROOT, 'data/SHH/daily_shh.csv'))
    df = pd.concat([shz, shh], ignore_index=True)
    df['delta'] = df['high'] - df['low']
    panel = {}
    for col in ['open', 'high', 'low', 'close', 'volume', 'delta']:
        panel[col] = df.pivot(index='date', columns='symbol', values=col).sort_index().astype(np.float32)
    return panel


def add_technical_factors(panel):
    """与 one_v2/add_technical_factor 完全一致，但面板级（date × symbol）。"""
    f = {}
    close = panel['close']
    high = panel['high']
    low = panel['low']
    volume = panel['volume']
    dlt = panel['delta']
    open_ = panel['open']

    f['ma3'] = close.rolling(3).mean()
    f['ma7'] = close.rolling(7).mean()
    f['ma31'] = close.rolling(31).mean()

    delta = close.diff()
    def calc_rsi(period):
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        return (100 - (100 / (1 + (gain / (loss + 1e-6))))) * 0.01
    f['rsi3'] = calc_rsi(3)
    f['rsi31'] = calc_rsi(31)

    f['atr3'] = (dlt.rolling(3).mean()) * 31
    f['atr7'] = (dlt.rolling(7).mean()) * 31

    obv = delta * volume
    f['obv3'] = (obv.rolling(3).mean()) * 31
    f['obv7'] = (obv.rolling(7).mean()) * 31
    f['obv31'] = (obv.rolling(31).mean()) * 31

    f['corr3'] = volume.rolling(3).corr(close)
    f['corr7'] = volume.rolling(7).corr(close)
    f['corr31'] = volume.rolling(31).corr(close)

    f['curvature'] = (close.diff().diff()) * 31
    f['vma_3_7'] = volume.rolling(3).mean() / volume.rolling(7).mean() - 1
    f['factor'] = (close.pct_change(3) - volume.rolling(7).std())

    overnight = open_ * close / close.shift(1).replace(0, np.nan) - 1
    f['overnight3'] = overnight.rolling(3).mean() * 31
    f['overnight7'] = overnight.rolling(7).mean() * 31
    f['overnight31'] = overnight.rolling(31).mean() * 31

    rk_h = high.rolling(10).rank(pct=True)
    rk_v = volume.rolling(10).rank(pct=True)
    inner_corr = rk_h.rolling(3).corr(rk_v)
    f['alpha15_wq'] = -1 * inner_corr.rolling(10).rank(pct=True)

    adv20 = volume.rolling(20).mean()
    f['alpha128_gtja'] = -1 * (close.diff(1) * (volume / (adv20 + 1e-6))).rolling(5).mean()

    low_30 = low.rolling(30).min()
    high_30 = high.rolling(30).max()
    f['alpha101_gtja'] = (close - low_30) / (high_30 - low_30 + 1e-6)

    rc3 = (high * close).rolling(3).corr(volume)
    f['aplha22_3_7'] = -1 * (rc3.diff(3) * close.rolling(7).std()) * 31
    rc7 = (high * close).rolling(7).corr(volume)
    f['aplha22_7_31'] = -1 * (rc7.diff(7) * close.rolling(31).std()) * 31

    return f


def compute_labels(close_panel):
    labels = {}
    for h in HORIZONS:
        base = close_panel.shift(-1)
        wmax = close_panel.rolling(h, min_periods=h).max().shift(-(h + 1))
        wmed = close_panel.rolling(h, min_periods=h).median().shift(-(h + 1))
        wmin = close_panel.rolling(h, min_periods=h).min().shift(-(h + 1))
        labels[h] = np.log(wmax) + np.log(wmed) + np.log(wmin) - 3.0 * np.log(base)
    return labels


def icir(factor, label):
    fr = factor.rank(axis=1, pct=True)
    lr = label.rank(axis=1, pct=True)
    fc = fr.sub(fr.mean(axis=1), axis=0)
    lc = lr.sub(lr.mean(axis=1), axis=0)
    denom = np.sqrt(fc.pow(2).sum(axis=1)) * np.sqrt(lc.pow(2).sum(axis=1))
    ic = (fc * lc).sum(axis=1) / denom
    ic = ic.dropna()
    if len(ic) < 20:
        return np.nan
    std = ic.std()
    return float(ic.mean() / std) if std > 1e-8 else np.nan


def main():
    panel = load_panel()
    labels = compute_labels(panel['close'])
    factors = add_technical_factors(panel)
    rows = []
    for name, fac in factors.items():
        icirs = [icir(fac, labels[h]) for h in HORIZONS]
        mean_icir = float(np.nanmean(icirs))
        row = {'factor': name, 'mean_icir': mean_icir}
        for h, v in zip(HORIZONS, icirs):
            row[f'icir_{h}'] = v
        rows.append(row)
    df = pd.DataFrame(rows).sort_values('mean_icir', key=abs)
    out = os.path.join(HERE, 'current_ic.csv')
    df.to_csv(out, index=False)
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
