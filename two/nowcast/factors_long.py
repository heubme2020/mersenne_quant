"""
factors_long.py — two 技术因子的「拉长尺度」版。

原 add_technical_factor 的窗口是 3/7/31 交易日（约 1 周~1.5 个月），
但我们要预测的是「季度级财务变化」（1Q≈63 天、3Q≈189 天、7Q≈440 天），尺度不匹配。
这里把窗口拉长：3→7、7→31、31→127 交易日（约 2 周~半年）。
"""

import numpy as np

# 24 个拉长后因子（顺序与 add_technical_factor_long 输出一致）
FACTORS = [
    'ma7', 'ma31', 'ma127',
    'rsi7', 'rsi127',
    'atr7', 'atr31',
    'obv7', 'obv31', 'obv127',
    'corr7', 'corr31', 'corr127',
    'curvature', 'vma_7_31', 'factor',
    'overnight7', 'overnight31', 'overnight127',
    'alpha15_wq', 'alpha128_gtja', 'alpha101_gtja',
    'aplha22_7_31', 'aplha22_31_127',
]

# 量纲无关（无需归一化）的因子
SCALE_FREE = {'rsi7', 'rsi127', 'corr7', 'corr31', 'corr127', 'vma_7_31',
              'alpha15_wq', 'alpha101_gtja', 'factor', 'curvature'}


def add_technical_factor_long(data):
    """窗口拉长版技术因子：3→7、7→31、31→127 交易日。"""
    data = data.sort_values('date') if 'date' in data.columns else data

    # --- 均线 ---
    data['ma7'] = data['close'].rolling(7).mean()
    data['ma31'] = data['close'].rolling(31).mean()
    data['ma127'] = data['close'].rolling(127).mean()

    # --- RSI ---
    delta = data['close'].diff()
    def calc_rsi(ser, period):
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        return (100 - (100 / (1 + (gain / (loss + 1e-6))))) * 0.01
    data['rsi7'] = calc_rsi(data['close'], 7)
    data['rsi127'] = calc_rsi(data['close'], 127)

    # --- 波动率 / 成交量 ---
    data['atr7'] = (data['delta'].rolling(7).mean()) * 127
    data['atr31'] = (data['delta'].rolling(31).mean()) * 127

    obv = delta * data['volume']
    data['obv7'] = (obv.rolling(7).mean()) * 127
    data['obv31'] = (obv.rolling(31).mean()) * 127
    data['obv127'] = (obv.rolling(127).mean()) * 127

    # --- 相关性 ---
    data['corr7'] = data['volume'].rolling(7).corr(data['close'])
    data['corr31'] = data['volume'].rolling(31).corr(data['close'])
    data['corr127'] = data['volume'].rolling(127).corr(data['close'])

    # --- curvature / vma ---
    data['curvature'] = (data['close'].diff().diff()) * 127
    data['vma_7_31'] = data['volume'].rolling(7).mean() / data['volume'].rolling(31).mean() - 1

    # --- factor ---
    data['factor'] = data['close'].pct_change(7) - data['volume'].rolling(31).std()

    # --- overnight ---
    overnight = data['open'] * data['close'] / data['close'].shift(1).replace(0, np.nan) - 1
    data['overnight7'] = overnight.rolling(7).mean() * 127
    data['overnight31'] = overnight.rolling(31).mean() * 127
    data['overnight127'] = overnight.rolling(127).mean() * 127

    # --- alpha15_wq（31→127、7→31） ---
    rk_h = data['high'].rolling(127).rank(pct=True)
    rk_v = data['volume'].rolling(127).rank(pct=True)
    inner_corr = rk_h.rolling(31).corr(rk_v)
    data['alpha15_wq'] = -1 * inner_corr.rolling(127).rank(pct=True)

    # --- alpha128_gtja（31→127、7→31） ---
    adv127 = data['volume'].rolling(127).mean()
    data['alpha128_gtja'] = -1 * (data['close'].diff(1) * (data['volume'] / (adv127 + 1e-6))).rolling(31).mean()

    # --- alpha101_gtja（31→127） ---
    low_127 = data['low'].rolling(127).min()
    high_127 = data['high'].rolling(127).max()
    data['alpha101_gtja'] = (data['close'] - low_127) / (high_127 - low_127 + 1e-6)

    # --- alpha22（3→7、7→31、31→127） ---
    rolling_corr_7 = (data['high'] * data['close']).rolling(7).corr(data['volume'])
    delta_corr_7 = rolling_corr_7.diff(7)
    std_close_31 = data['close'].rolling(31).std()
    data['aplha22_7_31'] = -1 * (delta_corr_7 * std_close_31) * 127

    rolling_corr_31 = (data['high'] * data['close']).rolling(31).corr(data['volume'])
    delta_corr_31 = rolling_corr_31.diff(31)
    std_close_127 = data['close'].rolling(127).std()
    data['aplha22_31_127'] = -1 * (delta_corr_31 * std_close_127) * 127

    return data
