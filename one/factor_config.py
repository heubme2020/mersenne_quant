"""因子配置：baseline / plan1 / plan2 三套技术因子（模型保持 31 特征 = 7 底料 + 24 技术）。"""

# 当前 24 个技术因子（add_technical_factor 里的顺序）
CURRENT_24 = [
    'ma3', 'ma7', 'ma31', 'rsi3', 'rsi31', 'atr3', 'atr7',
    'obv3', 'obv7', 'obv31', 'corr3', 'corr7', 'corr31',
    'curvature', 'vma_3_7', 'factor',
    'overnight3', 'overnight7', 'overnight31',
    'alpha15_wq', 'alpha128_gtja', 'alpha101_gtja',
    'aplha22_3_7', 'aplha22_7_31',
]

# 【2026-09-14 定案】用海外市场数据贪心选出的 24 个（对 A 股纯样本外）。
# 实测：3 次集成 combined ICIR 0.5264 vs 现有 24 的 0.4534（+0.0731，P<0.001）。
# 见 one/实验结果_5win.md（2026-09-28 前在 one_5win/实验结果.md，那个目录已清理成 _trash）。
# 启用方式：FACTOR_MODE=new24（现已是默认）。
NEW_24 = ['std127', 'rsi127', 'mom31', 'atr7', 'mom1', 'rsv127', 'obv31', 'atr127', 'maspread3_7', 'mom127', 'corr31', 'std31', 'alpha15_wq', 'std7', 'mom3', 'rank127', 'maspread7_31', 'vma7_31', 'amihud31', 'maspread31_127', 'skew127', 'vma31_127', 'obv127', 'ma3']

# 最差 7 个（|ICIR| 最小，含退化 rsi3）
WORST_7 = ['rsi3', 'alpha15_wq', 'aplha22_3_7', 'aplha22_7_31', 'curvature', 'corr7', 'vma_3_7']

# 新因子（top 22，按 |ICIR| 排序，单股可算；academic_smb / gtja191_062 是截面算子已排除）
NEW_22 = [
    'gtja191_150', 'alpha101_042', 'gtja191_070', 'gtja191_095', 'alpha101_024',
    'gtja191_173', 'gtja191_121', 'gtja191_012', 'alpha101_094', 'gtja191_010',
    'qlib158_rsv60', 'gtja191_164', 'qlib158_ma60', 'gtja191_153', 'qlib158_resi30',
    'qlib158_min30', 'alpha101_013', 'gtja191_046', 'gtja191_167', 'gtja191_067',
    'gtja191_187', 'qlib158_resi20',
]

# plan1：换掉最差 7，换成 top 7
PLAN1_SWAP_IN = NEW_22[:7]
PLAN1_KEEP = [f for f in CURRENT_24 if f not in WORST_7]

# plan2：22 新 + 2 个最好的旧（obv31、ma3）
PLAN2_KEEP_OLD = ['obv31', 'ma3']

# plan1_ts：纯时序因子 top 7（去相关后，无截面算子，单股直接可算）
TIME_SERIES_TOP_7 = ['gtja191_150', 'gtja191_070', 'gtja191_095', 'alpha101_024',
                     'gtja191_173', 'qlib158_rsv60', 'gtja191_164']

# plan2_ts：去相关后的 23 个纯时序 + 1 补充 = 24
TIME_SERIES_TOP_24 = [
    'gtja191_150', 'gtja191_070', 'gtja191_095', 'alpha101_024', 'gtja191_173',
    'qlib158_rsv60', 'gtja191_164', 'academic_smb', 'qlib158_ma60', 'gtja191_153',
    'qlib158_resi30', 'qlib158_min30', 'gtja191_046', 'gtja191_167', 'gtja191_067',
    'gtja191_187', 'qlib158_rsv30', 'qlib158_min20', 'gtja191_059', 'gtja191_055',
    'gtja191_031', 'gtja191_144', 'qlib158_roc30', 'gtja191_132',
]


def get_technical_factors(mode):
    """返回该 mode 下的 24 个技术因子名。"""
    if mode == 'new24':
        return list(NEW_24)
    if mode == 'baseline':
        return list(CURRENT_24)
    if mode == 'plan1':
        return PLAN1_KEEP + PLAN1_SWAP_IN
    if mode == 'plan2':
        return NEW_22 + PLAN2_KEEP_OLD
    if mode == 'plan1_ts':
        return PLAN1_KEEP + TIME_SERIES_TOP_7
    if mode == 'plan2_ts':
        return TIME_SERIES_TOP_24
    raise ValueError(mode)
