"""normalize.py — 特征截面 z-score 归一化（three 训练/推理共用，保证两边口径一致）。

mean/std 表（data/{EX}/mean_*.csv、std_*.csv）由 get_stock_data.py:get_mean_std_data 生成，
它只对「当季已披露股票数 ≥ 127」的季度写一行。所以某些季度（典型是最新几季、小交易所）在
表里没有对应行 —— 此时**无法**做 z-score。

契约：normalize_by_mean_std 要么把 data 完整归一化并返回它，要么返回 None（该季度的
mean/std 缺失/重复/NaN）。调用方拿到 None 必须【跳过】这个样本并告警/计数，绝不能像旧代码
那样默认 mean=0/std=1 静默喂原始值 —— 模型是在 z 空间特征上训练的，喂原始值会静默产出
垃圾预测，且没有任何报错。

⚠️ three/ 与 seven/ 各有一份，必须保持【逐字相同】：
refresh_candidates.py 把 three/ 排在 seven/ 之前加进 sys.path，所以 seven 的推理脚本被它
import 时，`from normalize import ...` 实际命中的是 three/ 这份（seven/ 那份只在
`python seven/gen_train_data.py` 这种直接运行时才被用到）。两份一旦漂移，seven 的每日推理
就会静默用到 three 的版本 —— 改任何一边都要同步另一边。
"""

import pandas as pd


def normalize_by_mean_std(data, endDate, mean_data, std_data):
    """按 endDate 对 data 的每一列做 (x−mean)/std；无法完整归一化时返回 None。

    * data：只含特征列的 DataFrame（symbol/endDate 已由调用方去掉），原地修改。
    * 只在 mean/std 列集里的列才归一化 —— indicator 的 42 列不在 mean/std 表里，保持原值，
      与历史口径一致。
    * 该 endDate 在 mean/std 里没有唯一一行（缺失或重复），或某个需归一化列的 mean/std 为
      NaN -> 返回 None。
    """
    m = mean_data[mean_data['endDate'] == int(endDate)]
    s = std_data[std_data['endDate'] == int(endDate)]
    if len(m) != 1 or len(s) != 1:
        return None
    m, s = m.iloc[0], s.iloc[0]
    mean_cols = set(mean_data.columns)
    for col in data.columns:
        if col not in mean_cols:
            continue
        mean_v, std_v = m[col], s[col]
        if pd.isna(mean_v) or pd.isna(std_v):
            return None
        if std_v != 0:
            data[col] = (data[col] - mean_v) / std_v
        else:
            data[col] = 0.0
    return data
