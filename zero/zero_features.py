"""
zero_features.py — zero 模型的 7 个输入特征定义（训练 / 预测共用，保证对齐）。

由 validate_label.py 全量 IC 验证选出（见 zero标签与特征改造论证.md）：
  - 保留 4 个：interestCoverage / debtToEquity / debtRatio / cashRatio
  - 新增 3 个：ocfToLiab（现金流/总负债）、goodwillToEquity（商誉/净资产）、accrual（应计背离）

这 3 个新比值已集中到 indicator 表里计算（见 write_stock_data.py 的
write_exchange_indicator_data），zero 直接从这里截取列，无需再合并 income/balance/cashflow。
"""

FEATURE_COLUMNS = [
    'interestCoverage',
    'ocfToLiab',        # 经营现金流 / 总负债
    'debtToEquity',
    'goodwillToEquity', # 商誉 / 净资产
    'debtRatio',
    'accrual',          # (净利润 - 经营现金流) / 总资产
    'cashRatio',
]

LOOKBACK_QUARTERS = 3
PRICE_WINDOW_DAYS = 127 * 3
CLIP_VALUE = 127.0
