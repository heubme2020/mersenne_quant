"""
validate_label.py — zero 模型「换标签 / 换特征」前的数据验证（zero待改进.md 第五节）。

目标：用数据回答三个问题
  1. 候选标签里，哪个对「财务 ratio」可预测性最强（IC/ICIR 最高）？
     - mom        : 现有标签，price_fore - price_past（对称动量）
     - negMdd     : -未来最大回撤（越大越安全）
     - downRet    : min(0, 未来对数收益)
     - downVol    : 下行半波动（越大越差）
     - upVol      : 上行半波动
     - upMinusDown: upVol - downVol（收益不对称度）
     - asym       : (upVol - downVol)/(upVol + downVol)（归一化不对称度）
  2. 现有 7 个 ratio 之间有多冗余（相关矩阵），interestCoverage 负值/极端值多严重？
  3. 新增的 4 个比值特征（现金流覆盖 / 应计背离 / 商誉占比 / 应收占比）是否比现有 ratio 更能避雷？

输出：
  - zero/validate_ic_feature_x_label.csv   feature × label 的 IC / ICIR / t
  - zero/validate_corr_features.csv        特征相关矩阵
  - zero/validate_dist.csv                 各特征分布统计（含 interestCoverage 负值占比）

用法：
  python validate_label.py --exchanges AMEX,CNQ        # 快速子集
  python validate_label.py                              # 全部训练交易所
"""

import os
import math
import sys
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

# Windows 控制台默认 GBK，强制 UTF-8 输出，避免中文/emoji 乱码或报错
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

pd.set_option('future.no_silent_downcasting', True)

# --------------------------------------------------------------------------- #
# 配置（与 gen_train_data.py 保持一致）
# --------------------------------------------------------------------------- #
FEATURE_COLUMNS = [
    'debtRatio',
    'debtToEquity',
    'interestCoverage',
    'cashRatio',
    'quickRatio',
    'currentRatio',
    'inventoryTurnover',
]
NEW_FEATURES = [
    'ocfToLiab',        # 经营现金流 / 总负债（现金流覆盖）
    'accrual',          # (净利润 - 经营现金流) / 总资产（应计背离，抓利润造假）
    'goodwillToEquity', # 商誉 / 净资产（商誉暴雷）
    'arToRevenue',      # 应收账款 / 营收（收入造假）
]
LABEL_COLUMNS = [
    'mom', 'negMdd', 'downRet', 'downVol', 'upVol', 'upMinusDown', 'asym',
]

LOOKBACK_QUARTERS = 3
PRICE_WINDOW_DAYS = 127 * 3


# --------------------------------------------------------------------------- #
# 数据加载：把 indicator / income / balance / cashflow 合并，算新比值特征
# --------------------------------------------------------------------------- #
def load_financial(exchange, current_dir):
    upper = exchange[0].upper() + exchange[1:]
    base = os.path.join(current_dir, '..', 'data', upper)
    low = exchange.lower()

    indicator = pd.read_csv(os.path.join(base, f'indicator_{low}.csv'), encoding='utf-8')
    income = pd.read_csv(os.path.join(base, f'income_{low}.csv'), encoding='utf-8')
    balance = pd.read_csv(os.path.join(base, f'balance_{low}.csv'), encoding='utf-8')
    cashflow = pd.read_csv(os.path.join(base, f'cashflow_{low}.csv'), encoding='utf-8')

    bal = balance[['symbol', 'endDate', 'goodwill', 'totalStockholdersEquity',
                   'totalAssets', 'totalLiabilities', 'accountsReceivables']]
    inc = income[['symbol', 'endDate', 'revenue', 'netIncome']]
    cf = cashflow[['symbol', 'endDate', 'netCashProvidedByOperatingActivities']]

    fin = indicator.merge(bal, on=['symbol', 'endDate'], how='left')
    fin = fin.merge(inc, on=['symbol', 'endDate'], how='left')
    fin = fin.merge(cf, on=['symbol', 'endDate'], how='left')

    # 新增比值（在标准化之前显式计算，避免 z-score 打散比值关系）
    fin['ocfToLiab'] = fin['netCashProvidedByOperatingActivities'] / fin['totalLiabilities']
    fin['accrual'] = (fin['netIncome'] - fin['netCashProvidedByOperatingActivities']) / fin['totalAssets']
    fin['goodwillToEquity'] = fin['goodwill'] / fin['totalStockholdersEquity']
    fin['arToRevenue'] = fin['accountsReceivables'] / fin['revenue']

    fin.replace([np.inf, -np.inf], np.nan, inplace=True)
    return fin


# --------------------------------------------------------------------------- #
# 单个股票：生成样本（特征 + 候选标签），返回 DataFrame
# --------------------------------------------------------------------------- #
def build_symbol_samples(symbol, fin_group, dates, closes):
    """
    fin_group : 该股票按 endDate 升序的财务数据（含 7 旧 + 4 新特征）
    dates     : 该股票按 date 升序的交易日（int）
    closes    : 对应收盘价（float）
    """
    fin_group = fin_group.sort_values('endDate').reset_index(drop=True)
    feature_cols = FEATURE_COLUMNS + NEW_FEATURES
    n = len(dates)
    rows = []

    # 从第 LOOKBACK_QUARTERS-1 个季度开始（保证能取到 3 个季度输入）
    for j in range(LOOKBACK_QUARTERS - 1, len(fin_group)):
        end_date = int(fin_group['endDate'].iloc[j])

        # 用 searchsorted 定位「endDate 之后第一个交易日」的索引 i
        i = np.searchsorted(dates, end_date, side='right')
        if i + PRICE_WINDOW_DAYS > n:      # 未来交易日不足
            break
        if i < PRICE_WINDOW_DAYS:          # 历史交易日不足
            continue

        fore_closes = closes[i:i + PRICE_WINDOW_DAYS]
        past_closes = closes[i - PRICE_WINDOW_DAYS:i]
        close_now = closes[i - 1]           # 决策日收盘价

        if (fore_closes <= 0).any() or (past_closes <= 0).any():
            continue

        # ---- 候选标签（均在 log 空间） ----
        # 1) 现有动量标签：log(min)+log(median)+log(max) 之差
        mom = (math.log(fore_closes.min()) + math.log(float(np.median(fore_closes))) + math.log(fore_closes.max())
               - (math.log(past_closes.min()) + math.log(float(np.median(past_closes))) + math.log(past_closes.max())))

        # 2) 未来最大回撤（取负，越大越安全）
        running_max = np.maximum.accumulate(fore_closes)
        mdd = float(((running_max - fore_closes) / running_max).max())
        neg_mdd = -mdd

        # 3) 未来对数收益的下行部分
        fwd_ret = math.log(fore_closes[-1]) - math.log(close_now)
        down_ret = min(0.0, fwd_ret)

        # 4/5/6/7) 上行/下行半波动 & 不对称度
        p = np.concatenate([[close_now], fore_closes])
        r = np.diff(np.log(p))
        down_vol = math.sqrt(float(np.mean(np.minimum(r, 0.0) ** 2)))
        up_vol = math.sqrt(float(np.mean(np.maximum(r, 0.0) ** 2)))
        up_minus_down = up_vol - down_vol
        asym = (up_vol - down_vol) / (up_vol + down_vol + 1e-9)

        # ---- 特征（取最新一季度，点在当时截面） ----
        feat = fin_group.iloc[j][feature_cols].to_dict()

        row = {'symbol': symbol, 'endDate': end_date}
        row.update(feat)
        row.update({
            'mom': mom,
            'negMdd': neg_mdd,
            'downRet': down_ret,
            'downVol': down_vol,
            'upVol': up_vol,
            'upMinusDown': up_minus_down,
            'asym': asym,
        })
        rows.append(row)

    return pd.DataFrame(rows)


def run_exchange(exchange, current_dir):
    fin = load_financial(exchange, current_dir)
    daily_path = os.path.join(current_dir, '..', 'data',
                              exchange[0].upper() + exchange[1:],
                              f'daily_{exchange.lower()}.csv')
    daily = pd.read_csv(daily_path, encoding='utf-8')

    fin_groups = {sym: g for sym, g in fin.groupby('symbol', sort=False)}
    parts = []
    for sym, gdaily in tqdm(daily.groupby('symbol', sort=False), desc=exchange, leave=False):
        if sym not in fin_groups:
            continue
        gdaily = gdaily.sort_values('date')
        dates = gdaily['date'].values
        closes = gdaily['close'].values.astype('float64')
        df = build_symbol_samples(sym, fin_groups[sym], dates, closes)
        if len(df):
            parts.append(df)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------- #
# IC / 指标
# --------------------------------------------------------------------------- #
def per_date_ic(df, feat_col, label_col, min_stocks):
    def _ic(g):
        if len(g) < min_stocks:
            return np.nan
        return g[feat_col].rank().corr(g[label_col].rank())

    return df.groupby('endDate')[[feat_col, label_col]].apply(_ic).dropna()


def summarize(ic_series):
    s = ic_series.dropna()
    if len(s) == 0:
        return dict(ic=np.nan, icir=np.nan, t=np.nan, n=0)
    mean, std = float(s.mean()), float(s.std())
    return dict(ic=mean, icir=mean / std if std > 0 else np.nan,
                t=mean / std * math.sqrt(len(s)) if std > 0 else np.nan, n=len(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exchanges', default=None, help='逗号分隔，如 AMEX,CNQ；默认用 train_exchanges.csv')
    ap.add_argument('--min-stocks', type=int, default=10)
    args = ap.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))

    if args.exchanges:
        exchanges = [e.strip() for e in args.exchanges.split(',') if e.strip()]
    else:
        train_csv = os.path.join(current_dir, '..', 'train_exchanges.csv')
        exchanges = pd.read_csv(train_csv, encoding='utf-8')['exchange'].tolist()

    print(f"交易所：{exchanges}")
    frames = []
    for ex in exchanges:
        print(f"\n=== {ex} ===")
        df = run_exchange(ex, current_dir)
        print(f"样本数：{len(df)}")
        if len(df):
            frames.append(df)

    if not frames:
        print("没有样本。")
        return

    samples = pd.concat(frames, ignore_index=True)
    samples.to_csv(os.path.join(current_dir, 'validate_samples.csv'), index=False)
    print(f"\n总样本：{len(samples)}，endDate 范围 [{samples['endDate'].min()}, {samples['endDate'].max()}]")

    feature_cols = FEATURE_COLUMNS + NEW_FEATURES

    # ---- 1) feature × label 的 IC ----
    print(f"\n{'=' * 100}")
    print(f" feature × label 截面 Rank IC（min_stocks={args.min_stocks}）")
    print(f"{'=' * 100}")
    rows = []
    for fc in feature_cols:
        for lc in LABEL_COLUMNS:
            s = summarize(per_date_ic(samples, fc, lc, args.min_stocks))
            rows.append({'feature': fc, 'label': lc, **s})
    ic_df = pd.DataFrame(rows)
    ic_df.to_csv(os.path.join(current_dir, 'validate_ic_feature_x_label.csv'), index=False)

    # 打印：label 列 × feature 行
    pivot = ic_df.pivot(index='feature', columns='label', values='ic')
    pivot_icir = ic_df.pivot(index='feature', columns='label', values='icir')
    # 对每个 label，取 |IC| 最大的特征
    print("\n[IC] 行=特征，列=标签：")
    print(pivot.round(4).to_string())
    print("\n[ICIR] 行=特征，列=标签：")
    print(pivot_icir.round(3).to_string())

    print("\n每个标签的最强特征（按 |IC|）：")
    for lc in LABEL_COLUMNS:
        sub = ic_df[ic_df['label'] == lc]
        best = sub.loc[sub['ic'].abs().idxmax()]
        print(f"  {lc:<12} <- {best['feature']:<18} IC={best['ic']:+.4f}  ICIR={best['icir']:+.3f}  (n={best['n']})")

    # ---- 2) 特征相关矩阵 ----
    corr = samples[feature_cols].corr()
    corr.to_csv(os.path.join(current_dir, 'validate_corr_features.csv'))
    print(f"\n[特征相关矩阵]（7 旧 + 4 新）：\n{corr.round(3).to_string()}")

    # ---- 3) 分布统计 ----
    dist_rows = []
    for fc in feature_cols:
        s = samples[fc]
        dist_rows.append({
            'feature': fc,
            'n': int(s.notna().sum()),
            'nan_pct': round(float(s.isna().mean()), 4),
            'neg_pct': round(float((s < 0).mean()), 4),
            'zero_pct': round(float((s == 0).mean()), 4),
            'median': round(float(s.median()), 4),
            'p1': round(float(s.quantile(0.01)), 4),
            'p99': round(float(s.quantile(0.99)), 4),
        })
    dist = pd.DataFrame(dist_rows)
    dist.to_csv(os.path.join(current_dir, 'validate_dist.csv'), index=False)
    print(f"\n[特征分布]（neg_pct=负值占比，zero_pct=0占比，p1/p99=1%/99%分位）：")
    print(dist.to_string(index=False))

    print("\n完成。输出文件：validate_ic_feature_x_label.csv / validate_corr_features.csv / validate_dist.csv")


if __name__ == '__main__':
    main()
