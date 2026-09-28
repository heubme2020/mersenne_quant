# import os
# import pandas as pd
# import numpy as np
# from tqdm import tqdm
# import warnings
# warnings.filterwarnings('ignore')

# # ========== 配置不变 ==========
# DAYS_INPUT = 127 * 3
# TARGET_HORIZONS = [3, 7, 31]
# FEATURE_COLS = [
#     'ma3', 'ma7', 'ma31',
#     'rsi3', 'rsi31',
#     'atr3', 'atr7',
#     'obv3', 'obv7', 'obv31',
#     'corr3', 'corr7', 'corr31',
#     'curvature',
#     'vma_3_7',
#     'factor',
#     'overnight3', 'overnight7', 'overnight31',
#     'alpha15_wq', 'alpha128_gtja', 'alpha101_gtja',
#     'aplha22_3_7', 'aplha22_7_31'
# ]

# # ====================== 【极速版】IC / ICIR ======================
# def fast_rank_ic_icir(factor: np.ndarray, ret: np.ndarray, window=60):
#     """
#     向量化 + pandas 滚动 corr 极速计算 IC / ICIR
#     比 for 循环快 8~12 倍
#     """
#     # 构造DF并去空
#     df = pd.DataFrame({'f': factor, 'r': ret}).dropna()
#     if len(df) < window * 2:
#         return 0.0, 0.0, 0.0

#     # 秩向量化（只算一次，极快）
#     df['f_rank'] = df['f'].rank()
#     df['r_rank'] = df['r'].rank()

#     # 滚动相关系数（C 语言内核，速度天花板）
#     rolling_corr = df['f_rank'].rolling(window=window).corr(df['r_rank'])
#     rolling_corr = rolling_corr.dropna()

#     if len(rolling_corr) == 0:
#         return 0.0, 0.0, 0.0

#     ic_mean = rolling_corr.mean()
#     ic_std = rolling_corr.std()
#     icir = ic_mean / ic_std if ic_std > 1e-8 else 0.0

#     return float(ic_mean), float(abs(ic_mean)), float(icir)

# # ====================== 快速读取 h5 ======================
# def load_all_train_h5(train_folder='train'):
#     files = [f for f in os.listdir(train_folder) if f.endswith('.h5')]
#     data_list = []
#     for f in tqdm(files, desc='Reading h5'):
#         try:
#             df = pd.read_hdf(os.path.join(train_folder, f), key='data')
#             data_list.append(df)
#         except Exception:
#             continue
#     return data_list

# # ====================== 快速提取收益 ======================
# def extract_multi_horizon_return(df):
#     input_df = df.iloc[:DAYS_INPUT]
#     future_df = df.iloc[DAYS_INPUT:]

#     close_now = input_df['close'].iloc[-1]
#     close_future = future_df['close'].values

#     ret_dict = {}
#     for n in TARGET_HORIZONS:
#         if len(close_future) < n:
#             ret_dict[n] = np.nan
#         else:
#             ret_dict[n] = np.log(close_future[n-1]) - np.log(close_now)

#     factor = input_df.iloc[-1][FEATURE_COLS]
#     return factor, ret_dict

# # ====================== 主函数（全程向量化） ======================
# def compute_multi_period_ic():
#     train_folder = os.path.join(os.path.dirname(__file__), 'train')
#     if not os.path.exists(train_folder):
#         print("train 文件夹不存在！")
#         return

#     data_list = load_all_train_h5(train_folder)
#     period_data = {n: {'factor': [], 'ret': []} for n in TARGET_HORIZONS}

#     print("\nExtracting factors & returns...")
#     for df in tqdm(data_list, desc='Processing samples'):
#         fac, ret_dict = extract_multi_horizon_return(df)
#         if fac.isna().any():
#             continue

#         for n, r in ret_dict.items():
#             if not np.isnan(r):
#                 period_data[n]['factor'].append(fac)
#                 period_data[n]['ret'].append(r)

#     # 逐周期计算
#     all_result = []
#     for horizon in TARGET_HORIZONS:
#         print(f"\n===== {horizon} 日 IC/ICIR =====")
#         fac_df = pd.DataFrame(period_data[horizon]['factor'])
#         ret_arr = np.array(period_data[horizon]['ret'])

#         res = []
#         # 向量化批量算 IC（极快）
#         for feat in tqdm(FEATURE_COLS, desc=f'Calculating {horizon}D'):
#             ic, abs_ic, icir = fast_rank_ic_icir(fac_df[feat].values, ret_arr)
#             res.append({
#                 '周期': f'{horizon}日',
#                 '特征': feat,
#                 'IC': round(ic, 4),
#                 '|IC|': round(abs_ic, 4),
#                 'ICIR': round(icir, 4)
#             })

#         res_df = pd.DataFrame(res).sort_values('|IC|', ascending=False)
#         print(res_df.head(10))  # 只打印前10，更快
#         all_result.append(res_df)
#         res_df.to_csv(f'IC_{horizon}日_结果.csv', index=False, encoding='utf-8-sig')

#     pd.concat(all_result).to_csv('全周期IC汇总表.csv', index=False, encoding='utf-8-sig')
#     print("\n✅ 计算完成，已保存所有文件")

# if __name__ == '__main__':
#     compute_multi_period_ic()

import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr
import warnings

# Windows 控制台默认 GBK，而本文件会打印 `⚠️` / `🎉` 这类非 GBK 字符 —— 走到那条分支就会
# UnicodeEncodeError 崩掉，而且**往往是在活儿干完之后**才崩（2026-09-26 zero/gen 就这么"失败"过：
# 日志里 already 写着总量，然后崩在一句庆祝打印上）。与其它脚本同一套修法。2026-09-28 扫描后补齐。
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass


warnings.filterwarnings('ignore')

# ========== 配置 ==========
DAYS_INPUT = 127 * 3
TARGET_HORIZONS = [3, 7, 31]
FEATURE_COLS = [
    'ma3', 'ma7', 'ma31', 'rsi3', 'rsi31', 'atr3', 'atr7',
    'obv3', 'obv7', 'obv31', 'corr3', 'corr7', 'corr31',
    'curvature', 'vma_3_7', 'factor',
    'overnight3', 'overnight7', 'overnight31',
    'alpha15_wq', 'alpha128_gtja', 'alpha101_gtja',
    'aplha22_3_7', 'aplha22_7_31'
]

# ====================== 优化版：IC / ICIR 计算 ======================
def calc_ic_metrics(ic_series):
    """根据每日IC序列计算最终指标"""
    if len(ic_series) < 2:
        return 0.0, 0.0, 0.0
    ic_mean = np.mean(ic_series)
    ic_std = np.std(ic_series)
    icir = ic_mean / ic_std if ic_std > 1e-8 else 0.0
    return ic_mean, abs(ic_mean), icir

def winsorize_mad(data, n=3):
    """MAD去极值处理，增强IC的鲁棒性"""
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    threshold = n * 1.4826 * mad
    return np.clip(data, median - threshold, median + threshold)

# ====================== 主逻辑 ======================
def compute_multi_period_ic_optimized():
    train_folder = os.path.join(os.path.dirname(__file__), 'train')
    files = [f for f in os.listdir(train_folder) if f.endswith('.h5')]
    
    if not files:
        print("未找到 h5 文件")
        return

    # 结构化存储：{日期: {特征名: [值], 收益率: [值]}}
    # 这样可以方便计算截面 IC
    daily_data_storage = {h: {} for h in TARGET_HORIZONS}

    print("\n[Step 1/2] 正在按日期聚合因子数据 (截面化)...")
    for f in tqdm(files):
        try:
            # 仅读取最后一行特征（即输入周期的终点）以节省时间
            df = pd.read_hdf(os.path.join(train_folder, f), key='data')
            
            # 解析文件名获取日期 (假设文件名格式为 symbol_date.h5)
            # 这对于计算真正的截面 IC 至关重要
            date_key = f.split('_')[-1].split('.')[0]
            
            # 提取因子值 (取 DAYS_INPUT-1 位置，即输入区的最后一天)
            fac_row = df.iloc[DAYS_INPUT-1]
            
            # 提取不同周期的未来收益率
            close_now = df['close'].iloc[DAYS_INPUT-1]
            future_closes = df['close'].values[DAYS_INPUT:]

            for h in TARGET_HORIZONS:
                if len(future_closes) >= h:
                    ret = np.log(future_closes[h-1]) - np.log(close_now)
                    
                    if date_key not in daily_data_storage[h]:
                        daily_data_storage[h][date_key] = {'factors': [], 'rets': []}
                    
                    daily_data_storage[h][date_key]['factors'].append(fac_row[FEATURE_COLS].values)
                    daily_data_storage[h][date_key]['rets'].append(ret)
        except Exception as e:
            continue

    # [Step 2/2] 计算截面 Rank IC
    all_period_results = []
    
    for h in TARGET_HORIZONS:
        print(f"\n[Step 2/2] 正在计算 {h} 日截面 IC...")
        daily_ic_dict = {feat: [] for feat in FEATURE_COLS}
        
        for date, val in daily_data_storage[h].items():
            factors_mat = np.array(val['factors']) # (股票数, 特征数)
            rets_vec = np.array(val['rets'])       # (股票数,)
            
            if len(rets_vec) < 5: # 样本太少不计入当日IC
                continue
                
            for i, feat in enumerate(FEATURE_COLS):
                f_vec = factors_mat[:, i]
                # 去极值
                f_vec = winsorize_mad(f_vec)
                
                # 计算 Spearman Rank IC
                # 相比 pd.rank().corr()，spearmanr 在处理大规模向量时更高效
                ic, _ = spearmanr(f_vec, rets_vec)
                if not np.isnan(ic):
                    daily_ic_dict[feat].append(ic)

        # 汇总
        res = []
        for feat in FEATURE_COLS:
            mean_ic, abs_mean_ic, icir = calc_ic_metrics(daily_ic_dict[feat])
            res.append({
                '周期': f'{h}日',
                '特征': feat,
                'IC': round(mean_ic, 4),
                '|IC|': round(abs_mean_ic, 4),
                'ICIR': round(icir, 4),
                '样本天数': len(daily_ic_dict[feat])
            })
        
        res_df = pd.DataFrame(res).sort_values('|IC|', ascending=False)
        print(f"\n===== {h} 日结果前10 =====")
        print(res_df.head(10))
        all_period_results.append(res_df)

    # 保存
    full_df = pd.concat(all_period_results)
    full_df.to_csv('优化后_全周期IC汇总.csv', index=False, encoding='utf-8-sig')
    print("\n✅ 计算完成，IC 指标已根据截面数据优化")

if __name__ == '__main__':
    compute_multi_period_ic_optimized()