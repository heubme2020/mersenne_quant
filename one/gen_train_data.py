
import os
import sys
import pandas as pd
from tqdm import tqdm
import random
import numpy as np
import multiprocessing
from multiprocessing import Pool
import math
import shutil

# 必须在 Windows 上使用多进程时引入
if os.name == 'nt':
    multiprocessing.freeze_support()

pd.set_option('future.no_silent_downcasting', True)

# ========== 输出/标签定义（与 train.py 保持一致） ==========
# close 标量头：5 个时间窗口（1/3/7/31/127），相对明天收盘价
HORIZONS = [1, 3, 7, 31, 127]
DAYS_INPUT = 127 * 7            # 输入窗口：889 个交易日
AUX_OUTPUT_DAYS = 128           # 未来窗口长度 = 明天 1 天 + 127 天（辅助序列头重建）
WINDOW_TOTAL = DAYS_INPUT + AUX_OUTPUT_DAYS   # 889 + 128 = 1017
REF_IDX = DAYS_INPUT - 1        # 888：输入最后一天（today，归一化基准）
TOMORROW_IDX = DAYS_INPUT       # 889：明天（close 标签基准）

# 因子方案：baseline / plan1 / plan2（通过环境变量传入）
FACTOR_MODE = os.environ.get('FACTOR_MODE', 'new24')   # 2026-09-14 起默认用新 24 个
from factor_config import get_technical_factors  # noqa: E402
from factor_pool import add_pool_factors  # noqa: E402

# Windows 控制台默认 GBK，而本文件会打印 `⚠️` / `🎉` 这类非 GBK 字符 —— 走到那条分支就会
# UnicodeEncodeError 崩掉，而且**往往是在活儿干完之后**才崩（2026-09-26 zero/gen 就这么"失败"过：
# 日志里 already 写着总量，然后崩在一句庆祝打印上）。与其它脚本同一套修法。2026-09-28 扫描后补齐。
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass


NEW_FACTOR_H5 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'factor_screen', 'new_factors.h5')


def load_new_factors():
    """加载预计算的新因子（dict: factor_id -> (date × symbol) DataFrame）。"""
    store = pd.HDFStore(NEW_FACTOR_H5, mode='r')
    factors = {}
    for k in store.keys():
        fid = k.strip('/')[2:]  # 'f_xxx' -> 'xxx'
        factors[fid] = store[k].astype(np.float32)
    store.close()
    return factors


def compute_gain_label(values, base):
    """
    与 one 完全一致的 close 标签公式：
        log(max) + log(median) + log(min) - 3 * log(base)
    values: 未来窗口的 close 序列（pandas Series）
    base  : 明天（TOMORROW_IDX）的 close 值
    """
    return (math.log(float(values.max())) + math.log(float(values.median()))
            + math.log(float(values.min())) - 3.0 * math.log(float(base)))


def add_technical_factor(data):
    """
    计算股票技术因子
    已根据 IC 分析结果替换掉弱势因子：vma_7_31, rsi7, atr31
    集成强势因子：Alpha 15 (WQ), Alpha 128 (GTJA), Alpha 101 (GTJA)
    """
    # 确保数据按时间排序
    data = data.sort_values('date') if 'date' in data.columns else data

    # --- 1. 均线类 (原有保留) ---
    data['ma3'] = data['close'].rolling(3).mean()
    data['ma7'] = data['close'].rolling(7).mean()
    data['ma31'] = data['close'].rolling(31).mean()

    # --- 2. RSI类 (仅保留效果尚可的 rsi3 和 rsi31) ---
    delta = data['close'].diff()
    def calc_rsi(ser, period):
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        return (100 - (100 / (1 + (gain / (loss + 1e-6))))) * 0.01

    data['rsi3'] = calc_rsi(data['close'], 3)
    data['rsi31'] = calc_rsi(data['close'], 31)

    # --- 3. 波动率与成交量类 (原有保留) ---
    data['atr3'] = (data['delta'].rolling(3).mean()) * 31
    data['atr7'] = (data['delta'].rolling(7).mean()) * 31

    # OBV
    obv = delta * data['volume']
    data['obv3'] = (obv.rolling(3).mean()) * 31
    data['obv7'] = (obv.rolling(7).mean()) * 31
    data['obv31'] = (obv.rolling(31).mean()) * 31

    # Correlation
    data['corr3'] = data['volume'].rolling(3).corr(data['close'])
    data['corr7'] = data['volume'].rolling(7).corr(data['close'])
    data['corr31'] = data['volume'].rolling(31).corr(data['close'])

    # Curvature & VMA
    data['curvature'] = (data['close'].diff().diff()) * 31
    data['vma_3_7'] = data['volume'].rolling(3).mean() / data['volume'].rolling(7).mean() - 1

    # Factor
    data['factor'] = (data['close'].pct_change(3) - data['volume'].rolling(7).std())

    # Overnight
    overnight = data['open'] * data['close'] / data['close'].shift(1).replace(0, np.nan) - 1
    data['overnight3'] = overnight.rolling(3).mean() * 31
    data['overnight7'] = overnight.rolling(7).mean() * 31
    data['overnight31'] = overnight.rolling(31).mean() * 31

    # --- 4. 三个 Alpha 因子 ---
    rk_h = data['high'].rolling(10).rank(pct=True)
    rk_v = data['volume'].rolling(10).rank(pct=True)
    inner_corr = rk_h.rolling(3).corr(rk_v)
    data['alpha15_wq'] = -1 * inner_corr.rolling(10).rank(pct=True)

    adv20 = data['volume'].rolling(20).mean()
    data['alpha128_gtja'] = -1 * (data['close'].diff(1) * (data['volume'] / (adv20 + 1e-6))).rolling(5).mean()

    low_30 = data['low'].rolling(30).min()
    high_30 = data['high'].rolling(30).max()
    data['alpha101_gtja'] = (data['close'] - low_30) / (high_30 - low_30 + 1e-6)

    # --- 5. Alpha22 ---
    rolling_corr_3 = (data['high'] * data['close']).rolling(3).corr(data['volume'])
    delta_corr_3 = rolling_corr_3.diff(3)
    std_close_7 = data['close'].rolling(7).std()
    data['aplha22_3_7'] = -1 * (delta_corr_3 * std_close_7) * 31

    rolling_corr_7 = (data['high'] * data['close']).rolling(7).corr(data['volume'])
    delta_corr_7 = rolling_corr_7.diff(7)
    std_close_31 = data['close'].rolling(31).std()
    data['aplha22_7_31'] = -1 * (delta_corr_7 * std_close_31) * 31

    return data


def gen_group_train_data(groups, train_folder, sample_rate):
    """
    生成训练数据，并在多进程中运行。
    groups: 股票分组列表
    train_folder: 训练数据保存路径
    sample_rate: 抽样率
    """
    files_created = 0
    for i in tqdm(range(len(groups))):
        symbol = groups[i][0]
        daily_group = groups[i][1].copy(deep=True).reset_index(drop=True)

        # 除去最新 127 天的数据（预留评估期）
        daily_group = daily_group.iloc[:-127].reset_index(drop=True)
        daily_group['date'] = pd.to_numeric(daily_group['date'], errors='coerce').astype('Int64')

        group_data_length = len(daily_group)
        if group_data_length < WINDOW_TOTAL:
            continue

        for j in range(DAYS_INPUT, group_data_length - AUX_OUTPUT_DAYS - 1):
            date = daily_group['date'].iloc[j]
            data_basename = symbol + '_' + str(date) + '.h5'
            data_name = os.path.join(train_folder, data_basename)

            if random.random() < sample_rate:  # 按抽样率随机保存
                # 输入 889 天 + 未来 128 天 = 1017 行
                data = daily_group.iloc[j - DAYS_INPUT + 1: j + AUX_OUTPUT_DAYS + 1].copy(deep=True)
                data = data.reset_index(drop=True)

                if len(data) != WINDOW_TOTAL:
                    continue

                # --- 归一化 ---
                ref_close = data['close'].iloc[REF_IDX]
                ref_volume = data['volume'].iloc[REF_IDX]

                if ref_close <= 0 or ref_volume <= 0:
                    continue

                data['open'] = data['open'] / ref_close
                data['high'] = data['high'] / ref_close
                data['low'] = data['low'] / ref_close
                data['delta'] = data['high'] - data['low']
                data['close'] = data['close'] / ref_close
                data['volume'] = data['volume'] / ref_volume

                # --- 目标变量（用于过滤异常样本）---
                close_tomorrow = data['close'].iloc[TOMORROW_IDX]
                volume_tomorrow = data['volume'].iloc[TOMORROW_IDX]
                if close_tomorrow <= 0 or volume_tomorrow <= 0:
                    continue

                # 用最长的 127 周期（从后天起）计算 close 过滤标签
                close_fore = data['close'].iloc[TOMORROW_IDX + 1: WINDOW_TOTAL]
                close_127 = compute_gain_label(close_fore, close_tomorrow)

                if abs(close_127) > 127:
                    continue

                # --- 特征工程 ---
                data_input = data[:DAYS_INPUT].reset_index(drop=True)
                data_fore = data[DAYS_INPUT:].reset_index(drop=True)
                # 池子先算，one 原公式覆盖重名项（重名项两者秩等价）
                data_input = add_technical_factor(add_pool_factors(data_input))
                data_fore = add_technical_factor(add_pool_factors(data_fore))
                data = pd.concat([data_input, data_fore], axis=0)
                data = data.reset_index(drop=True)

                # --- 选定 24 个技术因子（6 raw + 24 tech）---
                keep_cols = ['open', 'high', 'low', 'close', 'volume', 'delta'] + get_technical_factors(FACTOR_MODE)
                data = data[keep_cols]

                # --- 最终处理 ---
                data['idx'] = data.index / (DAYS_INPUT - 1.0)

                data = data.fillna(0)
                data.replace([np.inf, -np.inf], 0, inplace=True)

                for col in data.columns:
                    try:
                        data[col] = data[col].astype('float32')
                    except Exception:
                        pass

                data[(data > 127.0)] = 127.0
                data[(data < -127.0)] = -127.0

                data.to_hdf(data_name, key='data', mode='w', format='fixed')
                files_created += 1

    return files_created          # 与另外四个 gen 一致：报"生成的文件数"而不是"股票组数"


def gen_exchange_one_train_data(exchange, train_folder, sample_rate):
    upper_exchange = exchange[0].upper() + exchange[1:]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    daily_path = os.path.join(current_dir, '../data/' + upper_exchange + '/daily_' + exchange + '.csv')

    try:
        daily_data = pd.read_csv(daily_path, encoding="utf-8")
    except FileNotFoundError:
        print(f"Error: Data files not found for {exchange}.")
        return

    if FACTOR_MODE in ('plan1', 'plan2', 'plan1_ts', 'plan2_ts'):
        new_factors = load_new_factors()
        for fid, fac in new_factors.items():
            long = fac.stack().rename(fid).reset_index()
            daily_data = daily_data.merge(long, on=['date', 'symbol'], how='left')
        print(f"合并了 {len(new_factors)} 个新因子（{FACTOR_MODE}）。", flush=True)

    print(f"Loading {upper_exchange} data...", flush=True)
    print(f"  {len(daily_data):,} rows")

    groups = list(daily_data.groupby('symbol'))
    random.shuffle(groups)
    group_count = len(groups)

    NUM_PROCESSES = multiprocessing.cpu_count()
    split_size = group_count // NUM_PROCESSES

    groups_list_for_starmap = []

    for i in range(NUM_PROCESSES):
        start = i * split_size
        end = (i + 1) * split_size if i < NUM_PROCESSES - 1 else group_count
        groups_list_for_starmap.append((groups[start:end], train_folder, sample_rate))

    print(f"Starting {NUM_PROCESSES} processes for {group_count} groups.")

    with Pool(processes=NUM_PROCESSES) as pool:
        results = pool.starmap(gen_group_train_data, groups_list_for_starmap)

    n_created = sum(results)          # 与另外四个 gen 一样：报"生成的文件数"而不是"股票组数"
    print(f"Finished {upper_exchange}. Total files created: {n_created}", flush=True)
    return n_created


def gen_one_train_data(exchanges, train_folder, sample_rate):
    # 无条件清空 —— 与 zero/three/seven 逐字一致（2026-09-27 对齐，原来这句是
    # "Cleaning existing folder: ..." + 不带 ignore_errors 的 rmtree，措辞和健壮性两处都不一样）。
    print("Clearing and creating 'train' folder...")
    shutil.rmtree(train_folder, ignore_errors=True)
    os.makedirs(train_folder, exist_ok=True)

    failed_list = []

    total_files = 0
    for exchange in exchanges:
        print(f"\n--- Starting processing for {exchange} ---", flush=True)
        try:
            total_files += gen_exchange_one_train_data(exchange, train_folder, sample_rate) or 0
        except Exception as e:
            failed_list.append((exchange, str(e)))
            print(f"FATAL error for exchange {exchange}: {e}")

    # 与 three/seven/zero 的 gen 同一套收尾
    print("\n--- Summary ---")
    print(f"Total HDF5 files created: {total_files}")
    if failed_list:
        print(f"Failed to process exchanges: {failed_list} ⚠️")
    else:
        print("All exchanges processed successfully! 🎉")


def main():
    import argparse
    p = argparse.ArgumentParser(description='生成训练数据')
    p.add_argument('--exchanges', nargs='+', default=['SHZ', 'SHH'])
    p.add_argument('--folder', default='train')
    p.add_argument('--rate', type=float, default=1.0 / 31.0)
    args = p.parse_args()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_folder = args.folder if os.path.isabs(args.folder) else os.path.join(current_dir, args.folder)
    gen_one_train_data(args.exchanges, train_folder, args.rate)


if __name__ == '__main__':
    main()
