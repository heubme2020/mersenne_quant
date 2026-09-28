import os
import sys
import pandas as pd
import random
import numpy as np
from tqdm import tqdm
import shutil
from multiprocessing import Pool, cpu_count

from zero_features import (
    FEATURE_COLUMNS, LOOKBACK_QUARTERS, PRICE_WINDOW_DAYS, CLIP_VALUE,
)

# Windows 控制台默认 GBK，而本文件末尾会 print 一个 emoji（"🎉"）—— GBK 编不出来，
# 于是**在干完所有活之后**抛 UnicodeEncodeError、退出码 1。
# 2026-09-26 踩过：日志里已经写着 "Total HDF5 files created: 1765796"（数据全好了），
# 然后崩在一句庆祝打印上，编排脚本把它当成失败、中止了整条流水线。
# three/seven 的同类脚本本来就有这个 try 块，zero 漏了 —— 这里补齐。
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

pd.set_option('future.no_silent_downcasting', True)


def process_group(group_tuple, daily_data, train_folder):
    """
    处理单个股票数据组并生成训练文件。
    这是一个独立的函数，适合在多进程中运行。
    """
    symbol = group_tuple[0]
    group = group_tuple[1]

    # 1. 数据清洗和准备
    missing_columns = [col for col in FEATURE_COLUMNS if col not in group.columns]
    if missing_columns:
        print(f"Skip {symbol}, missing feature columns: {missing_columns}")
        return 0

    group = group.sort_values('endDate').fillna(0).reset_index(drop=True)
    group_data_length = len(group)

    if group_data_length < LOOKBACK_QUARTERS:
        return 0  # 返回成功处理的文件数，这里是 0

    # 优化：提前筛选 daily_data
    group_daily = daily_data[daily_data['symbol'] == symbol].sort_values('date').reset_index(drop=True)
    if group_daily.empty:
        return 0

    # 提前转换类型
    group['endDate'] = group['endDate'].astype('int32')

    files_created = 0

    # 循环生成训练样本
    for j in range(LOOKBACK_QUARTERS - 1, group_data_length):
        endDate = group['endDate'].iloc[j]  # 当前样本的最新财务报告期

        # 预测期 daily 数据（未来 127*3 天）
        fore_daily = group_daily[group_daily['date'] > int(endDate)]
        fore_daily = fore_daily.iloc[:PRICE_WINDOW_DAYS]
        if len(fore_daily) != PRICE_WINDOW_DAYS:
            continue
        # 检查预测期价格是否有非正数（0或负数）
        if (fore_daily['close'] <= 0).any():
            continue

        # 历史期 daily 数据（endDate 之前取最近 127*3 天，作为「上市历史足够长」的过滤）
        past_daily = group_daily[group_daily['date'] <= int(endDate)]
        past_daily = past_daily.iloc[-PRICE_WINDOW_DAYS:]
        if len(past_daily) != PRICE_WINDOW_DAYS:
            continue
        if (past_daily['close'] <= 0).any():
            continue

        # 标签：未来最大回撤取负（negMdd，越大越安全，值域 (-1, 0]）
        fore_closes = fore_daily['close'].values.astype(float)
        running_max = np.maximum.accumulate(fore_closes)
        zero = -float(((running_max - fore_closes) / running_max).max())

        # 截取 3 个季度的财务数据 (j-2, j-1, j)
        data = group.loc[j - LOOKBACK_QUARTERS + 1:j, FEATURE_COLUMNS].copy().reset_index(drop=True)
        if len(data) != LOOKBACK_QUARTERS:
            continue
        data = data.assign(zero=zero)
        data = data.fillna(0)
        data.replace([np.inf, -np.inf], 0, inplace=True)
        data[(data > CLIP_VALUE)] = CLIP_VALUE
        data[(data < -CLIP_VALUE)] = -CLIP_VALUE
        for col in data.columns:
            if data[col].dtype in ['int64', 'float64', 'object']:
                data[col] = data[col].astype('float32')

        # 保存为 HDF5
        data_basename = f"{symbol}_{endDate}.h5"
        data_name = os.path.join(train_folder, data_basename)
        data.to_hdf(data_name, key='data', mode='w')
        files_created += 1

    return files_created


def gen_exchange_zero_train_data(exchange):
    """
    加载数据，分割任务并使用多进程处理。
    """
    upper_exchange = exchange[0].upper() + exchange[1:]
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. 加载数据
    daily_path = os.path.join(current_dir, f'../data/{upper_exchange}/daily_{exchange}.csv')
    indicator_path = os.path.join(current_dir, f'../data/{upper_exchange}/indicator_{exchange}.csv')

    print(f"Loading {exchange} data...")
    try:
        daily_data = pd.read_csv(daily_path, encoding="utf-8")
        indicator_data = pd.read_csv(indicator_path, encoding="utf-8")
    except FileNotFoundError:
        print(f"Error: Data files not found for {exchange}.")
        return 0

    # 2. 分组并打乱顺序
    groups = list(indicator_data.groupby('symbol'))
    random.shuffle(groups)

    # 3. 设置多进程参数
    train_folder = os.path.join(current_dir, 'train')
    # 使用所有可用 CPU 核心，或根据需要设置一个固定值
    num_processes = cpu_count()
    print(f"Starting {num_processes} processes for {len(groups)} groups.")

    # 准备 Pool.starmap 需要的参数列表
    # (group_tuple, daily_data, train_folder)
    task_args = [(group_tuple, daily_data, train_folder) for group_tuple in groups]

    # 4. 运行多进程池
    total_files_created = 0
    try:
        # 使用 Pool.starmap 并行处理所有 groups
        with Pool(processes=num_processes) as pool:
            # 进程池会返回一个结果列表，每个结果是 process_group 的返回值 (files_created)
            results = list(tqdm(pool.starmap(process_group, task_args), total=len(groups), desc=f"Processing {exchange} groups"))

        total_files_created = sum(results)
        print(f"Finished {exchange}. Total files created: {total_files_created}")

    except Exception as e:
        print(f"An error occurred during multiprocessing for {exchange}: {e}")
        return 0

    return total_files_created


def gen_zero_train_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_folder = os.path.join(current_dir, 'train')

    # 清理和创建训练目录
    print("Clearing and creating 'train' folder...")
    shutil.rmtree(train_folder, ignore_errors=True)
    os.makedirs(train_folder, exist_ok=True)

    # 获取交易所列表：data/ 下全部交易所（已在上游过滤，不再套 num_symbols 阈值）
    data_dir = os.path.join(current_dir, '..', 'data')
    exchange_list = sorted(d for d in os.listdir(data_dir)
                           if os.path.isdir(os.path.join(data_dir, d)))

    failed_list = []
    total_files = 0

    for exchange in exchange_list:
        try:
            print(f"\n--- Starting processing for {exchange} ---")
            files_created = gen_exchange_zero_train_data(exchange)
            if files_created == 0:
                failed_list.append(exchange)
            total_files += files_created
        except Exception as e:
            print(f"FATAL error for exchange {exchange}: {e}")
            failed_list.append(exchange)

    print("\n--- Summary ---")
    print(f"Total HDF5 files created: {total_files}")
    if failed_list:
        print(f"Failed to process exchanges: {failed_list} ⚠️")
    else:
        print("All exchanges processed successfully! 🎉")


if __name__ == '__main__':
    gen_zero_train_data()
