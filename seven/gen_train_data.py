import os
import sys
import pandas as pd
import random
import numpy as np
import math
import threading
from tqdm import tqdm
import shutil
import datetime
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed

from normalize import normalize_by_mean_std

# Windows 控制台默认 GBK，强制 UTF-8 输出，避免 emoji 乱码/报错
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

pd.set_option('future.no_silent_downcasting', True)

# mean/std 改成每个 worker 自己读一份，任务参数里只剩 group_tuple。
# 2026-09-29 改：原来每个任务都重复序列化一遍 mean/std（几千个任务就是几千遍）；three 在
# TSX 上撞内存、一个 worker 猝死、整条流水线静默卡死 4 小时的详细说明见 three/gen_train_data.py。
# seven 的 mean/std 只有几百 KB，风险比 three 小得多，但同一个坑不踩第二遍。
_W = {}


def _init_worker(mean_path, std_path, train_folder, exchange):
    """每个 worker 进程启动时读一次 mean/std（spawn 下父进程的内存不会被子进程继承）。"""
    _W['mean'] = pd.read_csv(mean_path, encoding='utf-8')
    _W['std'] = pd.read_csv(std_path, encoding='utf-8')
    _W['train_folder'] = train_folder
    _W['exchange'] = exchange.upper()


def process_group(group_tuple):
    """
    处理单个股票数据组并生成训练文件。
    这是一个独立的函数，适合在多进程中运行。
    """
    symbol = group_tuple[0]
    group = group_tuple[1]
    mean_data = _W['mean']
    std_data = _W['std']
    train_folder = _W['train_folder']

    # 1. 数据清洗和准备
    # NaN 财务字段不在这里填 0：留到 normalize 之后、per-sample 的 data.fillna(0) 再填，
    # 让「缺失 = 中性(0)」与推理端一致。否则缺失会被当成"0 美元"、z-score 成 -mean/std。
    group = group.reset_index(drop=True)
    group_data_length = len(group)

    if group_data_length < 62:
        return 0, 0 # 返回 (成功处理的文件数, 因 mean/std 缺失跳过的锚点数)

    # 提前转换类型
    group['endDate'] = group['endDate'].astype('int32')

    files_created = 0
    n_skip_norm = 0

    # 循环生成训练样本
    for j in range(30, group_data_length-1):
        endDate = group['endDate'].iloc[j] # 这里的 endDate 应该是当前样本的最新财务报告期

        data = group.iloc[j - 30:j + 1].copy().reset_index(drop=True) # 使用 .copy() 避免 SettingWithCopyWarning
        data_fore = group.iloc[j + 1:j + 32].copy().reset_index(drop=True) # 使用 .copy() 避免 SettingWithCopyWarning
        if len(data) != 31:
            continue
        if len(data_fore) != 31:
            continue
        totalStockholdersEquity = data['totalStockholdersEquity'].iloc[-1]
        # 检查是不是有点净资产（524287.0 为缺失哨兵值；NaN 来自三表 outer merge 没对齐）
        if not np.isfinite(totalStockholdersEquity) or totalStockholdersEquity < 524287.0:
            continue

        # 三个预测期限（未来季度数）及其名称，与 seven_model.HORIZONS / HORIZON_NAMES 对应
        HORIZONS = [3, 7, 31]
        HORIZON_NAMES = ['three', 'seven', 'thirty_one']

        # 分支一：自由现金流和 / 当前净资产（原有逻辑）
        # 分支二：股息和 / 当前净资产（现金流表中 commonDividendsPaid 为负的现金流出，取反使“派息越多越大”）
        # 分支三：净资产和 / 当前净资产
        # 标签按 branch-major 顺序写入（fcf 三期限 -> dividend 三期限 -> netasset 三期限），
        # 之后 train.py 按 (3, 3) reshape 即得 [branch, horizon]，与 seven_model.label_columns() 一致。
        fcf_fore = data_fore['freeCashFlow']
        dividend_fore = -data_fore['commonDividendsPaid']
        netasset_fore = data_fore['totalStockholdersEquity']

        labels = {}
        for h, name in zip(HORIZONS, HORIZON_NAMES):
            labels[f'fcf_{name}'] = fcf_fore.iloc[:h].sum() / totalStockholdersEquity
        for h, name in zip(HORIZONS, HORIZON_NAMES):
            labels[f'dividend_{name}'] = dividend_fore.iloc[:h].sum() / totalStockholdersEquity
        for h, name in zip(HORIZONS, HORIZON_NAMES):
            labels[f'netasset_{name}'] = netasset_fore.iloc[:h].sum() / totalStockholdersEquity

        # 异常值过滤（9 个标签任一超界则丢弃该样本）
        if any(abs(v) > 127 for v in labels.values()):
            continue
            
        # 2. 训练输入 (财务比率) 准备
        
        data.drop(columns=['symbol', 'endDate'], inplace=True)
        if normalize_by_mean_std(data, endDate, mean_data, std_data) is None:
            n_skip_norm += 1
            continue
        data = data.assign(**labels)
        data = data.fillna(0)
        data.replace([np.inf, -np.inf], 0, inplace=True)
        data[(data > 8191.0)] = 8191.0
        data[(data < -8191.0)] = -8191.0
        for col in data.columns:
            if data[col].dtype in ['int64', 'float64', 'object']:
                data[col] = data[col].astype('float32')

        # 保存为 HDF5
        data_basename = f"{_W['exchange']}_{symbol}_{endDate}.h5"
        data_name = os.path.join(train_folder, data_basename)
        
        # 使用 'a' 模式写入单个文件更安全，虽然 'w' 也可以
        data.to_hdf(data_name, key='data', mode='w')
        files_created += 1

    return files_created, n_skip_norm


def gen_exchange_seven_train_data(exchange):
    """
    加载数据，分割任务并使用多进程处理。
    """
    upper_exchange = exchange[0].upper() + exchange[1:]
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. 加载数据
    income_path = os.path.join(current_dir, f'../data/{upper_exchange}/income_{exchange}.csv')
    balance_path = os.path.join(current_dir, f'../data/{upper_exchange}/balance_{exchange}.csv')
    cashflow_path = os.path.join(current_dir, f'../data/{upper_exchange}/cashflow_{exchange}.csv')
    mean_path = os.path.join(current_dir, f'../data/{upper_exchange}/mean_{exchange}.csv')
    std_path = os.path.join(current_dir, f'../data/{upper_exchange}/std_{exchange}.csv')
    indicator_path = os.path.join(current_dir, f'../data/{upper_exchange}/indicator_{exchange}.csv')
    features_path = os.path.join(current_dir, f'../three/features_importance.csv')

    print(f"Loading {exchange} data...")
    try:
        income_data = pd.read_csv(income_path, encoding="utf-8")
        balance_data = pd.read_csv(balance_path, encoding="utf-8")
        cashflow_data = pd.read_csv(cashflow_path, encoding="utf-8")
        mean_data = pd.read_csv(mean_path, encoding="utf-8")
        std_data = pd.read_csv(std_path, encoding="utf-8")
        indicator_data = pd.read_csv(indicator_path, encoding="utf-8")
        features = pd.read_csv(features_path, encoding="utf-8")
        # 合并财务相关数据
        financial_data = pd.merge(income_data, balance_data, on=['symbol', 'endDate'], how='outer')
        financial_data = pd.merge(financial_data, cashflow_data, on=['symbol', 'endDate'], how='outer')
        financial_data = financial_data.dropna(subset=['symbol', 'endDate'])
        financial_data = financial_data.reset_index(drop=True)
        financial_data.drop_duplicates(subset=['symbol', 'endDate'], keep='first', inplace=True)
        financial_data = financial_data.reset_index(drop=True)
        # 截取指定特征的部分
        features_list = features['feature'].to_list()
        features_data = pd.DataFrame()
        features_data['symbol'] = financial_data['symbol']
        features_data['endDate'] = financial_data['endDate'].astype('int64')
        for feature in features_list:
            features_data[feature] = financial_data[feature]
        features_data = pd.merge(indicator_data, features_data, on=['symbol', 'endDate'], how='outer')
        features_data = features_data.dropna(subset=['symbol', 'endDate'])
        features_data = features_data.reset_index(drop=True)
        features_data.drop_duplicates(subset=['symbol', 'endDate'], keep='first', inplace=True)
        features_data = features_data.reset_index(drop=True)

        #删除features_data中endDate小于mean std data中最早的endDate的所有行
        features_data = features_data[features_data['endDate'].astype(float) >= mean_data['endDate'].iloc[0].astype(float)]
        features_data = features_data.reset_index(drop=True)
        features_data = features_data[features_data['endDate'].astype(float) >= std_data['endDate'].iloc[0].astype(float)]
        features_data = features_data.reset_index(drop=True)
    except FileNotFoundError:
        print(f"Error: Data files not found for {exchange}.")
        return 0
        
    # 2. 分组并打乱顺序
    groups = list(features_data.groupby('symbol'))
    random.shuffle(groups)

    # 3. 设置多进程参数
    train_folder = os.path.join(current_dir, 'train')
    # 使用所有可用 CPU 核心，或根据需要设置一个固定值
    num_processes = cpu_count() 
    print(f"Starting {num_processes} processes for {len(groups)} groups.")
    
    # 任务参数只传 group_tuple —— mean/std 交给 _init_worker，每个 worker 读一份
    task_args = list(groups)
    
    # 4. 运行多进程池
    total_files_created = 0
    total_skipped_norm = 0
    try:
        # 用 ProcessPoolExecutor 而不是 Pool：worker 意外死掉（内存不够等）时它会抛
        # BrokenProcessPool，被下面的 except 接住、记成"这个交易所失败"继续跑下一个所，
        # 而不是像 Pool 那样永远静默等下去。
        # 进度条改成 as_completed —— Pool.starmap 返回的是 list，原来那根条是结果全到手
        # 之后才瞬间刷完的（显示 100% / 00:00 那根全是假象）。
        with ProcessPoolExecutor(max_workers=num_processes, initializer=_init_worker,
                                 initargs=(mean_path, std_path, train_folder, exchange)) as ex:
            futs = [ex.submit(process_group, t) for t in task_args]
            for fut in tqdm(as_completed(futs), total=len(futs), desc=f"Processing {exchange} groups"):
                n, skipped = fut.result()
                total_files_created += n
                total_skipped_norm += skipped
        print(f"Finished {exchange}. Total files created: {total_files_created}")
        if total_skipped_norm:
            print(f"  skipped {total_skipped_norm} anchors（mean/std 未覆盖该季度，跳过）")
        
    except Exception as e:
        print(f"An error occurred during multiprocessing for {exchange}: {e}")
        return 0

    return total_files_created


def gen_seven_train_data():
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
            files_created = gen_exchange_seven_train_data(exchange)
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
    gen_seven_train_data()
    # gen_exchange_three_train_data_pre('KOE')