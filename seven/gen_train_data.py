import os
import pandas as pd
import random
import numpy as np
import math
import threading
from tqdm import tqdm
import shutil
import datetime
from multiprocessing import Pool, cpu_count

pd.set_option('future.no_silent_downcasting', True)

def process_group(group_tuple, mean_data, std_data, train_folder):
    """
    处理单个股票数据组并生成训练文件。
    这是一个独立的函数，适合在多进程中运行。
    """
    symbol = group_tuple[0]
    group = group_tuple[1]

    # 1. 数据清洗和准备
    group = group.fillna(0).reset_index(drop=True)
    group_data_length = len(group)

    if group_data_length < 62:
        return 0 # 返回成功处理的文件数，这里是 0
    
    # 提前转换类型
    group['endDate'] = group['endDate'].astype('int32')

    files_created = 0

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
        three_dcf_fore = data_fore['freeCashFlow'].iloc[:3].sum()
        seven_dcf_fore = data_fore['freeCashFlow'].iloc[:7].sum()
        thirty_one_dcf_fore = data_fore['freeCashFlow'].iloc[:31].sum()
        # 检查是不是有点净资产
        if totalStockholdersEquity < 524287.0:
            continue
        
        three = three_dcf_fore/totalStockholdersEquity
        seven = seven_dcf_fore/totalStockholdersEquity
        thirty_one = thirty_one_dcf_fore/totalStockholdersEquity
        
        if abs(three) > 127 or abs(seven) > 127 or abs(thirty_one) > 127: # 异常值过滤
            continue
            
        # 2. 训练输入 (财务比率) 准备
        
        data.drop(columns=['symbol', 'endDate'], inplace=True)
        col_names = data.columns.values
        mean_col_names = mean_data.columns.values
        for k in range(len(data.columns)):
            col_name = col_names[k]
            if col_name not in mean_col_names:
                continue
            mean_value = mean_data.loc[mean_data['endDate'] == int(endDate), col_name].item()
            std_value = std_data.loc[std_data['endDate'] == int(endDate), col_name].item()
            if std_value != 0:
                data[col_name] = data[col_name] - mean_value
                data[col_name] = data[col_name] / std_value
            else:
                data[col_name] = 0
        data = data.assign(three=three)
        data = data.assign(seven=seven)
        data = data.assign(thirty_one=thirty_one)
        data = data.fillna(0)
        data.replace([np.inf, -np.inf], 0, inplace=True)
        data[(data > 8191.0)] = 8191.0
        data[(data < -8191.0)] = -8191.0
        for col in data.columns:
            if data[col].dtype in ['int64', 'float64', 'object']:
                data[col] = data[col].astype('float32')

        # 保存为 HDF5
        data_basename = f"{symbol}_{endDate}.h5"
        data_name = os.path.join(train_folder, data_basename)
        
        # 使用 'a' 模式写入单个文件更安全，虽然 'w' 也可以
        data.to_hdf(data_name, key='data', mode='w') 
        files_created += 1
        
    return files_created


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
        financial_data = financial_data.fillna(0).reset_index(drop=True)
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
        features_data = features_data.fillna(0).reset_index(drop=True)
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
    
    # 准备 Pool.starmap 需要的参数列表
    # (group_tuple, daily_data, train_folder)
    task_args = [(group_tuple, mean_data, std_data, train_folder) for group_tuple in groups]
    
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


def gen_seven_train_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_folder = os.path.join(current_dir, 'train')
    
    # 清理和创建训练目录
    print("Clearing and creating 'train' folder...")
    shutil.rmtree(train_folder, ignore_errors=True)
    os.makedirs(train_folder, exist_ok=True)
    
    # 获取交易所列表
    csv_path = os.path.join(current_dir, '..', 'train_exchanges.csv')
    csv_path = os.path.abspath(csv_path)

    try:
        exchanges_data = pd.read_csv(csv_path, encoding="utf-8")
        exchange_list = exchanges_data['exchange'].tolist()
        # exchange_list = ['KOE']
    except FileNotFoundError:
        print(f"Error: train_exchanges.csv not found at {csv_path}")
        return

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