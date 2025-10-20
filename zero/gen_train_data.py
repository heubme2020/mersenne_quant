# import os
# import pandas as pd
# import random
# import numpy as np
# import threading
# import math
# from tqdm import tqdm
# import shutil
# pd.set_option('future.no_silent_downcasting', True)


# def gen_group_train_data(groups, daily_data):
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     train_folder = os.path.join(current_dir, 'train')
#     for i in tqdm(range(len(groups))):
#         group = groups[i][1]
#         # 去除最后7个季度的数据
#         group = group.iloc[:-7]
#         group = group.fillna(0).reset_index(drop=True)
#         group_data_length = len(group)
#         # 目前必须上市3个季度后才可以预测
#         if group_data_length < 3:
#             continue
#         symbol = groups[i][0]
#         group_daily = daily_data[(daily_data['symbol'] == symbol)].reset_index(drop=True)

#         for j in range(3, group_data_length-1):
#             group['endDate'] = group['endDate'].astype('int32')
#             endDate = group['endDate'].iloc[j]
#             data_basename = symbol + '_' + str(endDate) + '.h5'
#             # data_name = 'train/' + data_basename
#             data_name = os.path.join(train_folder, data_basename)

#             # 截取daily数据
#             fore_daily = group_daily[(group_daily['date'] > int(endDate))]
#             fore_daily = fore_daily.iloc[:127*3]
#             fore_daily = fore_daily.reset_index(drop=True)
#             if len(fore_daily) != 127*3:
#                 continue
#             min_fore = fore_daily['close'].min()
#             median_fore = fore_daily['close'].median()
#             max_fore = fore_daily['close'].max()
#             price_fore = math.log(min_fore) + math.log(median_fore) + math.log(max_fore)

#             past_daily = group_daily[(group_daily['date'] <= int(endDate))]
#             past_daily = past_daily.iloc[-127*3:]
#             past_daily = past_daily.reset_index(drop=True)
#             if len(past_daily) != 127*3:
#                 continue
#             min_past = past_daily['close'].min()
#             median_past = past_daily['close'].median()
#             max_past = past_daily['close'].max()
#             price_past = math.log(min_past) + math.log(median_past) + math.log(max_past)
#             three = price_fore - price_past
#             if three > 127 or three < -127:
#                 continue
#             # 下面进行归一化
#             data = group.iloc[j - 2:j + 1].reset_index(drop=True)
#             if len(data) != 3:
#                 continue
#             data.drop(columns=['symbol'], inplace=True)
#             data.drop(columns=['endDate'], inplace=True)
#             data = data.assign(three=three)
#             data = data.fillna(0)
#             data.replace([np.inf, -np.inf], 0, inplace=True)
#             for col in data.select_dtypes(include=['int64']).columns:
#                 data[col] = data[col].astype('float32')
#             for col in data.select_dtypes(include=['float64']).columns:
#                 data[col] = data[col].astype('float32')
#             for col in data.select_dtypes(include=['object']).columns:
#                 data[col] = data[col].astype('float32')
#             data.to_hdf(data_name, key='data', mode='w')


# def gen_exchange_ratio_train_data(exchange):
#     upper_exchange = exchange[0].upper() + exchange[1:]

#     # 加载数据
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     daily_path = os.path.join(current_dir, '../data/'+ upper_exchange + '/daily_' + exchange + '.csv')
#     daily_data = pd.read_csv(daily_path, encoding="utf-8")
#     ratio_path = os.path.join(current_dir, '../data/'+ upper_exchange + '/ratio_' + exchange + '.csv')
#     ratio_data = pd.read_csv(ratio_path, encoding="utf-8")
#     print(ratio_data)

#     groups = list(ratio_data.groupby('symbol'))
#     random.shuffle(groups)
#     group_count = len(groups)
#     split_count = int(0.333*group_count)
#     groups_0 = groups[:split_count]
#     groups_1 = groups[split_count:2*split_count]
#     groups_2 = groups[2*split_count:]

#     # 创建线程并启动它们
#     thread0 = threading.Thread(target=gen_group_train_data, args=(groups_0, daily_data))
#     thread1 = threading.Thread(target=gen_group_train_data, args=(groups_1, daily_data))
#     thread2 = threading.Thread(target=gen_group_train_data, args=(groups_2, daily_data))

#     thread0.start()
#     thread1.start()
#     thread2.start()

#     thread0.join()
#     thread1.join()
#     thread2.join()


# def gen_ratio_train_data():
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     train_folder = os.path.join(current_dir, 'train')
#     shutil.rmtree(train_folder, ignore_errors=True)
#     os.makedirs(train_folder, exist_ok=True)
#     # 获取当前脚本所在目录
#     csv_path = os.path.join(current_dir, '..', 'train_exchanges.csv')
#     csv_path = os.path.abspath(csv_path)  # 转成绝对路径

#     exchanges_data = pd.read_csv(csv_path, encoding="utf-8")
#     exchange_list = exchanges_data['exchange'].tolist()
#     failed_list = []
#     for exchange in exchange_list:
#         try:
#             print(exchange)
#             gen_exchange_ratio_train_data(exchange)
#         except:
#             failed_list.append(exchange)

#     print(failed_list)


# if __name__ == '__main__':
#     gen_ratio_train_data()


import os
import pandas as pd
import random
import numpy as np
import math
from tqdm import tqdm
import shutil
from multiprocessing import Pool, cpu_count

pd.set_option('future.no_silent_downcasting', True)


def process_group(group_tuple, daily_data, train_folder):
    """
    处理单个股票数据组并生成训练文件。
    这是一个独立的函数，适合在多进程中运行。
    """
    symbol = group_tuple[0]
    group = group_tuple[1]

    # 1. 数据清洗和准备
    # 去除最后7个季度的数据
    group = group.iloc[:-7]
    group = group.fillna(0).reset_index(drop=True)
    group_data_length = len(group)

    if group_data_length < 3:
        return 0 # 返回成功处理的文件数，这里是 0

    # 优化：提前筛选 daily_data
    group_daily = daily_data[daily_data['symbol'] == symbol].reset_index(drop=True)
    if group_daily.empty:
        return 0
    
    # 提前转换类型
    group['endDate'] = group['endDate'].astype('int32')

    files_created = 0

    # 循环生成训练样本
    for j in range(2, group_data_length-1):
        endDate = group['endDate'].iloc[j] # 这里的 endDate 应该是当前样本的最新财务报告期

        # 预测期 daily 数据
        fore_daily = group_daily[group_daily['date'] > int(endDate)]
        fore_daily = fore_daily.iloc[:127*3] # 截取前 3 个季度（约 127*3 天）
        
        if len(fore_daily) != 127*3:
            continue
        # 检查预测期价格是否有非正数（0或负数）
        if (fore_daily['close'] <= 0).any():
            # 发现非正数价格，跳过此样本
            continue
        # 优化：使用 .agg(['min', 'median', 'max']) 简化操作
        fore_stats = fore_daily['close'].agg(['min', 'median', 'max'])
        price_fore = sum(math.log(x) for x in fore_stats)

        # 历史期 daily 数据 (在 endDate_split 之前，取最近 127*3 天)
        past_daily = group_daily[group_daily['date'] <= int(endDate)]
        past_daily = past_daily.iloc[-127*3:]
        
        if len(past_daily) != 127*3:
            continue
        # 检查历史期价格是否有非正数（0或负数）
        if (past_daily['close'] <= 0).any():
            # 发现非正数价格，跳过此样本
            continue
        # 优化：使用 .agg(['min', 'median', 'max']) 简化操作
        past_stats = past_daily['close'].agg(['min', 'median', 'max'])
        price_past = sum(math.log(x) for x in past_stats)
        
        three = price_fore - price_past
        
        if abs(three) > 127: # 异常值过滤
            continue           
        
        # 截取 3 个季度的财务数据 (j-2, j-1, j)
        data = group.iloc[j - 2:j + 1, 5:12].copy().reset_index(drop=True) # 使用 .copy() 避免 SettingWithCopyWarning
        if len(data) != 3:
            continue
        # data.drop(columns=['symbol', 'endDate', 'netAssetValuePerShare', 'dcfPerShare', 'dividendPerShare'], inplace=True)
        data = data.assign(three=three)
        data = data.fillna(0)
        data.replace([np.inf, -np.inf], 0, inplace=True)
        data[(data > 127.0)] = 127.0
        data[(data < -127.0)] = -127.0
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
    
    # 获取交易所列表
    csv_path = os.path.join(current_dir, '..', 'train_exchanges.csv')
    csv_path = os.path.abspath(csv_path)

    try:
        exchanges_data = pd.read_csv(csv_path, encoding="utf-8")
        exchange_list = exchanges_data['exchange'].tolist()
    except FileNotFoundError:
        print(f"Error: train_exchanges.csv not found at {csv_path}")
        return

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