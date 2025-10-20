# import os
# import pandas as pd
# from tqdm import tqdm
# import random
# import numpy as np
# import threading
# import statistics
# import math
# import shutil

# pd.set_option('future.no_silent_downcasting', True)


# def add_technical_factor(data):
#     # 均线
#     data['ma3'] = data['close'].rolling(3).mean()
#     data['ma7'] = data['close'].rolling(7).mean()
#     data['ma31'] = data['close'].rolling(31).mean()
#     # rsi
#     delta = data['close'].diff()
#     gain3 = delta.where(delta > 0, 0).rolling(3).mean()
#     loss3 = -delta.where(delta < 0, 0).rolling(3).mean()
#     data['rsi3'] = (100 - (100 / (1 + (gain3 / (loss3 + 1e-6))))) * 0.01
#     gain7 = delta.where(delta > 0, 0).rolling(7).mean()
#     loss7 = -delta.where(delta < 0, 0).rolling(7).mean()
#     data['rsi7'] = (100 - (100 / (1 + (gain7 / (loss7 + 1e-6))))) * 0.01
#     gain31 = delta.where(delta > 0, 0).rolling(31).mean()
#     loss31 = -delta.where(delta < 0, 0).rolling(31).mean()
#     data['rsi31'] = (100 - (100 / (1 + (gain31 / (loss31 + 1e-6))))) * 0.01
#     # atr
#     data['atr3'] = (data['delta'].rolling(3).mean())*31
#     data['atr7'] = (data['delta'].rolling(7).mean())*31
#     data['atr31'] = (data['delta'].rolling(31).mean())*31
#     # obv
#     obv = delta * data['volume']
#     data['obv3'] = (obv.rolling(3).mean())*31
#     data['obv7'] = (obv.rolling(7).mean())*31
#     data['obv31'] = (obv.rolling(31).mean())*31
#     # corr
#     data['corr3'] = data['volume'].rolling(3).corr(data['close'])
#     data['corr7'] = data['volume'].rolling(7).corr(data['close'])
#     data['corr31'] = data['volume'].rolling(31).corr(data['close'])
#     # curvature
#     data['curvature'] = (data['close'].diff().diff())*31
#     # vma
#     data['vma_3_7'] = data['volume'].rolling(window=3).mean()/data['volume'].rolling(window=7).mean() - 1
#     data['vma_7_31'] = data['volume'].rolling(window=7).mean()/data['volume'].rolling(window=31).mean() - 1
#     # factor
#     data['factor'] = (data['close'].pct_change(3) - data['volume'].rolling(7).std()) 
#     # overnight
#     overnight = data['open']*data['close'] / data['close'].shift(1) - 1
#     data['overnight3'] = overnight.rolling(3).mean()*31
#     data['overnight7'] = overnight.rolling(7).mean()*31
#     data['overnight31'] = overnight.rolling(31).mean()*31
#     # aplha22
#     # 3_7
#     rolling_corr_3 = (data['high']*data['close']).rolling(3).corr(data['volume'])
#     delta_corr_3 = rolling_corr_3.diff(3)
#     std_close_7 = data['close'].rolling(7).std()
#     data['aplha22_3_7'] = -1 * (delta_corr_3 * std_close_7)*31
#     # 7_31
#     rolling_corr_7 = (data['high']*data['close']).rolling(7).corr(data['volume'])
#     delta_corr_7 = rolling_corr_7.diff(7)
#     std_close_31 = data['close'].rolling(31).std()
#     data['aplha22_7_31'] = -1 * (delta_corr_7 * std_close_31)*31
#     return data

# def gen_group_train_data(groups):
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     train_folder = os.path.join(current_dir, 'train')
#     days_input = 127*3
#     days_output = 31
#     for i in tqdm(range(len(groups))):
#         symbol = groups[i][0]
#         daily_group = groups[i][1].reset_index(drop=True)
#         #除去最新127天的数据
#         daily_group = daily_group.iloc[:-127]
#         daily_group['date'] = pd.to_numeric(daily_group['date'], errors='coerce').astype('Int64')
#         # 我们用127*3 + 31个交易日数据进行训练，127*3个交易日作为输入，31个交易日作为输出
#         group_data_length = len(daily_group)
#         if group_data_length < 127*3 + 31:
#             continue
#         for j in range(days_input, group_data_length - days_output):
#             date = daily_group['date'].iloc[j]
#             data_basename = symbol + '_' + str(date) + '.h5'
#             # data_name = 'train/' + data_basename
#             data_name = os.path.join(train_folder, data_basename)
#             if random.random() < 1.0/31.0:  # 随机保存1/31的数据
#                 data = daily_group.iloc[j - days_input + 1:j + days_output + 1]
#                 data = data.reset_index(drop=True)
#                 if len(data) != (127*3 + 31):
#                     continue
#                 data['open'] = data['open']/(data['close'] + 1e-7)
#                 data['high'] = data['high']/(data['close'] + 1e-7)
#                 data['low'] = data['low']/(data['close'] + 1e-7)
#                 data['delta'] = data['high'] - data['low']
#                 ref_close = data['close'].iloc[-32]
#                 ref_volume = data['volume'].iloc[-32]
#                 if ref_close <= 0 or ref_volume <= 0:
#                     continue
#                 data['close'] = data['close'] / ref_close
#                 data['volume'] = data['volume'] / ref_volume
#                 close_tomorrow = data['close'].iloc[-31]
#                 volume_tomorrow = data['volume'].iloc[-31]
#                 if close_tomorrow <= 0 or volume_tomorrow <= 0:
#                     continue
#                 close_fore = data['close'].iloc[127*3:127*3+31]
#                 close_fore_median = statistics.median(close_fore)
#                 close_fore_min = min(close_fore)
#                 close_fore_max = max(close_fore)
#                 close_31 = (math.log(close_fore_median) + math.log(close_fore_min) + math.log(close_fore_max) - 3*math.log(close_tomorrow))
#                 if abs(close_31) > 127:
#                     continue   
#                 data = data.fillna(0)
#                 data.replace([np.inf, -np.inf], 0, inplace=True)
#                 # data['date'] = data['date'].astype('int32')
#                 # 添加一些日线技术指标相关的因子
#                 data = data[:127*3].reset_index(drop=True)
#                 data_fore = data[127*3:].reset_index(drop=True)
#                 data = add_technical_factor(data)
#                 data_fore = add_technical_factor(data_fore)
#                 data = pd.concat([data, data_fore], axis=0)
#                 data = data.reset_index(drop=True)
#                 #***********
#                 data['idx'] = data.index / (127.0*3 - 1.0)
#                 data.drop(columns=['symbol'], inplace=True)
#                 data.drop(columns=['date'], inplace=True)
#                 #***************************
#                 data = data.fillna(0)
#                 data.replace([np.inf, -np.inf], 0, inplace=True)
#                 for col in data.select_dtypes(include=['int64']).columns:
#                     data[col] = data[col].astype('float32')
#                 for col in data.select_dtypes(include=['float64']).columns:
#                     data[col] = data[col].astype('float32')
#                 for col in data.select_dtypes(include=['object']).columns:
#                     data[col] = data[col].astype('float32')
#                 data[(data > 127.0)] = 127.0
#                 data[(data < -127.0)] = -127.0
#                 data.to_hdf(data_name, key='data', mode='w')


# def gen_exchange_one_train_data(exchange):
#     upper_exchange = exchange[0].upper() + exchange[1:]

#     # 加载数据
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     daily_path = os.path.join(current_dir, '../data/'+ upper_exchange + '/daily_' + exchange + '.csv')
#     daily_data = pd.read_csv(daily_path, encoding="utf-8")
#     print(daily_data)

#     groups = list(daily_data.groupby('symbol'))
#     random.shuffle(groups)
#     group_count = len(groups)
#     split_count = int(0.333*group_count)
#     groups_0 = groups[:split_count]
#     groups_1 = groups[split_count:2*split_count]
#     groups_2 = groups[2*split_count:]

#     # 创建线程并启动它们
#     thread0 = threading.Thread(target=gen_group_train_data, args=(groups_0, ))
#     thread1 = threading.Thread(target=gen_group_train_data, args=(groups_1, ))
#     thread2 = threading.Thread(target=gen_group_train_data, args=(groups_2, ))

#     thread0.start()
#     thread1.start()
#     thread2.start()

#     thread0.join()
#     thread1.join()
#     thread2.join()

# def gen_one_train_data():
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     train_folder = os.path.join(current_dir, 'train')
#     shutil.rmtree(train_folder, ignore_errors=True)
#     os.makedirs(train_folder, exist_ok=True)
#     exchange_list = ['SHZ', 'SHH']
#     failed_list = []
#     exchanges = exchange_list
#     # random.shuffle(exchanges)
#     for exchange in exchanges:
#         try:
#             gen_exchange_one_train_data(exchange)
#         except:
#             failed_list.append(exchange)
#             pass
#     print(failed_list)


# if __name__ == '__main__':
#     gen_one_train_data()
#     # convert_data()
#     # clean_data()

import os
import pandas as pd
# 使用 tqdm 仍然可能导致多进程输出混乱，但为了保持原代码逻辑保留
from tqdm import tqdm
import random
import numpy as np
import multiprocessing 
from multiprocessing import Pool # 引入 Pool
import statistics
import math
import shutil

# 必须在 Windows 上使用多进程时引入
if os.name == 'nt':
    multiprocessing.freeze_support()

pd.set_option('future.no_silent_downcasting', True)


def add_technical_factor(data):
    """计算股票技术因子"""
    # 均线
    data['ma3'] = data['close'].rolling(3).mean()
    data['ma7'] = data['close'].rolling(7).mean()
    data['ma31'] = data['close'].rolling(31).mean()
    # rsi
    delta = data['close'].diff()
    gain3 = delta.where(delta > 0, 0).rolling(3).mean()
    loss3 = -delta.where(delta < 0, 0).rolling(3).mean()
    data['rsi3'] = (100 - (100 / (1 + (gain3 / (loss3 + 1e-6))))) * 0.01
    gain7 = delta.where(delta > 0, 0).rolling(7).mean()
    loss7 = -delta.where(delta < 0, 0).rolling(7).mean()
    data['rsi7'] = (100 - (100 / (1 + (gain7 / (loss7 + 1e-6))))) * 0.01
    gain31 = delta.where(delta > 0, 0).rolling(31).mean()
    loss31 = -delta.where(delta < 0, 0).rolling(31).mean()
    data['rsi31'] = (100 - (100 / (1 + (gain31 / (loss31 + 1e-6))))) * 0.01
    # atr
    data['atr3'] = (data['delta'].rolling(3).mean())*31
    data['atr7'] = (data['delta'].rolling(7).mean())*31
    data['atr31'] = (data['delta'].rolling(31).mean())*31
    # obv
    obv = delta * data['volume']
    data['obv3'] = (obv.rolling(3).mean())*31
    data['obv7'] = (obv.rolling(7).mean())*31
    data['obv31'] = (obv.rolling(31).mean())*31
    # corr
    data['corr3'] = data['volume'].rolling(3).corr(data['close'])
    data['corr7'] = data['volume'].rolling(7).corr(data['close'])
    data['corr31'] = data['volume'].rolling(31).corr(data['close'])
    # curvature
    data['curvature'] = (data['close'].diff().diff())*31
    # vma
    data['vma_3_7'] = data['volume'].rolling(window=3).mean()/data['volume'].rolling(window=7).mean() - 1
    data['vma_7_31'] = data['volume'].rolling(window=7).mean()/data['volume'].rolling(window=31).mean() - 1
    # factor
    data['factor'] = (data['close'].pct_change(3) - data['volume'].rolling(7).std()) 
    # overnight
    overnight = data['open']*data['close'] / data['close'].shift(1) - 1
    data['overnight3'] = overnight.rolling(3).mean()*31
    data['overnight7'] = overnight.rolling(7).mean()*31
    data['overnight31'] = overnight.rolling(31).mean()*31
    # aplha22
    # 3_7
    rolling_corr_3 = (data['high']*data['close']).rolling(3).corr(data['volume'])
    delta_corr_3 = rolling_corr_3.diff(3)
    std_close_7 = data['close'].rolling(7).std()
    data['aplha22_3_7'] = -1 * (delta_corr_3 * std_close_7)*31
    # 7_31
    rolling_corr_7 = (data['high']*data['close']).rolling(7).corr(data['volume'])
    delta_corr_7 = rolling_corr_7.diff(7)
    std_close_31 = data['close'].rolling(31).std()
    data['aplha22_7_31'] = -1 * (delta_corr_7 * std_close_31)*31
    return data

# <<<<<<<< 关键修改在这里：函数签名现在接收两个参数 >>>>>>>>
def gen_group_train_data(groups, train_folder):
    """
    生成训练数据，并在多进程中运行。
    groups: 股票分组列表
    train_folder: 训练数据保存路径
    """
    days_input = 127*3
    days_output = 31
    
    # 在进程中，使用 tqdm 可能会导致输出混乱，但为了显示进度，可以保留
    for i in tqdm(range(len(groups))):
        symbol = groups[i][0]
        # 复制数据以确保每个进程独立操作，防止意外修改共享数据
        daily_group = groups[i][1].copy(deep=True).reset_index(drop=True) 
        
        #除去最新127天的数据
        daily_group = daily_group.iloc[:-127].reset_index(drop=True)
        # 确保 date 是数值类型
        daily_group['date'] = pd.to_numeric(daily_group['date'], errors='coerce').astype('Int64')
        
        # 我们用127*3 + 31个交易日数据进行训练，127*3个交易日作为输入，31个交易日作为输出
        group_data_length = len(daily_group)
        if group_data_length < 127*3 + 31:
            continue
            
        window_total = days_input + days_output
        ref_idx = days_input - 1 # 归一化基准索引 (380)
        
        for j in range(days_input, group_data_length - days_output):
            date = daily_group['date'].iloc[j]
            data_basename = symbol + '_' + str(date) + '.h5'
            data_name = os.path.join(train_folder, data_basename)
            
            if random.random() < 1.0/31.0:  # 随机保存1/31的数据
                data = daily_group.iloc[j - days_input + 1:j + days_output + 1].copy(deep=True)
                data = data.reset_index(drop=True)
                
                if len(data) != window_total:
                    continue

                # --- 归一化 ---
                ref_close = data['close'].iloc[ref_idx]
                ref_volume = data['volume'].iloc[ref_idx]
                
                if ref_close <= 0 or ref_volume <= 0:
                    continue
                    
                data['open'] = data['open']/ref_close
                data['high'] = data['high']/ref_close
                data['low'] = data['low']/ref_close
                data['delta'] = data['high'] - data['low']
                data['close'] = data['close'] / ref_close
                data['volume'] = data['volume'] / ref_volume
                
                # --- 目标变量计算 ---
                close_tomorrow = data['close'].iloc[days_input]
                volume_tomorrow = data['volume'].iloc[days_input]
                if close_tomorrow <= 0 or volume_tomorrow <= 0:
                    continue
                    
                close_fore = data['close'].iloc[days_input:window_total]
                close_fore_median = statistics.median(close_fore)
                close_fore_min = min(close_fore)
                close_fore_max = max(close_fore)
                close_31 = (math.log(close_fore_median) + math.log(close_fore_min) + math.log(close_fore_max) - 3*math.log(close_tomorrow))
                
                if abs(close_31) > 127:
                    continue   
                    
                # --- 特征工程 ---
                data_input = data[:days_input].reset_index(drop=True)
                data_fore = data[days_input:].reset_index(drop=True)
                data_input = add_technical_factor(data_input)
                data_fore = add_technical_factor(data_fore)
                data = pd.concat([data_input, data_fore], axis=0)
                data = data.reset_index(drop=True)

                # --- 最终处理 ---
                data['idx'] = data.index / (days_input - 1.0)
                data.drop(columns=['symbol', 'date'], inplace=True, errors='ignore')
                
                data = data.fillna(0)
                data.replace([np.inf, -np.inf], 0, inplace=True)
                
                for col in data.columns:
                    try:
                        data[col] = data[col].astype('float32')
                    except Exception:
                        pass # 忽略无法转换的列

                data[(data > 127.0)] = 127.0
                data[(data < -127.0)] = -127.0
                
                # I/O 操作
                data.to_hdf(data_name, key='data', mode='w', format='fixed')

    return len(groups)

def gen_exchange_one_train_data(exchange):
    upper_exchange = exchange[0].upper() + exchange[1:]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    daily_path = os.path.join(current_dir, '../data/'+ upper_exchange + '/daily_' + exchange + '.csv')
    
    try:
        daily_data = pd.read_csv(daily_path, encoding="utf-8")
    except FileNotFoundError:
        print(f"Error: Data file not found at {daily_path}")
        return
        
    print(f"Loaded {upper_exchange} data with {len(daily_data)} rows.")

    groups = list(daily_data.groupby('symbol'))
    random.shuffle(groups)
    group_count = len(groups)
    
    # 进程数限制，使用 CPU 核心数
    NUM_PROCESSES = multiprocessing.cpu_count()
    split_size = group_count // NUM_PROCESSES
    
    groups_list_for_starmap = []
    train_folder = os.path.join(current_dir, 'train')

    for i in range(NUM_PROCESSES):
        start = i * split_size
        end = (i + 1) * split_size if i < NUM_PROCESSES - 1 else group_count
        # 创建元组 (groups_subset, train_folder)，供 starmap 解包
        groups_list_for_starmap.append((groups[start:end], train_folder))

    print(f"Starting {NUM_PROCESSES} processes for {upper_exchange}...")
    
    # 使用 Pool 运行多进程
    with Pool(processes=NUM_PROCESSES) as pool:
        # starmap 正确地解包 (groups, train_folder) 并传递给 gen_group_train_data(groups, train_folder)
        results = pool.starmap(gen_group_train_data, groups_list_for_starmap)
        
    print(f"Finished processing {upper_exchange}. Total groups processed: {sum(results)}")


def gen_one_train_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_folder = os.path.join(current_dir, 'train')
    
    # 清理和创建目录
    if os.path.exists(train_folder):
        print(f"Cleaning existing folder: {train_folder}")
        shutil.rmtree(train_folder)
    os.makedirs(train_folder, exist_ok=True)
    
    exchange_list = ['SHZ', 'SHH']
    failed_list = []
    exchanges = exchange_list
    
    for exchange in exchanges:
        try:
            gen_exchange_one_train_data(exchange)
        except Exception as e:
            failed_list.append((exchange, str(e)))
            print(f"Error processing {exchange}: {e}")
            pass
            
    print(f"Failed to process exchanges: {failed_list}")


if __name__ == '__main__':
    # 多进程必须在 if __name__ == '__main__': 块内运行
    print("Starting multi-process data generation...")
    gen_one_train_data()
    print("Data generation finished.")