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

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

pd.set_option('future.no_silent_downcasting', True)


# #求取上个季度的最后一天, 如果已经是季度末则返回原值
# def get_quarter_end_date(date_str):
#     date_str = str(date_str)
#     date_check = date_str[-4:]
#     if date_check == '1231' or date_check == '0331' or date_check == '0630' or date_check =='0930':
#         return date_str
#     date = datetime.datetime.strptime(date_str, '%Y%m%d').date()
#     quarter_month = ((date.month-1)//3) * 3 + 1
#     # 构建季度末日期
#     quarter_end_date = datetime.date(date.year, quarter_month, 1) + datetime.timedelta(days=-1)
#     # 将日期格式转换为字符串格式
#     quarter_end_date_str = quarter_end_date.strftime('%Y%m%d')
#     return quarter_end_date_str


# def gen_group_train_data(groups, mean_data, std_data, daily_data):
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     train_folder = os.path.join(current_dir, 'train2')
#     for i in tqdm(range(len(groups))):
#         group = groups[i][1]
#         # 去除最后7个季度的数据
#         group = group.iloc[:-7]
#         group = group.fillna(0).reset_index(drop=True)
#         group_data_length = len(group)
#         # 目前必须上市31个季度后才可以预测
#         if group_data_length < 31:
#             continue
#         symbol = groups[i][0]
#         group_daily = daily_data[(daily_data['symbol'] == symbol)].reset_index(drop=True)

#         for j in range(31, group_data_length-1):
#             group['endDate'] = group['endDate'].astype('int32')
#             endDate = group['endDate'].iloc[j]
#             data_basename = symbol + '_' + str(endDate) + '.h5'
#             data_name = os.path.join(train_folder, data_basename)

#             # 截取daily数据
#             past_daily = group_daily[(group_daily['date'] <= int(endDate))]
#             past_daily = past_daily.iloc[-127*3:]
#             past_daily = past_daily.reset_index(drop=True)
#             if len(past_daily) != 127*3:
#                 continue
#             max_past = past_daily['close'].max()
#             median_past = past_daily['close'].median()
#             min_past = past_daily['close'].min()
#             math_past = math.log(max_past) + math.log(median_past) + math.log(min_past)
#             ##获取one的label
#             fore_daily = group_daily[(group_daily['date'] > int(endDate))]
#             fore_daily = fore_daily.iloc[:127]
#             fore_daily = fore_daily.reset_index(drop=True)
#             if len(fore_daily) != 127:
#                 continue
#             max_fore = fore_daily['close'].max()
#             median_fore = fore_daily['close'].median()
#             min_fore = fore_daily['close'].min()
#             one = math.log(max_fore) + math.log(median_fore) + math.log(min_fore) - math_past
#             if one > 127 or one < -127:
#                 continue
#             ##获取three的label
#             fore_daily = group_daily[(group_daily['date'] > int(endDate))]
#             fore_daily = fore_daily.iloc[:127*3]
#             fore_daily = fore_daily.reset_index(drop=True)
#             if len(fore_daily) != 127*3:
#                 continue
#             max_fore = fore_daily['close'].max()
#             median_fore = fore_daily['close'].median()
#             min_fore = fore_daily['close'].min()
#             three = math.log(max_fore) + math.log(median_fore) + math.log(min_fore) - math_past
#             if three > 127 or three < -127:
#                 continue
#             ##获取seven的label
#             fore_daily = group_daily[(group_daily['date'] > int(endDate))]
#             fore_daily = fore_daily.iloc[:127*7]
#             fore_daily = fore_daily.reset_index(drop=True)
#             if len(fore_daily) != 127*7:
#                 continue
#             max_fore = fore_daily['close'].max()
#             median_fore = fore_daily['close'].median()
#             min_fore = fore_daily['close'].min()
#             seven = math.log(max_fore) + math.log(median_fore) + math.log(min_fore) - math_past
#             if seven > 127 or seven < -127:
#                 continue
#             # 截取daily数据
#             # 下面进行归一化
#             # data = group[(group['endDate'] <= int(endDate))].reset_index(drop=True)
#             # data = data.iloc[-17:].reset_index(drop=True)
#             # print(endDate)
#             # print(data)
#             data = group.iloc[j - 30:j + 1].reset_index(drop=True)
#             if len(data) != 31:
#                 continue
#             data.drop(columns=['symbol'], inplace=True)
#             data.drop(columns=['endDate'], inplace=True)
#             col_names = data.columns.values
#             mean_col_names = mean_data.columns.values
#             for k in range(len(data.columns)):
#                 col_name = col_names[k]
#                 if col_name not in mean_col_names:
#                     continue
#                 mean_value = mean_data.loc[mean_data['endDate'] == int(endDate), col_name].item()
#                 std_value = std_data.loc[std_data['endDate'] == int(endDate), col_name].item()
#                 if std_value != 0:
#                     data[col_name] = data[col_name] - mean_value
#                     data[col_name] = data[col_name] / std_value
#                 else:
#                     data[col_name] = 0

#             data = data.assign(one=one)
#             data = data.assign(three=three)
#             data = data.assign(seven=seven)
#             data = data.fillna(0)
#             data.replace([np.inf, -np.inf], 0, inplace=True)
#             for col in data.select_dtypes(include=['int64']).columns:
#                 data[col] = data[col].astype('float32')
#             for col in data.select_dtypes(include=['float64']).columns:
#                 data[col] = data[col].astype('float32')
#             for col in data.select_dtypes(include=['object']).columns:
#                 data[col] = data[col].astype('float32')
#             data.to_hdf(data_name, key='data', mode='w')

# def gen_exchange_three_train_data_pre(exchange):
#     upper_exchange = exchange[0].upper() + exchange[1:]
#     # 加载数据
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     daily_path = os.path.join(current_dir, '../data/'+ upper_exchange + '/daily_' + exchange + '.csv')
#     daily_data = pd.read_csv(daily_path, encoding="utf-8")
#     income_path = os.path.join(current_dir, '../data/'+ upper_exchange + '/income_' + exchange + '.csv')
#     income_data = pd.read_csv(income_path, encoding="utf-8")
#     balance_path = os.path.join(current_dir, '../data/'+ upper_exchange + '/balance_' + exchange + '.csv')
#     balance_data = pd.read_csv(balance_path, encoding="utf-8")
#     cashflow_path = os.path.join(current_dir, '../data/'+ upper_exchange + '/cashflow_' + exchange + '.csv')
#     cashflow_data = pd.read_csv(cashflow_path, encoding="utf-8")
#     mean_path = os.path.join(current_dir, '../data/'+ upper_exchange + '/mean_' + exchange + '.csv')
#     mean_data = pd.read_csv(mean_path, encoding="utf-8")
#     std_path = os.path.join(current_dir, '../data/'+ upper_exchange + '/std_' + exchange + '.csv')
#     std_data = pd.read_csv(std_path, encoding="utf-8")

#     # 合并财务相关数据
#     financial_data = pd.merge(income_data, balance_data, on=['symbol', 'endDate'], how='outer')
#     financial_data = pd.merge(financial_data, cashflow_data, on=['symbol', 'endDate'], how='outer')
#     financial_data = financial_data.dropna().reset_index(drop=True)
#     financial_data['endDate'] = financial_data['endDate'].apply(get_quarter_end_date)
#     financial_data.drop_duplicates(subset=['symbol', 'endDate'], keep='first', inplace=True)
#     financial_data = financial_data.reset_index(drop=True)

#     #删除financial_data中endDate小于mean std data中最早的endDate的所有行
#     financial_data = financial_data[financial_data['endDate'].astype(float) >= mean_data['endDate'].iloc[0].astype(float)]
#     financial_data = financial_data.reset_index(drop=True)
#     financial_data = financial_data[financial_data['endDate'].astype(float) >= std_data['endDate'].iloc[0].astype(float)]
#     financial_data = financial_data.reset_index(drop=True)
#     print(financial_data)

#     groups = list(financial_data.groupby('symbol'))
#     random.shuffle(groups)
#     group_count = len(groups)
#     split_count = int(0.333*group_count)
#     groups_0 = groups[:split_count]
#     groups_1 = groups[split_count:2*split_count]
#     groups_2 = groups[2*split_count:]

#     # 创建线程并启动它们
#     thread0 = threading.Thread(target=gen_group_train_data, args=(groups_0, mean_data, std_data, daily_data))
#     thread1 = threading.Thread(target=gen_group_train_data, args=(groups_1, mean_data, std_data, daily_data))
#     thread2 = threading.Thread(target=gen_group_train_data, args=(groups_2, mean_data, std_data, daily_data))

#     thread0.start()
#     thread1.start()
#     thread2.start()

#     thread0.join()
#     thread1.join()
#     thread2.join()


# 共享数据（daily/mean/std）改成每个 worker 自己读一份，任务参数里只剩 group_tuple。
# 2026-09-29 改：原来是**每个任务都塞一整份 daily_data** —— Windows 是 spawn，TSX 那 1033 个
# 任务就把 259MB 的 daily_tsx.csv（读进内存更大）分别序列化 1033 次、32 个 worker 各常驻一份，
# 峰值贴着内存上限；那次有个 worker 反序列化到一半抛 MemoryError 死掉，而
# multiprocessing.Pool **不会因为 worker 猝死而报错**，只是永远等那个丢掉的返回值
# → 整条流水线静默卡死（19:01:31 之后日志再无写入、父进程 CPU 增量 0.00 秒）。
_W = {}


def _init_worker(daily_path, mean_path, std_path, train_folder, exchange):
    """每个 worker 进程启动时读一次共享数据（spawn 下父进程的内存不会被子进程继承）。"""
    _W['daily'] = pd.read_csv(daily_path, encoding='utf-8')
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
    daily_data = _W['daily']
    train_folder = _W['train_folder']

    # 1. 数据清洗和准备
    # 去除最后7个季度的数据
    group = group.iloc[:-7]
    # NaN 财务字段不在这里填 0：留到 normalize 之后、per-sample 的 data.fillna(0) 再填，
    # 让「缺失 = 中性(0)」与推理端一致。否则缺失会被当成"0 美元"、z-score 成 -mean/std。
    group = group.reset_index(drop=True)
    group_data_length = len(group)

    if group_data_length < 31:
        return 0, 0 # 返回 (成功处理的文件数, 因 mean/std 缺失跳过的锚点数)

    # 优化：提前筛选 daily_data
    group_daily = daily_data[daily_data['symbol'] == symbol].reset_index(drop=True)
    if group_daily.empty:
        return 0, 0

    # 提前转换类型
    group['endDate'] = group['endDate'].astype('int32')

    files_created = 0
    n_skip_norm = 0

    # 循环生成训练样本
    for j in range(30, group_data_length-1):
        endDate = group['endDate'].iloc[j] # 这里的 endDate 应该是当前样本的最新财务报告期

        # 预测期 daily 数据
        one_fore_daily = group_daily[group_daily['date'] > int(endDate)].reset_index(drop=True)
        one_fore_daily = one_fore_daily.iloc[:127]
        
        if len(one_fore_daily) != 127:
            continue
        # 检查预测期价格是否有非正数（0或负数）
        if (one_fore_daily['close'] <= 0).any():
            # 发现非正数价格，跳过此样本
            continue
        # 优化：使用 .agg(['min', 'median']) 简化操作
        # one_fore_stats = one_fore_daily['close'].agg(['min', 'median'])
        # # one_fore_stats = one_fore_daily['close'].agg(['median'])
        # one_fore_price = sum(math.log(x) for x in one_fore_stats)
        # one_max_fore = one_fore_daily['close'].max()
        one_median_fore = one_fore_daily['close'].median()
        one_min_fore = one_fore_daily['close'].min()

        three_fore_daily = group_daily[group_daily['date'] > int(endDate)].reset_index(drop=True)
        three_fore_daily = three_fore_daily.iloc[:127*3] 
        
        if len(three_fore_daily) != 127*3:
            continue
        # 检查预测期价格是否有非正数（0或负数）
        if (three_fore_daily['close'] <= 0).any():
            # 发现非正数价格，跳过此样本
            continue
        # # 优化：使用 .agg(['min', 'median']) 简化操作
        # three_fore_stats = three_fore_daily['close'].agg(['min', 'median'])
        # # three_fore_stats = three_fore_daily['close'].agg(['median'])
        # three_fore_price = sum(math.log(x) for x in three_fore_stats)
        three_median_fore = three_fore_daily['close'].median()
        three_min_fore = three_fore_daily['close'].min()

        seven_fore_daily = group_daily[group_daily['date'] > int(endDate)].reset_index(drop=True)
        seven_fore_daily = seven_fore_daily.iloc[:127*7] 
        
        if len(seven_fore_daily) != 127*7:
            continue
        # 检查预测期价格是否有非正数（0或负数）
        if (seven_fore_daily['close'] <= 0).any():
            # 发现非正数价格，跳过此样本
            continue
        # # 优化：使用 .agg(['min', 'median']) 简化操作
        # seven_fore_stats = seven_fore_daily['close'].agg(['min', 'median'])
        # # seven_fore_stats = seven_fore_daily['close'].agg(['median'])
        # seven_fore_price = sum(math.log(x) for x in seven_fore_stats)
        seven_median_fore = seven_fore_daily['close'].median()
        seven_min_fore = seven_fore_daily['close'].min()

        # 历史期 daily 数据 (在 endDate_split 之前，取最近 127*3 天)
        past_daily = group_daily[group_daily['date'] <= int(endDate)].reset_index(drop=True)
        past_daily = past_daily.iloc[-127*3:]
        
        if len(past_daily) != 127*3:
            continue
        # 检查历史期价格是否有非正数（0或负数）
        if (past_daily['close'] <= 0).any():
            # 发现非正数价格，跳过此样本
            continue
        # # 优化：使用 .agg([ 'median', 'max']) 简化操作
        # past_stats = past_daily['close'].agg(['median', 'max'])
        # # past_stats = past_daily['close'].agg(['median'])
        # past_price = sum(math.log(x) for x in past_stats)
        max_past = past_daily['close'].max()
        median_past = past_daily['close'].median()
        
        growth_one = one_median_fore/max_past
        growth_three = three_median_fore/max_past
        growth_seven = seven_median_fore/max_past
        death_one = median_past/one_min_fore
        death_three = median_past/three_min_fore
        death_seven = median_past/seven_min_fore

        
        if abs(growth_one) > 7 or abs(growth_three) > 31 or abs(growth_seven) > 127: # 异常值过滤
            continue
        if abs(death_one) > 7 or abs(death_three) > 31 or abs(death_seven) > 127: # 异常值过滤
            continue
            
        # 2. 训练输入 (财务比率) 准备
        
        # 截取 3 个季度的财务数据 (j-2, j-1, j)
        data = group.iloc[j - 30:j + 1].copy().reset_index(drop=True) # 使用 .copy() 避免 SettingWithCopyWarning
        if len(data) != 31:
            continue
        
        data.drop(columns=['symbol', 'endDate'], inplace=True)
        if normalize_by_mean_std(data, endDate, mean_data, std_data) is None:
            n_skip_norm += 1
            continue
        data = data.assign(growth_one=growth_one)
        data = data.assign(growth_three=growth_three)
        data = data.assign(growth_seven=growth_seven)
        data = data.assign(death_one=death_one)
        data = data.assign(death_three=death_three)
        data = data.assign(death_seven=death_seven)
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


def gen_exchange_three_train_data(exchange):
    """
    加载数据，分割任务并使用多进程处理。
    """
    upper_exchange = exchange[0].upper() + exchange[1:]
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. 加载数据
    daily_path = os.path.join(current_dir, f'../data/{upper_exchange}/daily_{exchange}.csv')
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
    # 大所降并发：每个 worker 在 _init_worker 里把整个 daily CSV 读进内存（DataFrame 常驻约
    # 2× 文件体积、read_csv 峰值约 4×）。32 核 × NASDAQ(935MB→峰值~4GB) ≈ 128GB，会把 64GB
    # 机器打爆 → worker 在 read_csv 时 OOM 猝死 → BrokenProcessPool → 该所 0 文件。
    # 把「worker 数 × 峰值」压在 24GB 内；小所照旧用满全部核。
    daily_mb = os.path.getsize(daily_path) / 1048576
    per_worker_peak_mb = daily_mb * 4
    if per_worker_peak_mb > 0:
        num_processes = max(2, min(num_processes, int(24000 / per_worker_peak_mb)))
    print(f"Starting {num_processes} processes for {len(groups)} groups.")
    
    # 任务参数只传 group_tuple —— daily/mean/std 交给 _init_worker，每个 worker 读一份
    task_args = list(groups)
    
    # 4. 运行多进程池
    total_files_created = 0
    total_skipped_norm = 0
    try:
        # 用 ProcessPoolExecutor 而不是 Pool：worker 意外死掉（内存不够等）时它会抛
        # BrokenProcessPool，被下面的 except 接住、记成"这个交易所失败"继续跑下一个所，
        # 而不是像 Pool 那样永远静默等下去。
        # 进度条改成 as_completed —— Pool.starmap 返回的是 list，原来那根条是结果全到手
        # 之后才瞬间刷完的（日志里的 100% / 00:00 / 238 万 it/s 全是假象）。
        with ProcessPoolExecutor(max_workers=num_processes, initializer=_init_worker,
                                 initargs=(daily_path, mean_path, std_path, train_folder, exchange)) as ex:
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


def gen_three_train_data(exchanges=None):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_folder = os.path.join(current_dir, 'train')

    # 获取交易所列表：data/ 下全部交易所（已在上游过滤，不再套 num_symbols 阈值）
    data_dir = os.path.join(current_dir, '..', 'data')
    all_exchanges = sorted(d for d in os.listdir(data_dir)
                           if os.path.isdir(os.path.join(data_dir, d)))

    if exchanges:
        # --exchanges：只补跑这几个所，**不清空** train/（别的所的 h5 原样留着）
        print(f"指定交易所 {exchanges} —— 保留现有 train/ 内容，只补跑这几个所。")
        os.makedirs(train_folder, exist_ok=True)
        want = {e.upper() for e in exchanges}
        exchange_list = [e for e in all_exchanges if e.upper() in want]
        nosuch = sorted(want - {e.upper() for e in exchange_list})
        if nosuch:
            print(f"⚠️ --exchanges 里这些名字在 data/ 下不存在（已忽略）：{nosuch}")
        if not exchange_list:
            print("⚠️ 没有任何交易所匹配，什么都没跑。")
            return
        print(f"实际重跑：{exchange_list}")
    else:
        # 清理和创建训练目录（全量重跑才清空）
        print("Clearing and creating 'train' folder...")
        shutil.rmtree(train_folder, ignore_errors=True)
        os.makedirs(train_folder, exist_ok=True)
        exchange_list = all_exchanges

    failed_list = []
    total_files = 0
    
    for exchange in exchange_list:
        try:
            print(f"\n--- Starting processing for {exchange} ---")
            files_created = gen_exchange_three_train_data(exchange)
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


def main():
    import argparse
    p = argparse.ArgumentParser(description='three：生成训练 h5')
    p.add_argument('--exchanges', nargs='+', default=None,
                   help='只跑 data/ 下这几个交易所（如 TSX TSXV）。指定时**不清空** train/，'
                        '用于补跑；不指定＝全量重跑（先清空 train/），这也是 refresh_models 的用法')
    args = p.parse_args()
    gen_three_train_data(args.exchanges)


if __name__ == '__main__':
    main()