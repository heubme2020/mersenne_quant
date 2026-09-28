# import numpy as np
# import pandas as pd
# import cupy as cp
# import datetime
# from tqdm import tqdm
# from catboost import CatBoostRegressor
# import os
# import torch

# pd.set_option('future.no_silent_downcasting', True)
# # 求取上个季度的最后一天

# def get_exchange_dcf(exchange):
#     print(exchange)
#     # 检查GPU是否可用
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model_name = os.path.join(os.path.dirname(__file__), 'seven.pt')
#     model = torch.load(model_name).to(device)

#     exchange = exchange.lower()
#     upper_exchange = exchange[0].upper() + exchange[1:]
#     exchange = exchange[0].lower() + exchange[1:]
#     data_name = os.path.join(os.path.dirname(__file__), '../data/' + upper_exchange)
#     income_data = pd.read_csv(data_name + '/income_' + exchange + '.csv')
#     balance_data = pd.read_csv(data_name + '/balance_' + exchange + '.csv')
#     cashflow_data = pd.read_csv(data_name + '/cashflow_' + exchange + '.csv')
#     mean_data = pd.read_csv(data_name + '/mean_' + exchange + '.csv')
#     std_data = pd.read_csv(data_name + '/std_' + exchange + '.csv')
#     indicator_data = pd.read_csv(data_name + '/indicator_' + exchange + '.csv')
#     # indicator_data.drop(columns=['operatingCapitalPerShare'], inplace=True)
#     features_data = pd.read_csv(data_name + '/../../three/features_importance.csv')

#     #合并财务相关数据
#     financial_data = pd.merge(income_data, balance_data, on=['symbol', 'endDate'], how='outer')
#     financial_data = pd.merge(financial_data, cashflow_data, on=['symbol', 'endDate'], how='outer')
#     print("1. 三张表 outer 合并后的最新日期：", financial_data['endDate'].max())
#     financial_data = financial_data.dropna()
#     print("2. 财务表 dropna() 后的最新日期：", financial_data['endDate'].max())
#     financial_data = financial_data.reset_index(drop=True)
#     financial_data.drop_duplicates(subset=['symbol', 'endDate'], keep='first', inplace=True)
#     financial_data = financial_data.reset_index(drop=True)
#     # 截取指定特征的部分
#     features_list = features_data['feature'].to_list()
#     features_data = pd.DataFrame()
#     features_data['symbol'] = financial_data['symbol']
#     features_data['endDate'] = financial_data['endDate'].astype('int64')
#     for feature in features_list:
#         features_data[feature] = financial_data[feature]
#     features_data = pd.merge(indicator_data, features_data, on=['symbol', 'endDate'], how='outer')
#     print("3. 与 indicator 表 outer 合并后的最新日期：", features_data['endDate'].max())
#     features_data = features_data.dropna().reset_index(drop=True)
#     print("4. 最终 dropna() 后的最新日期：", features_data['endDate'].max())
#     features_data.drop_duplicates(subset=['symbol', 'endDate'], keep='first', inplace=True)
#     features_data = features_data.reset_index(drop=True)
#     # predict结果
#     predict_list = []
#     groups = list(features_data.groupby('symbol'))
#     print(len(groups))
#     # 调试代码：查看合并后各日期的样本数量
#     print("合并后各日期的样本量：")
#     print(features_data['endDate'].value_counts().sort_index())

#     # 检查 mean_data 里面有没有新日期
#     print("mean_data 中包含的日期：", mean_data['endDate'].unique())
#     # 生成growth_death_train_data
#     for i in tqdm(range(len(groups))):
#         group = groups[i][1]
#         if len(group) < 31:
#             continue
#         symbol = groups[i][0]
#         # if symbol not in three_list:
#         #     continue
#         digit0 = int(symbol[0])
#         if digit0 not in [0, 3, 6]:
#             # print(symbol)
#             continue
#         group = group[-31:].reset_index(drop=True)
#         group.drop(columns='symbol', inplace=True)
#         endDate = group['endDate'].iloc[-1]
#         group.drop(columns='endDate', inplace=True)
#         if len(group) != 31:
#             continue
#         for col in group.select_dtypes(include=['object']).columns:
#             group[col] = group[col].astype('float32')
#         for col in group.select_dtypes(include=['float64']).columns:
#             group[col] = group[col].astype('float32')
#         col_names = group.columns.values
#         mean_col_names = mean_data.columns.values
#         for k in range(len(col_names)):
#             col_name = col_names[k]
#             if col_name not in mean_col_names:
#                 continue
#             mean_value = mean_data.loc[mean_data['endDate'] == int(endDate), col_name].item()
#             std_value = std_data.loc[std_data['endDate'] == int(endDate), col_name].item()
#             if std_value != 0:
#                 group[col_name] = group[col_name] - mean_value
#                 group[col_name] = group[col_name] / std_value
#             else:
#                 group[col_name] = 0
#         data = group.fillna(0)
#         data.replace([np.inf, -np.inf], 0, inplace=True)
#         data[(data > 8191.0)] = 8191.0
#         data[(data < -8191.0)] = -8191.0
#         for col in data.select_dtypes(include=['int64']).columns:
#             data[col] = data[col].astype('float32')
#         for col in data.select_dtypes(include=['float64']).columns:
#             data[col] = data[col].astype('float32')
#         for col in data.select_dtypes(include=['object']).columns:
#             data[col] = data[col].astype('float32')
#         # print(data)
#         symbol_data = data.values
#         symbol_data = torch.tensor(symbol_data).float().unsqueeze(0).to(device)
#         three, seven, thirty_one = model(symbol_data)
#         three = three.cpu().detach().numpy()[0][0][0]
#         seven = seven.cpu().detach().numpy()[0][0][0]
#         thirty_one = thirty_one.cpu().detach().numpy()[0][0][0]
#         predict_result = {'symbol': symbol, 'endDate': int(endDate), 'three': three, 'seven': seven, 'thirty_one': thirty_one}
#         predict_list.append(predict_result)
#     predict_data = pd.DataFrame(predict_list)
#     predict_data['dcf'] = predict_data['three'] + predict_data['seven'] + predict_data['thirty_one']
#     # predict_data.drop(columns=['three'], inplace=True)
#     # predict_data.drop(columns=['seven'], inplace=True)
#     # predict_data.drop(columns=['thirty_one'], inplace=True)
#     # predict_data.drop(columns=['endDate'], inplace=True)
#     # predict_data = pd.merge(predict_data, three_data, on=['symbol'], how='inner').dropna().reset_index(drop=True)
#     predict_data = predict_data.sort_values('dcf', ascending=False)
#     predict_data = predict_data.reset_index(drop=True)
#     # #截取seven股票里大于median值的股票列表
#     # predict_data = predict_data[predict_data['dcf'] > predict_data['dcf'].median()].reset_index(drop=True)
#     print(predict_data)
#     predict_data.to_csv(data_name + '/dcf_predict.csv', index=False)


# def refresh_dcf():
#     get_exchange_dcf('SHZ')
#     get_exchange_dcf('SHH')
#     data_name = os.path.join(os.path.dirname(__file__), '../data/')
#     predict_data_shenzhen = pd.read_csv(data_name + 'SHZ/dcf_predict.csv', engine='pyarrow')
#     predict_data_shanghai = pd.read_csv(data_name + 'SHH/dcf_predict.csv', engine='pyarrow')

#     predict_data = pd.concat([predict_data_shenzhen, predict_data_shanghai], axis=0)
#     predict_data.sort_values(by='dcf', inplace=True, ascending=False)
#     predict_data = predict_data.reset_index(drop=True)
#     print(predict_data)
#     three_predict_name = os.path.join(os.path.dirname(__file__), 'seven_predict.csv')
#     predict_data.to_csv(three_predict_name, index=False)


# if __name__ == '__main__':
#     refresh_dcf()
#     # get_exchange_growth_death('SHZ')

import numpy as np
import pandas as pd
import os
import time
import sys
from tqdm import tqdm
import torch
from seven_model import SEVEN  # noqa: F401  确保 torch.load 整模型时类可反序列化

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 仓库根（本脚本在 <root>/<model>/）
VERBOSE = '--verbose' in sys.argv    # 默认只打摘要行；加 --verbose 才打整张表（旧行为）

pd.set_option('future.no_silent_downcasting', True)


def select_device():
    """选择设备：优先 CUDA，但会实际跑一个 matmul 验证 kernel 可用，否则退回 CPU。

    torch.cuda.is_available() 只检测设备是否存在，不保证 kernel 镜像存在
    （如 cu126 装在 RTX 50 系 Blackwell GPU 上，is_available=True 但一执行就崩）。
    """
    if torch.cuda.is_available():
        try:
            (torch.zeros(2, 2, device='cuda') @ torch.zeros(2, 2, device='cuda')).cpu()
            return torch.device('cuda')
        except Exception as e:
            print(f'[警告] CUDA 设备检测到但 kernel 不可用（{type(e).__name__}），退回 CPU 运行。')
    return torch.device('cpu')


def get_exchange_dcf(exchange):
    t0 = time.time()
    # 检查GPU是否可用
    device = select_device()
    model_name = os.path.join(os.path.dirname(__file__), 'seven.pt')
    model = torch.load(model_name, map_location=device, weights_only=False).to(device)
    model.eval()

    exchange = exchange.lower()
    upper_exchange = exchange[0].upper() + exchange[1:]
    exchange = exchange[0].lower() + exchange[1:]
    data_name = os.path.join(os.path.dirname(__file__), '../data/' + upper_exchange)
    income_data = pd.read_csv(data_name + '/income_' + exchange + '.csv')
    balance_data = pd.read_csv(data_name + '/balance_' + exchange + '.csv')
    cashflow_data = pd.read_csv(data_name + '/cashflow_' + exchange + '.csv')
    mean_data = pd.read_csv(data_name + '/mean_' + exchange + '.csv')
    std_data = pd.read_csv(data_name + '/std_' + exchange + '.csv')
    indicator_data = pd.read_csv(data_name + '/indicator_' + exchange + '.csv')
    features_data = pd.read_csv(data_name + '/../../three/features_importance.csv')

    # 合并财务相关数据
    financial_data = pd.merge(income_data, balance_data, on=['symbol', 'endDate'], how='outer')
    financial_data = pd.merge(financial_data, cashflow_data, on=['symbol', 'endDate'], how='outer')
    
    # 【修改点 1】不再直接使用全局 dropna() 导致新日期蒸发，仅对主键防空
    financial_data = financial_data.dropna(subset=['symbol', 'endDate']).reset_index(drop=True)
    financial_data.drop_duplicates(subset=['symbol', 'endDate'], keep='first', inplace=True)
    financial_data = financial_data.reset_index(drop=True)
    
    # 截取指定特征的部分
    features_list = features_data['feature'].to_list()
    features_data = pd.DataFrame()
    features_data['symbol'] = financial_data['symbol']
    features_data['endDate'] = financial_data['endDate'].astype('int64')
    for feature in features_list:
        features_data[feature] = financial_data[feature]
        
    features_data = pd.merge(indicator_data, features_data, on=['symbol', 'endDate'], how='outer')
    
    # 【修改点 2】同理，仅对主键防空，保留含有部分 NaN 的最新特征行
    features_data = features_data.dropna(subset=['symbol', 'endDate']).reset_index(drop=True)
    features_data.drop_duplicates(subset=['symbol', 'endDate'], keep='first', inplace=True)
    features_data = features_data.reset_index(drop=True)
    
    # predict结果
    predict_list = []
    groups = list(features_data.groupby('symbol'))
    print(f"[seven] {exchange} 预测：{len(groups)} 只股票", flush=True)
    
    # 生成growth_death_train_data
    for i in tqdm(range(len(groups))):
        group = groups[i][1]
        if len(group) < 31:
            continue
        symbol = groups[i][0]
        digit0 = int(symbol[0])
        if digit0 not in [0, 3, 6]:
            continue
            
        group = group[-31:].reset_index(drop=True)
        group.drop(columns='symbol', inplace=True)
        endDate = group['endDate'].iloc[-1]
        group.drop(columns='endDate', inplace=True)
        if len(group) != 31:
            continue
            
        for col in group.select_dtypes(include=['object']).columns:
            group[col] = group[col].astype('float32')
        for col in group.select_dtypes(include=['float64']).columns:
            group[col] = group[col].astype('float32')
            
        col_names = group.columns.values
        mean_col_names = mean_data.columns.values
        
        for k in range(len(col_names)):
            col_name = col_names[k]
            if col_name not in mean_col_names:
                continue
            
            # 【修改点 3】安全获取均值和标准差，防止新财报某些科目缺失导致 .item() 报错崩溃
            mean_series = mean_data.loc[mean_data['endDate'] == int(endDate), col_name]
            std_series = std_data.loc[std_data['endDate'] == int(endDate), col_name]
            
            mean_value = mean_series.item() if not mean_series.empty and pd.notna(mean_series.item()) else 0.0
            std_value = std_series.item() if not std_series.empty and pd.notna(std_series.item()) else 1.0
            
            if std_value != 0:
                group[col_name] = group[col_name] - mean_value
                group[col_name] = group[col_name] / std_value
            else:
                group[col_name] = 0
                
        # 统一填充新财报带进来的 NaN 空值
        data = group.fillna(0)
        data.replace([np.inf, -np.inf], 0, inplace=True)
        data[(data > 8191.0)] = 8191.0
        data[(data < -8191.0)] = -8191.0
        
        for col in data.select_dtypes(include=['int64']).columns:
            data[col] = data[col].astype('float32')
        for col in data.select_dtypes(include=['float64']).columns:
            data[col] = data[col].astype('float32')
        for col in data.select_dtypes(include=['object']).columns:
            data[col] = data[col].astype('float32')
            
        symbol_data = data.values
        symbol_data = torch.tensor(symbol_data).float().unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(symbol_data)  # [1, 3(branch), 3(horizon)]
        # 只用 fcf 分支（第 0 维），三个期限求和仍作为排序分
        fcf = out[0, 0, :].cpu().detach().numpy()  # [3]
        three = fcf[0]
        seven = fcf[1]
        thirty_one = fcf[2]

        predict_result = {'symbol': symbol, 'endDate': int(endDate), 'three': three, 'seven': seven, 'thirty_one': thirty_one}
        predict_list.append(predict_result)
        
    predict_data = pd.DataFrame(predict_list)
    predict_data['dcf'] = predict_data['three'] + predict_data['seven'] + predict_data['thirty_one']
    predict_data = predict_data.sort_values('dcf', ascending=False)
    predict_data = predict_data.reset_index(drop=True)
    _out = data_name + "/dcf_predict.csv"
    print(f"[seven] {exchange} 完成：{len(predict_data)} 行 × {predict_data.shape[1]} 列 -> {os.path.relpath(_out, ROOT)}（{time.time()-t0:.1f}s）", flush=True)
    if VERBOSE: print(predict_data)
    predict_data.to_csv(data_name + '/dcf_predict.csv', index=False)


def refresh_dcf():
    t0 = time.time()          # 总计时（下面的 per-exchange t0 是局部变量，互不影响）
    get_exchange_dcf('SHZ')
    get_exchange_dcf('SHH')
    data_name = os.path.join(os.path.dirname(__file__), '../data/')
    predict_data_shenzhen = pd.read_csv(data_name + 'SHZ/dcf_predict.csv', engine='pyarrow')
    predict_data_shanghai = pd.read_csv(data_name + 'SHH/dcf_predict.csv', engine='pyarrow')

    predict_data = pd.concat([predict_data_shenzhen, predict_data_shanghai], axis=0)
    predict_values = predict_data.sort_values(by='dcf', ascending=False).reset_index(drop=True)
    three_predict_name = os.path.join(os.path.dirname(__file__), 'seven_predict.csv')
    print(f"[seven] 汇总：{len(predict_values)} 行 -> {os.path.relpath(three_predict_name, ROOT)}"
          f"（{time.time()-t0:.1f}s）", flush=True)
    if VERBOSE: print(predict_values)
    predict_values.to_csv(three_predict_name, index=False)


if __name__ == '__main__':
    refresh_dcf()
