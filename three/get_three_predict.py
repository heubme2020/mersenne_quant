# import numpy as np
# import pandas as pd
# import cupy as cp
# import datetime
# from tqdm import tqdm
# # from catboost import CatBoostRegressor
# import os
# import torch


# pd.set_option('future.no_silent_downcasting', True)

# def get_exchange_growth_death(exchange):
#     print(exchange)
#     # 检查GPU是否可用
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model_name = os.path.join(os.path.dirname(__file__), 'three.pt')
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
#     seven_data = pd.read_csv(data_name + '/../../seven/seven_predict.csv')

#     #合并财务相关数据
#     financial_data = pd.merge(income_data, balance_data, on=['symbol', 'endDate'], how='outer')
#     financial_data = pd.merge(financial_data, cashflow_data, on=['symbol', 'endDate'], how='outer')
#     financial_data = financial_data.dropna()
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
#     features_data = features_data.dropna().reset_index(drop=True)
#     features_data.drop_duplicates(subset=['symbol', 'endDate'], keep='first', inplace=True)
#     features_data = features_data.reset_index(drop=True)
#     #截取seven股票里大于median值的股票列表
#     seven_data = seven_data[seven_data['dcf'] > seven_data['dcf'].median()].reset_index(drop=True)
#     seven_list = seven_data['symbol'].to_list()
#     # predict结果
#     predict_list = []
#     groups = list(features_data.groupby('symbol'))
#     print(len(groups))
#     # 生成growth_death_train_data
#     for i in tqdm(range(len(groups))):
#         group = groups[i][1]
#         if len(group) < 31:
#             continue
#         symbol = groups[i][0]
#         if symbol not in seven_list:
#             continue
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
#         growth, death= model(symbol_data)
#         growth = growth.cpu().detach().numpy()[0][0]
#         growth = np.mean(growth)
#         death = death.cpu().detach().numpy()[0][0]
#         death = np.mean(death)
#         predict_result = {'symbol': symbol, 'endDate': int(endDate), 'growth': growth, 'death': death}
#         predict_list.append(predict_result)
#     predict_data = pd.DataFrame(predict_list)
#     predict_data['growth_death'] = predict_data['growth']/predict_data['death'] 
#     predict_data.drop(columns=['endDate'], inplace=True)
#     predict_data = pd.merge(predict_data, seven_data, on=['symbol'], how='inner').dropna().reset_index(drop=True)
#     predict_data = predict_data.sort_values('growth_death', ascending=False)
#     predict_data = predict_data.reset_index(drop=True)
#     print(predict_data)
#     predict_data.to_csv(data_name + '/grow_death_predict.csv', index=False)


# def refresh_growth_death():
#     get_exchange_growth_death('SHZ')
#     get_exchange_growth_death('SHH')
#     data_name = os.path.join(os.path.dirname(__file__), '../data/')
#     predict_data_shenzhen = pd.read_csv(data_name + 'SHZ/grow_death_predict.csv', engine='pyarrow')
#     predict_data_shanghai = pd.read_csv(data_name + 'SHH/grow_death_predict.csv', engine='pyarrow')

#     predict_data = pd.concat([predict_data_shenzhen, predict_data_shanghai], axis=0)
#     predict_data.sort_values(by='growth_death', inplace=True, ascending=False)
#     predict_data = predict_data.reset_index(drop=True)
#     print(predict_data)
#     three_predict_name = os.path.join(os.path.dirname(__file__), 'three_predict.csv')
#     predict_data.to_csv(three_predict_name, index=False)


# if __name__ == '__main__':
#     refresh_growth_death()
#     # get_exchange_growth_death('SHZ')

import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import time
import sys
import torch
from three_model import THREE  # noqa: F401  确保 torch.load 整模型时类可反序列化

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


def get_exchange_growth_death(exchange):
    t0 = time.time()
    # 检查GPU是否可用
    device = select_device()
    model_name = os.path.join(os.path.dirname(__file__), 'three.pt')
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
    seven_data = pd.read_csv(data_name + '/../../seven/seven_predict.csv')

    # 合并财务相关数据
    financial_data = pd.merge(income_data, balance_data, on=['symbol', 'endDate'], how='outer')
    financial_data = pd.merge(financial_data, cashflow_data, on=['symbol', 'endDate'], how='outer')
    
    # 【修改点 1】去掉全局 dropna()，仅针对主键防空，保留最新披露、含部分 NaN 的新财报数据
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
    
    # 【修改点 2】同理，仅针对主键防空，保住最新日期
    features_data = features_data.dropna(subset=['symbol', 'endDate']).reset_index(drop=True)
    features_data.drop_duplicates(subset=['symbol', 'endDate'], keep='first', inplace=True)
    features_data = features_data.reset_index(drop=True)
    
    # 截取seven股票里大于median值的股票列表
    seven_data = seven_data[seven_data['dcf'] > seven_data['dcf'].median()].reset_index(drop=True)
    seven_list = seven_data['symbol'].to_list()
    
    # predict结果
    predict_list = []
    groups = list(features_data.groupby('symbol'))
    print(f"[three] {exchange} 预测：{len(groups)} 只股票", flush=True)
    
    # 生成growth_death_train_data
    for i in tqdm(range(len(groups))):
        group = groups[i][1]
        if len(group) < 31:
            continue
        symbol = groups[i][0]
        if symbol not in seven_list:
            continue
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
                
            # 【修改点 3】升级为安全取值逻辑，防止新日期均值单元格为空导致 .item() 异常崩溃
            mean_series = mean_data.loc[mean_data['endDate'] == int(endDate), col_name]
            std_series = std_data.loc[std_data['endDate'] == int(endDate), col_name]
            
            mean_value = mean_series.item() if not mean_series.empty and pd.notna(mean_series.item()) else 0.0
            std_value = std_series.item() if not std_series.empty and pd.notna(std_series.item()) else 1.0
            
            if std_value != 0:
                group[col_name] = group[col_name] - mean_value
                group[col_name] = group[col_name] / std_value
            else:
                group[col_name] = 0
                
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
            growth, death = model(symbol_data)
        growth = growth.cpu().detach().numpy()[0][0]
        growth = np.mean(growth)
        death = death.cpu().detach().numpy()[0][0]
        death = np.mean(death)
        
        predict_result = {'symbol': symbol, 'endDate': int(endDate), 'growth': growth, 'death': death}
        predict_list.append(predict_result)
        
    predict_data = pd.DataFrame(predict_list)
    
    # 【安全防御】避免分母 death 出现 0 导致报错或产生 inf 
    predict_data['growth_death'] = predict_data['growth'] / predict_data['death'].replace(0, np.nan)
    predict_data['growth_death'] = predict_data['growth_death'].fillna(0)
    
    # 【逻辑对齐】之前代码里直接 drop(columns=['endDate']) 会导致最后合并丢失预测截止日期
    # 在此保留，合并完 seven_data 之后再行选择处理，确保新旧日期在最终的合并中对齐
    predict_data = pd.merge(predict_data, seven_data, on=['symbol'], how='inner')
    # 两个表都带 endDate，merge 后成了 endDate_x / endDate_y（值相同，都是各自最新财报期）。
    # 统一成单个 endDate，去掉重复列。
    predict_data = predict_data.drop(columns=['endDate_y'])
    predict_data = predict_data.rename(columns={'endDate_x': 'endDate'})
    
    # 针对合并完特征后的空行采用更稳健的丢弃，仅限缺失重要因子的行
    predict_data = predict_data.dropna(subset=['growth_death', 'dcf']).reset_index(drop=True)
    predict_data = predict_data.sort_values('growth_death', ascending=False)
    predict_data = predict_data.reset_index(drop=True)
    
    _out = data_name + "/grow_death_predict.csv"
    print(f"[three] {exchange} 完成：{len(predict_data)} 行 × {predict_data.shape[1]} 列 -> {os.path.relpath(_out, ROOT)}（{time.time()-t0:.1f}s）", flush=True)
    if VERBOSE: print(predict_data)
    predict_data.to_csv(data_name + '/grow_death_predict.csv', index=False)


def refresh_growth_death():
    t0 = time.time()          # 总计时（下面的 per-exchange t0 是局部变量，互不影响）
    get_exchange_growth_death('SHZ')
    get_exchange_growth_death('SHH')
    data_name = os.path.join(os.path.dirname(__file__), '../data/')
    predict_data_shenzhen = pd.read_csv(data_name + 'SHZ/grow_death_predict.csv', engine='pyarrow')
    predict_data_shanghai = pd.read_csv(data_name + 'SHH/grow_death_predict.csv', engine='pyarrow')

    predict_data = pd.concat([predict_data_shenzhen, predict_data_shanghai], axis=0)
    predict_values = predict_data.sort_values(by='growth_death', ascending=False).reset_index(drop=True)
    three_predict_name = os.path.join(os.path.dirname(__file__), 'three_predict.csv')
    print(f"[three] 汇总：{len(predict_values)} 行 -> {os.path.relpath(three_predict_name, ROOT)}"
          f"（{time.time()-t0:.1f}s）", flush=True)
    if VERBOSE: print(predict_values)
    predict_values.to_csv(three_predict_name, index=False)


if __name__ == '__main__':
    refresh_growth_death()
