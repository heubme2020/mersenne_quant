import numpy as np
import pandas as pd
import cupy as cp
import datetime
from tqdm import tqdm
from catboost import CatBoostRegressor
import os
import torch

pd.set_option('future.no_silent_downcasting', True)
# 求取上个季度的最后一天

def get_exchange_dcf(exchange):
    print(exchange)
    # 检查GPU是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = os.path.join(os.path.dirname(__file__), 'seven.pt')
    model = torch.load(model_name).to(device)

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
    three_data = pd.read_csv(data_name + '/../../three/three_predict.csv')

    #合并财务相关数据
    financial_data = pd.merge(income_data, balance_data, on=['symbol', 'endDate'], how='outer')
    financial_data = pd.merge(financial_data, cashflow_data, on=['symbol', 'endDate'], how='outer')
    financial_data = financial_data.dropna()
    financial_data = financial_data.reset_index(drop=True)
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
    features_data = features_data.dropna().reset_index(drop=True)
    features_data.drop_duplicates(subset=['symbol', 'endDate'], keep='first', inplace=True)
    features_data = features_data.reset_index(drop=True)
    #截取three股票里大于median值的股票列表
    three_data = three_data[three_data['growth_death'] > three_data['growth_death'].median()].reset_index(drop=True)
    three_list = three_data['symbol'].to_list()
    # predict结果
    predict_list = []
    groups = list(features_data.groupby('symbol'))
    print(len(groups))
    # 生成growth_death_train_data
    for i in tqdm(range(len(groups))):
        group = groups[i][1]
        if len(group) < 31:
            continue
        symbol = groups[i][0]
        if symbol not in three_list:
            continue
        digit0 = int(symbol[0])
        if digit0 not in [0, 3, 6]:
            # print(symbol)
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
            mean_value = mean_data.loc[mean_data['endDate'] == int(endDate), col_name].item()
            std_value = std_data.loc[std_data['endDate'] == int(endDate), col_name].item()
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
        # print(data)
        symbol_data = data.values
        symbol_data = torch.tensor(symbol_data).float().unsqueeze(0).to(device)
        three, seven, thirty_one = model(symbol_data)
        three = three.cpu().detach().numpy()[0][0][0]
        seven = seven.cpu().detach().numpy()[0][0][0]
        thirty_one = thirty_one.cpu().detach().numpy()[0][0][0]
        predict_result = {'symbol': symbol, 'endDate': int(endDate), 'three': three, 'seven': seven, 'thirty_one': thirty_one}
        predict_list.append(predict_result)
    predict_data = pd.DataFrame(predict_list)
    predict_data['dcf'] = predict_data['three'] + predict_data['seven'] + predict_data['thirty_one']
    predict_data = predict_data.sort_values('dcf', ascending=False)
    predict_data = predict_data.reset_index(drop=True)
    print(predict_data)
    predict_data.to_csv(data_name + '/dcf_predict.csv', index=False)


def refresh_dcf():
    get_exchange_dcf('SHZ')
    get_exchange_dcf('SHH')
    data_name = os.path.join(os.path.dirname(__file__), '../data/')
    predict_data_shenzhen = pd.read_csv(data_name + 'SHZ/dcf_predict.csv', engine='pyarrow')
    predict_data_shanghai = pd.read_csv(data_name + 'SHH/dcf_predict.csv', engine='pyarrow')

    predict_data = pd.concat([predict_data_shenzhen, predict_data_shanghai], axis=0)
    predict_data.sort_values(by='dcf', inplace=True, ascending=False)
    predict_data = predict_data.reset_index(drop=True)
    print(predict_data)
    three_predict_name = os.path.join(os.path.dirname(__file__), 'seven_predict.csv')
    predict_data.to_csv(three_predict_name, index=False)


if __name__ == '__main__':
    refresh_dcf()
    # get_exchange_growth_death('SHZ')

