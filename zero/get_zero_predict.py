import numpy as np
import pandas as pd
import cupy as cp
import datetime
from tqdm import tqdm
from catboost import CatBoostRegressor
import os
import torch
pd.set_option('future.no_silent_downcasting', True)


def get_exchange_zero(exchange):
    print(exchange)
    # 检查GPU是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = os.path.join(os.path.dirname(__file__), 'zero.pt')
    model = torch.load(model_name).to(device)

    exchange = exchange.lower()
    upper_exchange = exchange[0].upper() + exchange[1:]
    exchange = exchange[0].lower() + exchange[1:]
    data_name = os.path.join(os.path.dirname(__file__), '../data/' + upper_exchange)
    indicator_data = pd.read_csv(data_name + '/indicator_' + exchange + '.csv')
    print(indicator_data)
    groups = list(indicator_data.groupby('symbol'))
    # predict结果
    predict_list = []
    print(len(groups))
    for i in tqdm(range(len(groups))):
        group = groups[i][1]
        if len(group) < 3:
            continue
        symbol = groups[i][0]
        digit0 = int(symbol[0])
        if digit0 not in [0, 3, 6]:
            # print(symbol)
            continue
        group = group.iloc[-3:, :12].copy().reset_index(drop=True) # 使用 .copy() 避免 SettingWithCopyWarning
        endDate = group['endDate'].iloc[-1]
        group.drop(columns=['symbol', 'endDate', 'netAssetValuePerShare', 'dcfPerShare', 'dividendPerShare'], inplace=True)
        if len(group) != 3:
            continue
        for col in group.select_dtypes(include=['object']).columns:
            group[col] = group[col].astype('float32')
        for col in group.select_dtypes(include=['float64']).columns:
            group[col] = group[col].astype('float32')
        data = group.fillna(0)
        data.replace([np.inf, -np.inf], 0, inplace=True)
        for col in data.select_dtypes(include=['int64']).columns:
            data[col] = data[col].astype('float32')
        for col in data.select_dtypes(include=['float64']).columns:
            data[col] = data[col].astype('float32')
        for col in data.select_dtypes(include=['object']).columns:
            data[col] = data[col].astype('float32')
        symbol_data = data.values
        symbol_data = torch.tensor(symbol_data).float().unsqueeze(0).to(device)
        _, three = model(symbol_data)
        three = three.cpu().detach().numpy()[0][0][0]
        predict_result = {'symbol': symbol, 'endDate': int(endDate), 'three': three}
        predict_list.append(predict_result)
    predict_data = pd.DataFrame(predict_list)
    predict_data = predict_data.sort_values('three', ascending=False)
    predict_data = predict_data.reset_index(drop=True)
    print(predict_data)
    predict_data.to_csv(data_name + '/zero_predict.csv', index=False)


def refresh_zero():
    get_exchange_zero('SHZ')
    get_exchange_zero('SHH')
    data_name = os.path.join(os.path.dirname(__file__), '../data/')
    predict_data_shenzhen = pd.read_csv(data_name + 'SHZ/zero_predict.csv', engine='pyarrow')
    predict_data_shanghai = pd.read_csv(data_name + 'SHH/zero_predict.csv', engine='pyarrow')

    predict_data = pd.concat([predict_data_shenzhen, predict_data_shanghai], axis=0)
    predict_data.sort_values(by='three', inplace=True, ascending=False)
    predict_data = predict_data.reset_index(drop=True)
    print(predict_data)
    zero_predict_name = os.path.join(os.path.dirname(__file__), 'zero_predict.csv')
    predict_data.to_csv(zero_predict_name, index=False)


if __name__ == '__main__':
    refresh_zero()
    # get_exchange_growth_death('SHZ')

