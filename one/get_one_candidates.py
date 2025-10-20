import torch
import os
import random
import pandas as pd
import numpy as np
from tqdm import tqdm

pd.set_option('future.no_silent_downcasting', True)

def add_technical_factor(data):
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

def get_one_candidates(check_days=0):
    # 检查GPU是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = os.path.join(os.path.dirname(__file__), 'one.pt')
    model = torch.load(model_name).to(device)
    # 加载daily数据
    data_name = os.path.join(os.path.dirname(__file__), '../data/')
    daily_shz = pd.read_csv(data_name + 'SHZ/daily_shz.csv')
    daily_shh = pd.read_csv(data_name + 'SHH/daily_shh.csv')
    daily_data = pd.concat([daily_shz, daily_shh], axis=0).reset_index(drop=True)
    seven_data = pd.read_csv(data_name + '../seven/seven_predict.csv')
    #截取seven股票里大于median值的股票列表
    seven_data = seven_data[seven_data['dcf'] > seven_data['dcf'].median()].reset_index(drop=True)
    seven_list = seven_data['symbol'].to_list()
    groups = list(daily_data.groupby('symbol'))
    predict_list = []
    for i in tqdm(range(len(groups))):
        symbol = groups[i][0]
        if symbol not in seven_list:
            continue
        daily_group = groups[i][1].reset_index(drop=True)
        if check_days != 0:
            daily_group = daily_group.iloc[:-check_days].reset_index(drop=True)
        daily_group = daily_group.iloc[-127*3:].reset_index(drop=True)
        group_data_length = len(daily_group)
        if group_data_length != 127*3:
            continue
        date = daily_group['date'].iloc[-1]
        daily_input = daily_group.copy()

        # --- 归一化 ---
        ref_close = daily_input['close'].iloc[-1]
        ref_volume = daily_input['volume'].iloc[-1]
            
        daily_input['open'] = daily_input['open']/ref_close
        daily_input['high'] = daily_input['high']/ref_close
        daily_input['low'] = daily_input['low']/ref_close
        daily_input['delta'] = daily_input['high'] - daily_input['low']
        daily_input['close'] = daily_input['close'] / ref_close
        daily_input['volume'] = daily_input['volume'] / ref_volume

        daily_input = daily_input.fillna(0)
        daily_input.replace([np.inf, -np.inf], 0, inplace=True)
        daily_input = add_technical_factor(daily_input)
        daily_input['idx'] = daily_input.index / (127.0*3 - 1.0)
        daily_input.drop(columns=['symbol'], inplace=True)
        daily_input.drop(columns=['date'], inplace=True)
        daily_input = daily_input.fillna(0)
        daily_input.replace([np.inf, -np.inf], 0, inplace=True)
        for col in daily_input.select_dtypes(include=['int64']).columns:
            daily_input[col] = daily_input[col].astype('float32')
        for col in daily_input.select_dtypes(include=['float64']).columns:
            daily_input[col] = daily_input[col].astype('float32')
        for col in daily_input.select_dtypes(include=['object']).columns:
            daily_input[col] = daily_input[col].astype('float32')
        daily_input[(daily_input > 127.0)] = 127.0
        daily_input[(daily_input < -127.0)] = -127.0
        daily_input = daily_input.values
        daily_input = torch.tensor(daily_input).unsqueeze(0).float().to(device)
        _, three, seven, thirty_one = model(daily_input)
        three = three.squeeze(0).squeeze(0).cpu().detach().numpy()[0]
        seven = seven.squeeze(0).squeeze(0).cpu().detach().numpy()[0]
        thirty_one = thirty_one.squeeze(0).squeeze(0).cpu().detach().numpy()[0]
        up_down = three + seven + thirty_one
        predict_result = {'symbol': symbol, 'date': int(date), 'three': three, 'seven': seven, 'thirty_one': thirty_one, 'up_down': up_down}
        predict_list.append(predict_result)
    predict_data = pd.DataFrame(predict_list)
    predict_data = predict_data.sort_values('up_down', ascending=False)
    predict_data = predict_data.reset_index(drop=True)
    print(predict_data)
    one_predict_name = os.path.join(os.path.dirname(__file__), 'one_predict.csv')
    predict_data.to_csv(one_predict_name, index=False)
 
def refresh_buy():
    get_one_candidates()
    data_name = os.path.join(os.path.dirname(__file__), '../data/')
    daily_shz = pd.read_csv(data_name + 'SHZ/daily_shz.csv')
    daily_shh = pd.read_csv(data_name + 'SHH/daily_shh.csv')
    daily_data = pd.concat([daily_shz, daily_shh], axis=0).reset_index(drop=True)
    latest_date = daily_data['date'].max()
    daily_data = daily_data[daily_data['date'] == latest_date].copy().reset_index(drop=True)
    print(daily_data)
    

    indicator_shz = pd.read_csv(data_name + 'SHZ/indicator_shz.csv')
    indicator_shh = pd.read_csv(data_name + 'SHH/indicator_shh.csv')
    indicator_data = pd.concat([indicator_shz, indicator_shh], axis=0).reset_index(drop=True)
    indicator_data = indicator_data.sort_values('endDate').groupby('symbol').tail(1).reset_index(drop=True)
    print(indicator_data)

    data = pd.merge(daily_data, indicator_data, on='symbol', how='inner').dropna().reset_index(drop=True)

    one_predict = pd.read_csv(data_name + '../one/one_predict.csv')
    seven_predict = pd.read_csv(data_name + '../seven/seven_predict.csv')
    data = pd.merge(data, one_predict, on=['symbol', 'date'], how='inner').dropna().reset_index(drop=True)
    data = pd.merge(data, seven_predict, on=['symbol', 'endDate'], how='inner').dropna().reset_index(drop=True)
    buy_data = pd.DataFrame()
    buy_data['symbol'] = data['symbol']
    buy_data['date'] = data['date']
    buy_data['endDate'] = data['endDate']
    buy_data['close'] = data['close']
    buy_data['netAssetValuePerShare'] = data['netAssetValuePerShare']
    buy_data['dcf'] = data['dcf']
    buy_data['up_down'] = data['up_down']
    buy_data['value'] = buy_data['dcf']*buy_data['netAssetValuePerShare']/buy_data['close']
    print(buy_data)
    buy_predict_name = os.path.join(os.path.dirname(__file__), '../buy_predict.csv')
    buy_data.to_csv(buy_predict_name, index=False)
if __name__ == "__main__":
    refresh_buy()