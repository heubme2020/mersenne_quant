import torch
import os
import random
import pandas as pd
import numpy as np
from tqdm import tqdm

pd.set_option('future.no_silent_downcasting', True)

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


def add_technical_factor(data):
    """
    计算股票技术因子
    已根据 IC 分析结果替换掉弱势因子：vma_7_31, rsi7, atr31
    集成强势因子：Alpha 15 (WQ), Alpha 128 (GTJA), Alpha 101 (GTJA)
    """
    # 确保数据按时间排序
    data = data.sort_values('date') if 'date' in data.columns else data

    # --- 1. 均线类 (原有保留) ---
    data['ma3'] = data['close'].rolling(3).mean()
    data['ma7'] = data['close'].rolling(7).mean()
    data['ma31'] = data['close'].rolling(31).mean()

    # --- 2. RSI类 (仅保留效果尚可的 rsi3 和 rsi31) ---
    delta = data['close'].diff()
    def calc_rsi(ser, period):
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        return (100 - (100 / (1 + (gain / (loss + 1e-6))))) * 0.01

    data['rsi3'] = calc_rsi(data['close'], 3)
    data['rsi31'] = calc_rsi(data['close'], 31)

    # --- 3. 波动率与成交量类 (原有保留) ---
    data['atr3'] = (data['delta'].rolling(3).mean()) * 31
    data['atr7'] = (data['delta'].rolling(7).mean()) * 31
    
    # OBV
    obv = delta * data['volume']
    data['obv3'] = (obv.rolling(3).mean()) * 31
    data['obv7'] = (obv.rolling(7).mean()) * 31
    data['obv31'] = (obv.rolling(31).mean()) * 31

    # Correlation
    data['corr3'] = data['volume'].rolling(3).corr(data['close'])
    data['corr7'] = data['volume'].rolling(7).corr(data['close'])
    data['corr31'] = data['volume'].rolling(31).corr(data['close'])

    # Curvature & VMA
    data['curvature'] = (data['close'].diff().diff()) * 31
    data['vma_3_7'] = data['volume'].rolling(3).mean() / data['volume'].rolling(7).mean() - 1

    # Factor
    data['factor'] = (data['close'].pct_change(3) - data['volume'].rolling(7).std()) 

    # Overnight
    overnight = data['open'] * data['close'] / data['close'].shift(1).replace(0, np.nan) - 1
    data['overnight3'] = overnight.rolling(3).mean() * 31
    data['overnight7'] = overnight.rolling(7).mean() * 31
    data['overnight31'] = overnight.rolling(31).mean() * 31

    # --- 4. 替换/新增的三个最强 Alpha 因子 ---

    # # 【替换 atr31 -> Alpha 15 (WQ 101)】
    # # 逻辑：量价相关性反转。A股中放量冲高往往是出货信号
    # # 使用滚动排名近似截面排名
    # rk_h = data['high'].rolling(10).rank(pct=True)
    # rk_v = data['volume'].rolling(10).rank(pct=True)
    # data['alpha15_wq'] = -1 * rk_h.rolling(3).corr(rk_v)
    # 1. 内部 Rank：捕捉股价和成交量各自在短周期内的相对高低
    # 原版是全市场 rank，单股逻辑下用 rolling(10) 模拟其相对强弱是合理的
    rk_h = data['high'].rolling(31).rank(pct=True)
    rk_v = data['volume'].rolling(31).rank(pct=True)

    # 2. 计算相关性：这是 Alpha 15 的核心，窗口必须是 3
    # 此时得到的是 -1.0 到 1.0 之间的值
    inner_corr = rk_h.rolling(7).corr(rk_v)

    # 3. 补齐“Rank 的 Rank”逻辑（关键优化点）
    # 原版公式最外层还有一个 Rank。在单股训练中，为了让这个因子和其它因子量纲一致，
    # 且减少 rolling(3) 带来的剧烈跳动，建议再套一个 rolling rank。
    data['alpha15_wq'] = -1 * inner_corr.rolling(31).rank(pct=True)

    # 【替换 rsi7 -> Alpha 128 (GTJA 191)】
    # 逻辑：均线量价背离。判断趋势的健康度，ICIR 非常稳定
    adv20 = data['volume'].rolling(31).mean()
    data['alpha128_gtja'] = -1 * (data['close'].diff(1) * (data['volume'] / (adv20 + 1e-6))).rolling(7).mean()

    # 【替换 vma_7_31 -> Alpha 101 (GTJA 191)】
    # 逻辑：价格区间位置 (RSV)。衡量超买超卖，捕捉震荡市反转
    low_30 = data['low'].rolling(31).min()
    high_30 = data['high'].rolling(31).max()
    data['alpha101_gtja'] = (data['close'] - low_30) / (high_30 - low_30 + 1e-6)

    # --- 5. 原有 Alpha22 逻辑 (保持不变) ---
    # 3_7
    rolling_corr_3 = (data['high'] * data['close']).rolling(3).corr(data['volume'])
    delta_corr_3 = rolling_corr_3.diff(3)
    std_close_7 = data['close'].rolling(7).std()
    data['aplha22_3_7'] = -1 * (delta_corr_3 * std_close_7) * 31
    
    # 7_31
    rolling_corr_7 = (data['high'] * data['close']).rolling(7).corr(data['volume'])
    delta_corr_7 = rolling_corr_7.diff(7)
    std_close_31 = data['close'].rolling(31).std()
    data['aplha22_7_31'] = -1 * (delta_corr_7 * std_close_31) * 31

    return data

def raise_on_empty_merge(merged, left_dates, right_dates, source):
    """按 (symbol, date) inner merge 后为空时报错，避免静默写出空 CSV。

    与 one/get_one_predict.py 里的同名函数一致：日期错位（上游股票池没跟着
    日线数据刷新）会让两边 (symbol, date) 完全不相交，这里直接报出两边日期。
    """
    if not merged.empty:
        return
    left_set, right_set = set(left_dates), set(right_dates)
    raise RuntimeError(
        f"与 {source} 按 (symbol, date) 合并后无任何匹配：\n"
        f"  本次预测日期: {sorted(left_set)[:5]}（共 {len(left_set)} 个）\n"
        f"  {source} 日期 : {sorted(right_set)[:5]}（共 {len(right_set)} 个）\n"
        f"  日期交集: {sorted(left_set & right_set)[:5]}（共 {len(left_set & right_set)} 个）\n"
        f"  请先确认 {source} 是否已随最新的日线数据一起刷新。"
    )


def get_two_candidates(check_days=0, target_date=None):
    # 检查GPU是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 2026-09-26：`two.pt` 现在是【生产模型】（nowcast 架构，Nowcast 类），而本脚本要的是
    # 旧的 TWO 架构（two_model.TWO）—— 旧权重已改名 `two_legacy.pt`。
    # 别再指回 two.pt：类不匹配会直接抛异常，而且是每天选票时才发现。
    model_name = os.path.join(os.path.dirname(__file__), 'two_legacy.pt')
    model = torch.load(model_name, weights_only=False).to(device)
    # 加载daily数据
    data_name = os.path.join(os.path.dirname(__file__), '../data/')
    daily_shz = pd.read_csv(data_name + 'SHZ/daily_shz.csv')
    daily_shh = pd.read_csv(data_name + 'SHH/daily_shh.csv')
    daily_data = pd.concat([daily_shz, daily_shh], axis=0).reset_index(drop=True)
    if target_date is not None:
        daily_data = daily_data[daily_data['date'] <= int(target_date)].reset_index(drop=True)
    zero_data = pd.read_csv(data_name + '../zero/zero_predict.csv')
    zero_data.drop(columns=['three'], inplace=True)
    zero_data.drop(columns=['seven'], inplace=True)
    zero_data.drop(columns=['thirty_one'], inplace=True)
    zero_data = zero_data[zero_data['zero'] > zero_data['zero'].median()].reset_index(drop=True)
    print(zero_data)
    schloss_data = pd.read_csv(data_name + '../schloss/schloss.csv')
    schloss_data.drop(columns=['endDate'], inplace=True)
    print(schloss_data)
    buffett_data = pd.merge(schloss_data, zero_data, on=['symbol'], how='inner').dropna().reset_index(drop=True)
    buffett_data['buffett'] = buffett_data['dcf']*buffett_data['netAssetValuePerShare']/buffett_data['close'] + buffett_data['schloss']
    buffett_data = buffett_data.sort_values('buffett', ascending=False)
    buffett_data = buffett_data.reset_index(drop=True)
    # buffett_data = buffett_data[buffett_data['buffett'] > buffett_data['buffett'].median()].reset_index(drop=True)
    buffett_list = buffett_data['symbol'].to_list()
    groups = list(daily_data.groupby('symbol'))
    predict_list = []
    for i in tqdm(range(len(groups))):
        symbol = groups[i][0]
        if symbol not in buffett_list:
            continue
        daily_group = groups[i][1].reset_index(drop=True)
        if check_days != 0:
            daily_group = daily_group.iloc[:-check_days].reset_index(drop=True)
        daily_group = daily_group.iloc[-127*7:].reset_index(drop=True)
        group_data_length = len(daily_group)
        if group_data_length != 127*7:
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
        _, seven, thirty_one, two_hundred_and_twenty_seven = model(daily_input)
        seven = seven.squeeze(0).squeeze(0).cpu().detach().numpy()[0]
        thirty_one = thirty_one.squeeze(0).squeeze(0).cpu().detach().numpy()[0]
        two_hundred_and_twenty_seven = two_hundred_and_twenty_seven.squeeze(0).squeeze(0).cpu().detach().numpy()[0]
        up_down = seven + thirty_one + two_hundred_and_twenty_seven
        predict_result = {'symbol': symbol, 'date': int(date), 'seven': seven, 'thirty_one': thirty_one, 'two_hundred_and_twenty_seven': two_hundred_and_twenty_seven, 'up_down': up_down}
        predict_list.append(predict_result)
    predict_data = pd.DataFrame(predict_list)
    predict_dates = predict_data['date'].tolist() if 'date' in predict_data.columns else []
    predict_data = pd.merge(predict_data, buffett_data, on=['symbol', 'date'], how='inner').dropna().reset_index(drop=True)
    raise_on_empty_merge(predict_data, predict_dates, buffett_data['date'].tolist(), 'schloss/schloss.csv')
    predict_data = predict_data.sort_values('up_down', ascending=False)
    predict_data = predict_data.reset_index(drop=True)
    print(predict_data)
    two_predict_name = os.path.join(os.path.dirname(__file__), 'two_predict.csv')
    predict_data.to_csv(two_predict_name, index=False)
 

    
if __name__ == "__main__":
    get_two_candidates()
    # get_two_all()