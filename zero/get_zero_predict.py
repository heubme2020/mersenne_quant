import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import time
import sys
import torch
from zero_model import ZERO  # noqa: F401  确保 torch.load 整模型时类可反序列化
from zero_features import FEATURE_COLUMNS, LOOKBACK_QUARTERS, CLIP_VALUE

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


def get_exchange_zero(exchange):
    t0 = time.time()
    # 检查GPU是否可用
    device = select_device()
    model_name = os.path.join(os.path.dirname(__file__), 'zero.pt')
    model = torch.load(model_name, map_location=device, weights_only=False).to(device)
    model.eval()

    exchange = exchange.lower()
    upper_exchange = exchange[0].upper() + exchange[1:]
    exchange = exchange[0].lower() + exchange[1:]
    data_name = os.path.join(os.path.dirname(__file__), '../data/' + upper_exchange)
    indicator_data = pd.read_csv(data_name + '/indicator_' + exchange + '.csv')
    three_data = pd.read_csv(data_name + '/../../three/three_predict.csv')

    # 截取three股票里大于median值的股票列表
    three_data = three_data[three_data['growth_death'] > three_data['growth_death'].median()].reset_index(drop=True)
    three_list = three_data['symbol'].to_list()
    groups = list(indicator_data.groupby('symbol'))

    # predict结果
    predict_list = []
    print(f"[zero] {exchange} 预测：{len(groups)} 只股票", flush=True)

    for i in tqdm(range(len(groups))):
        group = groups[i][1]
        if len(group) < LOOKBACK_QUARTERS:
            continue
        symbol = groups[i][0]
        if symbol not in three_list:
            continue

        missing_columns = [col for col in FEATURE_COLUMNS if col not in group.columns]
        if missing_columns:
            print(f"Skip {symbol}, missing feature columns: {missing_columns}")
            continue

        group = group.sort_values('endDate').reset_index(drop=True)
        endDate = group['endDate'].iloc[-1]
        group = group.loc[len(group) - LOOKBACK_QUARTERS:, FEATURE_COLUMNS].copy().reset_index(drop=True)

        if len(group) != LOOKBACK_QUARTERS:
            continue

        for col in group.select_dtypes(include=['object']).columns:
            group[col] = group[col].astype('float32')
        for col in group.select_dtypes(include=['float64']).columns:
            group[col] = group[col].astype('float32')

        data = group.fillna(0)
        data.replace([np.inf, -np.inf], 0, inplace=True)
        data[(data > CLIP_VALUE)] = CLIP_VALUE
        data[(data < -CLIP_VALUE)] = -CLIP_VALUE

        for col in data.select_dtypes(include=['int64']).columns:
            data[col] = data[col].astype('float32')
        for col in data.select_dtypes(include=['float64']).columns:
            data[col] = data[col].astype('float32')
        for col in data.select_dtypes(include=['object']).columns:
            data[col] = data[col].astype('float32')

        symbol_data = data.values
        symbol_data = torch.tensor(symbol_data).float().unsqueeze(0).to(device)

        with torch.no_grad():
            three_pred = model(symbol_data)
        three_pred = three_pred.cpu().detach().numpy()[0][0][0]

        predict_result = {'symbol': symbol, 'endDate': int(endDate), 'zero': three_pred}
        predict_list.append(predict_result)

    predict_data = pd.DataFrame(predict_list)

    # 【优化】先不忙 drop 掉 endDate，确保跟 three_data 进行 inner merge 时有更安全的对齐依据
    # 如果 three_data 里也带有 endDate 字段，merge 后会自动处理
    if 'endDate' in three_data.columns:
        predict_data = pd.merge(predict_data, three_data, on=['symbol', 'endDate'], how='inner')
    else:
        predict_data = pd.merge(predict_data, three_data, on=['symbol'], how='inner')

    predict_data = predict_data.dropna(subset=['zero']).reset_index(drop=True)
    predict_data = predict_data.sort_values('zero', ascending=False)
    predict_data = predict_data.reset_index(drop=True)

    _out = data_name + "/zero_predict.csv"
    print(f"[zero] {exchange} 完成：{len(predict_data)} 行 × {predict_data.shape[1]} 列 -> {os.path.relpath(_out, ROOT)}（{time.time()-t0:.1f}s）", flush=True)
    if VERBOSE: print(predict_data)
    predict_data.to_csv(data_name + '/zero_predict.csv', index=False)


def refresh_zero():
    t0 = time.time()          # 总计时（下面的 per-exchange t0 是局部变量，互不影响）
    get_exchange_zero('SHZ')
    get_exchange_zero('SHH')
    data_name = os.path.join(os.path.dirname(__file__), '../data/')
    predict_data_shenzhen = pd.read_csv(data_name + 'SHZ/zero_predict.csv', engine='pyarrow')
    predict_data_shanghai = pd.read_csv(data_name + 'SHH/zero_predict.csv', engine='pyarrow')

    predict_data = pd.concat([predict_data_shenzhen, predict_data_shanghai], axis=0)
    predict_values = predict_data.sort_values(by='zero', ascending=False).reset_index(drop=True)
    zero_predict_name = os.path.join(os.path.dirname(__file__), 'zero_predict.csv')
    print(f"[zero] 汇总：{len(predict_values)} 行 -> {os.path.relpath(zero_predict_name, ROOT)}"
          f"（{time.time()-t0:.1f}s）", flush=True)
    if VERBOSE: print(predict_values)
    predict_values.to_csv(zero_predict_name, index=False)


if __name__ == '__main__':
    refresh_zero()
