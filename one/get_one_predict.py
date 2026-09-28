import torch
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

pd.set_option('future.no_silent_downcasting', True)

HORIZONS = [1, 3, 7, 31, 127]
CLOSE_COLS = [f'close_{h}' for h in HORIZONS]
DAYS_INPUT = 127 * 7

# 因子方案：baseline / plan1 / plan2（通过环境变量传入）
FACTOR_MODE = os.environ.get('FACTOR_MODE', 'new24')   # 2026-09-14 起默认用新 24 个
from factor_config import get_technical_factors  # noqa: E402
from factor_pool import add_pool_factors  # noqa: E402

NEW_FACTOR_H5 = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'factor_screen', 'new_factors.h5')


def load_new_factors():
    store = pd.HDFStore(NEW_FACTOR_H5, mode='r')
    factors = {}
    for k in store.keys():
        fid = k.strip('/')[2:]
        factors[fid] = store[k].astype(np.float32)
    store.close()
    return factors


def merge_new_factors(daily_data):
    """把预计算的新因子按 (date, symbol) 合并进 daily_data。"""
    new_factors = load_new_factors()
    for fid, fac in new_factors.items():
        long = fac.stack().rename(fid).reset_index()
        daily_data = daily_data.merge(long, on=['date', 'symbol'], how='left')
    return daily_data

# ---- close 标签全局 z-score 统计（由 train.py 生成）----
STATS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'target_stats.npz')


def _load_close_stats():
    if not os.path.exists(STATS_PATH):
        raise FileNotFoundError(f"{STATS_PATH} 不存在。请先运行 train.py 生成统计。")
    s = np.load(STATS_PATH)
    return s['close_mean'].astype(np.float32), s['close_std'].astype(np.float32)


CLOSE_MEAN, CLOSE_STD = _load_close_stats()


def add_technical_factor(data):
    """
    计算股票技术因子（与 gen_train_data.py 保持一致）
    """
    data = data.sort_values('date') if 'date' in data.columns else data

    data['ma3'] = data['close'].rolling(3).mean()
    data['ma7'] = data['close'].rolling(7).mean()
    data['ma31'] = data['close'].rolling(31).mean()

    delta = data['close'].diff()
    def calc_rsi(ser, period):
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        return (100 - (100 / (1 + (gain / (loss + 1e-6))))) * 0.01

    data['rsi3'] = calc_rsi(data['close'], 3)
    data['rsi31'] = calc_rsi(data['close'], 31)

    data['atr3'] = (data['delta'].rolling(3).mean()) * 31
    data['atr7'] = (data['delta'].rolling(7).mean()) * 31

    obv = delta * data['volume']
    data['obv3'] = (obv.rolling(3).mean()) * 31
    data['obv7'] = (obv.rolling(7).mean()) * 31
    data['obv31'] = (obv.rolling(31).mean()) * 31

    data['corr3'] = data['volume'].rolling(3).corr(data['close'])
    data['corr7'] = data['volume'].rolling(7).corr(data['close'])
    data['corr31'] = data['volume'].rolling(31).corr(data['close'])

    data['curvature'] = (data['close'].diff().diff()) * 31
    data['vma_3_7'] = data['volume'].rolling(3).mean() / data['volume'].rolling(7).mean() - 1
    data['factor'] = (data['close'].pct_change(3) - data['volume'].rolling(7).std())

    overnight = data['open'] * data['close'] / data['close'].shift(1).replace(0, np.nan) - 1
    data['overnight3'] = overnight.rolling(3).mean() * 31
    data['overnight7'] = overnight.rolling(7).mean() * 31
    data['overnight31'] = overnight.rolling(31).mean() * 31

    rk_h = data['high'].rolling(10).rank(pct=True)
    rk_v = data['volume'].rolling(10).rank(pct=True)
    inner_corr = rk_h.rolling(3).corr(rk_v)
    data['alpha15_wq'] = -1 * inner_corr.rolling(10).rank(pct=True)

    adv20 = data['volume'].rolling(20).mean()
    data['alpha128_gtja'] = -1 * (data['close'].diff(1) * (data['volume'] / (adv20 + 1e-6))).rolling(5).mean()

    low_30 = data['low'].rolling(30).min()
    high_30 = data['high'].rolling(30).max()
    data['alpha101_gtja'] = (data['close'] - low_30) / (high_30 - low_30 + 1e-6)

    rolling_corr_3 = (data['high'] * data['close']).rolling(3).corr(data['volume'])
    delta_corr_3 = rolling_corr_3.diff(3)
    std_close_7 = data['close'].rolling(7).std()
    data['aplha22_3_7'] = -1 * (delta_corr_3 * std_close_7) * 31

    rolling_corr_7 = (data['high'] * data['close']).rolling(7).corr(data['volume'])
    delta_corr_7 = rolling_corr_7.diff(7)
    std_close_31 = data['close'].rolling(31).std()
    data['aplha22_7_31'] = -1 * (delta_corr_7 * std_close_31) * 31

    return data


def load_model(device):
    model_name = os.path.join(os.path.dirname(__file__), 'one.pt')
    return torch.load(model_name, weights_only=False).to(device)


def prepare_input(daily_group, device):
    """对单只股票最近 889 天做归一化 + 特征工程，返回 (1, 889, 31) 张量。"""
    daily_input = daily_group.copy()

    ref_close = daily_input['close'].iloc[-1]
    ref_volume = daily_input['volume'].iloc[-1]

    daily_input['open'] = daily_input['open'] / ref_close
    daily_input['high'] = daily_input['high'] / ref_close
    daily_input['low'] = daily_input['low'] / ref_close
    daily_input['delta'] = daily_input['high'] - daily_input['low']
    daily_input['close'] = daily_input['close'] / ref_close
    daily_input['volume'] = daily_input['volume'] / ref_volume

    daily_input = daily_input.fillna(0)
    daily_input.replace([np.inf, -np.inf], 0, inplace=True)
    daily_input = add_technical_factor(add_pool_factors(daily_input))
    keep_cols = ['open', 'high', 'low', 'close', 'volume', 'delta'] + get_technical_factors(FACTOR_MODE)
    daily_input = daily_input[keep_cols]
    daily_input['idx'] = daily_input.index / (DAYS_INPUT - 1.0)
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
    return torch.tensor(daily_input).unsqueeze(0).float().to(device)


def predict_close_raw(model, daily_input):
    """模型输出 z-scored 的 5 个 close 值，反标准化回原始尺度。

    ⚠️ 2026-09-25 修 bug：这里**必须** `model.eval()`。
    `one.pt` 是「整个模型对象」pickle 存下来的，而它落盘时的 `training` 标志是 **True**
    （训练脚本没在存盘前切 eval），加载后 nn.Module 的 training 就跟着是 True。
    只要不显式 eval()，模型里的 nn.Dropout(0.1)（以及注意力里的 dropout）就会在**推理时生效**
    —— 同一个输入每次前向都会随机丢不同的神经元，输出每次都不一样。
    实测：同一输入 forward 4 次，输出 −35.15 / −32.72 / −32.54 / −27.81（极差 7.34），
    直接导致每日最终选票在 buffett 档位内按 up_down 排序时**随机漂移**。
    （seven/three/zero 的预测脚本本来就调了 .eval()，只有 one 漏了。）
    """
    model.eval()                          # ← 关掉 dropout / 固定 BatchNorm
    with torch.no_grad():                 # 推理不需要梯度
        _, close_preds = model(daily_input)  # (1, 1, 5)
    close_z = close_preds.squeeze(0).squeeze(0).cpu().detach().numpy().astype(np.float32)  # (5,)
    # 2026-09-26：反标准化（×close_std + close_mean）已**焼进模型的 buffer**（one_model.set_output_scale），
    # 模型直接输出真实单位 -> 这里不能再乘一次（否则双重换算）。已做逐元素等价验证。
    return close_z


def build_predict_result(symbol, date, close_preds):
    result = {'symbol': symbol, 'date': int(date)}
    for name, value in zip(CLOSE_COLS, close_preds):
        result[name] = float(value)
    result['up_down'] = float(sum(close_preds))
    return result


def compute_aligned_updown(df):
    """up_down = 5 个 horizon 预测按日期截面 z-score 后求和。

    pearson loss 是尺度无关的，各 horizon 预测尺度可能不同；求和前做截面 z-score
    让 5 个 horizon 等权，避免尺度大的 horizon 暗中主导 up_down。
    """
    zs = []
    for h in HORIZONS:
        col = f'close_{h}'
        g = df.groupby('date')[col]
        z = (df[col] - g.transform('mean')) / (g.transform('std') + 1e-8)
        zs.append(z)
    return sum(zs)


def raise_on_empty_merge(merged, left_dates, right_dates, source):
    """按 (symbol, date) inner merge 后为空时报错。

    merge 为空基本只有一个原因：上游股票池（two/schloss）没跟着日线数据一起刷新，
    日期错位导致两边 (symbol, date) 完全不相交。以前这种情况会静默写出空 CSV，
    一路传到 buy_predict.csv 才在 send_candidates 里以 IndexError 的形式爆出来，
    这里直接报出两边的日期，定位更快。
    """
    if not merged.empty:
        return
    left_set, right_set = set(left_dates), set(right_dates)
    raise RuntimeError(
        f"与 {source} 按 (symbol, date) 合并后无任何匹配：\n"
        f"  本次预测日期: {sorted(left_set)[:5]}（共 {len(left_set)} 个）\n"
        f"  {source} 日期 : {sorted(right_set)[:5]}（共 {len(right_set)} 个）\n"
        f"  日期交集: {sorted(left_set & right_set)[:5]}（共 {len(left_set & right_set)} 个）\n"
        f"  请先确认 {source} 是否已随最新的日线数据一起刷新（schloss -> two -> one 顺序不能反）。"
    )


def get_one_candidates(check_days=0, target_date=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    data_name = os.path.join(os.path.dirname(__file__), '../data/')
    daily_shz = pd.read_csv(data_name + 'SHZ/daily_shz.csv')
    daily_shh = pd.read_csv(data_name + 'SHH/daily_shh.csv')
    daily_data = pd.concat([daily_shz, daily_shh], axis=0).reset_index(drop=True)
    if target_date is not None:
        daily_data = daily_data[daily_data['date'] <= int(target_date)].reset_index(drop=True)
    if FACTOR_MODE in ('plan1', 'plan2', 'plan1_ts', 'plan2_ts'):
        daily_data = merge_new_factors(daily_data)

    two_data = pd.read_csv(data_name + '../two/two_predict.csv')
    # 2026-09-25：two 阶段换成 nowcast 模型，三个头的列名随之改成 gpMed7/revMed7/taMed7
    # （旧名 seven/thirty_one/two_hundred_and_twenty_seven 语义已完全不同）。
    # ⚠️ 这个列表必须与 two/get_two_predict.py 写出的列名一致，否则 KeyError。
    two_data.drop(columns=['gpMed7', 'revMed7', 'taMed7'], inplace=True)
    two_data = two_data[two_data['up_down'] > two_data['up_down'].median()].reset_index(drop=True)
    two_data.drop(columns=['up_down'], inplace=True)
    two_data = two_data.sort_values('buffett', ascending=False).reset_index(drop=True)
    buffett_list = two_data['symbol'].to_list()

    groups = list(daily_data.groupby('symbol'))
    predict_list = []
    for i in tqdm(range(len(groups))):
        symbol = groups[i][0]
        if symbol not in buffett_list:
            continue
        daily_group = groups[i][1].reset_index(drop=True)
        if check_days != 0:
            daily_group = daily_group.iloc[:-check_days].reset_index(drop=True)
        daily_group = daily_group.iloc[-DAYS_INPUT:].reset_index(drop=True)
        if len(daily_group) != DAYS_INPUT:
            continue
        date = daily_group['date'].iloc[-1]

        daily_input = prepare_input(daily_group, device)
        close_preds = predict_close_raw(model, daily_input)
        predict_list.append(build_predict_result(symbol, date, close_preds))

    predict_data = pd.DataFrame(predict_list)
    predict_dates = predict_data['date'].tolist() if 'date' in predict_data.columns else []
    predict_data = pd.merge(predict_data, two_data, on=['symbol', 'date'], how='inner').dropna().reset_index(drop=True)
    raise_on_empty_merge(predict_data, predict_dates, two_data['date'].tolist(), 'two/two_predict.csv')
    predict_data['up_down'] = compute_aligned_updown(predict_data)
    predict_data = predict_data.sort_values('buffett', ascending=False).reset_index(drop=True)
    print(predict_data)
    one_predict_name = os.path.join(os.path.dirname(__file__), 'one_predict.csv')
    predict_data.to_csv(one_predict_name, index=False)


def get_one_all(check_days=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    data_name = os.path.join(os.path.dirname(__file__), '../data/')
    daily_shz = pd.read_csv(data_name + 'SHZ/daily_shz.csv')
    daily_shh = pd.read_csv(data_name + 'SHH/daily_shh.csv')
    daily_data = pd.concat([daily_shz, daily_shh], axis=0).reset_index(drop=True)
    if FACTOR_MODE in ('plan1', 'plan2', 'plan1_ts', 'plan2_ts'):
        daily_data = merge_new_factors(daily_data)

    groups = list(daily_data.groupby('symbol'))
    predict_list = []
    for i in tqdm(range(len(groups))):
        symbol = groups[i][0]
        daily_group = groups[i][1].reset_index(drop=True)
        if check_days != 0:
            daily_group = daily_group.iloc[:-check_days].reset_index(drop=True)
        daily_group = daily_group.iloc[-DAYS_INPUT:].reset_index(drop=True)
        if len(daily_group) != DAYS_INPUT:
            continue
        date = daily_group['date'].iloc[-1]

        daily_input = prepare_input(daily_group, device)
        close_preds = predict_close_raw(model, daily_input)
        predict_list.append(build_predict_result(symbol, date, close_preds))

    predict_data = pd.DataFrame(predict_list)
    predict_data['up_down'] = compute_aligned_updown(predict_data)
    predict_data = predict_data.sort_values('up_down', ascending=False).reset_index(drop=True)
    latest_date = predict_data['date'].max()
    print(latest_date)
    predict_data = predict_data[predict_data['date'] == latest_date].copy().reset_index(drop=True)
    one_predict_name = os.path.join(os.path.dirname(__file__), 'one_all_predict.csv')
    predict_data.to_csv(one_predict_name, index=False)


def refresh_buy(target_date=None):
    get_one_candidates(target_date=target_date)
    data_name = os.path.join(os.path.dirname(__file__), '../data/')
    daily_shz = pd.read_csv(data_name + 'SHZ/daily_shz.csv')
    daily_shh = pd.read_csv(data_name + 'SHH/daily_shh.csv')
    daily_data = pd.concat([daily_shz, daily_shh], axis=0).reset_index(drop=True)
    if target_date is not None:
        daily_data = daily_data[daily_data['date'] <= int(target_date)].reset_index(drop=True)
    latest_date = daily_data['date'].max()
    daily_data = daily_data[daily_data['date'] == latest_date].copy().reset_index(drop=True)
    daily_data.drop(columns=['open', 'low', 'high', 'volume'], inplace=True)

    buy_data = pd.read_csv(data_name + '../one/one_predict.csv')
    if buy_data.empty:
        raise RuntimeError(
            "one_predict.csv 为空（本次没有任何候选通过筛选）。"
            "不要写出空的 buy_predict.csv，否则下游会以 IndexError 的形式报错。"
        )
    buy_data = buy_data.sort_values('buffett', ascending=False).reset_index(drop=True)
    buy_data = buy_data.sort_values('up_down', ascending=False).reset_index(drop=True)
    print(buy_data)
    buy_predict_name = os.path.join(os.path.dirname(__file__), '../buy_predict.csv')
    buy_data.to_csv(buy_predict_name, index=False)


if __name__ == "__main__":
    refresh_buy()
    # get_one_all()
