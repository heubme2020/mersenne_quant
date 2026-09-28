import os
import math
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from one_model import ONE, HORIZONS, AUX_OUTPUT_DAYS, set_output_scale
from torch.utils.data import Dataset, DataLoader

DAYS_INPUT = 127 * 7          # 输入窗口 889 天
TOMORROW_IDX = DAYS_INPUT     # 明天（close 标签基准）索引
FORE_COLS = ['close', 'volume', 'delta']

TEST_N = 127 * 7              # 测试集股票数（127×7 = 889）
TEST_SEED = int(os.environ.get('TEST_SEED', 42))  # 可用环境变量 TEST_SEED 覆盖（每次训练换测试集）

# loss 方案（命令行参数传入）：mse / smooth_l1 / logcosh / asymmetric / pearson
LOSS_MODES = ['mse', 'smooth_l1', 'logcosh', 'asymmetric', 'pearson']
LOSS_MODE = sys.argv[1] if len(sys.argv) > 1 else 'pearson'
DATA_MODE = sys.argv[2] if len(sys.argv) > 2 else 'ashare'  # ashare / global
FACTOR_MODE = os.environ.get('FACTOR_MODE', 'baseline')     # baseline / plan1 / plan2
TRAIN_DIR = os.environ.get('TRAIN_DIR', 'train')            # 训练数据目录
GLOBAL_FOLDER = 'D:/quant_data/train_global'                # 全球数据目录（D 盘）
RUN_TAG = os.environ.get('RUN_TAG', '')
MODEL_NAMES = {m: f'one_{DATA_MODE}_{m}_{FACTOR_MODE}{RUN_TAG}.pt' for m in LOSS_MODES}
EVAL_NAMES = {m: f'{DATA_MODE}_{m}_{FACTOR_MODE}{RUN_TAG}' for m in LOSS_MODES}


def is_ashare(symbol):
    return symbol.endswith('.SS') or symbol.endswith('.SZ')


def load_h5_files(train_folder, data_mode):
    dirs = [train_folder]
    if data_mode == 'global':
        dirs.append(GLOBAL_FOLDER)
    files = []
    for d in dirs:
        if os.path.isdir(d):
            files += [os.path.join(d, f) for f in os.listdir(d) if f.endswith('.h5')]
    return files


def select_device():
    if torch.cuda.is_available():
        try:
            (torch.zeros(2, 2, device='cuda') @ torch.zeros(2, 2, device='cuda')).cpu()
            return torch.device('cuda')
        except Exception as e:
            print(f'[警告] CUDA 检测到但 kernel 不可用（{type(e).__name__}），退回 CPU。')
    return torch.device('cpu')


def symbol_of(f):
    return os.path.basename(f)[:-3].rsplit('_', 1)[0]


def compute_raw_labels(data):
    """5 个 close 标签（原始 log 比值，未标准化）。"""
    close_tomorrow = data['close'].iloc[TOMORROW_IDX]
    labels = []
    for h in HORIZONS:
        w = data['close'].iloc[TOMORROW_IDX + 1: TOMORROW_IDX + 1 + h]
        g = math.log(w.max()) + math.log(w.median()) + math.log(w.min()) - 3.0 * math.log(close_tomorrow)
        labels.append(g)
    return np.array(labels, dtype=np.float64)


def estimate_target_stats(h5_files, max_samples=50000):
    """在【训练集】上估计全局 μ/σ：5 个 close 标签 + 3 个辅助特征（用于 z-score）。"""
    files = h5_files
    if max_samples and len(files) > max_samples:
        step = len(files) / max_samples
        files = [files[int(i * step)] for i in range(max_samples)]

    n_close = len(HORIZONS)
    close_sum = np.zeros(n_close, dtype=np.float64)
    close_sumsq = np.zeros(n_close, dtype=np.float64)
    n_samples = 0

    aux_sum = np.zeros(3, dtype=np.float64)
    aux_sumsq = np.zeros(3, dtype=np.float64)
    n_aux = 0

    for f in tqdm(files, desc='估算标签统计'):
        try:
            data = pd.read_hdf(f, key='data')
        except Exception:
            continue
        close_tomorrow = data['close'].iloc[TOMORROW_IDX]
        if close_tomorrow <= 0:
            continue
        raw = compute_raw_labels(data)
        close_sum += raw
        close_sumsq += raw * raw
        n_samples += 1

        fore = data.iloc[DAYS_INPUT: DAYS_INPUT + AUX_OUTPUT_DAYS][FORE_COLS].values.astype(np.float64)
        aux_sum += fore.sum(axis=0)
        aux_sumsq += (fore * fore).sum(axis=0)
        n_aux += fore.shape[0]

    close_mean = (close_sum / n_samples).astype(np.float32)
    close_std = np.sqrt(np.maximum(close_sumsq / n_samples - close_mean ** 2, 1e-12)).astype(np.float32)

    aux_mean = (aux_sum / n_aux).astype(np.float32)
    aux_std = np.sqrt(np.maximum(aux_sumsq / n_aux - aux_mean ** 2, 1e-12)).astype(np.float32)

    return close_mean, close_std, aux_mean, aux_std


class Dataset(Dataset):
    def __init__(self, h5_file_list, close_mean, close_std, aux_mean, aux_std):
        self.h5_file_list = h5_file_list
        self.close_mean = close_mean
        self.close_std = close_std
        self.aux_mean = aux_mean
        self.aux_std = aux_std

    def __len__(self):
        return len(self.h5_file_list)

    def __getitem__(self, idx):
        data = pd.read_hdf(self.h5_file_list[idx], key='data')
        data_input, data_fore, labels = get_data_input(
            data, self.close_mean, self.close_std, self.aux_mean, self.aux_std)
        return (torch.tensor(data_input.values).float(),
                torch.tensor(data_fore).float(),
                torch.tensor(labels).float())


def get_data_input(data, close_mean, close_std, aux_mean, aux_std):
    """提取输入、辅助序列目标、5 个 close 标签，并对目标做全局 z-score。"""
    data_input = data.iloc[:DAYS_INPUT].reset_index(drop=True)

    fore = data.iloc[DAYS_INPUT: DAYS_INPUT + AUX_OUTPUT_DAYS][FORE_COLS].values.astype(np.float32)
    data_fore = (fore - aux_mean) / aux_std

    raw = compute_raw_labels(data)
    # z-score 目标（减均值 + 除尺度）。模型输出因此在 z 空间 -> 由 train.py 存盘时
    # 焼进模型的 out_mean/out_scale buffer 还原成真实单位（见 set_output_scale）。
    # 若想改成 seven/three 那种「两边同除、不减均值」（少存一个常数），
    # **必须把损失里的预测也一起除**（`f(pred/std, label/std)`）—— 只除目标不改预测
    # 只会把模型输出变成"除以尺度后的值"，既得不到真实单位、又要多一次换算（2026-09-26 踩过）。
    labels = (raw.astype(np.float32) - close_mean) / close_std

    return data_input, data_fore, labels


def _logcosh(pred, target):
    x = pred - target
    return (x + F.softplus(-2.0 * x) - math.log(2.0)).mean()


def compute_loss(cvd, close_preds, data_fore, labels, criterion_mse):
    """aux 固定 MSE（正则），close 头按 LOSS_MODE 切换损失函数。"""
    loss = criterion_mse(cvd, data_fore)

    if LOSS_MODE == 'mse':
        for k in range(len(HORIZONS)):
            loss = loss + criterion_mse(close_preds[:, k], labels[:, k])
    elif LOSS_MODE == 'smooth_l1':
        for k in range(len(HORIZONS)):
            loss = loss + F.smooth_l1_loss(close_preds[:, k], labels[:, k])
    elif LOSS_MODE == 'logcosh':
        for k in range(len(HORIZONS)):
            loss = loss + _logcosh(close_preds[:, k], labels[:, k])
    elif LOSS_MODE == 'asymmetric':
        for k in range(len(HORIZONS)):
            err = labels[:, k] - close_preds[:, k]  # 真实 - 预测，<0 即高估
            loss = loss + torch.where(err < 0, 3.0 * err * err, err * err).mean()
    elif LOSS_MODE == 'pearson':
        for k in range(len(HORIZONS)):
            p = close_preds[:, k] - close_preds[:, k].mean()
            t = labels[:, k] - labels[:, k].mean()
            corr = (p * t).sum() / (p.norm() * t.norm() + 1e-8)
            loss = loss + (1.0 - corr)
    return loss


def train_one_model():
    device = select_device()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_name = os.path.join(current_dir, MODEL_NAMES[LOSS_MODE])
    train_folder = os.path.join(current_dir, TRAIN_DIR)
    h5_files = load_h5_files(train_folder, DATA_MODE)
    if not h5_files:
        raise RuntimeError("train 文件夹为空或无 h5 文件")

    # ---- 按股票隔离划分：测试集 = 889 只 A 股（seed 42），val = A 股，train = 其余 A 股 + 全球 ----
    symbols = sorted({symbol_of(f) for f in h5_files})
    ashare_symbols = [s for s in symbols if is_ashare(s)]
    if len(ashare_symbols) <= TEST_N:
        raise ValueError(f"A 股总数 {len(ashare_symbols)} 不足，无法抽出 {TEST_N} 只测试股票")

    # 固定测试集（2026-09-26）：已存在 test_symbols.txt 就直接沿用，不再按股票全集重抽。
    # 动机：数据里加入新股票后，下面那次 seed 洗牌的【输入】就变了 -> 整个测试集都会漂，
    # 新旧测试指标就不可比了。用户要求：测试的股票固定，只是测试数据要有新的行。
    _pin = os.path.join(current_dir, 'test_symbols.txt')
    if os.path.exists(_pin):
        with open(_pin) as fp:
            _want = {l.strip() for l in fp if l.strip()}
        test_symbols = {s for s in ashare_symbols if s in _want}
        _miss = sorted(_want - set(ashare_symbols))
        print(f'[固定测试集] 沿用 test_symbols.txt：命中 {len(test_symbols)} / 名单 {len(_want)}',
              flush=True)
        if _miss:
            print(f'  ⚠️ 名单里 {len(_miss)} 只在新数据里不存在，已剔出测试集：'
                  f'{_miss[:10]}{" ..." if len(_miss) > 10 else ""}', flush=True)
    else:
        # 首次运行：仍是"固定随机"——独立 Random 实例 + 固定种子，不污染全局随机状态
        rng_test = random.Random(TEST_SEED)
        test_pool = ashare_symbols[:]
        rng_test.shuffle(test_pool)
        test_symbols = set(test_pool[:TEST_N])
        print('[首次运行] 无 test_symbols.txt -> 按 TEST_SEED 重抽并落盘', flush=True)

    remaining_ashare = [s for s in ashare_symbols if s not in test_symbols]
    rng_split = random.Random(int(os.environ.get('VAL_SPLIT_SEED', 12345)))  # 可用环境变量覆盖（每次训练换验证集划分）
    rng_split.shuffle(remaining_ashare)
    val_length = int(len(remaining_ashare) * 0.3)
    val_symbols = set(remaining_ashare[:val_length])
    train_symbols = set(remaining_ashare[val_length:]) | {s for s in symbols if not is_ashare(s)}

    train_files = [f for f in h5_files if symbol_of(f) in train_symbols]
    val_files = [f for f in h5_files if symbol_of(f) in val_symbols]
    test_files = [f for f in h5_files if symbol_of(f) in test_symbols]

    test_symbols_path = os.path.join(current_dir, 'test_symbols.txt')
    with open(test_symbols_path, 'w') as fp:
        fp.write('\n'.join(sorted(test_symbols)))
    print(f"[{DATA_MODE}] 测试A股 {len(test_symbols)}（样本 {len(test_files)}）；"
          f"训练 {len(train_symbols)} 股 / 验证 {len(val_symbols)} 股；"
          f"训练样本 {len(train_files)} / 验证样本 {len(val_files)}；loss={LOSS_MODE}", flush=True)

    # ---- 全局统计（仅训练集）+ 保存 ----
    close_mean, close_std, aux_mean, aux_std = estimate_target_stats(train_files)
    stats_path = os.path.join(current_dir, 'target_stats.npz')
    np.savez(stats_path, close_mean=close_mean, close_std=close_std,
             aux_mean=aux_mean, aux_std=aux_std)
    print(f'close_std: {close_std.round(6).tolist()}', flush=True)

    train_dataset = Dataset(train_files, close_mean, close_std, aux_mean, aux_std)
    val_dataset = Dataset(val_files, close_mean, close_std, aux_mean, aux_std)

    _seed = int(os.environ.get('TORCH_SEED', '0'))
    if _seed:
        torch.manual_seed(_seed); np.random.seed(_seed); random.seed(_seed)
        print(f'[种子] TORCH_SEED={_seed}', flush=True)
    model = ONE([31, DAYS_INPUT]).to(device)
    if os.path.exists(model_name):
        model = torch.load(model_name, map_location=device, weights_only=False).to(device)
        print('loaded existing model:', model_name)

    criterion_mse = nn.MSELoss()
    learning_rate = 0.001
    num_epochs = 7
    batch_size = 64
    num_workers = os.cpu_count() // 4 if os.cpu_count() else 4
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, persistent_workers=True, pin_memory=(device.type == 'cuda'))
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, persistent_workers=True, pin_memory=(device.type == 'cuda'))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        mean_train_loss = 0.0
        step_num = 0
        for data_input, data_fore, labels in train_dataloader:
            data_input = data_input.to(device, non_blocking=True)
            data_fore = data_fore.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()

            cvd, close_preds = model(data_input)
            close_preds = close_preds.squeeze(1)
            loss = compute_loss(cvd, close_preds, data_fore, labels, criterion_mse)

            loss.backward()
            optimizer.step()

            mean_train_loss = (mean_train_loss * step_num + loss.item()) / float(step_num + 1)
            step_num += 1
            if step_num % 50 == 0:
                print("Epoch: %d, step: %d, train loss: %1.5f, mean loss: %1.5f, min val loss: %1.5f" %
                      (epoch, step_num, loss.item(), mean_train_loss, best_val_loss), flush=True)

        mean_val_loss = 0.0
        with torch.no_grad():
            for data_input, data_fore, labels in val_dataloader:
                data_input = data_input.to(device, non_blocking=True)
                data_fore = data_fore.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                cvd, close_preds = model(data_input)
                close_preds = close_preds.squeeze(1)
                mean_val_loss += compute_loss(cvd, close_preds, data_fore, labels, criterion_mse).item()

        mean_val_loss = mean_val_loss / len(val_dataloader)
        print("Epoch: %d, validate loss: %1.5f" % (epoch, mean_val_loss), flush=True)
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            print('best_val_loss:' + str(best_val_loss) + ' saving model:' + model_name, flush=True)
            # 顺序必须是「装上 affine -> 存 -> 还原恒等」：
            #   装上 affine 存下来 -> .pt 直接输出真实单位（evaluate_model 读的就是它，内部不再换算）
            #   存完还原恒等     -> 训练/验证的标签和损失都是 z 空间，带着 affine 会错
            set_output_scale(model, close_mean, close_std)
            torch.save(model, model_name)
            set_output_scale(model, np.zeros_like(close_mean), np.ones_like(close_std))

    from evaluate import evaluate_model
    print("\n训练完成，开始在留出的测试股票上评估...", flush=True)
    evaluate_model(
        model_path=model_name,
        train_folder=train_folder,
        test_symbols=test_symbols,
        name=EVAL_NAMES[LOSS_MODE],
        output_dir=os.path.join(current_dir, 'eval_results'),
        device=device,
    )


if __name__ == '__main__':
    train_one_model()
