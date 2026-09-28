import numpy as np
import pandas as pd
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from zero_model import ZERO
from torch.utils.data import Dataset, DataLoader

class Dataset(Dataset):
    def __init__(self, h5_file_list):
        self.h5_file_list = h5_file_list

    def __len__(self):
        return len(self.h5_file_list)  # 返回文件列表的长度

    def __getitem__(self, idx):
        # 随机选择两个 HDF5 文件
        # selected_files = random.sample(self.h5_file_list, 1)
        data = pd.read_hdf(self.h5_file_list[idx])

        input_data, three = get_data_input(data)
        input_data = input_data.values
        input_data = torch.tensor(input_data).float()
        three = torch.tensor(three).unsqueeze(0).unsqueeze(0).float()
        return input_data, three


def get_data_input(data):
    input_data = data.iloc[:, :-1]
    three = data.iloc[0, -1]
    return input_data, three


def pairwise_rank_loss(pred, target, margin=0.1):
    """pairwise 排序 loss：直接对齐 ICIR（秩相关）。

    pred/target 均为 [B]（预测分与 negMdd 标签）。对每个标签有差异的对子 (i,j)，
    若预测顺序与标签顺序一致且差距超过 margin 则 loss=0，否则按 margin 惩罚。
    只对 |target_i - target_j| > 1e-3 的对子算，避免同标签对子的噪声。
    """
    pd_ = pred.unsqueeze(1) - pred.unsqueeze(0)      # [B, B] 预测分差
    td = target.unsqueeze(1) - target.unsqueeze(0)   # [B, B] 标签差
    mask = td.abs() > 1e-3
    if not mask.any():
        return (pred * 0).sum()
    return torch.relu(margin - pd_ * td.sign())[mask].mean()



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

def train_zero_model():
    # --- 设备设置 ---
    device = select_device()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_name = os.path.join(current_dir, 'zero.pt')
    model = ZERO([7, 3], [1, 1]).to(device)
    batch_size = 512
    train_folder = os.path.join(current_dir, 'train')
    h5_files = [os.path.join(train_folder, f) for f in os.listdir(train_folder) if f.endswith('.h5')]

    # 按股票(symbol)隔离划分：先随机抽出 TEST_N 只股票作为测试集（不参与训练），
    # 其余股票再按 7:3 隔离成 train/val。
    # 文件名形如 {symbol}_{endDate}.h5，symbol 可能含下划线，故按最后一个下划线切。
    def _symbol_of(f):
        return os.path.basename(f)[:-3].rsplit('_', 1)[0]

    TEST_N = 127 * 7  # 测试集股票数（127×7，扩大测试样本以降低 IC 噪声，对齐 seven/three）
    TEST_SEED = int(os.environ.get('TEST_SEED', 42))  # 可用环境变量 TEST_SEED 覆盖（每次训练换测试集）
    symbols = sorted({_symbol_of(f) for f in h5_files})
    if len(symbols) <= TEST_N:
        raise ValueError(f"股票总数 {len(symbols)} 不足，无法抽出 {TEST_N} 只测试股票")

    # 固定测试集（2026-09-26）：已存在 test_symbols.txt 就直接沿用，不再按股票全集重抽。
    # 动机：数据里加入新股票后，下面那次 seed 洗牌的【输入】就变了 -> 整个测试集都会漂，
    # 新旧测试指标就不可比了。用户要求：测试的股票固定，只是测试数据要有新的行。
    _pin = os.path.join(current_dir, 'test_symbols.txt')
    if os.path.exists(_pin):
        with open(_pin) as fp:
            _want = {l.strip() for l in fp if l.strip()}
        test_symbols = {s for s in symbols if s in _want}
        _miss = sorted(_want - set(symbols))
        print(f'[固定测试集] 沿用 test_symbols.txt：命中 {len(test_symbols)} / 名单 {len(_want)}',
              flush=True)
        if _miss:
            print(f'  ⚠️ 名单里 {len(_miss)} 只在新数据里不存在，已剔出测试集：'
                  f'{_miss[:10]}{" ..." if len(_miss) > 10 else ""}', flush=True)
    else:
        # 首次运行：仍是"固定随机"——独立 Random 实例 + 固定种子，不污染全局随机状态
        rng_test = random.Random(TEST_SEED)
        test_pool = symbols[:]
        rng_test.shuffle(test_pool)
        test_symbols = set(test_pool[:TEST_N])
        print('[首次运行] 无 test_symbols.txt -> 按 TEST_SEED 重抽并落盘', flush=True)

    # 训练/验证：不固定随机——用全局 random（系统熵播种），每次运行划分不同
    rest_symbols = [s for s in symbols if s not in test_symbols]
    random.shuffle(rest_symbols)
    train_length = int(len(rest_symbols) * 0.7)
    train_symbols = set(rest_symbols[:train_length])
    val_symbols = set(rest_symbols[train_length:])

    test_files = [f for f in h5_files if _symbol_of(f) in test_symbols]
    train_files = [f for f in h5_files if _symbol_of(f) in train_symbols]
    val_files = [f for f in h5_files if _symbol_of(f) in val_symbols]

    # 保存测试股票列表，供后续复用
    test_symbols_path = os.path.join(current_dir, 'test_symbols.txt')
    with open(test_symbols_path, 'w') as fp:
        fp.write('\n'.join(sorted(test_symbols)))
    print(f"已随机抽出测试股票 {len(test_symbols)} 只（样本 {len(test_files)}）并保存到 {test_symbols_path}；"
          f"训练股票 {len(train_symbols)} / 验证股票 {len(val_symbols)}；"
          f"训练样本 {len(train_files)} / 验证样本 {len(val_files)}")

    train_dataset = Dataset(train_files)
    val_dataset = Dataset(val_files)
    test_dataset = Dataset(test_files)      # 2026-09-26：测试集以前抽了却没用，现在补上评估

    if os.path.exists(model_name):
        loaded_model = torch.load(model_name, map_location=device, weights_only=False)
        if hasattr(loaded_model, 'input_proj'):
            model = loaded_model.to(device)
            print('loaded existing ZERO model:', model_name)
        else:
            print('old ZERO checkpoint detected, training the optimized model from scratch:', model_name)

    criterion_mse = nn.MSELoss()
    learning_rate = 0.001
    # 7 = 五个模型共用的口径（2026-09-27 从 31 改过来）：one/three/seven/two 的 train.py
    # 都是 7 个 epoch，只有 zero 是 31 —— 同样的 7 轮里 zero 早就收敛了（见 zero_featnorm_7ep /
    # zero_mse_7ep 那两份日志：第 6 轮 val loss 0.03189 / 0.03258，与 31 轮的 0.03184 无差别），
    # 留着 31 只是让"重训全部模型"多花 4 倍时间、指标还没法横向比。
    num_epochs = 7
     # 推荐设置为 CPU 核心数减一
    num_workers = os.cpu_count() // 4 if os.cpu_count() else 4
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        persistent_workers=True,
        pin_memory=(device.type == 'cuda')
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False, # 验证集无需 shuffle
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=(device.type == 'cuda')
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=(device.type == 'cuda')
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        # 训练阶段
        mean_train_loss = 0.0
        step_num = 0
        for data_input, three in train_dataloader:
            data_input = data_input.to(device, non_blocking=True)
            three = three.to(device, non_blocking=True)
            optimizer.zero_grad()
            three_predict = model(data_input)
            loss = criterion_mse(three, three_predict)
            # loss = criterion(predict0, fore0) + criterion(predict1, fore1) + 3*criterion(compare, gain)
            # print(criterion(compare, gain).item())
            loss.backward()  # 计算梯度
            # optimizer.step()
            optimizer.step()
            mean_train_loss = (mean_train_loss*step_num + loss.item())/float(step_num + 1)
            step_num = step_num + 1
            try:
                print("Epoch: %d, train loss: %1.5f, mean loss: %1.5f, min val loss: %1.5f" %
                      (epoch, loss.item(), mean_train_loss, best_val_loss))
            except:
                pass

        # 验证阶段
        mean_val_loss = 0
        with ((torch.no_grad())):
            for data_input, three in val_dataloader:
                data_input = data_input.to(device, non_blocking=True)
                three = three.to(device, non_blocking=True)
                three_predict = model(data_input)
                loss = criterion_mse(three, three_predict)
                # growth_death_loss = criterion_mse(decoder, growth_death
                # loss = criterion(predict0, fore0) + criterion(predict1, fore1) + 3 * criterion(compare, gain)
                mean_val_loss += loss.item()

        mean_val_loss = mean_val_loss / len(val_dataloader)
        print("Epoch: %d, validate loss: %1.5f" % (epoch, mean_val_loss))
        # 如果当前模型比之前的模型性能更好，则保存当前模型
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            print('best_val_loss:' + str(best_val_loss) + ' saving model:' + model_name)
            torch.save(model, model_name)

    # ---- 测试集评估（2026-09-26 加）：用留出的 test_files，口径与验证集完全一致 ----
    # 让重训流程（训 3 遍选验证最优、并汇报测试集结果）有一个统一的机器可读输出。
    if test_files:
        best = torch.load(model_name, map_location=device, weights_only=False).eval()
        tl = 0.0
        with torch.no_grad():
            for data_input, three in test_dataloader:
                data_input = data_input.to(device, non_blocking=True)
                three = three.to(device, non_blocking=True)
                tl += criterion_mse(three, best(data_input)).item()
        mean_test_loss = tl / max(len(test_dataloader), 1)
        print(f'[TEST] test_loss={mean_test_loss:.6f}')
    print(f'[VAL]  best_val_loss={best_val_loss:.6f}')


if __name__ == '__main__':
    train_zero_model()
