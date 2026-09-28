# import numpy as np
# import pandas as pd
# import os
# import random
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from three_model import THREE
# from torch.utils.data import Dataset, DataLoader

# class Dataset(Dataset):
#     def __init__(self, h5_file_list):
#         self.h5_file_list = h5_file_list

#     def __len__(self):
#         return len(self.h5_file_list)  # 返回文件列表的长度

#     def __getitem__(self, idx):
#         # 随机选择两个 HDF5 文件
#         # selected_files = random.sample(self.h5_file_list, 1)
#         data = pd.read_hdf(self.h5_file_list[idx])

#         input_data, growth_death_one, growth_death_three, growth_death_seven= get_data_input(data)
#         input_data = input_data.values
#         input_data = torch.tensor(input_data).float()
#         growth_death_one = growth_death_one.values
#         growth_death_one = torch.tensor(growth_death_one).unsqueeze(0).unsqueeze(0).float()
#         growth_death_three = growth_death_three.values
#         growth_death_three = torch.tensor(growth_death_three).unsqueeze(0).unsqueeze(0).float()
#         growth_death_seven = growth_death_seven.values
#         growth_death_seven = torch.tensor(growth_death_seven).unsqueeze(0).unsqueeze(0).float()
#         return input_data.to('cuda'), growth_death_one.to('cuda'), growth_death_three.to('cuda'), growth_death_seven.to('cuda')


# def get_data_input(data):
#     input_data = data.iloc[:, :-3]
#     growth_death_one = data.iloc[0, -3]
#     growth_death_three = data.iloc[0, -2]
#     growth_death_seven = data.iloc[0, -1]
#     return input_data, growth_death_one, growth_death_three, growth_death_seven


# def train_three_model():
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     model_name = os.path.join(current_dir, 'three.pt')
#     model = THREE([127, 31], [3, 1]).to('cuda')
#     batch_size = 256

#     train_folder = os.path.join(current_dir, 'train')
#     h5_files = [os.path.join(train_folder, f) for f in os.listdir(train_folder) if f.endswith('.h5')]
#     random.shuffle(h5_files)
#     # h5_files = h5_files[:131071]
#     train_length = int(len(h5_files)*0.7)
#     train_files = h5_files[:train_length]
#     val_files = h5_files[train_length:]
#     train_dataset = Dataset(train_files)
#     val_dataset = Dataset(val_files)

#     if os.path.exists(model_name):
#         model = torch.load(model_name)

#     criterion_mse = nn.MSELoss()
#     # criterion = nn.CrossEntropyLoss()
#     criterion = nn.L1Loss()
#     learning_rate = 0.001
#     num_epochs = 7
#     train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
#     val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     best_val_loss = float('inf')
#     # val_compare_loss = float('inf')
#     for epoch in range(num_epochs):
#         # 训练阶段
#         mean_train_loss = 0.0
#         mean_up_down_loss = 0.0
#         step_num = 0
#         for data_input, growth_death_one,  growth_death_three, growth_death_seven in train_dataloader:
#             up_down = torch.sign(growth_death)
#             optimizer.zero_grad()
#             up_down_predict, growth_death_predict = model(data_input)
#             growth_death_loss = criterion_mse(growth_death, growth_death_predict)
#             up_down_loss = criterion(up_down, up_down_predict)
#             loss = growth_death_loss + up_down_loss
#             # loss = criterion(predict0, fore0) + criterion(predict1, fore1) + 3*criterion(compare, gain)
#             # print(criterion(compare, gain).item())
#             loss.backward()  # 计算梯度
#             # optimizer.step()
#             optimizer.step()
#             mean_up_down_loss = (mean_up_down_loss*step_num + up_down_loss.item())/float(step_num + 1)
#             # running_train_loss += loss.item() * inputs.size(0)
#             mean_train_loss = (mean_train_loss*step_num + loss.item())/float(step_num + 1)
#             step_num = step_num + 1
#             try:
#                 print("Epoch: %d, train loss: %1.5f, mean loss: %1.5f, mean_up_down_loss:%1.5f, min val loss: %1.5f" %
#                       (epoch, loss.item(), mean_train_loss, mean_up_down_loss, best_val_loss))
#             except:
#                 pass

#         # 验证阶段
#         mean_val_loss = 0
#         with ((torch.no_grad())):
#             for data_input, growth_death in val_dataloader:
#                 up_down = torch.sign(growth_death)
#                 up_down_predict, growth_death_predict = model(data_input)
#                 up_down_loss = criterion(up_down, up_down_predict)
#                 # growth_death_loss = criterion_mse(decoder, growth_death
#                 # loss = criterion(predict0, fore0) + criterion(predict1, fore1) + 3 * criterion(compare, gain)
#                 mean_val_loss += up_down_loss.item()

#         mean_val_loss = mean_val_loss / len(val_dataloader)
#         print("Epoch: %d, validate loss: %1.5f" % (epoch, mean_val_loss))
#         # 如果当前模型比之前的模型性能更好，则保存当前模型
#         if mean_val_loss < best_val_loss:
#             best_val_loss = mean_val_loss
#             print('best_val_loss:' + str(best_val_loss) + ' saving model:' + model_name)
#             torch.save(model, model_name)


# if __name__ == '__main__':
#     train_three_model()


import numpy as np
import pandas as pd
import os
import random
import datetime
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from seven_model import SEVEN, BRANCHES, HORIZONS
from evaluate import evaluate_model
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

        input_data, labels = get_data_input(data)
        input_data = input_data.values
        input_data = torch.tensor(input_data).float()
        labels = torch.tensor(labels).float().reshape(len(BRANCHES), len(HORIZONS))
        return input_data, labels


def get_data_input(data):
    num_labels = len(BRANCHES) * len(HORIZONS)  # 9 个标签（最后 9 列）
    input_data = data.iloc[:, :-num_labels]
    labels = data.iloc[0, -num_labels:].values  # branch-major 顺序的 9 个标签
    return input_data, labels


def estimate_label_std(h5_files, num_labels, max_samples=20000):
    """抽样估计每个标签格子的标准差（branch-major，[num_labels]）。

    用于逆方差加权 loss：3x3 的 9 个标签量级差异巨大（netasset≈3~31、fcf 有大有小、dividend≈0~1），
    直接 MSE 会被大量级的格子主导。除以各自标准差后，各格子对梯度的贡献更均衡。
    """
    files = h5_files
    if len(files) > max_samples:
        step = len(files) / max_samples
        files = [files[int(i * step)] for i in range(max_samples)]
    rows = []
    for f in files:
        d = pd.read_hdf(f)
        rows.append(d.iloc[0, -num_labels:].values.astype('float32'))
    arr = np.asarray(rows)          # [n_samples, num_labels]
    std = arr.std(axis=0)           # [num_labels]
    std = np.maximum(std, 1e-3)     # 防止某格 std≈0 导致权重爆炸
    return std



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

def train_seven_model():
    # --- 设备设置 ---
    device = select_device()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_name = os.path.join(current_dir, 'seven.pt')
    model = SEVEN([127, 31]).to(device)
    batch_size = 128
    train_folder = os.path.join(current_dir, 'train')
    h5_files = [os.path.join(train_folder, f) for f in os.listdir(train_folder) if f.endswith('.h5')]

    # 按股票(symbol)隔离划分：先随机抽出 TEST_N 只股票作为测试集（不参与训练），
    # 其余股票再按 7:3 隔离成 train/val。
    # 文件名形如 {symbol}_{endDate}.h5，symbol 可能含下划线，故按最后一个下划线切。
    def _symbol_of(f):
        return os.path.basename(f)[:-3].rsplit('_', 1)[0]

    TEST_N = 127 * 7  # 测试集股票数（127×7，扩大测试样本以降低 IC 噪声）
    TEST_SEED = int(os.environ.get('TEST_SEED', 42))  # 可用环境变量 TEST_SEED 覆盖（每次训练换测试集）
                      # （seed 固定 => 旧 127 只是新 889 只的子集，历史 IC 仍可比）
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

    # 保存测试股票列表，供 evaluate.py 复用（--test-symbols test_symbols.txt）
    test_symbols_path = os.path.join(current_dir, 'test_symbols.txt')
    with open(test_symbols_path, 'w') as fp:
        fp.write('\n'.join(sorted(test_symbols)))
    print(f"已随机抽出测试股票 {len(test_symbols)} 只（样本 {len(test_files)}）并保存到 {test_symbols_path}；"
          f"训练股票 {len(train_symbols)} / 验证股票 {len(val_symbols)}；"
          f"训练样本 {len(train_files)} / 验证样本 {len(val_files)}")

    # 逆方差加权的尺度：只在【训练集】上估计每个标签格子的标准差，避免把验证集统计量泄漏进训练
    label_std = torch.tensor(
        estimate_label_std(train_files, len(BRANCHES) * len(HORIZONS)),
        device=device, dtype=torch.float32,
    ).reshape(1, len(BRANCHES), len(HORIZONS))
    print('标签标准差 (branch×horizon):', label_std.cpu().numpy().round(4).tolist())

    train_dataset = Dataset(train_files)
    val_dataset = Dataset(val_files)

    if os.path.exists(model_name):
        try:
            loaded_model = torch.load(model_name, map_location=device, weights_only=False)
        except Exception as e:
            print('failed to load checkpoint, training from scratch:', e)
            loaded_model = None
        # 新版模型才有 netasset_head；旧版(3 个标量头)不兼容，直接重训
        if loaded_model is not None and hasattr(loaded_model, 'netasset_head'):
            model = loaded_model.to(device)
            print('loaded existing SEVEN model:', model_name)
        else:
            print('old/incompatible SEVEN checkpoint detected, training from scratch:', model_name)

    criterion_mse = nn.MSELoss()
    learning_rate = 0.001
    num_epochs = 7
    # 推荐设置为 CPU 核心数减一，但至少为 1（persistent_workers 要求 num_workers > 0）
    num_workers = max(1, os.cpu_count() // 4 if os.cpu_count() else 4)
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
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        # 训练阶段
        mean_train_loss = 0.0
        step_num = 0
        for data_input, labels in train_dataloader:
            data_input = data_input.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            predict = model(data_input)  # [batch, 3(branch), 3(horizon)]
            # 逆方差加权 MSE：每个格子除以各自标准差，平衡 3x3 的量级差异
            loss = criterion_mse(labels / label_std, predict / label_std)
            loss.backward()  # 计算梯度
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
            for data_input, labels in val_dataloader:
                data_input = data_input.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                predict = model(data_input)  # [batch, 3, 3]
                loss = criterion_mse(labels / label_std, predict / label_std)
                mean_val_loss += loss.item()

        mean_val_loss = mean_val_loss / len(val_dataloader)
        print("Epoch: %d, validate loss: %1.5f" % (epoch, mean_val_loss))
        # 如果当前模型比之前的模型性能更好，则保存当前模型
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            print('best_val_loss:' + str(best_val_loss) + ' saving model:' + model_name)
            torch.save(model, model_name)

    # 训练完成后，备份一份带日期的模型快照，便于按时间回溯对比。
    # 统一时间戳：备份文件名与评估历史(history.csv)里的 timestamp 用同一个时刻，
    # 这样「当前最好」那一行的 timestamp 能精确对应到 backup/ 里的 .pt 文件。
    run_ts = datetime.datetime.now()
    backup_dir = os.path.join(current_dir, 'backup')
    os.makedirs(backup_dir, exist_ok=True)
    backup_path = os.path.join(
        backup_dir, f"seven_{run_ts.strftime('%Y%m%d_%H%M%S')}.pt")
    shutil.copy2(model_name, backup_path)
    print('已备份最新模型到:', backup_path)

    # 训练完成后，直接在留出的测试股票上评估
    print("\n训练完成，开始在留出的测试股票上评估...")
    evaluate_model(
        model_path=model_name,
        train_folder=train_folder,
        test_symbols=test_symbols,
        name='after_train',
        output_dir=os.path.join(current_dir, 'eval_results'),
        device=device,
        timestamp=run_ts.strftime('%Y-%m-%d %H:%M:%S'),
    )


if __name__ == '__main__':
    train_seven_model()
