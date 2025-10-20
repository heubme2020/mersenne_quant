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


def train_zero_model():
    # --- 设备设置 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_name = os.path.join(current_dir, 'zero.pt')
    model = ZERO([7, 3], [1, 1]).to(device)
    batch_size = 512
    train_folder = os.path.join(current_dir, 'train')
    h5_files = [os.path.join(train_folder, f) for f in os.listdir(train_folder) if f.endswith('.h5')]
    random.shuffle(h5_files)
    # h5_files = h5_files[:131071]
    train_length = int(len(h5_files)*0.7)
    train_files = h5_files[:train_length]
    val_files = h5_files[train_length:]
    train_dataset = Dataset(train_files)
    val_dataset = Dataset(val_files)

    if os.path.exists(model_name):
        model = torch.load(model_name)

    criterion_mse = nn.MSELoss()
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.L1Loss()
    learning_rate = 0.001
    num_epochs = 31
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
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        # 训练阶段
        mean_train_loss = 0.0
        mean_up_down_loss = 0.0
        step_num = 0
        for data_input, three in train_dataloader:
            data_input = data_input.to(device, non_blocking=True)
            three = three.to(device, non_blocking=True)
            up_down = torch.sign(three)
            optimizer.zero_grad()
            up_down_predict, three_predict = model(data_input)
            three_loss = criterion_mse(three, three_predict)
            up_down_loss = criterion(up_down, up_down_predict)
            loss = three_loss + up_down_loss
            # loss = criterion(predict0, fore0) + criterion(predict1, fore1) + 3*criterion(compare, gain)
            # print(criterion(compare, gain).item())
            loss.backward()  # 计算梯度
            # optimizer.step()
            optimizer.step()
            mean_up_down_loss = (mean_up_down_loss*step_num + up_down_loss.item())/float(step_num + 1)
            mean_train_loss = (mean_train_loss*step_num + loss.item())/float(step_num + 1)
            step_num = step_num + 1
            try:
                print("Epoch: %d, train loss: %1.5f, mean loss: %1.5f, mean_up_down_loss:%1.5f, min val loss: %1.5f" %
                      (epoch, loss.item(), mean_train_loss, mean_up_down_loss, best_val_loss))
            except:
                pass

        # 验证阶段
        mean_val_loss = 0
        with ((torch.no_grad())):
            for data_input, three in val_dataloader:
                data_input = data_input.to(device, non_blocking=True)
                three = three.to(device, non_blocking=True)
                up_down = torch.sign(three)
                up_down_predict, _ = model(data_input)
                loss = criterion(up_down, up_down_predict)
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


if __name__ == '__main__':
    train_zero_model()
