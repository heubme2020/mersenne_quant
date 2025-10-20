import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import math
from one_model import ONE
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


class Dataset(Dataset):
    def __init__(self, h5_file_list):
        self.h5_file_list = h5_file_list

    def __len__(self):
        return len(self.h5_file_list)  # 返回文件列表的长度

    def __getitem__(self, idx):
        data = pd.read_hdf(self.h5_file_list[idx])

        data_input, data_fore, three_gain, seven_gain, thirty_one_gain = get_data_input(data)
        data_input = data_input.values
        data_input = torch.tensor(data_input).float()
        data_fore = data_fore.values
        data_fore = torch.tensor(data_fore).float()
        # gain = gain.values
        three_gain = torch.tensor(three_gain).unsqueeze(0).unsqueeze(0).float()
        seven_gain = torch.tensor(seven_gain).unsqueeze(0).unsqueeze(0).float()
        thirty_one_gain = torch.tensor(thirty_one_gain).unsqueeze(0).unsqueeze(0).float()
        return data_input, data_fore, three_gain, seven_gain, thirty_one_gain

def get_data_input(data):
    input_idx = 127*3
    data_input = data.iloc[:input_idx].reset_index(drop=True)
    selected_columns = ['close', 'volume', 'delta']
    close_tomorrow = data['close'].iloc[-31]

    three_data_fore = data.iloc[input_idx:input_idx+3].reset_index(drop=True)
    three_data_fore = three_data_fore[selected_columns]
    three_max_fore = three_data_fore['close'].max()
    three_median_fore = three_data_fore['close'].median()
    three_min_fore = three_data_fore['close'].min()
    three_gain = (math.log(three_max_fore) + math.log(three_median_fore) + math.log(three_min_fore) - 3*math.log(close_tomorrow))

    seven_data_fore = data.iloc[input_idx:input_idx+7].reset_index(drop=True)
    seven_data_fore = seven_data_fore[selected_columns]
    seven_max_fore = seven_data_fore['close'].max()
    seven_median_fore = seven_data_fore['close'].median()
    seven_min_fore = seven_data_fore['close'].min()
    seven_gain = (math.log(seven_max_fore) + math.log(seven_median_fore) + math.log(seven_min_fore) - 3*math.log(close_tomorrow))

    thirty_one_data_fore = data.iloc[input_idx:input_idx+31].reset_index(drop=True)
    thirty_one_data_fore = thirty_one_data_fore[selected_columns]
    thirty_one_max_fore = thirty_one_data_fore['close'].max()
    thirty_one_median_fore = thirty_one_data_fore['close'].median()
    thirty_one_min_fore = thirty_one_data_fore['close'].min()
    thirty_one_gain = (math.log(thirty_one_max_fore) + math.log(thirty_one_median_fore) + math.log(thirty_one_min_fore) - 3*math.log(close_tomorrow))  

    #缩放
    data_fore = thirty_one_data_fore*7.0
    three_gain = np.array(three_gain)*31.0 
    seven_gain = np.array(seven_gain)*31.0 
    thirty_one_gain = np.array(thirty_one_gain)*31.0 
    return data_input, data_fore, three_gain, seven_gain, thirty_one_gain


def train_one_model():
    # --- 设备设置 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_name = os.path.join(current_dir, 'one.pt')
    model = ONE([31, 127*3], [3, 31]).to('cuda')
    batch_size = 64
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
        mean_gain_loss = 0.0
        step_num = 0
        for data_input, data_fore, three_gain, seven_gain, thirty_one_gain in train_dataloader:
            data_input = data_input.to(device, non_blocking=True)
            data_fore = data_fore.to(device, non_blocking=True)
            three_gain = three_gain.to(device, non_blocking=True)
            seven_gain = seven_gain.to(device, non_blocking=True)
            thirty_one_gain = thirty_one_gain.to(device, non_blocking=True)
            optimizer.zero_grad()
            data_fore_predict, three_gain_predict, seven_gain_predict, thirty_one_gain_predict= model(data_input)
            gain_loss = criterion_mse(three_gain, three_gain_predict) + criterion_mse(seven_gain, seven_gain_predict) + criterion_mse(thirty_one_gain, thirty_one_gain_predict)
            loss = criterion_mse(data_fore, data_fore_predict)  + gain_loss
            # loss = criterion(predict0, fore0) + criterion(predict1, fore1) + 3*criterion(compare, gain)
            # print(criterion(compare, gain).item())
            loss.backward()  # 计算梯度
            # optimizer.step()
            optimizer.step()
            mean_gain_loss = (mean_gain_loss*step_num + gain_loss.item())/float(step_num + 1)
            # running_train_loss += loss.item() * inputs.size(0)
            mean_train_loss = (mean_train_loss*step_num + loss.item())/float(step_num + 1)
            step_num = step_num + 1
            try:
                print("Epoch: %d, train loss: %1.5f, mean loss: %1.5f, mean gain loss:%1.5f, min val loss: %1.5f" %
                      (epoch, loss.item(), mean_train_loss, mean_gain_loss, best_val_loss))
            except:
                pass

        # 验证阶段
        mean_val_loss = 0
        with ((torch.no_grad())):
            for data_input, data_fore, three_gain, seven_gain, thirty_one_gain in val_dataloader:
                data_input = data_input.to(device, non_blocking=True)
                data_fore = data_fore.to(device, non_blocking=True)
                three_gain = three_gain.to(device, non_blocking=True)
                seven_gain = seven_gain.to(device, non_blocking=True)
                thirty_one_gain = thirty_one_gain.to(device, non_blocking=True)
                data_fore_predict, three_gain_predict, seven_gain_predict, thirty_one_gain_predict= model(data_input)
                gain_loss = criterion_mse(three_gain, three_gain_predict) + criterion_mse(seven_gain, seven_gain_predict) + criterion_mse(thirty_one_gain, thirty_one_gain_predict)
                mean_val_loss += gain_loss.item()

        mean_val_loss = mean_val_loss / len(val_dataloader)
        print("Epoch: %d, validate loss: %1.5f" % (epoch, mean_val_loss))
        # 如果当前模型比之前的模型性能更好，则保存当前模型
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            print('best_val_loss:' + str(best_val_loss) + ' saving model:' + model_name)
            torch.save(model, model_name)



if __name__ == '__main__':
    train_one_model()
