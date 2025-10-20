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
import torch
import torch.nn as nn
import torch.optim as optim
from three_model import THREE
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

        input_data, growth_death_one, growth_death_three, growth_death_seven= get_data_input(data)
        input_data = input_data.values
        input_data = torch.tensor(input_data).float()
        growth_death_one = torch.tensor(growth_death_one).unsqueeze(0).unsqueeze(0).float()
        growth_death_three = torch.tensor(growth_death_three).unsqueeze(0).unsqueeze(0).float()
        growth_death_seven = torch.tensor(growth_death_seven).unsqueeze(0).unsqueeze(0).float()
        return input_data, growth_death_one, growth_death_three, growth_death_seven


def get_data_input(data):
    input_data = data.iloc[:, :-3]
    growth_death_one = data.iloc[0, -3]
    growth_death_three = data.iloc[0, -2]
    growth_death_seven = data.iloc[0, -1]
    return input_data, growth_death_one, growth_death_three, growth_death_seven

def train_three_model():
    # --- 设备设置 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_name = os.path.join(current_dir, 'three.pt')
    model = THREE([127, 31], [1, 1]).to(device)
    batch_size = 128
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
        step_num = 0
        for data_input, one, three, seven in train_dataloader:
            # print(data_input.shape)
            data_input = data_input.to(device, non_blocking=True)
            one = one.to(device, non_blocking=True)
            three = three.to(device, non_blocking=True)
            seven = seven.to(device, non_blocking=True)
            optimizer.zero_grad()
            one_predict, three_predict, seven_predict,= model(data_input)
            one_loss = criterion_mse(one, one_predict)
            three_loss = criterion_mse(three, three_predict)
            seven_loss = criterion_mse(seven, seven_predict)
            loss = one_loss + three_loss + seven_loss
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
            for data_input, one, three, seven in val_dataloader:
                data_input = data_input.to(device, non_blocking=True)
                one = one.to(device, non_blocking=True)
                three = three.to(device, non_blocking=True)
                seven = seven.to(device, non_blocking=True)
                one_predict, three_predict, seven_predict = model(data_input)
                one_loss = criterion_mse(one, one_predict)
                three_loss = criterion_mse(three, three_predict)
                seven_loss = criterion_mse(seven, seven_predict)
                loss = one_loss + three_loss + seven_loss
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
    train_three_model()
