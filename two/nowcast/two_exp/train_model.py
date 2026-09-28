"""
train_model.py — two 任务的参数化训练（复用 two/train.py 的 Dataset/loss/训练循环）。

用法：
  python train_model.py --data <h5目录> --out <模型路径> [--epochs 7] [--batch 8]
"""

import os
import sys
import math
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'two')))
from two_model import TWO  # noqa: F401

DAYS_INPUT = 127 * 7
SELECTED_COLUMNS = ['close', 'volume', 'delta']


class DS(Dataset):
    def __init__(self, h5_files):
        self.h5_files = h5_files

    def __len__(self):
        return len(self.h5_files)

    def __getitem__(self, idx):
        data = pd.read_hdf(self.h5_files[idx])
        data_input, data_fore, seven_gain, thirty_one_gain, one_twenty_seven_gain = get_data_input(data)
        data_input = torch.tensor(data_input.values).float()
        data_fore = torch.tensor(data_fore.values).float()
        seven_gain = torch.tensor(seven_gain).unsqueeze(0).unsqueeze(0).float()
        thirty_one_gain = torch.tensor(thirty_one_gain).unsqueeze(0).unsqueeze(0).float()
        one_twenty_seven_gain = torch.tensor(one_twenty_seven_gain).unsqueeze(0).unsqueeze(0).float()
        return data_input, data_fore, seven_gain, thirty_one_gain, one_twenty_seven_gain


def get_data_input(data):
    """与 two/train.py 完全一致：从 1016 天窗口提取 889 输入 + 3 个 gain label + data_fore。"""
    input_idx = DAYS_INPUT
    data_input = data.iloc[:input_idx].reset_index(drop=True)
    close_tomorrow = data['close'].iloc[-127]

    seven_data_fore = data.iloc[input_idx:input_idx + 7].reset_index(drop=True)[SELECTED_COLUMNS]
    seven_gain = (math.log(seven_data_fore['close'].max()) + math.log(seven_data_fore['close'].median())
                  + math.log(seven_data_fore['close'].min()) - 3 * math.log(close_tomorrow))

    thirty_two_data_fore = data.iloc[input_idx:input_idx + 31].reset_index(drop=True)[SELECTED_COLUMNS]
    thirty_two_gain = (math.log(thirty_two_data_fore['close'].max()) + math.log(thirty_two_data_fore['close'].median())
                       + math.log(thirty_two_data_fore['close'].min()) - 3 * math.log(close_tomorrow))

    one_data_fore = data.iloc[input_idx:input_idx + 127].reset_index(drop=True)[SELECTED_COLUMNS]
    one_gain = (math.log(one_data_fore['close'].max()) + math.log(one_data_fore['close'].median())
                + math.log(one_data_fore['close'].min()) - 3 * math.log(close_tomorrow))

    data_fore = one_data_fore * 7.0
    return data_input, data_fore, np.array(seven_gain) * 31.0, np.array(thirty_two_gain) * 31.0, np.array(one_gain) * 31.0


def ic_loss(y_true, y_pred):
    pred_c = y_pred - y_pred.mean()
    true_c = y_true - y_true.mean()
    return F.cosine_similarity(pred_c.unsqueeze(0), true_c.unsqueeze(0), dim=1, eps=1e-7)


def asymmetric_loss(y_true, y_pred, penalty_ratio=1.0):
    error = y_true - y_pred
    reg_loss = torch.where(error < 0, penalty_ratio * torch.pow(error, 2), torch.pow(error, 2))
    ic_batch = ic_loss(y_true, y_pred)
    loss = reg_loss + (1 - ic_batch) * 31.0
    return loss.mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--epochs', type=int, default=7)
    p.add_argument('--batch', type=int, default=8)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TWO([31, 127 * 7], [3, 127]).to(device)

    dirs = args.data.split(',')
    h5_files = []
    for d in dirs:
        h5_files += [os.path.join(d, f) for f in os.listdir(d) if f.endswith('.h5')]
    random.shuffle(h5_files)
    print(f'数据目录 {dirs}，数据文件数: {len(h5_files)}')
    train_len = int(len(h5_files) * 0.7)
    train_files, val_files = h5_files[:train_len], h5_files[train_len:]

    train_ds = DS(train_files)
    val_ds = DS(val_files)
    num_workers = max(1, os.cpu_count() // 4 if os.cpu_count() else 4)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=num_workers,
                          persistent_workers=True, pin_memory=(device.type == 'cuda'))
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=num_workers,
                        persistent_workers=True, pin_memory=(device.type == 'cuda'))

    criterion_mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_val = float('inf')
    for epoch in range(args.epochs):
        model.train()
        step = 0
        mean_loss = 0.0
        for di, df, s7, s31, s127 in train_dl:
            di, df, s7, s31, s127 = (t.to(device, non_blocking=True) for t in (di, df, s7, s31, s127))
            optimizer.zero_grad()
            df_p, s7_p, s31_p, s127_p = model(di)
            gain_loss = (asymmetric_loss(s7, s7_p) + asymmetric_loss(s31, s31_p)
                         + asymmetric_loss(s127, s127_p))
            loss = criterion_mse(df, df_p) + gain_loss
            loss.backward()
            optimizer.step()
            mean_loss = (mean_loss * step + loss.item()) / (step + 1)
            step += 1
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for di, df, s7, s31, s127 in val_dl:
                di, df, s7, s31, s127 = (t.to(device, non_blocking=True) for t in (di, df, s7, s31, s127))
                df_p, s7_p, s31_p, s127_p = model(di)
                val_loss += (criterion_mse(s7, s7_p) + criterion_mse(s31, s31_p)
                             + criterion_mse(s127, s127_p)).item()
        val_loss /= len(val_dl)
        print(f'Epoch {epoch}: train_loss={mean_loss:.5f} val_gain_loss={val_loss:.5f}')
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model, args.out)
    print(f'训练完成，模型保存到 {args.out}')


if __name__ == '__main__':
    main()
