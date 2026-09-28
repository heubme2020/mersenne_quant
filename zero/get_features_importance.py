import torch
import sys
import torch.nn as nn
from captum.attr import IntegratedGradients
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ratio_model import RATIO
import os
import random
from tqdm import tqdm

# Windows 控制台默认 GBK，而本文件会打印 `⚠️` / `🎉` 这类非 GBK 字符 —— 走到那条分支就会
# UnicodeEncodeError 崩掉，而且**往往是在活儿干完之后**才崩（2026-09-26 zero/gen 就这么"失败"过：
# 日志里 already 写着总量，然后崩在一句庆祝打印上）。与其它脚本同一套修法。2026-09-28 扫描后补齐。
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass


def get_data_input(data):
    input_data = data.iloc[:, :-1]
    three = data.iloc[0, -1]
    return input_data, three

def get_features_importance(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📦 Using device: {device}")

    # ====== 模型加载 ======
    model = RATIO([14, 3], [1, 1]).to(device)
    # model.load(torch.load(model_name, map_location=device))
    model = torch.load(model_name, weights_only=False)
    model.eval()

    # ====== 数据准备 ======
    train_folder = 'train'
    h5_files_list = [os.path.join(train_folder, f) for f in os.listdir(train_folder) if f.endswith('.h5')]
    feature_list = []
    for i in tqdm(range(127)):
        random.shuffle(h5_files_list)
        h5_files = h5_files_list[:127]

        # 取第一个文件列名
        data = pd.read_hdf(h5_files[0])
        feature_names = data.columns.tolist()[:14]

        # 组装 batch
        X_test = []
        for h5_file in h5_files:
            data = pd.read_hdf(h5_file)
            input_data, _ = get_data_input(data)
            input_tensor = torch.tensor(input_data.values, dtype=torch.float32, device=device)
            X_test.append(input_tensor)

        X = torch.stack(X_test, dim=0).detach().clone().requires_grad_(True)  # [127, 7, 10]
        baseline = torch.zeros_like(X)

        # ====== Integrated Gradients ======
        def model_for_captum(x):
            three = model(x)
            return three  # 只返回第一个输出
        ig = IntegratedGradients(model_for_captum)

        attributions, _ = ig.attribute(inputs=X, baselines=baseline, return_convergence_delta=True)

        # ====== 汇总特征重要性 ======
        feature_importance = attributions.abs().mean(dim=(0, 1)).detach().cpu().numpy()
        # 转成 DataFrame
        features_ranking = pd.DataFrame({
            "feature": feature_names,
            "importance": feature_importance
        })
        # 按重要性降序排序
        features_ranking = features_ranking.sort_values(by="importance", ascending=False).reset_index(drop=True)
        features_ranking = features_ranking.iloc[:7]
        # print(features_ranking)
        features = features_ranking['feature'].to_list()
        feature_list = feature_list + features
    feature_series = pd.Series(feature_list)
    count_series = feature_series.value_counts()
    print(count_series)
    # print("统计结果 (Series):\n", count_series)
    # features_ranking.to_csv("features_importance.csv", index=False)

if __name__ == '__main__':
    model_name = 'ratio.pt'
    get_features_importance(model_name)
