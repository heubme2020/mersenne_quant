import torch
import torch.nn as nn
from captum.attr import IntegratedGradients
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from three_model import THREE
import os
import random
from tqdm import tqdm
from collections import Counter

def get_data_input(data):
    input_data = data.iloc[:, :-3]
    return input_data

def get_features_importance(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📦 Using device: {device}")

    # ====== 模型加载 ======
    model = THREE([118, 31], [1, 1]).to(device)
    # model.load(torch.load(model_name, map_location=device))
    model = torch.load(model_name)
    model.eval()

    # ====== 数据准备 ======
    train_folder = 'train'
    h5_files_list = [os.path.join(train_folder, f) for f in os.listdir(train_folder) if f.endswith('.h5')]
    feature_list = []
    for i in tqdm(range(127)):
        random.shuffle(h5_files_list)
        h5_files = h5_files_list[:31]

        # 取第一个文件列名
        data = pd.read_hdf(h5_files[0])
        feature_names = data.columns.tolist()[:118]

        # 组装 batch
        X_test = []
        for h5_file in h5_files:
            data = pd.read_hdf(h5_file)
            input_data = get_data_input(data)
            input_tensor = torch.tensor(input_data.values, dtype=torch.float32, device=device)
            X_test.append(input_tensor)

        X = torch.stack(X_test, dim=0).detach().clone().requires_grad_(True)  # [127, 7, 10]
        baseline = torch.zeros_like(X)

        # ====== Integrated Gradients ======
        def model_for_captum(x):
            _, one, _, three, _, seven = model(x)
            growth_death = one + three + seven
            return growth_death  # 只返回第一个输出
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
        features_ranking = features_ranking.iloc[:86]
        # print(features_ranking)
        features = features_ranking['feature'].to_list()
        feature_list = feature_list + features
    counter = Counter(feature_list)
    features_count = pd.DataFrame(list(counter.items()), columns=['feature', 'count'])
    features_count = features_count.sort_values('count', ascending=False)
    features_count = features_count.reset_index(drop=True)
    features_count = features_count.iloc[:86]
    print(features_count)
    features_count_pre = pd.read_csv('features_importance.csv')
    features_list_pre = features_count_pre['feature'].to_list()
    features_list = features_count['feature'].to_list()
    add_features = list(set(features_list) - set(features_list_pre))
    print('add_features', add_features)
    delete_features = list(set(features_list_pre) - set(features_list))
    print('delete_features', delete_features)
    features_count.to_csv('features_importance.csv', index=False)

if __name__ == '__main__':
    model_name = 'three.pt'
    get_features_importance(model_name)