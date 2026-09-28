"""
用 seven 模型对 127 个输入特征做 Integrated Gradients 重要性排序。
（适配自 three/get_features_importance.py，SEVEN 输出 [batch, 3 branch, 3 horizon]）

输入布局（与 gen_train_data.py 保持一致）：
    h5 形状 (31, 136) = 127 个特征 + 9 个 label
    label 顺序（branch-major）: fcf/dividend/netasset × three/seven/thirty_one
    特征顺序: [0:42]  来自 indicator_{exchange}.csv（未做 z-score，原始比值）
              [42:127] 来自 features_importance.csv（已按 endDate 做 z-score）

关于 baseline：
    zero-baseline 的归因量级正比于特征数值大小，前 42 个 indicator 特征没标准化，
    跟后 85 个 z-score 过的特征直接比不公平。所以同时算 imp_zero 和 imp_mean，
    跨组比较看 imp_mean。

三个分支（fcf/dividend/netasset）分别归因（取 HORIZON_IDX=1 即 7 季度），再平均。
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from tqdm import tqdm

from seven_model import SEVEN, BRANCHES, HORIZONS  # noqa: F401

N_FEATURES = 127
N_LABELS = len(BRANCHES) * len(HORIZONS)  # 9
SEQ_LEN = 31
N_INDICATOR = 42  # 前 42 列来自 indicator 表
HORIZON_IDX = 1   # 7 季度（seven 的 namesake 期限）


def get_data_input(data):
    """切掉末尾 9 个 label 列，留下 127 个特征。"""
    return data.iloc[:, :-N_LABELS]


def load_batch(h5_files, device):
    rows = []
    for h5_file in h5_files:
        data = pd.read_hdf(h5_file)
        rows.append(torch.tensor(get_data_input(data).values, dtype=torch.float32))
    return torch.stack(rows, dim=0).to(device)


def attribute(ig, X, baseline, n_steps, internal_batch_size):
    """返回每个特征的平均绝对归因，形状 [n_features]。"""
    attributions = ig.attribute(
        inputs=X,
        baselines=baseline,
        n_steps=n_steps,
        internal_batch_size=internal_batch_size,
    )
    return attributions.abs().mean(dim=(0, 1)).detach().cpu().numpy()


def get_features_importance(model_name, n_iter, batch_size, n_steps,
                            internal_batch_size, train_folder, out_csv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = torch.load(model_name, map_location=device, weights_only=False)
    model.eval()
    in_features = model.input_shape[0]
    if in_features != N_FEATURES:
        raise ValueError(
            f"模型输入维度 {in_features} != 预期 {N_FEATURES}，"
            f"检查 seven.pt 与当前 train/ 数据是否匹配"
        )

    h5_files_list = [
        os.path.join(train_folder, f)
        for f in os.listdir(train_folder)
        if f.endswith('.h5')
    ]
    if len(h5_files_list) < batch_size:
        raise ValueError(f"train/ 下只有 {len(h5_files_list)} 个 h5，不足一个 batch")
    print(f"train/ 下 {len(h5_files_list)} 个样本，"
          f"抽 {n_iter} 轮 x {batch_size} 个，IG n_steps={n_steps}")

    feature_names = pd.read_hdf(h5_files_list[0]).columns.tolist()[:N_FEATURES]

    # SEVEN.forward 返回 [batch, 3(branch), 3(horizon)]，三个分支分别归因
    def branch_fn(branch_idx):
        def fn(x):
            return model(x)[:, branch_idx, HORIZON_IDX]
        return fn

    igs = {b: IntegratedGradients(branch_fn(i)) for i, b in enumerate(BRANCHES)}

    acc = {f'{b}_{base}': [] for b in BRANCHES for base in ('zero', 'mean')}

    for _ in tqdm(range(n_iter), desc="IG"):
        h5_files = random.sample(h5_files_list, batch_size)
        X = load_batch(h5_files, device).requires_grad_(True)

        zero_baseline = torch.zeros_like(X)
        mean_baseline = X.detach().mean(dim=(0, 1), keepdim=True).expand_as(X)

        for b in BRANCHES:
            acc[f'{b}_zero'].append(attribute(igs[b], X, zero_baseline, n_steps, internal_batch_size))
            acc[f'{b}_mean'].append(attribute(igs[b], X, mean_baseline, n_steps, internal_batch_size))

    stacked = {k: np.stack(v, axis=0) for k, v in acc.items()}

    imp_zero = np.mean([stacked[f'{b}_zero'].mean(axis=0) for b in BRANCHES], axis=0)
    imp_mean = np.mean([stacked[f'{b}_mean'].mean(axis=0) for b in BRANCHES], axis=0)

    ranking = pd.DataFrame({
        'feature': feature_names,
        'group': ['indicator'] * N_INDICATOR + ['raw'] * (N_FEATURES - N_INDICATOR),
        'imp_zero': imp_zero,
        'imp_mean': imp_mean,
    })
    for b in BRANCHES:
        ranking[f'imp_mean_{b}'] = stacked[f'{b}_mean'].mean(axis=0)

    ranking['rank_mean'] = ranking['imp_mean'].rank(ascending=False).astype(int)
    ranking['rank_in_group'] = (
        ranking.groupby('group')['imp_mean'].rank(ascending=False).astype(int)
    )

    ranking = ranking.sort_values('imp_mean', ascending=False).reset_index(drop=True)
    ranking.to_csv(out_csv, index=False)
    print(f"\n完整排名已写入 {out_csv}")

    ind = ranking[ranking['group'] == 'indicator'].sort_values('imp_mean')
    print(f"\n=== indicator 42 列中最弱的 10 个（按 imp_mean）===")
    print(ind.head(10)[['feature', 'imp_mean', 'rank_mean', 'rank_in_group']].to_string(index=False))

    new = ranking[ranking['feature'].isin(['ocfToLiab', 'goodwillToEquity', 'accrual'])]
    print(f"\n=== 新加的 3 个 ratio 在 seven 里的排名 ===")
    print(new[['feature', 'imp_mean', 'rank_mean', 'rank_in_group']].to_string(index=False))

    return ranking


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='seven.pt')
    p.add_argument('--train-folder', default='train')
    p.add_argument('--out', default='features_ranking_127.csv')
    p.add_argument('--n-iter', type=int, default=127, help="采样轮数")
    p.add_argument('--batch-size', type=int, default=127, help="每轮样本数")
    p.add_argument('--n-steps', type=int, default=32, help="IG 积分步数")
    p.add_argument('--internal-batch-size', type=int, default=1024)
    args = p.parse_args()

    get_features_importance(
        model_name=args.model,
        n_iter=args.n_iter,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        internal_batch_size=args.internal_batch_size,
        train_folder=args.train_folder,
        out_csv=args.out,
    )
