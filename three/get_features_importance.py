"""
用 three 模型对 127 个输入特征做 Integrated Gradients 重要性排序。

输入布局（与 gen_train_data.py 保持一致）：
    h5 形状 (31, 133) = 127 个特征 + 6 个 label
    label 顺序: growth_one/three/seven, death_one/three/seven
    特征顺序: [0:42]  来自 indicator_{exchange}.csv（未做 z-score，原始比值）
              [42:127] 来自 features_importance.csv（已按 endDate 做 z-score）

关于 baseline：
    IG 的归因 = (x - baseline) x ∫梯度，所以 zero-baseline 下归因量级正比于特征本身的
    数值大小。前 42 个 indicator 特征没有标准化，跟后 85 个 z-score 过的特征直接比
    zero-baseline 归因是不公平的。因此这里同时算：
      - imp_zero: baseline=0，等价于原来的口径，保留用于纵向对比
      - imp_mean: baseline=各特征在样本上的均值，归因 ≈ |梯度| x 特征自身波动幅度，
                  跨 indicator / raw 两组可比。做组间比较时看这一列。

注意：本脚本只产出排名，不再回写 features_importance.csv。那个文件被
gen_train_data.py / get_three_predict.py 当作「只含 raw 财务列」的清单使用
（会去 income+balance+cashflow 里取数），把 indicator 列写进去会直接 KeyError。
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from tqdm import tqdm

from three_model import THREE

N_FEATURES = 127
N_LABELS = 6
SEQ_LEN = 31
N_INDICATOR = 42  # 前 42 列来自 indicator 表
HORIZON_IDX = 1   # growth/death 的第 2 个输出 = three（3 个季度）


def get_data_input(data):
    """切掉末尾 6 个 label 列，留下 127 个特征。"""
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
    # dim 0 = batch, dim 1 = 时间步(31)
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
            f"检查 three.pt 与当前 train/ 数据是否匹配"
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

    # THREE.forward 返回 (growth, death)，各自形状 [B, 1, 3]。
    # 两个头分开归因，避免正负号相互抵消（growth 越大越好、death 越大越差）。
    def growth_fn(x):
        growth, _ = model(x)
        return growth[:, 0, HORIZON_IDX]

    def death_fn(x):
        _, death = model(x)
        return death[:, 0, HORIZON_IDX]

    ig_growth = IntegratedGradients(growth_fn)
    ig_death = IntegratedGradients(death_fn)

    # 每轮一条记录，最后取均值 + 标准差（标准差用来看排名稳不稳）
    acc = {k: [] for k in ('growth_zero', 'death_zero', 'growth_mean', 'death_mean')}

    for _ in tqdm(range(n_iter), desc="IG"):
        h5_files = random.sample(h5_files_list, batch_size)
        X = load_batch(h5_files, device).requires_grad_(True)

        zero_baseline = torch.zeros_like(X)
        # 每个特征在本 batch 上的均值，形状 [1, 1, 127] 再广播
        mean_baseline = X.detach().mean(dim=(0, 1), keepdim=True).expand_as(X)

        acc['growth_zero'].append(attribute(ig_growth, X, zero_baseline, n_steps, internal_batch_size))
        acc['death_zero'].append(attribute(ig_death, X, zero_baseline, n_steps, internal_batch_size))
        acc['growth_mean'].append(attribute(ig_growth, X, mean_baseline, n_steps, internal_batch_size))
        acc['death_mean'].append(attribute(ig_death, X, mean_baseline, n_steps, internal_batch_size))

    stacked = {k: np.stack(v, axis=0) for k, v in acc.items()}

    ranking = pd.DataFrame({
        'feature': feature_names,
        'group': ['indicator'] * N_INDICATOR + ['raw'] * (N_FEATURES - N_INDICATOR),
        'imp_zero': (stacked['growth_zero'].mean(axis=0) + stacked['death_zero'].mean(axis=0)) / 2,
        'imp_mean': (stacked['growth_mean'].mean(axis=0) + stacked['death_mean'].mean(axis=0)) / 2,
        'imp_mean_growth': stacked['growth_mean'].mean(axis=0),
        'imp_mean_death': stacked['death_mean'].mean(axis=0),
        # 轮与轮之间的波动，相对值越大说明这个排名越不可信
        'imp_mean_cv': (
            (stacked['growth_mean'].std(axis=0) + stacked['death_mean'].std(axis=0))
            / (stacked['growth_mean'].mean(axis=0) + stacked['death_mean'].mean(axis=0) + 1e-12)
        ),
    })

    ranking['rank_zero'] = ranking['imp_zero'].rank(ascending=False).astype(int)
    ranking['rank_mean'] = ranking['imp_mean'].rank(ascending=False).astype(int)
    # 组内排名：回答「indicator 这 42 个里谁最弱」
    ranking['rank_in_group'] = (
        ranking.groupby('group')['imp_mean'].rank(ascending=False).astype(int)
    )

    ranking = ranking.sort_values('imp_mean', ascending=False).reset_index(drop=True)
    ranking.to_csv(out_csv, index=False)
    print(f"\n完整排名已写入 {out_csv}")

    ind = ranking[ranking['group'] == 'indicator'].sort_values('imp_mean')
    print(f"\n=== indicator 42 列中最弱的 8 个（按 imp_mean）===")
    print(ind.head(8)[['feature', 'imp_mean', 'rank_mean', 'rank_in_group', 'imp_mean_cv']].to_string(index=False))

    print(f"\n=== 两种 baseline 的排名分歧最大的 10 个 ===")
    ranking['rank_gap'] = ranking['rank_zero'] - ranking['rank_mean']
    gap = ranking.reindex(ranking['rank_gap'].abs().sort_values(ascending=False).index)
    print(gap.head(10)[['feature', 'group', 'rank_zero', 'rank_mean', 'rank_gap']].to_string(index=False))

    return ranking


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='three.pt')
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
