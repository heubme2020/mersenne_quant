"""汇总 train.py 各臂的测试集结果，打印对比表。

用法：
    python two/nowcast/compare_arms.py --tag _s1
    python two/nowcast/compare_arms.py            # 多 seed 时自动按臂聚合
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# 头名从 model 取（它又从 labels 取），不硬编码 —— 换字段时这里会自动跟上
from model import FLAT_NAMES as HEADS   # noqa: E402
ARMS = ['ashare', 'global', 'both']


def load(results_dir, tag):
    """-> {arm: [每条结果的 dict]}"""
    out = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(results_dir, f'*{tag}*.json'))):
        r = json.load(open(p))
        if 'icir' in r:
            out[r['arm']].append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=os.path.join(HERE, 'results'))
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    res = load(args.results, args.tag)

    print(f'{"头":10s}' + ''.join(f'{a:>18s}' for a in ARMS))
    print(f'{"":10s}' + ''.join(f'{"IC":>9s}{"ICIR":>9s}' for _ in ARMS))
    print('-' * (10 + 18 * len(ARMS)))
    for h in HEADS:
        row = f'{h:10s}'
        for a in ARMS:
            rs = res.get(a, [])
            if not rs:
                row += f'{"-":>9s}{"-":>9s}'
                continue
            ic = np.mean([r['ic'][h] for r in rs])
            ig = np.mean([r['icir'][h] for r in rs])
            row += f'{ic:>+9.3f}{ig:>+9.2f}'
        print(row)
    print('-' * (10 + 18 * len(ARMS)))
    row = f'{"平均":10s}'
    for a in ARMS:
        rs = res.get(a, [])
        if not rs:
            row += f'{"-":>9s}{"-":>9s}'
            continue
        row += (f'{np.mean([np.mean(list(r["ic"].values())) for r in rs]):>+9.3f}'
                f'{np.mean([np.mean(list(r["icir"].values())) for r in rs]):>+9.2f}')
    print(row)
    for a in ARMS:
        rs = res.get(a, [])
        if rs:
            print(f'{a:8s} seed={sorted(r["seed"] for r in rs)} '
                  f'训练样本={rs[0]["train"]:,} 测试样本={rs[0]["test"]:,} '
                  f'参数={rs[0]["n_params"]:,}  (n={len(rs)})')


if __name__ == '__main__':
    main()
