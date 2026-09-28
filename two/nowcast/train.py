"""nowcast 训练 + 评估。

三个臂（对照实验，而不是只跑一个）：
    ashare : 只用 A 股训练      —— 基线，回答「这个模型能否预测 A 股前瞻财务」
    global : 只用海外 10 个市场 —— 检验「全球财务标签能否迁移到 A 股」
    both   : 全球 + A 股

测试集固定 = one 的 889 只 A 股（one/test_symbols.txt），从**所有**臂的训练/验证里排除。
验证集 = 其余 A 股里 30%（seed 12345，与 one 一致），只用于 early stop。

损失 = 每个头 (1 − batch 内相关系数) 求和 + aux 序列 MSE。
batch 是「同一锚点日、不同股票」的横截面，所以 batch 内相关 = 逐截面 IC。
（one_v2 的结论：pearson > mse/smooth_l1；且 batch 必须横截面，否则该项无意义。）

用法：
    python two/nowcast/train.py --arm ashare --seed 1 --epochs 7
"""

import argparse
import gc
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'one'))
sys.path.insert(0, HERE)

from data import Store, MARKETS, GLOBAL_EX, ASHARE_EX   # noqa: E402
from model import (Nowcast, FLAT_NAMES, N_HEADS, N_FIELDS,   # noqa: E402
                   N_HORIZONS)
from one_model import set_output_scale                  # noqa: E402
import labels as _L                                    # noqa: E402
import variants as _V                                  # noqa: E402


def save_model_with_scale(model, path, y_mean, y_std):
    """存盘前把「反标准化」写进模型的 buffer，让 .pt **直接输出真实单位**。

    动机（2026-09-26）：训练时标签被 (Y−median)/scale 标准化，模型输出天然是 z 空间。
    以前这对常数放在推理脚本里（或硬编码）—— 漏掉就静默错一个量级（nowcast 的
    "毛利/市值" 曾因此被放大 76 倍）。放进模型后，任何调用方拿到的都是真实单位。

    ⚠️ 存完必须还原成恒等：训练/验证期间的损失和标签都是 z 空间，带着 affine 会错。
    """
    set_output_scale(model, y_mean, y_std)
    torch.save(model, path)
    model.out_mean = torch.zeros(1)
    model.out_scale = torch.ones(1)

# 头名按 variant 取（架构与头数不变，只是名字不同）；
# 张量顺序始终是「字段优先」，与 labels.build_labels 写出的 y 数组一致。
HEAD_NAMES = list(FLAT_NAMES)

ARMS = {'ashare': ASHARE_EX, 'global': GLOBAL_EX, 'both': ASHARE_EX + GLOBAL_EX}
DAYS_INPUT = 127 * 7
BATCH = 64
VAL_FRAC = 0.30
VAL_SEED = 12345
Z_CLIP = 8.0      # z-score 后的裁剪：标签有极重尾（IQR 尺度的 20~40 倍），不裁会被极值主导


def sym_key(s):
    return s.split(':')[-1]


def mem_str():
    """进程 RSS / 系统可用内存，用于盯着别被系统 OOM 杀掉。"""
    try:
        import psutil
        return (f'{psutil.Process().memory_info().rss/1e9:.1f}GB/'
                f'avail{psutil.virtual_memory().available/1e9:.0f}GB')
    except Exception:
        return '?'


def load_split(store, arm):
    """返回 (train_pos, val_pos, test_pos)。股票级隔离，绝不重叠。"""
    keys = np.array([sym_key(s) for s in store.symbol_of(np.arange(len(store.dates)))])
    test_keys = set(l.strip() for l in open(os.path.join(ROOT, 'one', 'test_symbols.txt'))
                    if l.strip())
    uniq = {k: i for i, k in enumerate(np.unique(keys))}
    kc = np.array([uniq[k] for k in keys])
    is_test = np.isin(kc, [uniq[k] for k in test_keys if k in uniq])
    is_cn = np.isin(store.ex_of_file[store.row_file], ASHARE_EX)

    # 验证集：非测试 A 股里 30%（与 one 同 seed）
    cn_keys = sorted(set(keys[is_cn & ~is_test]))
    rng = random.Random(VAL_SEED)
    rng.shuffle(cn_keys)
    val_keys = set(cn_keys[:int(len(cn_keys) * VAL_FRAC)])
    is_val = np.isin(kc, [uniq[k] for k in val_keys if k in uniq])

    if arm == 'global':
        train = (~is_cn) & (~is_test)
    else:
        train = (~is_test) & (~is_val)
        if arm == 'ashare':
            train &= is_cn
    return np.where(train)[0], np.where(is_val)[0], np.where(is_test)[0]


def aligned_test_symbols():
    """`gen_data2.py` 里被设成【日期对齐采样】的那些符号（'EX:SYM'）。

    必须和生成时用的名单**完全一致** —— 否则测试集里混进随机采样的股票，
    逐日截面 IC 就会既稀疏又混了两种采样。
    名单来源：`two/nowcast/test_symbols_global889.txt`（全球 889）+ `one/test_symbols.txt`（A 股 889）。
    裸代码（A 股，如 000001.SZ）按生成端的规则补成 SHZ/SHH 两个键。
    """
    # 只读这一份：从 31 个交易所【均匀随机】抽的 889 只（种子 42，对齐 one 的 TEST_SEED）。
    # 不再叠加 one/test_symbols.txt 的 A 股 889 —— 那会让测试集明显偏袒 A 股。
    want = set()
    p = os.path.join(HERE, 'test_symbols_all31_889.txt')
    if os.path.exists(p):
        for line in open(p, encoding='utf-8'):
            s = line.strip()
            if s:
                want.add(s)
    return want


def load_split_global(store, val_frac=VAL_FRAC, val_seed=None):
    """测试集 = 生成时被对齐的那些股票（见 aligned_test_symbols），其余按股票严格隔离。

    键用完整的 'EX:SYM'（`load_split` 只取 ':' 后面那段，跨交易所会撞名）。
    """
    full = np.array(store.symbol_of(np.arange(len(store.dates))))
    ukeys, inv = np.unique(full, return_inverse=True)
    test_keys = aligned_test_symbols() & set(ukeys.tolist())
    test_idx = np.array([i for i, k in enumerate(ukeys) if k in test_keys])
    is_test = np.isin(inv, test_idx)

    rest = sorted(set(ukeys) - test_keys)
    rng2 = random.Random(VAL_SEED if val_seed is None else val_seed)
    rng2.shuffle(rest)
    val_keys = set(rest[:int(len(rest) * val_frac)])
    val_idx = np.array([i for i, k in enumerate(ukeys) if k in val_keys])
    is_val = np.isin(inv, val_idx)

    train = (~is_test) & (~is_val)
    return np.where(train)[0], np.where(is_val)[0], np.where(is_test)[0]


def group_by_date(store, pos):
    d = store.dates[pos]
    order = np.argsort(d, kind='stable')
    pos = pos[order]
    d = d[order]
    uniq, start = np.unique(d, return_index=True)
    return {int(u): pos[a:b] for u, a, b in
            zip(uniq, start, np.r_[start[1:], len(d)])}


def pearson_loss(pred, tgt):
    p = pred - pred.mean()
    t = tgt - tgt.mean()
    return 1.0 - (p * t).sum() / (p.norm() * t.norm() + 1e-8)


def predict(model, store, pos, y_mean, y_std, device, bs=256):
    """对给定行位置做预测（分块，避免显存爆）。"""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(pos), bs):
            X = store.take_x(pos[i:i + bs])                          # 预测不需要标签
            x = torch.tensor(X[:, :DAYS_INPUT]).float().to(device)   # 只喂 889 天
            del X
            _, p = model(x)
            out.append(p.reshape(p.size(0), N_HEADS).cpu().numpy())
    return np.concatenate(out) if out else np.empty((0, N_HEADS))


def eval_icir(pred, y, dates, min_stocks=20):
    """逐 (锚点日) 截面 Spearman IC -> 每个头的 (IC, ICIR, n)。"""
    res = {}
    ud, inv = np.unique(dates, return_inverse=True)
    groups = [np.where(inv == i)[0] for i in range(len(ud))]
    groups = [g for g in groups if len(g) >= min_stocks]
    for k, name in enumerate(HEAD_NAMES):
        ics = []
        for m in groups:
            a, b = pred[m, k], y[m, k]
            ra = np.argsort(np.argsort(a))
            rb = np.argsort(np.argsort(b))
            ics.append(np.corrcoef(ra, rb)[0, 1])
        ics = np.array([v for v in ics if np.isfinite(v)])
        res[name] = (ics.mean(), ics.mean() / ics.std() if len(ics) > 1 else np.nan,
                     len(ics))
    return res


def eval_ic_overall(pred, y):
    """整体 IC：把全部行放在一起算 Spearman。

    随机采样后每个日期的测试样本很稀疏，"逐日截面 IC" 噪声极大，
    所以主口径改成整体 IC（再辅以逐日截面 IC 作参考）。
    """
    res = {}
    for k, name in enumerate(HEAD_NAMES):
        a, b = pred[:, k], y[:, k]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 20:
            res[name] = (np.nan, np.nan, 0)
            continue
        ra = np.argsort(np.argsort(a[ok]))
        rb = np.argsort(np.argsort(b[ok]))
        res[name] = (float(np.corrcoef(ra, rb)[0, 1]), np.nan, int(ok.sum()))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=list(ARMS), required=True)
    ap.add_argument('--root', default='D:/quant_data/nowcast')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=7)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--tag', default='')
    ap.add_argument('--out', default='results')
    ap.add_argument('--labels', choices=['level', 'resid'], default='level',
                    help='level=预测水平标签 | resid=只预测「实际−朴素基准」的增量')
    ap.add_argument('--suffix', default='',
                    help='标签变体后缀（见 variants.py）：""=基线 | v1 | v2 | u1')
    ap.add_argument('--test-mode', choices=['a889', 'global_random'], default='a889',
                    help='a889=one 的 889 只 A 股（旧行为）| global_random=全球随机抽')
    ap.add_argument('--test-n', type=int, default=889, help='global_random 的测试股票数')
    ap.add_argument('--test-seed', type=int, default=42)
    ap.add_argument('--val-seed', type=int, default=0,
                    help='0 = 每次运行随机（用户要求训练/验证划分不固定随机）；非 0 则固定')
    ap.add_argument('--loss', choices=['pearson', 'smooth_l1'], default='smooth_l1',
                    help='pearson=横截面相关（要求 batch 同一天）| '
                         'smooth_l1=逐样本（可随机抽 batch，配合随机采样）')
    ap.add_argument('--min-date-n', type=int, default=0,
                    help='pearson 用【变长同日 batch】的下限（该日全部样本作为一个 batch）。'
                         '0=旧行为（固定 BATCH=64，样本不足 64 的日子整段丢弃）。'
                         '设 16 可让样本覆盖 ~98.6%、步数 ~0.87x，与 smooth_l1 可比。')
    ap.add_argument('--sched', choices=['none', 'cosine'], default='none',
                    help='学习率调度：none=恒定 lr（旧行为）| cosine=线性 warmup 后 cosine 降到 lr*0.02')
    ap.add_argument('--warmup', type=int, default=200, help='--sched 的 warmup 步数')
    ap.add_argument('--warm-start', action='store_true',
                    help='从同 tag 的旧 nowcast_{arm}{tag}.pt 接着训（默认关：以往每个实验都是从头训，'
                         '不加开关会悄悄改变既有 u* 变体的语义）。载入后会把 out_mean/out_scale '
                         '复位成恒等 —— 存盘的 .pt 被 save_model_with_scale 烧进了真实单位 affine。')
    ap.add_argument('--ema', type=float, default=0.0,
                    help='EMA 权重衰减（0=关闭）。开启后验证与存盘都用 EMA 权重，训练仍用原权重。'
                         '起点建议 0.999（约 1000 步的有效窗口）')
    args = ap.parse_args()

    # 头数按 variant 走：u6 只有 3 个头（3 字段 × 1 期限），其余是 3×3=9
    global HEAD_NAMES, N_HEADS, N_FIELDS, N_HORIZONS
    _f, _h = _V.fields_for(args.suffix), _V.horizons_for(args.suffix)
    HEAD_NAMES = _L.flat_names(_f, _h)
    N_FIELDS, N_HORIZONS = len(_f), len(_h)
    N_HEADS = N_FIELDS * N_HORIZONS
    print(f'变体 {args.suffix!r}: {_V.label_desc(args.suffix)}', flush=True)

    if args.seed:
        torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    rng = np.random.RandomState(args.seed + 1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Store 必须始终含 A 股：验证集与测试集都是 A 股。global 臂若只装海外交易所，
    # 验证集与测试集会双双为空（early stop 失效、889 只 A 股评估不了），整轮白跑。
    # 训练集由 load_split 按臂过滤，与 Store 装了什么无关。
    store = Store(args.root, sorted(set(ARMS[args.arm]) | set(ASHARE_EX)),
                  min_per_date=BATCH, label=args.labels, suffix=args.suffix)
    print(f'[{args.arm}] {store.stats()}  label={args.labels}{args.suffix}', flush=True)
    if args.test_mode == 'global_random':
        _vs = args.val_seed or random.randrange(10 ** 9)
        tr, va, te = load_split_global(store, val_seed=_vs)
        print(f'[{args.arm}] 测试集 = 对齐的 {len(set(store.symbol_of(te))):,} 只'
              f'（31 所均匀随机 889）；验证集划分种子 = {_vs}', flush=True)
    else:
        tr, va, te = load_split(store, args.arm)
    print(f'[{args.arm}] 训练 {len(tr):,} / 验证 {len(va):,} / 测试 {len(te):,}', flush=True)

    tr_by_date = group_by_date(store, tr)
    dates = np.array(sorted(tr_by_date))

    # 标签标准化统计（训练集抽样估计）
    samp = np.sort(tr[rng.choice(len(tr), min(20000, len(tr)), replace=False)])
    Ys = store.take_y(samp)
    # 稳健尺度代替 std：全球数据的标签极重尾（|x| 中位数 0.3、p99 到 113、std 是 IQR 尺度的
    # 23~37 倍）。用 std 做 z-score 会把 90% 的样本压成 ~0.1、让 1% 的极值主导损失。
    # IQR/1.349 是正态下的稳健标准差估计，对尾部免疫。
    y_mean = np.median(Ys, 0)
    _iqr = np.percentile(Ys, 75, 0) - np.percentile(Ys, 25, 0)
    y_std = _iqr / 1.349 + 1e-6
    # aux 统计分块读：一次性物化 2 万条 X 是 (20000,1017,31) = 2.5 GB，stack 后翻倍。
    # 末尾 .copy() 必须有 —— 否则 reshape 出的只是视图，会把整块 Xc 一直吊住。
    aux_acc = []
    for i in range(0, len(samp), 1024):
        Xc = store.take_x(samp[i:i + 1024])
        aux_acc.append(Xc[:, DAYS_INPUT:DAYS_INPUT + 128][:, :, [3, 4, 5]]
                       .reshape(-1, 3).copy())
        del Xc
    aux_s = np.concatenate(aux_acc)
    del aux_acc
    aux_mean = np.median(aux_s, 0)
    _aiqr = np.percentile(aux_s, 75, 0) - np.percentile(aux_s, 25, 0)
    aux_std = _aiqr / 1.349 + 1e-6
    del aux_s, Ys
    print(f'[{args.arm}] aux mean/std = {np.round(aux_mean,3).tolist()} / '
          f'{np.round(aux_std,3).tolist()}', flush=True)
    print(f'[{args.arm}] 标签稳健尺度(IQR/1.349) = {np.round(y_std, 4).tolist()}'
          f'   z 裁剪 = ±{Z_CLIP}', flush=True)

    model = Nowcast(n_fields=N_FIELDS, n_horizons=N_HORIZONS).to(device)
    _prev = os.path.join(HERE, f'nowcast_{args.arm}{args.tag}.pt')
    if args.warm_start and os.path.exists(_prev):
        model = torch.load(_prev, map_location=device, weights_only=False).to(device)
        # ⚠️ 必做：若这个 .pt 是 save_model_with_scale 存的，out_mean/out_scale 里装的是
        #    【真实单位】的 affine（forward: close_preds * out_scale + out_mean），而训练用的
        #    标签是 z 空间 —— 不复位，预测会带着 ×scale+median 进损失，第一步就炸。
        #    这里用 set_output_scale 而不是 `model.out_scale = ...`：对「没有这两个 buffer 的
        #    旧 .pt」（如 2026-09-26 10:29 存的 u9），直接赋值会退化成普通属性、.to(device)
        #    不搬它 -> 设备不一致 —— one_model.py:212 记的正是这个坑。
        #    长度必须是 N_HEADS，否则 forward 的 numel 守卫会跳过（跳过的含义是 z 空间，也安全）。
        set_output_scale(model, np.zeros(N_HEADS), np.ones(N_HEADS))
        print(f'[{args.arm}] warm start: 载入 {os.path.basename(_prev)}'
              f'（out_mean/out_scale 复位为恒等 {N_HEADS} 维）', flush=True)
    elif args.warm_start:
        print(f'[{args.arm}] warm start 开着但 {os.path.basename(_prev)} 不存在 -> 从头训', flush=True)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # EMA 权重平均：验证与存盘都用 EMA 权重（这才是 EMA 的正确用法 —— 训练还是用原权重）。
    # 30% 的 dropout + 逐样本噪声大的回归目标，权重本身的抖动不小，平均掉通常能给 IC 加一点点。
    ema = None
    if args.ema > 0:
        ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ymt = torch.tensor(y_mean, dtype=torch.float32, device=device)
    yst = torch.tensor(y_std, dtype=torch.float32, device=device)
    amt = torch.tensor(aux_mean, dtype=torch.float32, device=device)
    ast = torch.tensor(aux_std, dtype=torch.float32, device=device)

    # 1 epoch = 训练集全量一遍，与 one 的 DataLoader(shuffle=True) 语义对齐（one 是 2,617 步/epoch）。
    # 原实现每个锚点日只抽 1 个 batch，而锚点日只有 ~75 个（季度末），于是 1 epoch 仅 ~65 步、
    # 4 千样本；全球臂数据多 5 倍却拿到与 A 股臂完全相同的步数，三臂对照失真。
    if args.loss == 'pearson':
        if args.min_date_n:
            # 变长 batch：把该日【全部】样本作为一个 batch（≥ min_date_n 只的日子）
            steps_per_epoch = sum(1 for p in tr_by_date.values() if len(p) >= args.min_date_n)
            _cov = sum(len(p) for p in tr_by_date.values() if len(p) >= args.min_date_n)
            print(f'  变长同日 batch：≥{args.min_date_n} 只/日，覆盖训练样本 '
                  f'{_cov:,}/{len(tr):,} = {_cov/len(tr)*100:.1f}%', flush=True)
        else:
            steps_per_epoch = sum(len(p) // BATCH for p in tr_by_date.values())
    else:
        steps_per_epoch = len(tr) // BATCH
    print(f'[{args.arm}] loss={args.loss}  每 epoch {steps_per_epoch:,} 步', flush=True)
    # 学习率调度：线性 warmup → cosine 衰减到 lr*0.02。恒定 lr=1e-3 跑 7 轮，末尾验证 IC 还在
    # 缓慢上升；尾部降 lr 让权重收敛到更平的点上（通常是白赚的一点）。
    _total_steps = steps_per_epoch * args.epochs
    _warm = max(1, args.warmup)

    def lr_at(step):
        if step < _warm:
            return args.lr * (step + 1) / _warm
        p = min(1.0, (step - _warm) / max(1, _total_steps - _warm))
        return args.lr * (0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * p)))

    if args.sched != 'none':
        print(f'  学习率：warmup {_warm} 步 → cosine 降到 {args.lr*0.02:.1e}'
              f'（共 {_total_steps:,} 步）', flush=True)
    if ema is not None:
        print(f'  EMA 衰减 {args.ema}（验证与存盘都用 EMA 权重）', flush=True)
    best = float('inf')
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        if args.loss == 'pearson':
            if args.min_date_n:
                sched = [d for d in dates if len(tr_by_date[d]) >= args.min_date_n]
                rng.shuffle(sched)
                batch_src = (tr_by_date[d] for d in sched)
            else:
                # 横截面：同一天抽 BATCH 只不同股票（batch 内相关 = 当日截面 IC）
                sched = [d for d in dates for _ in range(len(tr_by_date[d]) // BATCH)]
                rng.shuffle(sched)
                batch_src = (rng.choice(tr_by_date[d], BATCH, replace=False) for d in sched)
        else:
            # 逐样本：随机抽整份数据（混日期、混股票）—— 配合 per-stock 随机采样。
            # 先 permutation 再切片：直接 rng.choice(tr, ...) 在大池上是 O(len(tr))，会非常慢。
            perm = rng.permutation(tr)
            batch_src = (perm[i * BATCH:(i + 1) * BATCH]
                         for i in range(steps_per_epoch))
        run, n = 0.0, 0
        for idx in batch_src:
            X, Y, _ = store.take(idx)
            xf = torch.tensor(X).float().to(device)              # (B, 1017, 31) 全窗口
            x = xf[:, :DAYS_INPUT]                               # 模型只吃 889 天
            # z-score 目标（减中位数 + 除稳健尺度）。模型输出因此在 z 空间 ->
            # 存盘时由 save_model_with_scale 焼进 out_mean/out_scale buffer 还原成真实单位。
            # ⚠️ 若改成 seven/three 那种「两边同除、不减中位数」，**必须把损失里的预测也一起除**
            #   （`smooth_l1(pred/yst, Y/yst)`）；只除目标不改预测只会让输出变成"除以尺度后的值" ✗
            #   —— 2026-09-26 这么错过一次。
            y = torch.clamp((torch.tensor(Y).float().to(device) - ymt) / yst,
                            -Z_CLIP, Z_CLIP)
            aux_t = xf[:, DAYS_INPUT:DAYS_INPUT + 128][:, :, [3, 4, 5]]   # 未来 128 天 close/volume/delta
            aux_t = torch.clamp((aux_t - amt) / ast, -Z_CLIP, Z_CLIP)
            opt.zero_grad()
            aux_p, pred = model(x)          # pred (B, 3 字段, 3 期限)
            loss = F.mse_loss(aux_p, aux_t)
            if args.loss == 'pearson':
                for f in range(N_FIELDS):
                    for h in range(N_HORIZONS):
                        loss = loss + pearson_loss(pred[:, f, h], y[:, f * N_HORIZONS + h])
            else:
                # 逐样本 smooth_l1（标签已 z-score）。随机 batch 上算"相关"会退化成
                # 猜"这是哪一年"（衰退年标签普遍为负）→ 不能用 pearson。
                loss = loss + F.smooth_l1_loss(pred.reshape(-1, N_HEADS), y, beta=1.0) * N_HEADS
            loss.backward(); opt.step()
            if args.sched != 'none':
                opt.param_groups[0]['lr'] = lr_at(ep * steps_per_epoch + n)
            if ema is not None:
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        if ema[k].is_floating_point():
                            ema[k].mul_(args.ema).add_(v.detach(), alpha=1.0 - args.ema)
                        else:
                            ema[k].copy_(v)          # 非浮点 buffer（如 num_batches_tracked）直接拷
            run += loss.item(); n += 1
            if n % 500 == 0:
                # 每 500 步（约 4 GB 读入）释放一次 memmap，否则 RSS 会被读过的
                # 文件页顶到 100 GB 以上 —— global 臂一个 epoch 读 97 GB。
                store.release_mmaps()
                gc.collect()
                print(f'  ep{ep} {n}/{steps_per_epoch} loss={run/n:.4f} '
                      f'{n/(time.time()-t0):.1f} step/s mem={mem_str()}',
                      flush=True)

        # 验证（ICIR 口径，而不是 loss —— val loss 与 IC 会脱钩）
        # EMA 开着时：先把 EMA 权重换进模型再验证（验证完立刻换回来，否则下一 epoch 就在 EMA 上继续训）
        _bak = None
        if ema is not None:
            _bak = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema)
        vp = predict(model, store, va, y_mean, y_std, device)
        vy = store.take_y(va)
        r = eval_ic_overall(vp, vy)          # 整体 IC 为主口径
        score = float(np.nanmean([v[0] for v in r.values()]))
        del vp, vy
        gc.collect()          # 及时归还验证集那几 GB，别让它和下一 epoch 的读盘叠加
        print(f'[{args.arm}] ep{ep} 训练 loss={run/max(n,1):.4f}  '
              f'验证 mean IC={score:.4f}  ({time.time()-t0:.0f}s)', flush=True)
        if -score < best:
            best = -score
            # 此刻 model 里就是 EMA 权重（若 EMA 开启）→ 存下来的 .pt 直接是 EMA 的
            # 顺带把反标准化（×yst + ymt）焼进 buffer -> .pt 直接输出真实单位；存完还原恒等
            save_model_with_scale(model, os.path.join(HERE, f'nowcast_{args.arm}{args.tag}.pt'),
                                  y_mean, y_std)
        if _bak is not None:
            model.load_state_dict(_bak)
            del _bak

    # 测试集评估
    model = torch.load(os.path.join(HERE, f'nowcast_{args.arm}{args.tag}.pt'),
                       map_location=device, weights_only=False)
    tp = predict(model, store, te, y_mean, y_std, device)
    ty = store.take_y(te)
    tsyms = store.symbol_of(te)
    r_ov = eval_ic_overall(tp, ty)
    r = eval_icir(tp, ty, store.dates[te])
    # A 股子样本（用户实际要投的市场）—— 全球随机测试集里必须单独看它
    tsym_arr = np.array(tsyms)
    is_cn = np.array([s.split(':')[0] in ASHARE_EX for s in tsym_arr])
    r_cn = eval_ic_overall(tp[is_cn], ty[is_cn]) if is_cn.sum() > 100 else {}
    print(f'\n===== [{args.arm}{args.tag}] 测试集  '
          f'（全体 {len(te):,} 行，其中 A 股 {int(is_cn.sum()):,} 行）=====')
    print(f'{"头":12s}{"整体IC":>10s}{"A股IC":>10s}{"逐日IC":>10s}{"逐日ICIR":>10s}{"n":>8s}')
    for name in HEAD_NAMES:
        ov = r_ov.get(name, (np.nan,))[0]
        cn = r_cn.get(name, (np.nan,))[0] if r_cn else np.nan
        ic, icir, nn = r.get(name, (np.nan, np.nan, 0))
        print(f'{name:12s}{ov:>+10.4f}{cn:>+10.4f}{ic:>+10.4f}{icir:>+10.2f}{nn:>8d}')
    print(f'{"平均":12s}{np.nanmean([v[0] for v in r_ov.values()]):>+10.4f}'
          f'{(np.nanmean([v[0] for v in r_cn.values()]) if r_cn else np.nan):>+10.4f}')
    print('\n===== 逐日截面 IC（旧口径；随机采样后每格样本稀疏，仅参考）=====')

    # ---- 朴素持续性基准对照（本轮的 go/no-go）----
    # Phase 0b 的结论：基准（「去年同期那一轮重演」，锚点日即可算出、不用价格）
    # 在所有字段上 IC 都比价格因子高。所以「模型 IC 很漂亮」不能说明有价值，
    # 必须看扣掉基准之后的**增量**。两种 label 口径下形式统一：
    #     level 版：模型直接预测水平       → 水平预测 = tp
    #     resid 版：模型只预测增量         → 水平预测 = 基准 + tp（基准是已知量）
    dt = store.dates[te]
    ty_level = store.take_level(te)
    ty_bench = store.take_bench(te)
    ty_resid = ty_level - ty_bench
    lvl = ty_bench + tp if args.labels == 'resid' else tp
    r_full = eval_icir(lvl, ty_level, dt)
    r_bch = eval_icir(ty_bench, ty_level, dt)
    r_inc = eval_icir(lvl - ty_bench, ty_resid, dt)
    print(f'\n===== 基准对照（同一批测试样本）=====')
    print(f'{"头":16s}{"IC(模型,水平)":>14s}{"IC(基准,水平)":>14s}'
          f'{"IC(增量,残差)":>15s}')
    for name in HEAD_NAMES:
        print(f'{name:16s}{r_full[name][0]:>+14.4f}{r_bch[name][0]:>+14.4f}'
              f'{r_inc[name][0]:>+15.4f}')
    mf = np.mean([v[0] for v in r_full.values()])
    mb = np.mean([v[0] for v in r_bch.values()])
    mi = np.mean([v[0] for v in r_inc.values()])
    print(f'{"平均":16s}{mf:>+14.4f}{mb:>+14.4f}{mi:>+15.4f}')
    print(f'→ 增量 IC（{mi:+.4f}）显著为正才算有超出持续性的信息；'
          f'水平 IC（{mf:+.4f}）低于基准（{mb:+.4f}）说明水平预测还不如直接用上一期财报。')

    os.makedirs(os.path.join(HERE, args.out), exist_ok=True)
    r = r_ov
    with open(os.path.join(HERE, args.out,
                           f'{args.arm}{args.tag}_s{args.seed}.json'), 'w') as f:
        json.dump({'arm': args.arm, 'seed': args.seed, 'n_params': n_par,
                   'labels': args.labels,
                   'train': int(len(tr)), 'test': int(len(te)),
                   'icir': {k: v[1] for k, v in r.items()},
                   'ic': {k: v[0] for k, v in r.items()},
                   'ic_full': {k: v[0] for k, v in r_full.items()},
                   'ic_bench': {k: v[0] for k, v in r_bch.items()},
                   'ic_incr': {k: v[0] for k, v in r_inc.items()},
                   'icir_incr': {k: v[1] for k, v in r_inc.items()}}, f, indent=1)
    # 存预测供集成用
    np.savez(os.path.join(HERE, args.out, f'pred_{args.arm}{args.tag}_s{args.seed}.npz'),
             pred=tp, y=ty, date=dt, sym=np.array(tsyms))


if __name__ == '__main__':
    main()
