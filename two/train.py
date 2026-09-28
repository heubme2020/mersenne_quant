"""two 训练器：读 `two/train/*.h5` -> 训 3 头 Nowcast -> 写 `two/two.pt`。

    cd two
    python train.py            # 默认：two/train/*.h5 -> two/two.pt

## 配方与 nowcast 一致（逐条对应 `two/nowcast/train.py` 的 both 臂 + `--suffix u9 --loss smooth_l1`）

* 输入 = h5 的 `key='data'` 前 889 行（1017×31 里的输入段）；辅助头目标 = 后 128 行 × 列 [3,4,5]
  （= close/volume/delta，与 one 的 FORE_COLS 同）。
* 损失 = `mse(aux)` + `smooth_l1(pred, y, beta=1.0) * N_HEADS`（two/nowcast/train.py:414-422）。
* 标签按**训练集**做稳健标准化：`y = (Y − median) / (IQR/1.349 + 1e-6)`，再裁到 ±8
  （two/nowcast/train.py:297-299、408-409）。aux 同理（中位数 + IQR/1.349）。
* Adam lr=1e-3、**7 个 epoch**、batch 64、每 epoch 随机打乱整份训练集（= nowcast 的 `perm` 切块）。
  （7 是五个模型共用的口径：one/three/seven 的 train.py 写死 7，zero 2026-09-27 从 31 改成 7。）
* EMA 关闭（`--ema 0` 是 nowcast 的默认）。
* 验证口径 = 整体 Spearman IC（`eval_ic_overall`）的均值，取最好的那个 epoch 存盘。
* **终端输出与另外四个模型同口径**（2026-09-27）：进度打
  `Epoch: %d, step: %d, train loss: ... mean loss: ... min val loss: ...`，每个 epoch 打
  `Epoch: %d, validate loss: %1.5f`（= 上面那个损失在验证集上的均值，`batch_loss` 一处定义、
  训练/验证共用）。但**存盘仍按验证 mean IC 选** —— 排名模型该看 IC，loss 只是"优化到哪了"
  的体温计。所以紧跟着多打一行 two 特有的 `验证 mean IC=... 逐头 [...]`，那才是决定存哪一版的数。
* 存盘前把「反标准化 affine」焼进模型 buffer（`one_model.set_output_scale`）——`two/get_two_predict.py`
  按「模型直接输出真实单位」写的，漏了就会静默错一个量级（nowcast 的毛利/市值曾因此被放大 76 倍）。
* **warm start**（2026-09-27 补）：`--out` 已存在就从它接着训，与 one/three/seven/zero 的 train.py
  一致（`--no-warm-start` 关掉）。载入后**必须把 affine 还原成恒等**，理由见下面那段 ⚠️。

## 划分（与 one / nowcast 一致：**测试股票固定**）

* 测试集 = `two/test_symbols.txt`（'EX:SYM'，= gen_train_data 走全局同步网格的那 889 只）。
  沿用而不是重抽的理由见 `one/train.py` 里那段注释：数据里加入新股票后，重抽会让整个测试集漂移，
  新旧指标不可比。**这份名单同时是生成端"对齐采样"的名单**，所以测试股票在同一天都有锚点
  -> 才算得出逐日截面 IC。
* 验证集 = 其余股票里 30%（seed 12345，与 one/nowcast 同）；训练集 = 剩下的 70%。

## 与 nowcast 的两个必然差异（不是配方差异，是数据格式差异）

1. h5 是**每样本一个文件**（nowcast 是 memmap 分块），所以按文件读；`--val-max` 默认只抽
   2 万行算验证 IC（nowcast 用全部验证行）——纯粹为了让 7 个 epoch 跑得完，IC 口径没变。
2. 标签在**另一个 key** 里（`key='label'`，1 行 3 列）。生成端已经把 u9 窗口不完整
   （标签为 NaN）的锚点整段丢掉（等价于旧线里 `two/nowcast/data.py` 的 Store 对 y/b/r 的有限性过滤），
   所以落盘的行标签一定有限；启动时抽检 200 个 h5 确认一次列名与有限性。
   2026-09-27 起 `key='label'` 里的 `ok`（恒 True）和 `b_*`（恒 0）两列常量不再写盘，见
   `two_labels.label_frame`。**旧格式（7 列）的文件不用重生成**：取 `[HEAD_NAMES]` 拿到的
   是同一份 y、逐位等价，所以 2026-09-27 之前生成的那 163 万个 h5 照常能用，
   启动抽检只提示不报错（见下面的 `_legacy`）。
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:                                    # Windows 控制台默认 GBK，见 gen_train_data.py 的坑 2
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))

# 模型定义在 two 自己那份（逐字复制自 two/nowcast/model.py；one_model 用显式路径导入）
from two_nowcast_model import (Nowcast, HEAD_NAMES, N_HEADS, AUX_OUTPUT_DAYS,   # noqa: E402
                               set_output_scale)
from two_features import COLS, DAYS_INPUT                                       # noqa: E402

AUX = AUX_OUTPUT_DAYS            # 128
Z_CLIP = 8.0                     # two/nowcast/train.py:Z_CLIP
VAL_FRAC = 0.30
VAL_SEED = 12345                 # 与 one / nowcast 同
AUX_COLS = [3, 4, 5]             # COLS = [open,high,low,close,volume,delta] + ... -> close/volume/delta


def parse_name(path):
    """文件名 -> (EX, SYM, date, key)。

    两种命名都能认（见 gen_train_data 的 `--name-style`）：
        ex_sym（默认） `JKT_BBCA_20240102.h5`  -> ex='JKT'  key='JKT:BBCA'
        sym            `000001.SZ_19940623.h5` -> ex=''     key='000001.SZ'
    从【右边】切日期，所以符号里带 '_' 也不会错；只有"裸符号本身含 '_'"会被误认成 EX_SYM
    （本数据集 31 个交易所的代码里没有 '_'，实测过）。
    """
    base = os.path.basename(path)[:-3]
    head, date = base.rsplit('_', 1)
    ex, sym = head.split('_', 1) if '_' in head else ('', head)
    return ex, sym, int(date), (f'{ex}:{sym}' if ex else sym)


def select_device():
    if torch.cuda.is_available():
        try:
            (torch.zeros(2, 2, device='cuda') @ torch.zeros(2, 2, device='cuda')).cpu()
            return torch.device('cuda')
        except Exception as e:
            print(f'[警告] CUDA 检测到但 kernel 不可用（{type(e).__name__}），退回 CPU。', flush=True)
    return torch.device('cpu')


class Sample:
    """一个 h5 的元信息（不读盘）。"""

    __slots__ = ('path', 'ex', 'sym', 'date', 'key')

    def __init__(self, path):
        self.path = path
        self.ex, self.sym, self.date, self.key = parse_name(path)


class H5Dataset(Dataset):
    """读一个样本：输入 (889,31)、辅助目标 (128,3)、标签 (N_HEADS,)，后两者已标准化+裁剪。"""

    def __init__(self, samples, y_mean, y_std, aux_mean, aux_std):
        self.samples = samples
        self.y_mean, self.y_std = y_mean, y_std
        self.aux_mean, self.aux_std = aux_mean, aux_std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        x = pd.read_hdf(s.path, key='data').values            # (1017, 31) float32
        y = pd.read_hdf(s.path, key='label')[HEAD_NAMES].values[0].astype('float32')
        inp = torch.from_numpy(x[:DAYS_INPUT])
        aux = torch.from_numpy(x[DAYS_INPUT:DAYS_INPUT + AUX][:, AUX_COLS])
        aux = torch.clamp((aux - self.aux_mean) / self.aux_std, -Z_CLIP, Z_CLIP)
        y = torch.clamp((torch.from_numpy(y) - self.y_mean) / self.y_std, -Z_CLIP, Z_CLIP)
        return inp, aux, y


def estimate_stats(samples, n_max, rng):
    """训练集的稳健标准化统计：median + IQR/1.349（two/nowcast/train.py:297-312 同式）。"""
    idx = np.sort(rng.choice(len(samples), min(n_max, len(samples)), replace=False))
    Ys, aux = [], []
    for i in idx:
        s = samples[i]
        x = pd.read_hdf(s.path, key='data').values
        Ys.append(pd.read_hdf(s.path, key='label')[HEAD_NAMES].values[0].astype('float64'))
        aux.append(x[DAYS_INPUT:DAYS_INPUT + AUX][:, AUX_COLS].astype('float64').reshape(-1, 3))
    Ys = np.stack(Ys)
    aux = np.concatenate(aux)
    y_mean = np.median(Ys, 0).astype('float32')
    y_std = (np.percentile(Ys, 75, 0) - np.percentile(Ys, 25, 0)) / 1.349 + 1e-6
    aux_mean = np.median(aux, 0).astype('float32')
    aux_std = (np.percentile(aux, 75, 0) - np.percentile(aux, 25, 0)) / 1.349 + 1e-6
    return y_mean, y_std.astype('float32'), aux_mean, aux_std.astype('float32')


def rank_ic(a, b):
    """Spearman（rank 后的 Pearson）。a/b 已对齐、长度 >2 且无 NaN。"""
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def eval_ic_overall(pred, y):
    """整体 IC（把全部行放一起算 Spearman）—— nowcast 的验证主口径。"""
    res = {}
    for k, name in enumerate(HEAD_NAMES):
        a, b = pred[:, k], y[:, k]
        ok = np.isfinite(a) & np.isfinite(b)
        res[name] = rank_ic(a[ok], b[ok]) if ok.sum() >= 20 else np.nan
    return res


def eval_icir(pred, y, dates, min_stocks=20):
    """逐（锚点日）截面 Spearman IC -> 每个头 (IC, ICIR, n)。测试股票在同步网格上，这才有意义。"""
    res = {}
    ud, inv = np.unique(dates, return_inverse=True)
    groups = [np.where(inv == i)[0] for i in range(len(ud))]
    groups = [g for g in groups if len(g) >= min_stocks]
    for k, name in enumerate(HEAD_NAMES):
        ics = []
        for m in groups:
            v = rank_ic(pred[m, k], y[m, k])
            if np.isfinite(v):
                ics.append(v)
        ics = np.array(ics)
        res[name] = (ics.mean() if len(ics) else np.nan,
                     ics.mean() / ics.std() if len(ics) > 1 and ics.std() > 0 else np.nan,
                     len(ics))
    return res


def batch_loss(aux_p, pred, aux_t, y):
    """训练/验证共用的损失（= two/nowcast/train.py:414-422 的 `mse(aux) + N_HEADS·smooth_l1`）。

    只在这里定义一次：训练循环和验证循环各写一遍的话，改了一处忘了另一处，
    那句打出来的 "validate loss" 就悄悄不再是训练目标那个 loss 了。
    """
    return F.mse_loss(aux_p, aux_t) + F.smooth_l1_loss(
        pred.reshape(-1, N_HEADS), y, beta=1.0) * N_HEADS


def predict(model, loader, device):
    """跑一遍 loader -> (预测, 标签, 平均 loss)。

    loss 走的是训练同一个 `batch_loss`（同 z 空间、同裁 ±Z_CLIP），所以
    `Epoch: %d, validate loss:` 与另外四个模型是同一个口径。
    """
    model.eval()
    P, Y, ls, n = [], [], 0.0, 0
    with torch.no_grad():
        for inp, aux_t, y in loader:
            inp = inp.to(device, non_blocking=True)
            aux_t = aux_t.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            aux_p, p = model(inp)
            ls += batch_loss(aux_p, p, aux_t, y).item()
            n += 1
            P.append(p.reshape(p.size(0), N_HEADS).cpu().numpy())
            Y.append(y.cpu().numpy())          # y 已经搬到 device 了 -> 必须 .cpu() 再 numpy
    return np.concatenate(P), np.concatenate(Y), ls / max(n, 1)


def main():
    ap = argparse.ArgumentParser(description='two 训练（读 h5，产出 two.pt）')
    ap.add_argument('--data', default=os.path.join(HERE, 'train'), help='h5 目录（默认 two/train）')
    ap.add_argument('--out', default=os.path.join(HERE, 'two.pt'), help='产出（默认 two/two.pt）')
    ap.add_argument('--test-symbols', default=os.path.join(HERE, 'test_symbols.txt'))
    ap.add_argument('--epochs', type=int, default=7)
    ap.add_argument('--log-every', type=int, default=50,
                    help='每多少步打一行进度（one/train.py 也是 50；0 = 不打）')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--seed', type=int, default=0)
    # 验证集划分种子：默认 12345（与 one/nowcast 同，保证手工跑可复现），但**允许环境变量覆盖**
    # —— `refresh_models.py` 每次重训会注入随机的 VAL_SPLIT_SEED，让验证划分每轮不同
    # （否则报出来的"验证 mean IC"会长期盯着同一个划分、越比越乐观）。
    # one/train.py:226 是同一个写法；three/seven/zero 本来就用系统熵，不需要这个。
    ap.add_argument('--val-seed', type=int, default=int(os.environ.get('VAL_SPLIT_SEED', VAL_SEED)))
    ap.add_argument('--stats-n', type=int, default=20000, help='估标准化统计的样本数')
    ap.add_argument('--val-max', type=int, default=20000,
                    help='验证 IC 用多少行（0=全部）。h5 逐文件读，全量验证太慢')
    ap.add_argument('--min-stocks', type=int, default=20,
                    help='逐日截面 IC 只统计当天样本数 ≥ 这个值的锚点日（nowcast 的 eval_icir 同）')
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--limit-train', type=int, default=0, help='只用前 N 个训练样本（调试用）')
    ap.add_argument('--no-warm-start', action='store_false', dest='warm_start', default=True,
                    help='不从 --out 的旧 .pt 接着训（默认从它续训，与另外四个模型的 train.py 一致）')
    a = ap.parse_args()

    if a.seed:
        torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    rng = np.random.RandomState(a.seed + 1)
    device = select_device()
    workers = a.workers or max(2, min(8, (os.cpu_count() or 4) // 2))

    if not os.path.isdir(a.data):
        raise SystemExit(f'训练目录不存在：{a.data}\n（先在本目录跑 `python gen_train_data.py`）')
    _out_dir = os.path.dirname(os.path.abspath(a.out))
    if _out_dir:
        os.makedirs(_out_dir, exist_ok=True)
    files = sorted(os.path.join(a.data, f) for f in os.listdir(a.data) if f.endswith('.h5'))
    if not files:
        raise SystemExit(f'训练目录没有 h5：{a.data}（先跑 `python gen_train_data.py`）')
    samples = [Sample(f) for f in files]
    keys = sorted({s.key for s in samples})
    print(f'[{a.data}] h5 {len(samples):,} 个 / 股票 {len(keys):,} 只', flush=True)

    # 列名/形状自检（拿第一个文件）：列序漂移会静默出错，所以这里硬校验一次
    _x = pd.read_hdf(samples[0].path, key='data')
    assert list(_x.columns) == COLS, f'❌ h5 列名与 build_x 的 COLS 不一致：{list(_x.columns)}'
    assert _x.shape == (DAYS_INPUT + AUX, len(COLS)) and _x.values.dtype == np.float32, \
        f'❌ h5 形状/类型不对：{_x.shape} {_x.values.dtype}'
    print(f'  自检 OK：key=data {_x.shape} {_x.values.dtype}', flush=True)
    # 标签自检（抽 200 个）：gen_train_data 只在 u9 标签有限（ok 门）时才落盘，
    # 所以这里只需确认「三个头都在 + 值有限」。抽检是为了挡住"手搓/放错目录/列名漂移"的数据
    # —— 标签 NaN 会让 loss 直接变 NaN。
    #
    # ⚠️ **新旧两种格式都收**（2026-09-27）：旧的 label 是 7 列（y + b_* 恒 0 + ok 恒 True），
    # 新的是 3 列。旧格式取 [HEAD_NAMES] 拿到的是同一份 y，**逐位等价**，所以不做全量重生成
    # 也能混着训（全量重来是 240GB / 好几小时）。这里只提示、不报错。
    _chk = [samples[i] for i in np.random.RandomState(0).choice(
        len(samples), min(200, len(samples)), replace=False)]
    _bad, _legacy = [], 0
    for s in _chk:
        _lab = pd.read_hdf(s.path, key='label')
        if not all(c in _lab.columns for c in HEAD_NAMES) or \
                not np.isfinite(_lab[HEAD_NAMES].values[0]).all():
            _bad.append(s.path)
        elif list(_lab.columns) != list(HEAD_NAMES):
            _legacy += 1
    if _bad:
        raise SystemExit(f'❌ 抽检的 200 个 h5 里有 {len(_bad)} 个标签不可用（缺头名 '
                         f'{HEAD_NAMES} 或值不是有限数），例如 {_bad[:3]} —— '
                         f'请用 two/gen_train_data.py 重新生成')
    if _legacy:
        print(f'  提示：抽检的 200 个里有 {_legacy} 个是旧格式（label 多带 ok/b_* 常量列），'
              f'取 [HEAD_NAMES] 后与新版等价，可混用', flush=True)
    print(f'  标签抽检 OK（200 个 h5 的 label 都含 {len(HEAD_NAMES)} 个头的有限值）', flush=True)

    # ---- 划分：测试集固定（沿用 two/test_symbols.txt），验证集 = 其余 30%（seed 12345）----
    if os.path.exists(a.test_symbols):
        want = {l.strip() for l in open(a.test_symbols, encoding='utf-8') if l.strip()}
        # 名单是 'EX:SYM' 全键。数据若是 --name-style sym 的裸符号，就按裸符号匹配
        # （那种命名下同名代码本来就分不开，见 gen_train_data 的说明）。
        _bare = {s.split(':')[-1] for s in want}
        test_keys = {k for k in keys if k in want or k in _bare}
        miss = sorted(want - set(_bare) - set(keys))
        print(f'[固定测试集] 沿用 {os.path.relpath(a.test_symbols, ROOT)}：'
              f'命中 {len(test_keys)} / 名单 {len(want)}', flush=True)
        if miss:
            print(f'  警告：名单里 {len(miss)} 只在数据里没有（已剔出测试集）：'
                  f'{miss[:8]}{" ..." if len(miss) > 8 else ""}', flush=True)
    else:
        # 首次运行（没有名单）：独立 Random + 固定种子抽，并落盘 —— 与 one 的做法一致。
        # ⚠️ 这样抽出来的测试股票【不在生成端的对齐网格上】，逐日截面 IC 会很稀疏；
        #    正常路径是先跑 gen_train_data.py（它会把对齐名单落盘成这份文件）。
        rng_test = random.Random(a.seed or 42)
        pool = keys[:]
        rng_test.shuffle(pool)
        n_test = min(889, max(1, len(pool) // 3))
        test_keys = set(pool[:n_test])
        with open(a.test_symbols, 'w', encoding='utf-8') as fp:
            fp.write('\n'.join(sorted(test_keys)))
        print(f'[首次运行] 无 {os.path.basename(a.test_symbols)} -> 按种子抽 {len(test_keys)} 只'
              f'并落盘（⚠️ 它们不在对齐网格上，逐日截面 IC 会稀疏）', flush=True)

    rest = [k for k in keys if k not in test_keys]
    rng_split = random.Random(a.val_seed)
    rng_split.shuffle(rest)
    val_keys = set(rest[:int(len(rest) * VAL_FRAC)])
    train_keys = set(rest) - val_keys

    tr = [s for s in samples if s.key in train_keys]
    va = [s for s in samples if s.key in val_keys]
    te = [s for s in samples if s.key in test_keys]
    if a.limit_train:
        tr = tr[:a.limit_train]
    if not tr or not va or not te:
        raise SystemExit(f'划分后有空集：训练 {len(tr)} / 验证 {len(va)} / 测试 {len(te)}')
    if a.val_max and len(va) > a.val_max:
        va = [va[i] for i in np.sort(np.random.RandomState(0).choice(len(va), a.val_max, replace=False))]
    print(f'  训练 {len(tr):,} 行 / 验证 {len(va):,} 行（{len(val_keys):,} 只）'
          f' / 测试 {len(te):,} 行（{len(test_keys):,} 只）', flush=True)

    y_mean, y_std, aux_mean, aux_std = estimate_stats(tr, a.stats_n, rng)
    print(f'  标签稳健尺度(IQR/1.349) = {np.round(y_std, 5).tolist()}  z 裁 ±{Z_CLIP}', flush=True)
    print(f'  aux  mean/std = {np.round(aux_mean, 3).tolist()} / {np.round(aux_std, 3).tolist()}',
          flush=True)

    tr_loader = DataLoader(H5Dataset(tr, y_mean, y_std, aux_mean, aux_std),
                           batch_size=a.batch, shuffle=True, drop_last=True,
                           num_workers=workers, persistent_workers=workers > 0)
    va_loader = DataLoader(H5Dataset(va, y_mean, y_std, aux_mean, aux_std),
                           batch_size=a.batch, shuffle=False, num_workers=workers,
                           persistent_workers=workers > 0)
    te_loader = DataLoader(H5Dataset(te, y_mean, y_std, aux_mean, aux_std),
                           batch_size=a.batch, shuffle=False, num_workers=workers,
                           persistent_workers=workers > 0)

    model = Nowcast([len(COLS), DAYS_INPUT]).to(device)     # 3 头（u9），与现有 two.pt 同构
    n_par = sum(p.numel() for p in model.parameters())
    print(f'  模型 Nowcast({N_HEADS} 头) 参数 {n_par:,}；每 epoch {len(tr) // a.batch:,} 步；'
          f'设备 {device}', flush=True)

    # ---- warm start（2026-09-27 补：以前 two/train.py 是每次从零训，而用户定的规格是
    #      "5 个模型都在已有模型基础上继续训练"，one/three/seven/zero 的 train.py 也都是
    #      "有 .pt 就接着训"）----
    # ⚠️ 载入后**必须把反标准化 affine 还原成恒等**。存盘点焼进去的 out_mean/out_scale 是
    #    "z 空间 -> 真实单位"的还原常数（two 实测 out_scale = 0.0116~0.148，即输出被预先缩了
    #    ~7~85 倍），而训练时的标签是 z 空间的 —— 带着 affine 训等于让模型从一个量级错的输出
    #    出发。这正是存盘那段注释说的"训练/验证的标签和损失都是 z 空间，带着 affine 会错"，
    #    只是以前这条只在【存盘后】做了、**加载时**漏了。
    if a.warm_start and os.path.exists(a.out):
        model = torch.load(a.out, map_location=device, weights_only=False).to(device)
        set_output_scale(model, np.zeros(N_HEADS, 'float32'), np.ones(N_HEADS, 'float32'))
        print('loaded existing TWO model:', a.out, flush=True)
    else:
        print(f'  warm start 未启用（--out {a.out} '
              f'{"不存在" if not os.path.exists(a.out) else "存在但 --no-warm-start"}）'
              f'，本次从零训', flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    best = -float('inf')                 # 存盘判据：验证 mean IC
    best_val_loss = float('inf')         # 只用于打印 "min val loss"（= 训练目标在验证集上的最优）
    for ep in range(a.epochs):
        model.train()
        t0 = time.time()
        run, n = 0.0, 0
        for inp, aux_t, y in tr_loader:
            inp = inp.to(device, non_blocking=True)
            aux_t = aux_t.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad()
            aux_p, pred = model(inp)
            loss = batch_loss(aux_p, pred, aux_t, y)
            loss.backward(); opt.step()
            run += loss.item(); n += 1
            if a.log_every and n % a.log_every == 0:
                # 与 one/train.py:296 一字不差的行格式（另外三个模型同）
                print("Epoch: %d, step: %d, train loss: %1.5f, mean loss: %1.5f, "
                      "min val loss: %1.5f" % (ep, n, loss.item(), run / n, best_val_loss),
                      flush=True)
        vp, vy, val_loss = predict(model, va_loader, device)
        r = eval_ic_overall(vp, vy)
        score = float(np.nanmean([v for v in r.values()]))
        best_val_loss = min(best_val_loss, val_loss)
        print("Epoch: %d, validate loss: %1.5f" % (ep, val_loss), flush=True)
        # two 特有的一行：这才是决定存哪一版的数（loss 只是体温计，见模块 docstring）
        print(f'         验证 mean IC={score:.4f}  逐头 {[round(float(v), 4) for v in r.values()]}  '
              f'({time.time() - t0:.0f}s)', flush=True)
        if score > best:
            best = score
            # 顺序必须是「装上 affine -> 存 -> 还原恒等」（two/nowcast/train.py:46-58 同）：
            # 装上存下来 -> .pt 直接输出真实单位（get_two_predict.py 就按这个写的）
            # 存完还原   -> 后面的训练/验证仍在 z 空间
            set_output_scale(model, y_mean, y_std)
            torch.save(model, a.out)
            set_output_scale(model, np.zeros(N_HEADS, 'float32'), np.ones(N_HEADS, 'float32'))
            print('best_val_ic:' + str(best) + ' saving model:' + a.out, flush=True)

    # ---- 测试集：整体 IC + 逐日截面 IC（测试股票在同步网格上，每个锚点有完整截面）----
    model = torch.load(a.out, map_location=device, weights_only=False).to(device)
    tp, ty, _ = predict(model, te_loader, device)
    dates = np.array([s.date for s in te])
    # 反标准化（.pt 已焼进 affine，预测就是真实单位）；这里把标签也还原，两者同量纲只是可读性
    ty = ty * y_std + y_mean
    r_ov = eval_ic_overall(tp, ty)
    r_d = eval_icir(tp, ty, dates, min_stocks=a.min_stocks)
    print(f'\n===== 测试集（{len(te):,} 行 / {len(set(dates))} 个锚点日 / {len(test_keys)} 只股票）=====')
    print(f'{"头":14s}{"整体IC":>10s}{"逐日IC":>10s}{"逐日ICIR":>10s}{"n日":>8s}')
    for name in HEAD_NAMES:
        ov = r_ov.get(name, np.nan)
        ic, icir, nn = r_d.get(name, (np.nan, np.nan, 0))
        print(f'{name:14s}{ov:>+10.4f}{ic:>+10.4f}{icir:>+10.2f}{nn:>8d}')
    _d_ic = np.array([v[0] for v in r_d.values()], dtype='f8')
    _o_ic = np.array([v for v in r_ov.values()], dtype='f8')
    print(f'{"平均":14s}{np.nanmean(_o_ic) if np.isfinite(_o_ic).any() else np.nan:>+10.4f}'
          f'{np.nanmean(_d_ic) if np.isfinite(_d_ic).any() else np.nan:>+10.4f}')
    if not np.isfinite(_d_ic).any():
        print(f'  ⚠️ 没有任何锚点日的样本数 ≥ {a.min_stocks}，逐日截面 IC 算不出来 —— '
              f'上面的逐日列为 NaN。全量数据下测试集是 889 只对齐股票、每个锚点日一整个截面，'
              f'不会这样；小规模试跑可以加 --min-stocks 5 看口径是否正常。')
    print(f'\n验证集 mean IC（最好 epoch）= {best:.4f}；产出 {a.out}（{n_par:,} 参数）')
    with open(os.path.splitext(a.out)[0] + '_metrics.json', 'w', encoding='utf-8') as fp:
        json.dump({'n_params': n_par, 'train': len(tr), 'val': len(va), 'test': len(te),
                   'val_mean_ic_best': best, 'test_ic_overall': r_ov,
                   'test_ic_daily': {k: v[0] for k, v in r_d.items()},
                   'test_icir_daily': {k: v[1] for k, v in r_d.items()},
                   'test_days': {k: v[2] for k, v in r_d.items()}}, fp, indent=1,
                  ensure_ascii=False, default=float)


if __name__ == '__main__':
    main()
