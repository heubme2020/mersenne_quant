"""nowcast 数据装载：memmap 分块文件 + 逐截面 batch 采样。

关键约束：损失与评估都是「同一锚点日、不同股票」的横截面，所以 batch 必须由同一天
的多个股票构成。分块文件里每个锚点日天然有约 K 个不同股票，因此这里先把
(块, 行) 按锚点日聚成全局索引，采样时先抽日期、再在该日的全体样本里抽 n 个。

股票级隔离：one 的 889 只 A 股测试集（one/test_symbols.txt）要从所有训练臂里排除。
"""

import glob
import json
import os

import numpy as np

MARKETS = {
    'CN': ['SHZ', 'SHH'],
    'US': ['NASDAQ', 'NYSE', 'AMEX'],
    'CA': ['TSX', 'TSXV', 'CNQ'],
    'JP': ['JPX'], 'HK': ['HKSE'], 'UK': ['LSE'], 'DE': ['XETRA'],
    'KR': ['KSC', 'KOE'], 'TW': ['TAI', 'TWO'],
    'AU': ['ASX'], 'IN': ['BSE', 'NSE'], 'ID': ['JKT'], 'MY': ['KLS'],
    'NO': ['OSL'], 'FR': ['PAR'], 'BR': ['SAO'], 'SA': ['SAU'],
    'SG': ['SES'], 'TH': ['SET'], 'CH': ['SIX'], 'SE': ['STO'],
    'IL': ['TLV'], 'PL': ['WSE'],
}
GLOBAL_EX = [e for k, v in MARKETS.items() if k != 'CN' for e in v]
ASHARE_EX = MARKETS['CN']
N_LABELS = 9
# 每块里的三个标签数组：level 标签 / 朴素持续性基准 / 残差(= y − b)
LABEL_KINDS = ('level', 'resid', 'bench')


class Store:
    """一组分块文件 + 按锚点日聚好的全局索引（memmap，不把数据读进内存）。"""

    def __init__(self, root, exchanges, min_per_date=64, label='level', suffix=''):
        # label : 'level' 用 _y<suffix>.npy（水平标签）| 'resid' 用 _r<suffix>.npy
        #         | 'bench' 用 _b<suffix>.npy（朴素基准，不训练，只用于对照）
        # suffix: 标签变体（见 variants.py）。'' = 基线；'v1'/'v2' = 头 1/头 2 的对照实验。
        #         **同一份 X 共用多套标签**，所以不必复制 19 GB 特征。
        self.files, self.ex_of_file, self.symnames = [], [], {}
        self._label = label if label in LABEL_KINDS else 'level'
        self.suffix = suffix
        dates, symidx, labok = [], [], []
        for ex in exchanges:
            ex_dir = os.path.join(root, ex)
            if not os.path.isdir(ex_dir):
                continue
            self.symnames[ex] = json.load(open(os.path.join(ex_dir, 'symbols.json')))
            for xf in sorted(glob.glob(os.path.join(ex_dir, 'chunk_*_X.npy'))):
                base = xf[:-6]
                self.files.append({
                    'x': xf,
                    'level': base + f'_y{suffix}.npy',
                    'bench': base + f'_b{suffix}.npy',
                    'resid': base + f'_r{suffix}.npy',
                    'd': base + '_d.npy', 's': base + '_s.npy'})
                self.ex_of_file.append(ex)
                dates.append(np.load(base + '_d.npy'))
                symidx.append(np.load(base + '_s.npy'))
                # 三套标签（level/bench/resid）必须都有限才能用：丢掉行数极少
                # （A 股实测 166 / 156,517 ≈ 0.1%），但保证三种口径共用同一批样本、
                # 且 y/b/r 的 NaN 不会污染损失或 z-score。
                y = np.load(base + f'_y{suffix}.npy')
                b = np.load(base + f'_b{suffix}.npy')
                r = np.load(base + f'_r{suffix}.npy')
                labok.append(np.isfinite(y).all(1) & np.isfinite(b).all(1)
                             & np.isfinite(r).all(1))
        self.ex_of_file = np.array(self.ex_of_file)

        dates = np.concatenate(dates)
        symidx = np.concatenate(symidx)
        labok = np.concatenate(labok)
        self.n_rows = [len(np.load(f['d'])) for f in self.files]
        offsets = np.r_[0, np.cumsum(self.n_rows)]

        order = np.argsort(dates, kind='stable')
        order = order[labok[order]]          # 丢掉标签非有限的行
        self.dates = dates[order]
        self.row_file = (np.searchsorted(offsets, order, side='right') - 1).astype('int32')
        self.row_idx = (order - offsets[self.row_file]).astype('int32')
        self.row_sym = symidx[order].astype('int32')

        uniq, start = np.unique(self.dates, return_index=True)
        cnt = np.diff(np.r_[start, len(self.dates)])
        keep = cnt >= min_per_date
        self.date_list = uniq[keep]
        self.date_slice = {int(d): (int(a), int(b)) for d, a, b in
                           zip(uniq[keep], start[keep], (start + cnt)[keep])}
        self._mm = {}

    # ---- I/O ----
    def mm(self, fi):
        """该块的全部 memmap：X + 三套标签。一起缓存更省句柄。"""
        m = self._mm.get(fi)
        if m is None:
            f = self.files[fi]
            m = {k: np.load(f[k], mmap_mode='r')
                 for k in ('x', 'level', 'bench', 'resid')}
            self._mm[fi] = m
        return m

    def take(self, pos):
        """按全局行位置取样本 -> (X, y, symbols)。y 按 self._label 口径。"""
        fi, ri = self.row_file[pos], self.row_idx[pos]
        k = self._label
        X = np.stack([self.mm(f)['x'][r] for f, r in zip(fi, ri)])
        Y = np.stack([self.mm(f)[k][r] for f, r in zip(fi, ri)])
        syms = [f'{self.ex_of_file[f]}:{self.symnames[self.ex_of_file[f]][s]}'
                for f, s in zip(fi, self.row_sym[pos])]
        return X, Y, syms

    def take_x(self, pos):
        """只取 X（前向/统计用），不读 y。"""
        fi, ri = self.row_file[pos], self.row_idx[pos]
        return np.stack([self.mm(f)['x'][r] for f, r in zip(fi, ri)])

    def _take(self, pos, kind):
        fi, ri = self.row_file[pos], self.row_idx[pos]
        return np.stack([self.mm(f)[kind][r] for f, r in zip(fi, ri)])

    def take_y(self, pos):
        """只取 y（评估用），按 self._label 口径。

        验证/测试集有 3 万行，走 take() 会把 X 物化一遍（4.18 GB，np.stack 后
        峰值 8.36 GB）——而算 IC/ICIR 只需要标签。这是之前 OOM 被系统杀掉的主因。
        """
        return self._take(pos, self._label)

    def take_level(self, pos):
        """只取 level 标签（基准对照用，与 self._label 无关）。"""
        return self._take(pos, 'level')

    def take_bench(self, pos):
        """只取朴素持续性基准（对照用，不训练）。"""
        return self._take(pos, 'bench')

    def release_mmaps(self):
        """丢掉缓存的 memmap 句柄，让 OS 回收这些文件页。

        必须定期调用：memmap 随机读过的页会留在进程工作集里，RSS 随读取量线性
        上涨。A 股训练集只有 9.7 GB 所以撑得住；global 臂训练集 97 GB，单个 epoch
        就能把 RSS 顶到 100 GB 以上（机器只有 68 GB 内存）→ 被系统 OOM 杀掉。
        这里只是丢引用，由 GC 关闭 mmap；仍有引用的对象不会被回收，所以是安全的。
        """
        self._mm.clear()

    def batch(self, date, n, rng):
        """该锚点日抽 n 个样本（不同股票）。"""
        a, b = self.date_slice[date]
        m = b - a
        idx = a + rng.choice(m, n, replace=(m < n))
        return self.take(idx)

    def random_batch(self, pos_pool, n, rng):
        """从给定行池里随机抽 n 个样本（可自由混合日期与股票）。

        逐样本损失用（smooth_l1/MSE）——**不要求同一天**，所以不需要按日期组 batch。
        注意用 `rng.choice` 在大池子上是 O(len(pool))，训练循环里应该改成
        先 `rng.permutation(pos_pool)` 再切片（见 train.py）。
        """
        idx = rng.choice(pos_pool, min(n, len(pos_pool)), replace=False)
        return self.take(idx)

    def symbol_of(self, pos):
        return [f'{self.ex_of_file[f]}:{self.symnames[self.ex_of_file[f]][s]}'
                for f, s in zip(self.row_file[pos], self.row_sym[pos])]

    def stats(self):
        return dict(rows=len(self.dates), files=len(self.files),
                    dates=len(self.date_list))
