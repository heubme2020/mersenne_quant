"""two 的训练数据：**一步**生成 h5（特征 X + u9 标签），不依赖 nowcast/。

    cd two
    python gen_train_data.py        # 默认：31 个交易所 -> two/train/*.h5

## 这个脚本是原来【两步】的合并

原线（nowcast）：

    ① two/nowcast/gen_data2.py      从 data/{EX}/*.csv 造 X（889 天日线窗口 -> 1017×31）+
                                基础标签，写 chunk_*.npy 分块；**同时决定 (symbol, 锚点) 集合**
    ② two/nowcast/add_u6_labels.py  的 relabel() 事后**覆写** y/b/r，换成 u9 的 meddiff 标签

现在一次做完，落盘从「分块 npy」改成「每样本一个 h5」，与 one/three/seven/zero 对齐。

## h5 格式（`key='data'` 与 one 完全相同）

    key='data'   (1017, 31) float32 DataFrame，列名 = `two_features.build_x` 的 COLS
                 （与 one/train/*.h5 一样，`pd.read_hdf(path, key='data')` 直接可用）
    key='label'  (1, 3) DataFrame：gpMedP3F / revMedP3F / taMedP3F
                 （三个头的 u9 标签，列名同 variants.py 的 u9 段）

`key='label'` 是 two **唯一**比另外四个模型多的东西，而且是必须的：one 的标签能从 1017 行里的
close 现算，zero/three/seven 的标签来自"只装财务数据"的表，而 two 的 u9 标签要用**财报面板**
（收入表/资产负债表 + 过去 3 季 / 前向 7 季窗口）—— 日线窗口里没有这些列，只能落盘。

2026-09-27 之前 `key='label'` 是 (1, 7)，多写了 `b_*`（朴素基准，meddiff 口径下恒 0）和
`ok`（恒 True，因为 ok=False 的锚点在下面就被丢掉了）两列常量，现在删了。
其余四个模型的 h5 里没有这种常量列，删掉既是省磁盘也是格式对齐。见 `two_labels.label_frame`。

文件名：`{EX}_{SYM}_{锚点日}.h5`（**带交易所前缀**，用 `--name-style sym` 可改成裸符号）。

### 为什么默认带前缀（实测数据）

`two` 是 31 个交易所**混在一个扁平目录**里的（one/three/seven/zero 的 train/ 也长这样）。
全量扫了 31 个 `data/{EX}/daily_*.csv`：**109 个代码在两个以上交易所重名**（223 个 (EX,SYM) 组合），
例如 `CNL` 同时在 AMEX/NASDAQ/NYSE，`ARE`/`M` 同时在 BSE/NSE/NYSE，`L`/`S` 同时在 NSE/NYSE/SET。
裸文件名下，同名代码**同一天**的样本会互相覆盖 —— 静默丢数据、还把两个不同市场的股票混成一个
（用户已实测到 zero/train 有 4,395 个文件对不上，很可能就是这个原因）。

`--name-style sym` 保留"与其它 4 个模型文件名一致"的选择，但**同名代码会互相覆盖**，
只在确定不撞号的场合用。

`key='data'` 的**内部格式**与 one 逐字相同（列名/列序/形状/dtype），这一条不受命名影响。

## 采样（`sym_samples` 逐字照搬 nowcast/gen_data2.sym_samples）

* **测试股票**（`two/test_symbols.txt` 里的 'EX:SYM'）-> **全局同步网格**：以 `--grid-exchange`
  （默认 SHZ）的交易日历为准，从 2004-01-01 起每 `--key-step`（默认 31）个交易日取一个锚点，
  **所有测试股票在同一天**都有锚点 -> 这才算得出「逐日截面 IC」。
* **其余股票** -> **每股随机采样**，概率 `--train-rate`（默认 1/31）；种子 = `crc32(symbol)`，
  **不用内置 hash()**（Python 字符串 hash 每进程随机化，结果不可复现）。

## (symbol, 锚点) 集合与旧线【逐行可比】

锚点是否被采用，由两个门决定（与旧线完全一致，机制见 `two_labels.py` 的 docstring）：

    1. `ok` 门：锚点日前最后一个【已披露】季度上，u1（yoy_delta）的标签必须有限
       —— 旧线里是 `gen_data2.get_labels` 的 labok，ok=False 的锚点整段跳过
    2. 标签门：u9 标签必须有限 —— 旧线里是 `two/nowcast/data.py` 的 Store 对 y/b/r 的有限性过滤

两个门都在**生成端**执行，所以落盘的都是训练真正会用的行：

    本脚本产出的行集 == 旧线 gen_data2 写出的行集 ∩ Store 的 labok
    （第 2 个门在旧线里是训练时才执行的，实测 SHZ 丢掉 10.6% 的行）

日志/meta 里会同时报「被标签门丢掉的行数」，所以「本脚本产出 + 丢掉 == 旧线 gen_data2 写出的行数」
仍然可以逐位核对（见 `gen_train_data` 的 `_gen_meta.json`）。

## ⚠️ 四个坑（照做，别"优化"掉）

1. **不要从 nowcast 导入任何东西**。`two/` 在 `sys.path` 最前，裸 `from gen_train_data import ...`
   会命中 two 自己这份；one/ 下的模块一律走 `two_features` 的**显式路径导入**。
2. **Windows 控制台默认 GBK**：print 里放 emoji/非 GBK 字符会让脚本在**干完所有活之后**崩、
   退出码 1（2026-09-26 `zero/gen_train_data.py` 就这么"失败"过）。开头 reconfigure + 不放 emoji。
3. **内存：日线 CSV 必须分块读**（见 `load_daily` 的 docstring）。旧写法每个 worker 都整表读
   一遍，12 个 worker 同时持有 0.34GB(TAI)/0.96GB(NASDAQ)/1.4GB(NYSE) 的 DataFrame，
   父进程还有一份 —— 2026-09-27 跑到 TAI 时，一个 worker 申请 **40MB** 都失败、整条流水线
   崩在第 25 个交易所（前 24 个交易所的成果都在，没丢）。
4. **磁盘**：每样本一个 h5（≈141 KB，与 one/train 同尺寸）。**实测：跑完 24 个所
   = 1,697,296 个文件 = 234GB**（不是早期 docstring 里估的 68 万/93GB —— 那是按 1/31
   随机采样估的，实际 31 个交易所的股票总数远大于此）。剩下 6 个所按数据量外推约 +40~60GB
   （估的，不是实测）。C 盘 1.7TB 可用，够。验证/调试请用 `--exchanges JKT` 这种小交易所
   （实测 3.39 万样本 = 4.95GB / 340s）。

## 输出（2026-09-27 起与另外四个模型逐行对齐）

zero/one/three/seven 的 gen 都是这套措辞，two 以前是一串中文块进度，现在改成一样的：

    --- Starting processing for {EX} ---
    Loading {EX} data...
    Starting {workers} processes for {N} symbols.
    Processing {EX} symbols: 100%|██████████| 12/12 [124s<00:00]
    Finished {EX}. Total files created: 17124
    --- Summary ---
    Total HDF5 files created: N
    All exchanges processed successfully! 🎉

two 特有的东西（全局网格、对齐股票数、被标签门丢掉的锚点数）挂在 tqdm 的 postfix 上，
以及落进 `_gen_meta.json`，不再各占一行。

## 清空（2026-09-27 与 zero/one/three/seven 对齐）

和其它四个模型一样：**每次运行先无条件清空 train/**，再从头生成。

    python two/gen_train_data.py                    # 全量：清空 -> 重生成 31 个所
    python two/gen_train_data.py --exchanges JKT     # 也是先清空！train/ 会被清空，只重生成 JKT
    ... --no-clean                                   # 逃生阀：只增不删（默认关闭）

清空与否写在开头那一行计划里（`WIPE existing files` / `keep existing files (union)`）——
它是整个脚本唯一**破坏性**的一步，另外四个模型从来不告诉你要删什么。

**为什么要无条件清空**：采样参数（`--key-step` / `--train-rate` / `--grid-exchange`）或标签门的
代码一改，旧行会原地留着、与新行混在同一个目录里 —— 不报错、也不掉文件，只是训练集里同时
存在两套采样口径、静默变脏。zero/one/three/seven 的 gen 都是无条件 `rmtree`，天然免疫。

⚠️ **`--exchanges` 不是补跑开关**（这一点与另外四个模型一致）：它只决定"生成哪些所"，
不影响"先清空"。所以补跑断掉的那几个所**必须加 `--no-clean`**，否则前 24 个所的 169 万个文件
会被删掉：

    python two/gen_train_data.py --exchanges TAI TWO TSX TSXV WSE XETRA --no-clean

用法：
    python two/gen_train_data.py                             # 全量（24 所实测 234GB，全量估 ≈280GB）
    python two/gen_train_data.py --exchanges JKT             # 单交易所
    python two/gen_train_data.py --exchanges TAI TWO TSX --no-clean   # 补跑（不清空）
"""
import argparse
import json
import multiprocessing
import os
import shutil
import sys
import time
import zlib

import numpy as np
import pandas as pd
from tqdm import tqdm

if os.name == 'nt':                 # Windows 多进程必须
    multiprocessing.freeze_support()

# 坑 2：先修 stdout/stderr 编码，再干别的（否则活儿干完了才崩）
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
DATA = os.path.join(ROOT, 'data')

# 坑 1：只能从这里拿特征（two_features 内部用显式路径导入 one/ 的模块）
from two_features import build_x, COLS, RAW, WINDOW, DAYS_INPUT, AUX_DAYS, REF_IDX  # noqa: E402
import two_labels as L                                                             # noqa: E402

ALIGNED_FILE = os.path.join(HERE, 'test_symbols.txt')
# 兜底名单（只在 two/test_symbols.txt 缺失时用一次，随即落盘到 two/，之后不再依赖 nowcast）
NOWCAST_ALIGNED = os.path.join(ROOT, 'two', 'nowcast', 'test_symbols_all31_889.txt')

_panel_cache = {}
CHUNK_ROWS = 500_000        # 日线分块读的行数（≈30MB/块，见 load_daily）


def daily_path(exchange):
    return os.path.join(DATA, exchange.upper(), f'daily_{exchange.lower()}.csv')


def daily_cols():
    """列集与 nowcast/gen_data2.load_daily 一致：symbol/date + RAW 前 5 个（delta 是算出来的）。"""
    return ['symbol', 'date'] + RAW[:5]


def symbol_list(exchange):
    """该交易所的股票名单（只读 symbol 一列）。

    父进程只要名单。以前走 `load_daily(ex).symbol.unique()` —— 那等于在父进程里把
    NYSE 的 1.4GB 日线表整个建出来（见 load_daily 的 docstring），纯浪费。
    """
    syms, seen = [], set()
    for ch in pd.read_csv(daily_path(exchange), usecols=['symbol'], chunksize=CHUNK_ROWS):
        for s in ch['symbol'].dropna().astype(str).unique():
            if s not in seen:
                seen.add(s)
                syms.append(s)
    return syms


def load_daily(exchange, symbols=None):
    """该交易所的日线（列集 / 2000 年过滤 / 排序与 nowcast/gen_data2.load_daily 一致）。

    `symbols=None` 读全表；给了集合就**只留这些股票**，且分块读。

    ## 为什么必须分块（2026-09-27 的一次 OOM，见模块 docstring 坑 3）

    整表读进 pandas 的实测内存：TAI 0.34GB / SHZ 0.59GB / NASDAQ 0.96GB —— 比 CSV 本身
    大得多（7 列里 6 列 float64 + 一个 object 的 symbol）。而**每个 worker 都要读整个
    交易所的日线**（它只负责其中一块股票），12 个 worker 同时就是 4~12GB，父进程还有一份。
    TAI 那次就是在一个 worker 给 symbol 列做 `.copy()` 时申请 **40MB** 失败、整条流水线崩掉。

    分块读把每个 worker 的峰值从「整表」降到「一块」（50 万行 ≈30MB）。顺带还快：
    股票名单是按【文件里的顺序】切块发给 worker 的（`main` 里用 `symbol_list` 的顺序），
    而日线 CSV 按 symbol 连续分块排序，所以每个 worker 通常读到文件的 1/12 处就集齐了
    自己那批股票、直接停（`seen >= want`）。**这条早停不依赖文件排序**：它只要求
    "我想要的股票都已经见过了"，所以在任何交易所的文件上都不会漏数据。
    实测 TAI：单块 0.5s / 峰值 +34MB，整表 3.6s / 峰值 +338MB（6.8 倍）。

    ## 分块读会不会让 dtype 变？（pandas 的老坑，实测确认不会）

    `read_csv(chunksize=)` 是**每块各自推断 dtype** 的：纯数字的块可能是 int64，含字母的块
    是 object，同一个文件里前后不一致。symbol 列一旦变成 float（如 7203.0），
    `astype(str)` 就会给出 '7203.0'，与 `want` 里的 '7203' 对不上 -> **静默丢股票**。
    2026-09-27 把 31 个交易所的 symbol 列全读了一遍：dtype 全是 object、
    数字形式的代码占比 0.0000（NASDAQ 有 1057 个空 symbol，其余 30 个所 0 个）
    —— 也就是说所有代码都带非数字字符（'.TW'/'.T'/'.SZ'…），parser 在任何块里都只会
    推成 object，所以分块读与原整表读**逐位等价**（已用 CNQ/SES/TAI 三个所
    `assert_frame_equal` 验过，含 dtype）。

    ⚠️ 换数据源时若出现"纯数字股票代码"，这条就不成立了 —— 那时要给
    `read_csv` 加 `dtype={'symbol': str}`（两处：这里和 `symbol_list`）。
    """
    cols = daily_cols()
    path = daily_path(exchange)
    if symbols is None:
        d = pd.read_csv(path, usecols=cols)
    else:
        want = set(symbols)
        parts, seen = [], set()
        for ch in pd.read_csv(path, usecols=cols, chunksize=CHUNK_ROWS):
            ch = ch[ch.symbol.notna()]
            hit = ch[ch.symbol.astype(str).isin(want)]
            if len(hit):
                parts.append(hit)
                seen |= set(hit.symbol.astype(str).unique())
                if seen >= want:
                    break
        d = pd.concat(parts) if parts else pd.DataFrame(columns=cols)
    d = d[d.symbol.notna() & (d.date >= 20000101)].copy()
    d['symbol'] = d.symbol.astype(str)
    return d.sort_values(['symbol', 'date'])


def panel_of(exchange):
    """该交易所的面板 + 打包（进程内缓存，避免每个 chunk 重算）。"""
    p = _panel_cache.get(exchange)
    if p is None:
        p = L.pack(L.load_panel(exchange))
        _panel_cache[exchange] = p
    return p


def sym_samples(sym, daily, packed, aligned, grid=None, rate=None, seed=0):
    """该股票的样本 -> [(anchor_date, X, y, b, ok), ...]。

    采样部分（网格/随机、两个门）逐字复制自 `nowcast/gen_data2.sym_samples`；
    唯一改动是「拿到 X 和标签之后就地返回」，而不是塞进分块数组。
    """
    dates = daily['date'].values
    if len(daily) < WINDOW or packed is None:
        return []
    avail, ended, okq, rows, Y, B = packed
    lo, hi = REF_IDX, len(dates) - AUX_DAYS - 1
    if hi < lo:
        return []

    if aligned:
        pos = np.searchsorted(dates, grid, side='right') - 1     # 每格 -> 该股 ≤ 格的最后交易日
        cand = np.unique(pos[(pos >= lo) & (pos <= hi)])
    else:
        rs = np.random.RandomState(seed)
        n = hi - lo + 1
        cand = (lo + np.where(rs.random_sample(n) < rate)[0]).astype(np.int64)
    if len(cand) == 0:
        return []

    out = []
    for a in cand:
        t = int(dates[a])
        k = np.searchsorted(avail, t, side='right') - 1          # 最后【已披露】季度 -> ok 门
        if k < 0 or not okq[k]:
            # ok=False：该季度的标签不存在（未来窗口超出数据）-> 这个锚点没有样本。
            # **不能**退回更早的季度 —— 那会造出「未来窗口已实现」的假标签。
            continue
        pj = np.searchsorted(ended, t, side='right') - 1         # jc = 最后【已结束】季度 -> 标签基准
        if pj < 0:
            continue
        w = daily.iloc[a - REF_IDX:a + AUX_DAYS + 1]
        if len(w) != WINDOW:
            continue
        X = build_x(w.reset_index(drop=True))
        if X is None:
            continue
        gi = rows[pj]
        y, b = Y[gi], B[gi]
        out.append((t, X, y, b, bool(np.isfinite(y).all())))
    return out


def write_h5(path, X, y):
    """一个样本 -> 一个 h5（key='data' 与 one 同格式；key='label' 见模块 docstring）。"""
    with pd.HDFStore(path, mode='w') as st:
        st.put('data', pd.DataFrame(X, columns=COLS), format='fixed')
        st.put('label', L.label_frame(y), format='fixed')


def fname(style, ex, sym, date):
    """h5 文件名。ex_sym = `{EX}_{SYM}_{日期}.h5`（默认，见模块 docstring 的撞号实测）。"""
    return f'{ex}_{sym}_{date}.h5' if style == 'ex_sym' else f'{sym}_{date}.h5'


def chunk_worker(task):
    ex, syms, out_dir, aligned, grid, rate, seed, style = task
    # 只读自己这块股票的日线（分块读，见 load_daily 的 docstring）
    d = load_daily(ex, syms)
    daily_map = {s: g.reset_index(drop=True) for s, g in d.groupby('symbol', sort=False)}
    del d
    packed = panel_of(ex)
    n = n_skip = 0
    for s in syms:
        got = sym_samples(s, daily_map.get(s), packed.get(s),
                          aligned=f'{ex}:{s}' in aligned, grid=grid, rate=rate,
                          seed=seed + zlib.crc32(s.encode()))
        # `_b` = 朴素基准（meddiff 口径下恒 0）：只跟着 ok 门走到这里，不再落盘
        # （以前写成 h5 里的 b_* 列，2026-09-27 删掉，见 two_labels.label_frame）。
        for t, X, y, _b, ok in got:
            if not ok:
                # u9 标签的窗口不完整（未来 7 季 / 过去 3 季）—— 旧线里由 Store 的 labok 丢掉，
                # 这里直接在生成端丢掉：省一份磁盘（实测 SAU 3.6% / JKT 4.6% / SHZ 10.6%），
                # 也省掉训练端"先把所有 h5 的 ok 列扫一遍"的开销（几十万次读盘）。
                n_skip += 1
                continue
            write_h5(os.path.join(out_dir, fname(style, ex, s, t)), X, y)
            n += 1
    return ex, len(syms), n, n_skip


def load_aligned(path):
    """读对齐名单（'EX:SYM' 全键）。缺失时用 nowcast 的名单兜底并落到 path（只此一次）。"""
    if not os.path.exists(path):
        if os.path.exists(NOWCAST_ALIGNED):
            print(f'[对齐名单] {os.path.relpath(path, ROOT)} 不存在 -> 从 '
                  f'{os.path.relpath(NOWCAST_ALIGNED, ROOT)} 取一次并落盘（之后 two/ 自足）',
                  flush=True)
            shutil.copyfile(NOWCAST_ALIGNED, path)
        else:
            raise SystemExit(f'错误：对齐名单 {path} 不存在，且兜底名单 '
                             f'{NOWCAST_ALIGNED} 也不存在。\n'
                             f'（测试股票必须走全局同步网格，缺它就算不出逐日截面 IC）')
    out = set()
    for line in open(path, encoding='utf-8'):
        s = line.strip()
        if s:
            out.add(s)
    return out


def main():
    ap = argparse.ArgumentParser(description='two 训练数据（h5：key=data 特征 + key=label u9 标签）')
    ap.add_argument('--exchanges', nargs='+', default=None,
                    help='要生成哪些交易所（默认 data/ 下的全部子目录）')
    ap.add_argument('--out', default=os.path.join(HERE, 'train'), help='输出目录（默认 two/train）')
    ap.add_argument('--aligned-symbols', default=ALIGNED_FILE,
                    help="走全局网格的股票（'EX:SYM'，默认 two/test_symbols.txt = 测试集）")
    ap.add_argument('--key-step', type=int, default=31, help='对齐网格步长（交易日）')
    ap.add_argument('--train-rate', type=float, default=1.0 / 31.0, help='其余股票的随机采样率')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--grid-exchange', default='SHZ', help='全局网格的交易日历基准')
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--chunk', type=int, default=0,
                    help='每个进程一次处理多少只股票（0=自动：按股票数均分给 workers，'
                         '小交易所也能占满进程数）')
    ap.add_argument('--name-style', choices=['ex_sym', 'sym'], default='ex_sym',
                    help='文件名：ex_sym={EX}_{SYM}_{日期}.h5（默认，避免跨交易所同名覆盖）'
                         '| sym={SYM}_{日期}.h5（与其它 4 个模型一致，但同名会互相覆盖）')
    ap.add_argument('--no-clean', action='store_true',
                    help='不清空输出目录（默认：每次运行都先清空 train/，与另外四个模型一致）')
    ap.add_argument('--dry-run', action='store_true',
                    help='只打印计划（交易所/网格/对齐股票数/输出目录）就退出，不生成数据')
    a = ap.parse_args()

    exams = a.exchanges or sorted(d for d in os.listdir(DATA) if os.path.isdir(os.path.join(DATA, d)))
    aligned = load_aligned(a.aligned_symbols)

    # 全局网格：以参考交易所（默认 SHZ）的交易日历为准，每 key_step 取一格。
    # A 股日历做基准 -> A 股测试股天然对齐；其它交易所取「≤ 格日的最后交易日」。
    gd = pd.read_csv(os.path.join(DATA, a.grid_exchange.upper(),
                                  f'daily_{a.grid_exchange.lower()}.csv'), usecols=['date'])
    gdates = np.sort(gd['date'].unique())
    grid = gdates[gdates >= 20040101][::a.key_step]
    # 清空规则：**无条件清空**（与 zero/one/three/seven 的 gen 逐字一致，2026-09-27 对齐）。
    # 为什么必须清空：采样参数（--key-step / --train-rate / --grid-exchange）或标签门的代码一改，
    #   旧行会原地留着与新行混在一起 —— 不报错也不掉文件，只是训练集里同时存在两套口径、静默变脏。
    # --no-clean 是唯一的例外（逃生阀，默认关闭）：补跑时用，否则 --exchanges 只挑了几个所、
    #   却会把已经跑完的所一起删掉。
    wipe = not a.no_clean
    # 计划信息压成一行；进度细节挂 tqdm 的 postfix，历史细节进 _gen_meta.json
    # （wipe 写进这一行，是因为它是**破坏性**的那一步 —— 另外四个模型从不告诉你要删什么）
    print(f'{len(exams)} exchanges -> {a.out} ({a.name_style}) | '
          f'{"WIPE existing files" if wipe else "keep existing files (union)"} | aligned '
          f'{os.path.relpath(a.aligned_symbols, ROOT)} = {len(aligned):,} symbols | global grid '
          f'{len(grid)} anchors {grid[0]}->{grid[-1]} step {a.key_step} ({a.grid_exchange})',
          flush=True)
    # --dry-run 要在**动磁盘之前**返回。这条以前写错过：当时清空那一段排在它前面，
    # 于是 `--dry-run`（"只看看计划"）会先把 train/ 整个删掉再退出。
    if a.dry_run:
        print('(--dry-run: plan only, nothing generated)', flush=True)
        return

    if wipe:
        # 与 zero/one/three/seven 的 gen 一字不差的三行（ignore_errors：目录不存在时静默）
        print("Clearing and creating 'train' folder...", flush=True)
        shutil.rmtree(a.out, ignore_errors=True)
    elif os.path.isdir(a.out) and os.listdir(a.out):
        print(f'⚠️ 输出目录 {a.out} 非空 —— 本次只增不删（--no-clean；补跑口径）', flush=True)
    os.makedirs(a.out, exist_ok=True)

    workers = a.workers or max(1, min(12, (os.cpu_count() or 4) - 4))
    meta, failed = {}, []
    t_all = time.time()
    for ex in exams:
        # 每个交易所独立 try：一个所（或它下面某个 worker）出事，不该把前面 20 个所的成果
        # 一起带走 —— 2026-09-27 那次 OOM 就是这么整条流水线崩在 TAI 上的：前 24 个所的 h5
        # 都在盘上，但脚本非 0 退出、后面的 6 个所一个没跑。另外四个模型的 gen 也是这个结构。
        total = skip = 0
        try:
            print(f'\n--- Starting processing for {ex} ---', flush=True)
            print(f'Loading {ex} data...', flush=True)
            # 父进程只取股票名单（单列分块读）；**按【文件里的顺序】切块**，worker 才能靠
            # "凑齐了就停"少读几倍的 CSV —— 见 load_daily 的 docstring。
            syms = symbol_list(ex)
            n_al = sum(1 for s in syms if f'{ex}:{s}' in aligned)
            # 默认按股票数均分成 ≈workers 块：块太大会让小交易所只用上一两个进程
            # （SAU 394 只分 2 块 -> 543s；均分成 12 块就快得多）。
            chunk = a.chunk or max(1, -(-len(syms) // workers))
            print(f'Starting {workers} processes for {len(syms)} symbols.', flush=True)
            chunks = [syms[i:i + chunk] for i in range(0, len(syms), chunk)]
            tasks = [(ex, c, a.out, aligned, grid, a.train_rate, a.seed, a.name_style)
                     for c in chunks]
            t0 = time.time()
            with multiprocessing.Pool(workers) as pool:
                bar = tqdm(pool.imap_unordered(chunk_worker, tasks), total=len(tasks),
                           desc=f'Processing {ex} symbols')
                for _, ns, n, nsk in bar:
                    total += n; skip += nsk
                    bar.set_postfix(files=f'{total:,}', skipped=f'{skip:,}')
            print(f'Finished {ex}. Total files created: {total}', flush=True)
            meta[ex] = {'symbols': len(syms), 'aligned': n_al, 'samples': total,
                        'label_gate_skipped': skip, 'gated_total': total + skip,
                        'seconds': round(time.time() - t0)}
        except Exception as e:
            print(f'FATAL error for exchange {ex}: {e}', flush=True)
            failed.append(ex)
            meta.setdefault(ex, {'symbols': 0, 'aligned': 0, 'samples': total,
                                 'label_gate_skipped': skip, 'gated_total': total + skip,
                                 'error': f'{type(e).__name__}: {e}'})

    # _gen_meta.json **累加**写：补跑（`--exchanges TAI TWO ...`）时不能把已经跑过的所冲掉 ——
    # 2026-09-27 那次崩溃后，全靠这个文件才能核对"哪些所跑完了、各产出了多少"。
    meta_path = os.path.join(a.out, '_gen_meta.json')
    hist = {}
    if os.path.exists(meta_path):
        try:
            hist = json.load(open(meta_path, encoding='utf-8')).get('exchanges', {})
        except Exception as e:
            print(f'⚠️ {os.path.basename(meta_path)} 读不出来（{e}），本次按空历史写', flush=True)
    merged = {**hist, **meta}
    # `or 0`：历史记录里的计数可能是 None（那 24 个所的 _gen_meta.json 是崩后才按文件名补记的，
    # 只有 samples 数得出来，标签门的数只在 2026-09-27 的终端日志里）。
    run_tot = sum(v.get('samples') or 0 for v in meta.values())
    tot = sum(v.get('samples') or 0 for v in merged.values())
    gated = sum(v.get('gated_total') or 0 for v in merged.values())
    run_gated = sum(v.get('gated_total') or 0 for v in meta.values())
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({'exchanges': merged, 'grid': grid.tolist(), 'key_step': a.key_step,
                   'train_rate': a.train_rate, 'seed': a.seed,
                   'aligned_symbols': os.path.abspath(a.aligned_symbols),
                   'grid_exchange': a.grid_exchange, 'name_style': a.name_style,
                   'total_samples': tot, 'total_gated': gated,
                   'this_run': sorted(meta)}, f, indent=1)

    print('\n--- Summary ---', flush=True)
    print(f'Total HDF5 files created: {run_tot:,}', flush=True)
    print(f'Label-gate skipped: {run_gated - run_tot:,} (not written; per-exchange detail in '
          f'{os.path.basename(meta_path)})', flush=True)
    if hist:
        print(f'Accumulated on disk: {tot:,} files / {len(merged)} exchanges '
              f'({len(meta)} in this run)', flush=True)
    if failed:
        print(f'Failed to process exchanges: {failed} ⚠️', flush=True)
    else:
        print('All exchanges processed successfully! 🎉', flush=True)
    print(f'This run: {len(meta)} exchanges in {time.time() - t_all:.0f}s '
          f'-> {a.out}', flush=True)


if __name__ == '__main__':
    main()
