"""一键重训 5 个模型（one / two / three / seven / zero）：数据 → 训练 → 部署 → 记指标。

## 本文件是三个脚本合并而成（2026-09-28）

    full_retrain.py    数据编排：MySQL 导出 CSV → 重生各模型 h5 →（可选）nowcast 分块
    retrain_all.py     训练编排：五个模型续训 / 部署 / 解析验证与测试指标
    refresh_model.py   ← 更早的"数据+训练一把梭"（**注意是单数的旧名**，2026-09 已失效：
                       模块级裸导入直接 ImportError、缺 two、训练在同进程里跑）。
                       本次把它们三个重写成**复数**的 refresh_models.py（本文件）。

合并后语义不变，入口只剩这一个：`--stage data` 跑数据、`--stage train` 只训练、默认 all 全跑。

## 三个步骤（用 `--stage` 单独跑，默认 all）

    data   ① MySQL -> data/{EX}/*.csv                       (get_stock_data.get_data)
           ② 重生 h5：one / three / seven / zero / two       (各自的 gen_train_data.py)
           ③ 重生 nowcast 分块（**可选**，只在 `--nowcast` 时跑；实验用）
    train  训练 5 个模型（warm start + 部署 + 指标追加到 retrain_results.csv）
    all    先 data 再 train

⚠️ two 的 h5 全量重生是**全链最贵的一步：≈5~6 小时**（24 个所实测 4.4 小时 / 1,697,296 个
文件 / 234GB），比另外四个加起来还长 —— 所以它排在 H5_JOBS 最后，前面四个先训好、先能部署。

## 两处刻意的工程决定（都不要随手改）

1. **nowcast 数据生成到【新根目录】**（默认 `nowcast4`），不覆盖旧的 `nowcast3`
   （现在只在 `--nowcast` 下才会走到这一步）。
   原因：`chunk_*_X.npy` 是所有变体**共用**的输入，而 `chunk_*_y{u1..u8}.npy` 是各变体
   独立的标签。只重生 X + u9、把 u1..u8 留在原地，会让那些**旧标签和新 X 行序错位**
   （同一 chunk 里的行已经不是同一批锚点了）—— 将来谁再跑 u6 会静默拿到错数据。
   生成到新根目录就完全没这个问题，而且旧 88GB 原样保留、随时可回退。

2. **只生成 u9**（`--variants u9`）。生产上 `two` 用的是 u9；其余 u1~u8 是已放弃的实验。
   要多个变体就传 `--variants u1 u6 u9`。

## 导出前的安全检查

`get_data()` 是**覆盖写** `data/{EX}/*.csv`。它从 MySQL 重新导出，所以理论上只会变多；
但万一库里少了行，CSV 会**静默缩小**、后面全部模型都跟着退化。
所以本脚本在导出前后各记一次每个 CSV 的（大小, mtime），**任何文件缩小超过 5% 就大声报警**。

## 用法

    python refresh_models.py                 # 全套：MySQL导出 -> 重生全部数据 -> 续训5个模型 -> 部署 -> 记指标
    python refresh_models.py --dry-run       # 只打印要执行什么，不改任何东西（建议第一次先跑这个）

跑完看两个地方：

    retrain_logs/            每一步的完整日志（本脚本自己的 run_*.log + 各子任务/各模型各一份）
    retrain_results.csv      5 个模型的验证指标 + 测试指标

**出问题时才需要的细粒度开关**（日常不用管）：

    --stage export|data|train|all    只跑某一段（默认 all = 全套）
    --skip-export                    跳过 MySQL->CSV（导出已单独跑过时用）
    --data-only zero                 只重生指定模型的 h5（配 --stage data --skip-export）
    --only one two                   只训指定模型
    --nowcast                        额外重生 nowcast 分块（默认跳过；做变体实验才用）
    --variants u1 u6 u9              要生成哪些 nowcast 变体（默认只有 u9，需配 --nowcast）
    --nowcast-root <path>            改 nowcast 分块输出根目录（需配 --nowcast）

## 跑之前的状态前提

* **测试集已固定**：one/three/seven/zero 的 train.py 会**沿用已有的 `test_symbols.txt`**
  （各 889 只），不再按股票全集重抽 —— 所以新数据只会更新测试的**行**，不会换测试的**股票**。
  名单里若有个别股票在新数据中消失，会打警告并剔出。
* **5 个模型都是续训**（warm start，"有 .pt 就接着训"）：one 需要先有 `one/one.pt`
  （脚本会自动复制成 `one_ashare_pearson_new24.pt` 供 train.py 加载）；two 从自己的
  `two/two.pt` 续；three/seven/zero 各自从 `three.pt`/`seven.pt`/`zero.pt` 续。
  **每个模型的日志里都该出现加载标记**，本脚本会自检，没出现就告警（否则等于悄悄从零训）。
* **训练/验证划分每次打乱**（注入随机 `VAL_SPLIT_SEED`；three/seven/zero 本来就是系统熵），
  但**测试集不动**。
* **回退点**：`_snapshot_20260926/`（5 个部署模型 + 7 个 CSV + 原 data/ 的 15GB CSV 副本），
  还原命令见该目录的 README.md。模型本身也在 git 里有一份。
"""
import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time

# Windows 控制台默认 GBK：本文件会打印 `⚠️`（U+26A0，GBK 编不出来）—— 走到告警分支时
# （warm start 没加载上、指标没解析出来、训练失败、CSV 缩小……）正好是最需要看到那句话的时候。
# 装了 _Tee 之后不会崩（Tee 的 write 逐个流 try/except），但**会被静默吞掉**——
# 控制台上什么都不显示，只有日志里有。这里统一成 UTF-8，与其它脚本同一套修法。
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
LOG_DIR = os.path.join(ROOT, 'retrain_logs')
DATA_DIR = os.path.join(ROOT, 'data')
RESULTS = os.path.join(ROOT, 'retrain_results.csv')

# nowcast 分块生成到新根目录（见模块 docstring 第 1 条）
NOWCAST_ROOT = 'C:/quant_data/nowcast4'
NOWCAST_VARIANTS = ['u9']
NOWCAST_ALIGNED = os.path.join(ROOT, 'two', 'nowcast', 'test_symbols_all31_889.txt')

# gen_data2 用的**基础变体**：它顺手写的那份 y 只是占位（真标签由 relabel 覆写）。
# 必须是 `labels.build_labels` 认得的 form 的变体（u1 属于 yoy_delta/growth 那一类）。
# ⚠️ 千万别改成 u9 —— 见 stage_data 里那段说明，会直接崩。
BASE_VARIANT = 'u1'

# 每个目标变体的 relabel 参数。依据 `two/nowcast/variants.py` 各变体段自己的注释：
#   u6/u7/u9 前向窗口都是 7 季；u9 的【过去窗口】是 3 季（其余变体默认 = 前向窗口）；
#   u8 用 yoymed（同比变化的中位数，无量纲）。别名要加就在这里加。
VARIANT_RELABEL_ARGS = {
    'u6': ['--suffix', 'u6', '--horizons', '7'],
    'u7': ['--suffix', 'u7', '--horizons', '7'],
    'u8': ['--suffix', 'u8', '--horizons', '7', '--form', 'yoymed'],
    'u9': ['--suffix', 'u9', '--horizons', '7', '--past-win', '3'],
}

# ---- h5 数据生成（one 需要显式参数；其余四个是写死的配置，无参数）----
H5_JOBS = [
    ('one',   [PY, 'one/gen_train_data.py', '--exchanges', 'SHZ', 'SHH',
               '--folder', 'train', '--rate', str(1.0 / 31.0)]),
    ('three', [PY, 'three/gen_train_data.py']),
    ('seven', [PY, 'seven/gen_train_data.py']),
    ('zero',  [PY, 'zero/gen_train_data.py']),
    ('two',   [PY, 'two/gen_train_data.py']),
]

# ---- 测试指标的抓取方式（2026-09-27 修）----
# 以前 one/three/seven 用的是 `icir_combined=([0-9.]+)` 这类正则，但**全仓没有任何脚本会打印
# 那个串** —— `icir_*_combined` 只是 evaluate.py 写进 `eval_results/history.csv` 的**列名**，
# 控制台里它出现在一张表的一行上：
#
#     horizon            IC     ICIR         t  n_dates         MSE
#     ...
#     combined       0.1437    0.477     29.48     3813      0.7613   <- 实际排序分数   (one)
#     gd/combined    0.2909    2.293     19.18       70      1.7334                    (three)
#     fcf/combined   0.5862    3.701     31.62       73      8.2143   <- 实际排序用的分数  (seven)
#
# 所以正则改成"抓那行的第二列（ICIR）"。⚠️ 不能用 `^` 裸锚定：下面 re.findall 不带 flags，
# `^` 只会匹配整个字符串的开头 —— 所以模式里自带 `(?m)`。
# （three/seven 的分支表里还有 dividend/netasset 的 combined 行，取各自的 gd/fcf 那行才对。）
TEST_RE_ONE = r'(?m)^combined\s+[+-]?[0-9.]+\s+([+-]?[0-9.]+)'
TEST_RE_THREE = r'(?m)^gd/combined\s+[+-]?[0-9.]+\s+([+-]?[0-9.]+)'
TEST_RE_SEVEN = r'(?m)^fcf/combined\s+[+-]?[0-9.]+\s+([+-]?[0-9.]+)'

# ---- 每个模型的「部署配方」—— 必须和线上推理读的东西一致，改错就等于换了模型 ----
MODELS = {
    'one': dict(
        cmd=[PY, 'one/train.py', 'pearson', 'ashare'],      # LOSS_MODE=pearson, DATA_MODE=ashare
        env={'FACTOR_MODE': 'new24'},                        # 线上 one.pt 是 new24
        # warm start 的种子：one/train.py 找的是 one_ashare_pearson_new24.pt（RUN_TAG 默认 ''），
        # 但线上那份叫 one.pt、且该名字的文件并不存在 —— 不先复制，one 就会【从零训】。
        pre=[('one/one.pt', 'one/one_ashare_pearson_new24.pt')],
        # 线上推理读的是硬编码的 one/one.pt，而训练产物是 one_{DATA_MODE}_{LOSS}_{FACTOR}.pt
        deploy=[('one/one_ashare_pearson_new24.pt', 'one/one.pt')],
        val_re=r'validate loss:\s*([0-9.]+)', val_better='lower', val_name='val_loss',
        test_re=TEST_RE_ONE, test_name='icir_combined',
        warm_mark='loaded existing model:',
    ),
    # two 的数据线（2026-09-27 换成自己的 h5）：
    #   以前 two 走的是「nowcast 分块 + two/nowcast/train.py」两步线（数据在 C:/quant_data/nowcast3|4，
    #   产物 nowcast_both_u9.pt 再复制成 two/two.pt）。现在 `two/gen_train_data.py` 一步
    #   生成 `two/train/*.h5`（特征与标签都是逐字移植、行集与旧线逐行可比），
    #   `two/train.py` 直接读它、直接写 two/two.pt —— 与另外四个模型的形状一致，
    #   所以 `deploy` 是空的（产物就是部署路径本身）。
    #   变体仍是 **u9**（不是 u6）：定案依据见 two/nowcast/paired_u6_u9.py 与
    #   `two/nowcast/variants.py` 的 u9 段 —— 按「排序选股」的终点指标看，u9 至少不差于 u6。
    # warm start：`two/train.py` 从 two/two.pt 接着训（日志里打 'loaded existing TWO model:'）。
    #
    # ---- 回退面包屑：旧的那条 two 线（2026-09-27 之前用的，要回退就把下面这行换回去）----
    # 前提是先用 `--nowcast`（或手动跑 two/nowcast/gen_data2.py + add_u6_labels.py）
    # 把分块生成到 --nowcast-root 指的目录：
    #   cmd=[PY, 'two/nowcast/train.py', '--arm', 'both', '--suffix', 'u9', '--labels', 'level',
    #        '--test-mode', 'global_random', '--test-n', '889', '--test-seed', '42',
    #        '--seed', '1', '--epochs', '7', '--root', 'C:/quant_data/nowcast3', '--tag', '_u9',
    #        '--warm-start'],
    #   deploy=[('two/nowcast/nowcast_both_u9.pt', 'two/two.pt')],
    #   warm_mark='warm start: 载入',
    # 旧数据都还在盘上（C:/quant_data/nowcast3 及其 u9 标签），换成这条不会找不到东西。
    # ⚠️ 2026-09-28：`two/nowcast/nowcast_both_u9.pt` 已随 nowcast 整理移进 _trash
    #    —— 但它与 git 提交里的 `two/two.pt` **逐字节相同**（sha256 ff6c0d9e…，4,163,478 B），
    #    所以要回退时把上面那行的 deploy 换成：
    #       deploy=[('two/two.pt', 'two/two.pt')]   # 就是它自己（先 git checkout 回到交付版）
    #    或直接从 `_trash_20260928/nowcast/nowcast_both_u9.pt` 移回来。
    'two': dict(
        cmd=[PY, 'two/train.py', '--epochs', '7'],
        env={},                                              # 不给 --val-seed -> 验证划分每次随机
        deploy=[],
        # 验证 IC 每个 epoch 打一行 -> 取最大值（train.py 存的也是验证最优那个）
        val_re=r'验证 mean IC=([0-9.]+)', val_better='higher', val_name='val_ic',
        test_re=r'平均\s+([+-][0-9.]+)', test_name='test_ic_avg',
        warm_mark='loaded existing TWO model:',
    ),
    'three': dict(
        cmd=[PY, 'three/train.py'], env={},
        deploy=[],                                           # 产物就是 three/three.pt 本身
        val_re=r'validate loss:\s*([0-9.]+)', val_better='lower', val_name='val_loss',
        test_re=TEST_RE_THREE, test_name='icir_gd_combined',
        warm_mark='loaded existing THREE model:',
    ),
    'seven': dict(
        cmd=[PY, 'seven/train.py'], env={},
        deploy=[],
        val_re=r'validate loss:\s*([0-9.]+)', val_better='lower', val_name='val_loss',
        test_re=TEST_RE_SEVEN, test_name='icir_fcf_combined',
        warm_mark='loaded existing SEVEN model:',
    ),
    'zero': dict(
        cmd=[PY, 'zero/train.py'], env={},
        deploy=[],
        val_re=r'best_val_loss:\s*([0-9.]+)', val_better='lower', val_name='val_loss',
        test_re=r'\[TEST\]\s*test_loss=([0-9.]+)', test_name='test_loss',
        warm_mark='loaded existing ZERO model:',
    ),
}


class _Tee:
    """把本脚本自己的 stdout 同时写到控制台和一份日志。

    动机（2026-09-26）：子进程的输出本来就各自落盘，但**本脚本自己**的阶段横幅、
    "导出后比对：…缩小>5% N 个"这类关键结论以前只打在控制台 —— 事后无从复查，
    而这些恰恰是最该留痕的判据。
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
                st.flush()
            except Exception:
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


def sh(cmd, log_name, dry=False):
    """跑一个子进程，stdout+stderr 落盘。返回 (退出码, 耗时秒, 日志路径)。"""
    log = os.path.join(LOG_DIR, f'{log_name}_{time.strftime("%Y%m%d_%H%M%S")}.log')
    print(f'  命令: {" ".join(cmd)}', flush=True)
    print(f'  日志: {os.path.relpath(log, ROOT)}', flush=True)
    if dry:
        print('  [dry-run] 不执行'); return 0, 0.0, log
    t0 = time.time()
    # ⚠️ 必须给子进程设 PYTHONIOENCODING=utf-8。默认在 Windows 上子进程 stdout 走 GBK，
    #    而好几个脚本末尾会 print 一个 emoji（如 zero/gen_train_data.py 的 "🎉"）——
    #    GBK 编不出来 → **在干完所有活之后**抛 UnicodeEncodeError → 退出码 1
    #    → 编排脚本当成失败、中止整条流水线。2026-09-26 的 zero 就是这么"失败"的：
    #    日志里已经写着 "Total HDF5 files created: 1765796"，然后崩在一句庆祝打印上。
    #    顺带这也让子进程日志统一成 UTF-8（否则中文按 GBK 落盘、读回来乱码）。
    env = dict(os.environ)
    env['PYTHONIOENCODING'] = 'utf-8'
    with open(log, 'w', encoding='utf-8', errors='replace') as fp:
        p = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fp, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    print(f'  退出码 {p.returncode}  用时 {dt / 60:.1f} 分钟', flush=True)
    if p.returncode != 0:
        print('  ⚠️ 失败，tail:')
        txt = open(log, encoding='utf-8', errors='replace').read().splitlines()
        print('\n'.join('    ' + l for l in txt[-15:]))
    return p.returncode, dt, log


def failed_exchanges(log_path):
    """从 gen 的子进程日志里读 `Failed to process exchanges: [...]` 那行，返回名单（空 = 全成功）。

    为什么需要它：**五个 gen 脚本都是"某个交易所出错只记一笔、照常退 0"**
    （`get_data()` 和各个 gen 都用 `except: failed_list.append(ex)`）—— 所以只看退出码
    发现不了"数据缺了一两个交易所"，而缺数据会一路静默传到训练段（拿缺的数据训 5 小时）。
    two 的日志里这行只在有失败时出现；one 的是无条件打印（成功时是 `[]`），所以按内容判空。

    ⚠️ 这行**不等于出错**：交易所进了这个名单，也可能是"它的数据天生凑不出样本"。实测
    seven 每次都会有 KLS/KOE/SAU/TLV 进名单 —— 那四个市场在 indicator+127 特征合并、并按
    mean/std 起始日（20120331）截断后，每只股票最多 58 行季度数据，而 seven 的标签要
    31 行过去 + 31 行未来（`if group_data_length < 62: return 0`）→ 全所 0 个文件、无异常。
    three 只要 31 行、zero 只要 3 行，所以同样四个所它们照常出文件。见调用点的打印。
    """
    try:
        txt = open(log_path, encoding='utf-8', errors='replace').read()
    except Exception:
        return []
    m = re.findall(r'Failed to process exchanges:\s*(\[[^\]]*\])', txt)
    if not m:
        return []
    try:
        return json.loads(m[-1].replace("'", '"'))
    except Exception:                     # 解析不了就原样报出来，别装作没事
        return [m[-1][:80]]


def csv_inventory():
    """记录每个 CSV 的（大小, mtime）。导出前后各调一次，用于发现"静默缩小"。"""
    inv = {}
    if not os.path.isdir(DATA_DIR):
        return inv
    for ex in sorted(os.listdir(DATA_DIR)):
        d = os.path.join(DATA_DIR, ex)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith('.csv'):
                p = os.path.join(d, f)
                st = os.stat(p)
                inv[f'{ex}/{f}'] = (st.st_size, int(st.st_mtime))
    return inv


def compare_inventory(before, after):
    """导出后比对。返回是否一切正常。"""
    shrank, grew, gone, new = [], [], [], []
    for k, (sz0, _) in before.items():
        if k not in after:
            gone.append(k); continue
        sz1 = after[k][0]
        if sz1 < sz0 * 0.95:
            shrank.append((k, sz0, sz1))
        elif sz1 > sz0:
            grew.append((k, sz0, sz1))
    for k in after:
        if k not in before:
            new.append(k)
    print(f'  导出后比对：变大 {len(grew)} 个 / 新出现 {len(new)} 个 / 消失 {len(gone)} 个 '
          f'/ **缩小>5% {len(shrank)} 个**')
    for k, a, b in grew[:6]:
        print(f'    ↑ {k}: {a/1048576:.1f} -> {b/1048576:.1f} MB')
    if gone:
        print(f'  ⚠️ 消失的文件: {gone}')
    if shrank:
        print('  ⚠️⚠️ 以下文件缩小超过 5% —— MySQL 里可能少了数据，别继续，先查库：')
        for k, a, b in shrank:
            print(f'    ↓ {k}: {a/1048576:.1f} -> {b/1048576:.1f} MB ({b/a-1:+.1%})')
    return not shrank and not gone


def stage_export(a):
    """只做 MySQL -> CSV 这一步，并检查有没有文件被"导出没了"。"""
    print(f'\n{"="*100}\n  阶段 1/3：MySQL -> CSV\n{"="*100}', flush=True)
    before = csv_inventory()
    print(f'  导出前：{len(before)} 个 CSV，合计 {sum(v[0] for v in before.values())/2**30:.2f} GB')
    rc, _, _ = sh([PY, 'get_stock_data.py'], 'data_export', dry=a.dry_run)
    if rc != 0:
        return False
    if a.dry_run:
        return True
    ok = compare_inventory(before, csv_inventory())
    if not ok:
        print('\n⛔ 导出后 CSV 出现缩小/消失 —— 停下，先查 MySQL，不要继续重生数据。')
        print('   备份在 _snapshot_20260926/data_csv/，可直接复制回去。')
    return ok


def stage_data(a):
    if a.skip_export:
        print(f'\n{"="*100}\n  （--skip-export：跳过 MySQL 导出，直接用现有 data/ 重生）\n{"="*100}',
              flush=True)
    elif not stage_export(a):
        return False

    if a.skip_h5:
        print(f'\n{"="*100}\n  阶段 2/3：重生 h5（--skip-h5：跳过）\n{"="*100}', flush=True)
        print('  （--skip-h5：跳过 h5 重生）', flush=True)
    else:
        jobs = [(n, c) for n, c in H5_JOBS if a.data_only is None or n in a.data_only]
        # 横幅里列**实际会跑的那几个**（--data-only 时只跑一个，别让人以为五个都在跑）
        print(f'\n{"="*100}\n  阶段 2/3：重生 h5（{" / ".join(n for n, _ in jobs)}）\n{"="*100}',
              flush=True)
        if a.data_only:
            print(f'  （--data-only：只跑 {[n for n, _ in jobs]}）', flush=True)
        unknown = sorted(set(a.data_only or []) - {n for n, _ in H5_JOBS})
        if unknown:
            print(f'  ⚠️ --data-only 里有未知模型（已忽略）：{unknown}', flush=True)
        for name, cmd in jobs:
            print(f'\n───── [{name}] 重生 h5', flush=True)
            rc, _, log = sh(cmd, f'gendata_{name}', dry=a.dry_run)
            if rc != 0:
                print(f'  ⚠️ {name} 的 h5 重生失败 —— 后续 {name} 无法训练。')
                return False
            if not a.dry_run:
                d = os.path.join(ROOT, name, 'train')
                n = len(os.listdir(d)) if os.path.isdir(d) else 0
                print(f'  {name}/train 现有 {n:,} 个 h5', flush=True)
                _bad = failed_exchanges(log)
                if _bad:
                    print(f'  ⚠️ {name} 有 {len(_bad)} 个交易所产出 0 个文件：{_bad}')
                    print(f'      （这行是【报告】不是报错：gen 吞掉单所异常、照常退 0，所以只看'
                          f'退出码发现不了。但如果上次它们有产出、这次归零，那就是真出问题了。）')
                    print(f'      已知的正常例子：seven 的 KLS/KOE/SAU/TLV —— 四个市场历史不足 62 季，'
                          f'装不下 31 过去 + 31 未来的标签窗口（three 只要 31 季、zero 只要 3 季，'
                          f'所以它们照常出文件）。')

    if a.data_only is not None:
        print('\n  （--data-only：只重生上面那些 h5，**跳过 nowcast 分块**。'
              '要单独跑它用 `--stage data --skip-export --skip-h5 --nowcast`）', flush=True)
        return True
    # 阶段 3 从 2026-09-27 起是**可选**的：two 以前靠 nowcast 分块（gen_data2 + relabel），
    # 现在靠自己的 h5（已经并进上面的 H5_JOBS），所以这一阶段默认不跑 —— 它每次要花 1~2 小时，
    # 而产出的分块只有做 nowcast 变体实验时才用得上。要跑加 `--nowcast`。
    # 跳过时照样打一条完整横幅（而不是一句括号说明）：看日志的人一眼就知道"阶段 3 是被有意
    # 跳过的"，不会以为脚本缺了一段。
    if not a.nowcast:
        print(f'\n{"="*100}\n  阶段 3/3：重生 nowcast 分块 —— 已跳过（默认）\n{"="*100}', flush=True)
        print('  two 现在读自己的 two/train/*.h5（阶段 2 生成），不再读 nowcast 分块；')
        print('  这一步的产物在生产链上没有任何消费者，所以默认不跑（每次省 1~2 小时）。')
        print('  要做 nowcast 变体实验（u1~u8）加 `--nowcast`；旧分块仍在 '
              'C:/quant_data/nowcast3（6030 个 .npy，含 603 个 u9 标签），一个都没删。', flush=True)
        return True

    print(f'\n{"="*100}\n  阶段 3/3：重生 nowcast 分块（--nowcast，实验用）-> {NOWCAST_ROOT}\n{"="*100}',
          flush=True)
    exs = sorted(d for d in os.listdir(DATA_DIR)
                 if os.path.isdir(os.path.join(DATA_DIR, d)))
    print(f'  交易所 {len(exs)} 个；目标变体 {NOWCAST_VARIANTS}', flush=True)

    # ⚠️ u9/u6/u8 的标签【不是】gen_data2 能直接生成的。它们的 form 是 'med' / 'yoymed'，
    #    而 `two/nowcast/labels.py::build_labels` 只认 level_flow / level_flow_eq / growth /
    #    level_stock / yoy_delta / growth_signed —— 遇到别的直接 `raise ValueError(form)`。
    #    所以 `gen_data2.py --variant u9` 会**一跑就崩**（2026-09-27 读代码确认）。
    #    正确流程是【两步】（nowcast3 里的 u6/u8/u9 也是这么来的）：
    #      ① gen_data2 用基础变体生成 X（X 与变体无关，只需一次）+ 一份占位 y
    #      ② add_u6_labels.relabel 覆写 `_y{suffix}` / `_b{suffix}` / `_r{suffix}`
    #    训练端按 `--suffix u9` 读的就是 ② 写出来的那份。
    print(f'\n───── [①] gen_data2 生成 X（只做一次；基础变体 {BASE_VARIANT} 的 y 是占位）', flush=True)
    rc, _, _ = sh([PY, 'two/nowcast/gen_data2.py', '--exchanges', *exs,
                   '--out', NOWCAST_ROOT, '--variant', BASE_VARIANT,
                   '--aligned-symbols', NOWCAST_ALIGNED],
                  'gendata_nowcast_X', dry=a.dry_run)
    if rc != 0:
        print('  ⚠️ X 生成失败。')
        return False

    for v in NOWCAST_VARIANTS:
        print(f'\n───── [②] relabel -> {v}', flush=True)
        extra = VARIANT_RELABEL_ARGS.get(v)
        if extra is None:
            print(f'  ⚠️ 不知道变体 {v!r} 该怎么 relabel —— 跳过。'
                  f'（已知：{sorted(VARIANT_RELABEL_ARGS)}；要加别名见 add_u6_labels.py 的 --past-win/--form）')
            continue
        rc, _, _ = sh([PY, 'two/nowcast/add_u6_labels.py', '--exchanges', *exs,
                       '--root', NOWCAST_ROOT, *extra],
                      f'relabel_nowcast_{v}', dry=a.dry_run)
        if rc != 0:
            print(f'  ⚠️ {v} 的 relabel 失败。')
            return False
    return True


def run_one(name, cfg, dry=False):
    """训练一个模型：warm start 种子 → 跑 train.py → 自检/解析指标 → 部署。返回结果行或 None。"""
    env = dict(os.environ)
    env.update(cfg['env'])
    # 同 sh() 的理由：Windows 下子进程 stdout 默认 GBK，脚本里任何非 GBK 字符（emoji 等）
    # 都会让它在【干完活之后】崩掉、退出码 1 —— 而这里会把退出码 1 当成"训练失败"，
    # 于是**跳过部署、几小时的训练白跑**。这一行是防这个的。
    env['PYTHONIOENCODING'] = 'utf-8'
    # 训练/验证划分打乱：只影响 val，不影响测试集（各 train.py 的 TEST_SEED 保持默认）
    env['VAL_SPLIT_SEED'] = str(random.randrange(10 ** 9))
    log = os.path.join(LOG_DIR, f'{name}_{time.strftime("%Y%m%d_%H%M%S")}.log')
    print(f'\n{"="*100}\n  [{name}] {cfg["cmd"][1]}   (VAL_SPLIT_SEED={env["VAL_SPLIT_SEED"]}'
          f'{", " + str(cfg["env"]) if cfg["env"] else ""})\n{"="*100}', flush=True)
    if dry:
        print(f'  命令: {" ".join(cfg["cmd"])}')
        for src, dst in cfg.get('pre', []):
            print(f'  先复制: {src}  ->  {dst}（warm start 种子）')
        for src, dst in cfg['deploy']:
            print(f'  训完复制: {src}  ->  {dst}（部署路径）')
        print('  [dry-run] 不执行'); return None
    # warm start 种子：把「线上那份 .pt」复制成 train.py 实际会加载的文件名。
    # one 必须走这一步 —— 否则脚本找不到文件、静默从零训（2026-09-26 踩过）。
    for src, dst in cfg.get('pre', []):
        s, d = os.path.join(ROOT, src), os.path.join(ROOT, dst)
        if not os.path.exists(s):
            print(f'  ⚠️ warm start 种子不存在 {src} -> 跳过，该模型会【从零训】', flush=True)
            continue
        shutil.copy(s, d)
        print(f'  种子: {src}  ->  {dst}', flush=True)
    t0 = time.time()
    with open(log, 'w', encoding='utf-8', errors='replace') as fp:
        p = subprocess.run(cfg['cmd'], cwd=ROOT, env=env, stdout=fp, stderr=subprocess.STDOUT)
    txt = open(log, encoding='utf-8', errors='replace').read()
    print(f'  退出码 {p.returncode}  用时 {(time.time()-t0)/60:.1f} 分钟  日志 {os.path.relpath(log, ROOT)}')
    if p.returncode != 0:
        print('  ⚠️ 训练失败，tail:'); print('\n'.join(txt.splitlines()[-12:])); return None
    # warm start 自检：日志里必须出现各 train.py 的加载标记。不加这道校验的话，
    # 「找不到 .pt 于是从零训」是完全静默的 —— 结果照出，但模型换了血统。
    _warm = cfg.get('warm_mark')
    if _warm and _warm not in txt:
        print(f'  ⚠️ 没在日志里找到 warm start 标记 {_warm!r} —— 这个模型可能是【从零训】的，'
              f'结果不可与其它模型横向比较')
    # 解析验证指标（取最优方向上的那个）
    vals = [float(v) for v in re.findall(cfg['val_re'], txt)]
    val = (max if cfg['val_better'] == 'higher' else min)(vals) if vals else None
    # 解析测试指标
    tests = re.findall(cfg['test_re'], txt)
    test = float(tests[-1]) if tests else None
    # 覆盖到部署路径
    for src, dst in cfg['deploy']:
        s = os.path.join(ROOT, src)
        if not os.path.exists(s):
            print(f'  ⚠️ 训练产物不存在，无法部署: {src}'); continue
        shutil.copy(s, os.path.join(ROOT, dst))
        print(f'  部署: {src}  ->  {dst}')
    # 解析不出来就**大声说**：以前是静默写进 None，结果表里那一格空着、看不出是"没算出来"
    # （one/three/seven 的测试正则错了半年就是这么藏住的）。
    if val is None:
        print(f'  ⚠️ 验证指标没解析出来（{cfg["val_re"]!r} 在日志里没命中）—— 结果行里会是 None')
    if test is None:
        print(f'  ⚠️ 测试指标没解析出来（{cfg["test_re"]!r} 在日志里没命中）—— 结果行里会是 None')
    print(f'  验证 {cfg["val_name"]} = {val}    测试 {cfg["test_name"]} = {test}')
    return dict(model=name, val_name=cfg['val_name'], val=val,
                test_name=cfg['test_name'], test=test, log=os.path.relpath(log, ROOT))


def stage_train(a):
    """逐个训练 5 个模型（warm start），训完部署 + 把指标追加到 retrain_results.csv。"""
    names = a.only or list(MODELS)
    print(f'\n{"="*100}\n  训练 {len(names)} 个模型：{" ".join(names)}（warm start）\n{"="*100}',
          flush=True)
    rows, failed = [], []
    for n in names:
        r = run_one(n, MODELS[n], dry=a.dry_run)
        if r:
            rows.append(r)
        else:
            failed.append(n)
    if rows:
        new = not os.path.exists(RESULTS)
        with open(RESULTS, 'a', encoding='utf-8') as fp:
            if new:
                fp.write('timestamp,model,val_metric,val,test_metric,test,log\n')
            ts = time.strftime('%Y-%m-%d %H:%M:%S')
            for r in rows:
                fp.write(f'{ts},{r["model"]},{r["val_name"]},{r["val"]},'
                         f'{r["test_name"]},{r["test"]},{r["log"]}\n')
        print(f'\n{"="*100}\n  汇总（已追加到 {os.path.relpath(RESULTS, ROOT)}）\n{"="*100}')
        print(f'  {"模型":8s}{"验证指标":>16s}{"验证":>12s}{"测试指标":>18s}{"测试":>12s}')
        for r in rows:
            print(f'  {r["model"]:8s}{r["val_name"]:>16s}{r["val"]:>12}{r["test_name"]:>18s}{r["test"]:>12}')
    if a.dry_run:
        return True
    # 退出码/返回值：单个模型失败不影响其余四个（上面逐个跑、逐个部署），
    # 但整体必须让对方看得见 —— 否则"五个模型全失败"也会报"全套完成 · 成功"。
    if failed or not rows:
        print(f'\n  ⚠️ 有模型没跑成：{failed or "（一个都没产出）"} —— 训练段判为失败'
              f'（日志见 retrain_logs/）。已训完的模型已经部署过了，没训的保持原样。')
        return False
    print(f'\n{"="*100}\n  全套完成。测试指标见 {os.path.relpath(RESULTS, ROOT)}\n{"="*100}')
    return True


def main():
    # global 必须在任何引用之前（否则 default=NOWCAST_VARIANTS 会触发
    # SyntaxError: name used prior to global declaration）
    global NOWCAST_VARIANTS, NOWCAST_ROOT
    ap = argparse.ArgumentParser(
        description='一键重训 5 个模型：数据（MySQL→CSV→h5）→ 训练（warm start）→ 部署 → 记指标')
    ap.add_argument('--stage', choices=['export', 'data', 'train', 'all'], default='all',
                    help='export=只做 MySQL->CSV（含"有没有导出没"的检查）；'
                         'data=export + 重生全部数据；train=只训练；all=全套（默认）')
    ap.add_argument('--nowcast', action='store_true',
                    help='额外重生 nowcast 分块（默认跳过 —— two 现在用自己的 h5，'
                         '这一阶段只有做 nowcast 变体实验时才需要，每次要 1~2 小时）')
    ap.add_argument('--variants', nargs='+', default=NOWCAST_VARIANTS,
                    help=f'要生成的 nowcast 变体（默认 {NOWCAST_VARIANTS}；只在 --nowcast 下生效）')
    ap.add_argument('--nowcast-root', default=NOWCAST_ROOT,
                    help='nowcast 分块输出根目录（只在 --nowcast 下生效，见模块 docstring）')
    ap.add_argument('--skip-nowcast', action='store_true',
                    help='【已无必要，只为兼容旧命令行】不跑 nowcast 分块 —— 现在这是默认行为')
    ap.add_argument('--skip-h5', action='store_true',
                    help='跳过 h5 重生（one/three/seven/zero/two）。配合 --stage data --skip-export '
                         '就等于"只生成 nowcast 分块"（需再加 --nowcast），便于一步一步单独跑。')
    ap.add_argument('--skip-export', action='store_true',
                    help='跳过 MySQL->CSV，直接用现有 data/ 重生（导出已单独跑过时用）')
    ap.add_argument('--only', nargs='+', choices=list(MODELS), default=None,
                    help='只训指定的模型（与 --stage train 配合）')
    ap.add_argument('--data-only', nargs='+', default=None,
                    help='只重生这些模型的 h5，例如 `--data-only zero`（其余步骤照常跳过）。'
                         '配合 --stage data --skip-export 就能一个模型一个模型地跑。')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    NOWCAST_VARIANTS, NOWCAST_ROOT = a.variants, a.nowcast_root

    os.makedirs(LOG_DIR, exist_ok=True)
    # 本脚本自己的输出也落盘（见 _Tee 的说明）
    _run_log = open(os.path.join(LOG_DIR, f'run_{time.strftime("%Y%m%d_%H%M%S")}.log'),
                    'w', encoding='utf-8', errors='replace')
    sys.stdout = _Tee(sys.__stdout__, _run_log)
    t0 = time.time()
    print(f'{"="*100}\n  一键重训（stage={a.stage}）  开始 {time.strftime("%Y-%m-%d %H:%M:%S")}\n{"="*100}')

    ok = True
    if a.stage == 'export':
        ok = stage_export(a)
    elif a.stage in ('data', 'all'):
        ok = stage_data(a)
    if ok and a.stage in ('train', 'all'):
        if a.stage == 'train':
            print('  （--stage train：跳过数据重生，直接用现有数据训练）', flush=True)
        ok = stage_train(a)

    print(f'\n  总用时 {(time.time()-t0)/60:.1f} 分钟   结果 {"成功" if ok else "失败"}')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
