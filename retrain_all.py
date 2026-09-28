"""统一重训 5 个模型（one / two / three / seven / zero）并记录测试集指标。

用户 2026-09-26 定的规格：
  * 每个模型**只训一遍**（不做「跑 3 遍选最优」）
  * **测试集固定不动**（TEST_SEED 保持默认 42，不用环境变量覆盖）
  * **在已有模型基础上继续训练**（warm start）—— 五个模型统一成"有 .pt 就接着训"：
    one/three/seven/zero 由各自的 train.py 自动加载旧 .pt；two 2026-09-27 起也是
    （`two/train.py` 从 `--out`（默认 two/two.pt）续训，载入后把 affine 还原成恒等，
    否则训练时的输出量级是错的 —— 见那边的 ⚠️）；one 还需先把线上 one.pt
    复制成脚本要找的文件名（见 MODELS['one']['pre']）
  * 训练/验证划分**打乱**（用随机 VAL_SPLIT_SEED；three/seven/zero 本来就是系统熵）
  * 训完把新模型覆盖到**部署路径**，并把**测试集指标**追加到 retrain_results.csv

用法：
    python retrain_all.py                   # 全部 5 个
    python retrain_all.py --only one zero   # 只跑指定的
    python retrain_all.py --dry-run         # 只打印要执行的命令，不真跑
"""
import argparse
import os
import random
import re
import shutil
import subprocess
import sys
import time

# ⚠️ Windows 控制台默认 GBK，而本文件会打印 `⚠️`（U+26A0，GBK 编不出来）—— 一旦走到告警分支
# （warm start 没加载上、指标没解析出来、训练失败……）就当场 UnicodeEncodeError，**正好是最需要
# 看到那句话的时候**崩掉。2026-09-27 实测复现。子进程有 full_retrain 传的 PYTHONIOENCODING=utf-8
# 保护，但**本脚本直接跑时**没有 —— 这里补上与其它脚本同一套的 reconfigure。
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

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
LOG_DIR = os.path.join(ROOT, 'retrain_logs')
RESULTS = os.path.join(ROOT, 'retrain_results.csv')

# 每个模型的「部署配方」—— 必须和线上推理读的东西一致，改错就等于换了模型
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
    # 前提是先用 `full_retrain.py --nowcast`（或手动跑 two/nowcast/gen_data2.py + add_u6_labels.py）
    # 把分块生成到 --root 指的目录：
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


# 2026-09-27 删掉了 `_override_root`：它只服务于「把 two 的数据根指到新生成的 nowcast 分块」。
# two 现在读自己的 `two/train/*.h5`，没有 --root 这个参数，再传进去会让 argparse 直接报错。


def run_one(name, cfg, dry=False):
    env = dict(os.environ)
    env.update(cfg['env'])
    # 同 full_retrain.sh 的理由：Windows 下子进程 stdout 默认 GBK，脚本里任何非 GBK 字符
    # （emoji 等）都会让它在【干完活之后】崩掉、退出码 1 —— 而这里会把退出码 1 当成
    # "训练失败"，于是**跳过部署、几小时的训练白跑**。这一行是防这个的。
    env['PYTHONIOENCODING'] = 'utf-8'
    # 训练/验证划分打乱：只影响 val，不影响测试集（TEST_SEED 保持默认 42）
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='+', choices=list(MODELS), default=None)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--nowcast-root', default=None,
                    help='【已失效，仅为兼容旧命令行而保留】two 现在读自己的 two/train/*.h5，'
                         '不再读 nowcast 分块。传了也只会打印一句提醒。')
    a = ap.parse_args()
    if a.nowcast_root:
        print(f'  [two] ⚠️ --nowcast-root {a.nowcast_root} 已失效：two 现在用自己的 h5'
              f'（two/gen_train_data.py -> two/train/*.h5），不再读 nowcast 分块', flush=True)
    names = a.only or list(MODELS)
    os.makedirs(LOG_DIR, exist_ok=True)
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
    # 退出码（2026-09-27 加）：以前不管几个模型挂掉，本脚本都退 0 -> `full_retrain.py`
    # 的 stage_train 只看本脚本的退出码，于是"五个模型全失败"也会报"全套完成 · 成功"。
    # 单个模型失败不影响其余四个（上面逐个跑、逐个部署），但整体必须让对方看得见。
    if not a.dry_run and (failed or not rows):
        print(f'\n  ⚠️ 有模型没跑成：{failed or "（一个都没产出）"} —— 退出码置 1，'
              f'让 full_retrain.py 把训练段判为失败（日志见 retrain_logs/）')
        sys.exit(1)


if __name__ == '__main__':
    main()
