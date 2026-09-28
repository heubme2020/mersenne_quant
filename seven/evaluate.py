"""
evaluate.py — 样本外(out-of-time)排序能力评估，用于横向对比不同模型/方法是否有改进。

它做的事：
  1. 读取 train/ 文件夹里的 .h5 样本（每个文件已含模型输入 + 真实 9 个标签）
  2. 只取 endDate 在指定测试期内的样本（样本外测试集）
  3. 跑模型得到 [batch, 3(branch), 3(horizon)] 预测，和真实标签按 endDate 横截面对比
  4. 对每个分支（fcf/dividend/netasset）分别输出各 horizon 及合并分的 Rank IC / ICIR / 分档回测
  5. 可选：和上一版 per-date IC 结果做逐日期配对比较（默认比 fcf 分支的合并分，即实际排序分数）

★★★ 必读前提 ★★★
这个脚本只负责"算指标"，不保证指标有效。要让对比有意义，被评估的模型
必须没有在测试期(endDate >= --test-start)上训练过。也就是说，训练时要先把
测试期从训练文件里剔除（见随附说明里给 train.py 的几行改动）。否则模型见过这些
样本，算出来的 IC 是泄漏后的虚高值。

典型用法：
  python evaluate.py --model seven.pt --test-start 20210101 --name new \
                     --compare-to eval_results/old_ic_by_date.csv

endDate 用整数比较（你的数据是什么整数格式就传什么，如 YYYYMMDD 传 20210101）。
"""

import os
import math
import argparse
import datetime

import numpy as np
import pandas as pd
import torch

# 必须 import 模型类：torch.load 整模型(whole-model pickle)反序列化时需要类在命名空间里
from seven_model import SEVEN, BRANCHES, HORIZONS, HORIZON_NAMES, label_columns  # noqa: F401

try:
    from tqdm import tqdm
except Exception:  # tqdm 不存在也能跑
    def tqdm(x, **kwargs):
        return x


NUM_LABELS = len(BRANCHES) * len(HORIZONS)  # 9


# --------------------------------------------------------------------------- #
# 模型加载：同时兼容"整模型 pickle"和"state_dict"两种保存方式
# --------------------------------------------------------------------------- #
def load_model(model_path, device, input_shape):
    try:
        # weights_only=False 是因为你现在 seven.pt 存的是整模型对象；
        # 若以后改成存 state_dict，下面的 isinstance 分支会自动处理
        obj = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        # 老版本 PyTorch 没有 weights_only 参数
        obj = torch.load(model_path, map_location=device)

    if isinstance(obj, torch.nn.Module):
        model = obj
    elif isinstance(obj, dict):
        model = SEVEN(input_shape)
        model.load_state_dict(obj)
    else:
        raise TypeError(f"无法识别的模型文件内容：{type(obj)}")

    return model.to(device).eval()


# --------------------------------------------------------------------------- #
# 圈定样本外测试集（按文件名里的 endDate 过滤）
# --------------------------------------------------------------------------- #
def list_test_files(train_folder, test_start, test_end, symbols=None):
    files = []
    for fn in os.listdir(train_folder):
        if not fn.endswith('.h5'):
            continue
        # 文件名形如 {symbol}_{endDate}.h5；symbol 可能含下划线，按最后一个下划线切
        try:
            symbol, end_str = fn[:-3].rsplit('_', 1)
            end = int(end_str)
        except (ValueError, IndexError):
            continue
        if symbols is not None and symbol not in symbols:
            continue
        if test_start is not None and end < test_start:
            continue
        if test_end is not None and end > test_end:
            continue
        files.append((os.path.join(train_folder, fn), symbol, end))
    return files


# --------------------------------------------------------------------------- #
# 批量推理：读 .h5 -> 攒成 batch -> 过模型 -> 收集预测与真实标签
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_predictions(model, files, device, batch_size):
    rows = []
    buf_x, buf_meta = [], []

    def flush():
        if not buf_x:
            return
        x = torch.tensor(np.stack(buf_x)).float().to(device)
        out = model(x)  # [B, 3(branch), 3(horizon)]
        out = out.detach().cpu().numpy()
        for i, (sym, end, labels) in enumerate(buf_meta):
            row = {'symbol': sym, 'endDate': end}
            for bi, b in enumerate(BRANCHES):
                for hi, h in enumerate(HORIZON_NAMES):
                    row[f'pred_{b}_{h}'] = out[i, bi, hi]
                    row[f'real_{b}_{h}'] = labels[bi * len(HORIZON_NAMES) + hi]
                # 每个分支的合并分 = 三个期限求和（fcf 分支的合并分即实际排序用的分数）
                row[f'pred_{b}_combined'] = out[i, bi, :].sum()
                row[f'real_{b}_combined'] = labels[bi * len(HORIZON_NAMES):(bi + 1) * len(HORIZON_NAMES)].sum()
            rows.append(row)
        buf_x.clear()
        buf_meta.clear()

    for path, sym, end in tqdm(files, desc="预测中"):
        try:
            data = pd.read_hdf(path)
        except Exception:
            continue  # 跳过损坏文件
        x = data.iloc[:, :-NUM_LABELS].values.astype('float32')   # 特征（最后 9 列是标签）
        labels = data.iloc[0, -NUM_LABELS:].values.astype('float32')  # branch-major 顺序的 9 个标签
        buf_x.append(x)
        buf_meta.append((sym, end, labels))
        if len(buf_x) >= batch_size:
            flush()
    flush()

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 指标：Rank IC（每个 endDate 横截面的 Spearman 秩相关，再按日期看分布）
# --------------------------------------------------------------------------- #
def per_date_ic(df, pred_col, real_col, min_stocks):
    def _ic(g):
        if len(g) < min_stocks:
            return np.nan
        # 秩的 Pearson 相关 == Spearman 相关，纯 pandas，无需 scipy
        return g[pred_col].rank().corr(g[real_col].rank())

    return df.groupby('endDate')[[pred_col, real_col]].apply(_ic).dropna()


def summarize(ic_series):
    s = ic_series.dropna()
    n = len(s)
    if n == 0:
        return dict(ic=np.nan, icir=np.nan, t=np.nan, n=0)
    mean, std = s.mean(), s.std()
    icir = mean / std if std > 0 else np.nan
    # 注意：这个 t-stat 偏乐观——标签横跨 31 季、日期高度自相关，有效样本数 < n
    t = mean / std * math.sqrt(n) if std > 0 else np.nan
    return dict(ic=mean, icir=icir, t=t, n=n)


# --------------------------------------------------------------------------- #
# 分档回测：每期按预测分排成 q 档，看每档真实标签均值（好模型应单调递增）
# --------------------------------------------------------------------------- #
def decile_backtest(df, pred_col, real_col, q, min_stocks):
    per_date = []
    for _, g in df.groupby('endDate'):
        if len(g) < max(q, min_stocks):
            continue
        bucket = pd.qcut(g[pred_col].rank(method='first'), q, labels=False)
        per_date.append(g[real_col].groupby(bucket).mean())
    if not per_date:
        return None
    # (档位 × 日期) 取每档跨日期均值，等权各期
    return pd.concat(per_date, axis=1).mean(axis=1)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="样本外排序能力评估")
    p.add_argument('--model', required=True, help="模型文件路径（整模型 pickle 或 state_dict 均可）")
    p.add_argument('--train-folder', default=None, help="样本 .h5 所在目录，默认脚本同级 train/")
    p.add_argument('--test-start', type=int, default=None,
                   help="测试期起始 endDate（整数）。不传则用全部样本——仅用于调试，不算样本外！")
    p.add_argument('--test-end', type=int, default=None, help="测试期结束 endDate（整数），可选")
    p.add_argument('--test-symbols', default=None,
                   help="测试集股票列表文件（每行一个 symbol，如 train.py 生成的 test_symbols.txt）。"
                        "提供后只在这些股票上评估（股票隔离的测试集）。")
    p.add_argument('--name', default='model', help="本次评估的名字（用于输出文件命名）")
    p.add_argument('--compare-to', default=None,
                   help="上一版的 *_ic_by_date.csv，做逐日期配对比较")
    p.add_argument('--output-dir', default='eval_results', help="结果输出目录")
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--min-stocks', type=int, default=5, help="某 endDate 横截面少于该数则跳过（太小的截面 IC 是噪声）")
    p.add_argument('--deciles', type=int, default=10)
    p.add_argument('--features', type=int, default=127, help="输入特征数（仅 state_dict 加载时用于重建模型）")
    p.add_argument('--seq-len', type=int, default=31, help="输入季度数（仅 state_dict 加载时用于重建模型）")
    return p.parse_args()


def append_history(history_csv, row, ranking_metric='icir_fcf_combined'):
    """把一次评估的汇总行追加进历史 CSV，并标注当前最好的一行。

    - ranking_metric：用来判定「最好」的列名（默认排序分 fcf 的 ICIR）。
      想改按 IC 判定就传 'ic_fcf_combined'。
    - 追加后重写整个 CSV，给该指标最优的那一行打上 is_best='BEST'。
    """
    new_df = pd.DataFrame([row])
    if os.path.exists(history_csv):
        hist = pd.read_csv(history_csv)
        hist = hist.drop(columns=['is_best'], errors='ignore')
        hist = pd.concat([hist, new_df], ignore_index=True)
    else:
        hist = new_df

    best_idx = None
    if ranking_metric in hist.columns and not hist[ranking_metric].isna().all():
        best_idx = int(hist[ranking_metric].idxmax())
        hist['is_best'] = ['BEST' if i == best_idx else '' for i in range(len(hist))]
    else:
        hist['is_best'] = ''

    hist.to_csv(history_csv, index=False)

    if best_idx is not None:
        b = hist.iloc[best_idx]
        print(f"  当前最好（按 {ranking_metric}）：{b['timestamp']}  {b['model']}  "
              f"{ranking_metric}={b[ranking_metric]:.4f}")
    print(f"  评估历史 -> {history_csv}")



def select_device():
    """选择设备：优先 CUDA，但会实际跑一个 matmul 验证 kernel 可用，否则退回 CPU。

    torch.cuda.is_available() 只检测设备是否存在，不保证 kernel 镜像存在
    （如 cu126 装在 RTX 50 系 Blackwell GPU 上，is_available=True 但一执行就崩）。
    """
    if torch.cuda.is_available():
        try:
            (torch.zeros(2, 2, device='cuda') @ torch.zeros(2, 2, device='cuda')).cpu()
            return torch.device('cuda')
        except Exception as e:
            print(f'[警告] CUDA 设备检测到但 kernel 不可用（{type(e).__name__}），退回 CPU 运行。')
    return torch.device('cpu')

def evaluate_model(model_path, train_folder, test_symbols=None, name='model',
                   output_dir='eval_results', device=None, batch_size=256,
                   min_stocks=5, deciles=10, test_start=None, test_end=None,
                   compare_to=None, features=127, seq_len=31, timestamp=None):
    """对一个已训练好的模型做样本外评估，返回 (ic_df, df)。

    main() 和 train.py 共用的入口：
      - test_symbols: 股票列表文件路径或 set；给定后只在这些股票上评估（股票隔离的测试集）。
      - 返回逐日期 IC 表 (ic_df) 与逐样本预测 (df)，调用方可进一步处理。
    """
    if device is None:
        device = select_device()
    os.makedirs(output_dir, exist_ok=True)

    # 1) 加载模型
    model = load_model(model_path, device, input_shape=[features, seq_len])

    # 2) 圈定测试集
    if isinstance(test_symbols, str):
        with open(test_symbols) as fp:
            test_symbols = {line.strip() for line in fp if line.strip()}
        print(f"已加载测试股票列表：{len(test_symbols)} 只")

    files = list_test_files(train_folder, test_start, test_end, test_symbols)
    if not files:
        print("没有符合条件的测试样本，请检查 --train-folder / --test-start / --test-symbols。")
        return None, None
    ends = [e for _, _, e in files]
    if test_symbols:
        n_sym = len({s for _, s, _ in files})
        print(f"测试集：股票数 = {n_sym}，样本数 = {len(files)}，endDate ∈ [{min(ends)}, {max(ends)}]")
    else:
        print(f"测试集：endDate ∈ [{min(ends)}, {max(ends)}]，样本数 = {len(files)}")

    # 3) 推理
    df = run_predictions(model, files, device, batch_size)
    if df.empty:
        print("预测结果为空（文件可能都读取失败）。")
        return None, None

    # 4) 每个分支的 Rank IC 表
    print(f"\n{'=' * 80}")
    print(f" 样本外评估：{name}")
    print(f"{'=' * 80}")
    print(f"分支/horizon{'':<4}{'IC':>9}{'ICIR':>9}{'t(乐观)':>10}{'n_dates':>9}{'MSE':>12}")

    ic_by_date = {}  # 保存每个分支每个 horizon 的逐日期 IC，供落盘
    summary = {}     # 保存每个分支每个 horizon 的 IC/ICIR 汇总，用于写评估历史
    for b in BRANCHES:
        print(f"\n【{b} 分支】")
        for h in HORIZON_NAMES + ['combined']:
            pred_col, real_col = f'pred_{b}_{h}', f'real_{b}_{h}'
            s = summarize(per_date_ic(df, pred_col, real_col, min_stocks))
            mse = ((df[pred_col] - df[real_col]) ** 2).mean()
            tag = "  <- 实际排序用的分数" if (b == 'fcf' and h == 'combined') else ""
            print(f"{b}/{h:<10}{s['ic']:>9.4f}{s['icir']:>9.3f}{s['t']:>10.2f}"
                  f"{s['n']:>9d}{mse:>12.4f}{tag}")
            key = f'ic_{b}_{h}' if h != 'combined' else f'ic_{b}_combined'
            ic_by_date[key] = per_date_ic(df, pred_col, real_col, min_stocks)
            base = f'{b}_{h}' if h != 'combined' else f'{b}_combined'
            summary[f'ic_{base}'] = s['ic']
            summary[f'icir_{base}'] = s['icir']

    # 5) 分档回测（fcf 合并分，即实际排序分数）
    print(f"\n分档回测（fcf/combined，{deciles} 档，每档真实 FCF/净资产均值，应从低到高单调递增）：")
    tbl = decile_backtest(df, 'pred_fcf_combined', 'real_fcf_combined', deciles, min_stocks)
    if tbl is not None:
        for b, v in tbl.items():
            mark = " (最低)" if b == 0 else (" (最高)" if b == deciles - 1 else "")
            print(f"  档 {int(b)}{mark:<6}: {v:>10.4f}")
        spread = tbl.iloc[-1] - tbl.iloc[0]
        print(f"  多空价差 (最高档 - 最低档): {spread:>10.4f}")
    print("  注：combined 是三段嵌套累加和(早期季度被重复计入)，两边口径一致，IC 仍可比。")

    print("\n判读提示：IC>0 且 ICIR 越大越稳；分档要尽量单调、价差要明显。"
          "\n标签横跨 31 季、日期高度自相关 -> 有效样本数远小于 n_dates，"
          "IC 差零点几个百分点很可能是噪声，别据此下结论。")

    # 6) 落盘
    sample_csv = os.path.join(output_dir, f"{name}_predictions.csv")
    df.to_csv(sample_csv, index=False)
    ic_df = pd.DataFrame(ic_by_date)
    ic_csv = os.path.join(output_dir, f"{name}_ic_by_date.csv")
    ic_df.to_csv(ic_csv)  # 保留 endDate 索引，供 --compare-to 用
    print(f"\n已保存：\n  逐样本预测 -> {sample_csv}\n  逐日期 IC  -> {ic_csv}")

    # 追加到评估历史：每次评估一行，含各 branch/horizon 的 IC/ICIR，并标注当前最好的一行
    history_csv = os.path.join(output_dir, 'history.csv')
    hist_row = {
        'timestamp': timestamp or datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'name': name,
        'model': os.path.basename(model_path),
        'n_symbols': len({s for _, s, _ in files}),
        'n_samples': len(df),
    }
    hist_row.update(summary)
    append_history(history_csv, hist_row, ranking_metric='icir_fcf_combined')

    # 7) 逐日期配对比较（fcf 合并分，即实际排序分数）
    if compare_to:
        prev = pd.read_csv(compare_to).set_index('endDate')['ic_fcf_combined']
        merged = pd.concat([ic_df['ic_fcf_combined'].rename('new'),
                            prev.rename('old')], axis=1).dropna()
        if merged.empty:
            print("\n配对比较失败：两次结果没有共同日期（测试集需一致）。")
        else:
            diff = merged['new'] - merged['old']
            k = len(diff)
            win = (diff > 0).mean()
            t = diff.mean() / diff.std() * math.sqrt(k) if diff.std() > 0 else float('nan')
            print(f"\n{'=' * 64}")
            print(f" 与 {os.path.basename(compare_to)} 配对比较（fcf combined IC）")
            print(f"{'=' * 64}")
            print(f"  共同日期数        : {k}")
            print(f"  平均 IC 提升      : {diff.mean():+.4f}")
            print(f"  胜率(新>旧的日期) : {win:.1%}")
            print(f"  配对 t-stat       : {t:.2f}   (|t|>2 才比较可信，且记得它仍偏乐观)")

    return ic_df, df


def main():
    args = parse_args()
    device = select_device()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_folder = args.train_folder or os.path.join(current_dir, 'train')

    if args.test_start is None:
        print("[警告] 未指定 --test-start：正在用【全部】样本评估，这不是样本外结果，"
              "只能用于调试，不能用来对比模型！\n")

    evaluate_model(
        model_path=args.model,
        train_folder=train_folder,
        test_symbols=args.test_symbols,
        name=args.name,
        output_dir=args.output_dir,
        device=device,
        batch_size=args.batch_size,
        min_stocks=args.min_stocks,
        deciles=args.deciles,
        test_start=args.test_start,
        test_end=args.test_end,
        compare_to=args.compare_to,
        features=args.features,
        seq_len=args.seq_len,
    )


if __name__ == '__main__':
    main()
