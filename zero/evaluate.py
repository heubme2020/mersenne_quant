"""
evaluate.py — zero 模型的样本外排序能力评估（对齐 three/evaluate.py，单分数版）。

zero 模型：输入 [7, 3]（7 个财务 ratio × 3 个季度），输出单个避雷分（negMdd，越大越安全）。
标签：negMdd = 未来 127×3 天最大回撤取负，值域 (-1, 0]。
排序用的分数 = 模型输出的单个 scalar（get_zero_predict.py 里按它降序排序）。

典型用法：
  python evaluate.py --model zero.pt --test-symbols test_symbols.txt --name run1
"""

import os
import math
import argparse
import datetime

import numpy as np
import pandas as pd
import torch

from zero_model import ZERO  # noqa: F401  反序列化整模型需要类在命名空间

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


FEATURES = 7
SEQ_LEN = 3  # LOOKBACK_QUARTERS


# --------------------------------------------------------------------------- #
# 模型加载
# --------------------------------------------------------------------------- #
def load_model(model_path, device, input_shape, output_shape):
    try:
        obj = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        obj = torch.load(model_path, map_location=device)

    if isinstance(obj, torch.nn.Module):
        model = obj
    elif isinstance(obj, dict):
        model = ZERO(input_shape, output_shape)
        model.load_state_dict(obj)
    else:
        raise TypeError(f"无法识别的模型文件内容：{type(obj)}")

    return model.to(device).eval()


# --------------------------------------------------------------------------- #
# 圈定测试集（按文件名里的 endDate 过滤，可选按 symbol 过滤）
# --------------------------------------------------------------------------- #
def list_test_files(train_folder, test_start, test_end, symbols=None):
    files = []
    for fn in os.listdir(train_folder):
        if not fn.endswith('.h5'):
            continue
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
# 批量推理
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_predictions(model, files, device, batch_size):
    rows = []
    buf_x, buf_meta = [], []

    def flush():
        if not buf_x:
            return
        x = torch.tensor(np.stack(buf_x)).float().to(device)
        out = model(x)  # [B, 1, 1]
        pred = out.reshape(len(buf_x)).cpu().numpy()
        for i, (sym, end, lab) in enumerate(buf_meta):
            rows.append({'symbol': sym, 'endDate': end,
                         'pred_zero': float(pred[i]), 'real_zero': lab})
        buf_x.clear()
        buf_meta.clear()

    for path, sym, end in tqdm(files, desc="预测中"):
        try:
            data = pd.read_hdf(path)
        except Exception:
            continue
        x = data.iloc[:, :-1].values.astype('float32')   # [3, 7] 特征
        lab = float(data.iloc[0, -1])                    # negMdd 标签（标量）
        buf_x.append(x)
        buf_meta.append((sym, end, lab))
        if len(buf_x) >= batch_size:
            flush()
    flush()

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def per_date_ic(df, pred_col, real_col, min_stocks):
    def _ic(g):
        if len(g) < min_stocks:
            return np.nan
        return g[pred_col].rank().corr(g[real_col].rank())

    return df.groupby('endDate')[[pred_col, real_col]].apply(_ic).dropna()


def summarize(ic_series):
    s = ic_series.dropna()
    n = len(s)
    if n == 0:
        return dict(ic=np.nan, icir=np.nan, t=np.nan, n=0)
    mean, std = s.mean(), s.std()
    icir = mean / std if std > 0 else np.nan
    t = mean / std * math.sqrt(n) if std > 0 else np.nan
    return dict(ic=mean, icir=icir, t=t, n=n)


def decile_backtest(df, pred_col, real_col, q, min_stocks):
    per_date = []
    for _, g in df.groupby('endDate'):
        if len(g) < max(q, min_stocks):
            continue
        bucket = pd.qcut(g[pred_col].rank(method='first'), q, labels=False)
        per_date.append(g[real_col].groupby(bucket).mean())
    if not per_date:
        return None
    return pd.concat(per_date, axis=1).mean(axis=1)


# --------------------------------------------------------------------------- #
# 历史记录
# --------------------------------------------------------------------------- #
def append_history(history_csv, row, ranking_metric='icir_zero'):
    """把一次评估的汇总行追加进历史 CSV，并标注当前最好的一行。"""
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
                   timestamp=None):
    if device is None:
        device = select_device()
    os.makedirs(output_dir, exist_ok=True)

    model = load_model(model_path, device, input_shape=[FEATURES, SEQ_LEN], output_shape=[1, 1])

    if isinstance(test_symbols, str):
        with open(test_symbols) as fp:
            test_symbols = {line.strip() for line in fp if line.strip()}
        print(f"已加载测试股票列表：{len(test_symbols)} 只")

    files = list_test_files(train_folder, test_start, test_end, test_symbols)
    if not files:
        print("没有符合条件的测试样本。")
        return None, None
    ends = [e for _, _, e in files]
    n_sym = len({s for _, s, _ in files})
    print(f"测试集：股票数 = {n_sym}，样本数 = {len(files)}，endDate ∈ [{min(ends)}, {max(ends)}]")

    df = run_predictions(model, files, device, batch_size)
    if df.empty:
        print("预测结果为空。")
        return None, None

    print(f"\n{'=' * 80}")
    print(f" 样本外评估：{name}")
    print(f"{'=' * 80}")
    print(f"分数{'':<8}{'IC':>9}{'ICIR':>9}{'t(乐观)':>10}{'n_dates':>9}{'MSE':>12}")

    pred_col, real_col = 'pred_zero', 'real_zero'
    s = summarize(per_date_ic(df, pred_col, real_col, min_stocks))
    mse = ((df[pred_col] - df[real_col]) ** 2).mean()
    print(f"zero{'':<8}{s['ic']:>9.4f}{s['icir']:>9.3f}{s['t']:>10.2f}"
          f"{s['n']:>9d}{mse:>12.4f}  <- 实际排序用的分数")

    # 分档回测
    print(f"\n分档回测（zero 分，{deciles} 档，每档真实 negMdd 均值，应从低到高单调递增）：")
    tbl = decile_backtest(df, pred_col, real_col, deciles, min_stocks)
    if tbl is not None:
        for b, v in tbl.items():
            mark = " (最低)" if b == 0 else (" (最高)" if b == deciles - 1 else "")
            print(f"  档 {int(b)}{mark:<6}: {v:>10.4f}")
        spread = tbl.iloc[-1] - tbl.iloc[0]
        print(f"  多空价差 (最高档 - 最低档): {spread:>10.4f}")

    # 落盘
    sample_csv = os.path.join(output_dir, f"{name}_predictions.csv")
    df.to_csv(sample_csv, index=False)
    ic_series = per_date_ic(df, pred_col, real_col, min_stocks)
    ic_df = pd.DataFrame({'ic_zero': ic_series})
    ic_csv = os.path.join(output_dir, f"{name}_ic_by_date.csv")
    ic_df.to_csv(ic_csv)
    print(f"\n已保存：\n  逐样本预测 -> {sample_csv}\n  逐日期 IC  -> {ic_csv}")

    # 追加历史
    history_csv = os.path.join(output_dir, 'history.csv')
    hist_row = {
        'timestamp': timestamp or datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'name': name,
        'model': os.path.basename(model_path),
        'n_symbols': n_sym,
        'n_samples': len(df),
        'ic_zero': s['ic'],
        'icir_zero': s['icir'],
    }
    append_history(history_csv, hist_row, ranking_metric='icir_zero')

    return ic_df, df


def parse_args():
    p = argparse.ArgumentParser(description="zero 模型样本外排序能力评估")
    p.add_argument('--model', required=True)
    p.add_argument('--train-folder', default=None)
    p.add_argument('--test-start', type=int, default=None)
    p.add_argument('--test-end', type=int, default=None)
    p.add_argument('--test-symbols', default=None,
                   help="测试集股票列表文件（每行一个 symbol，如 train.py 生成的 test_symbols.txt）")
    p.add_argument('--name', default='model')
    p.add_argument('--output-dir', default='eval_results')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--min-stocks', type=int, default=5)
    p.add_argument('--deciles', type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    device = select_device()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_folder = args.train_folder or os.path.join(current_dir, 'train')

    if args.test_start is None and not args.test_symbols:
        print("[警告] 未指定 --test-start 也未指定 --test-symbols：正在用【全部】样本评估，"
              "这不是样本外结果，只能用于调试！\n")

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
    )


if __name__ == '__main__':
    main()
