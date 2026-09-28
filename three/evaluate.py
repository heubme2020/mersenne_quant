"""
evaluate.py — three 模型的样本外排序能力评估（与 seven/evaluate.py 同款）。

three 模型：输入 [127, 31]，输出 growth / death 两个头，各 3 个期限（1/3/7 季度）。
标签是价格型：
  growth = 未来 N 季度收盘价中位数 / 过去最高价   （上行，越大越好）
  death  = 过去收盘价中位数 / 未来 N 季度最低价   （下行，越大越差）
排序用的分数 = growth / death（get_three_predict.py 里就是这么用的）。

典型用法：
  python evaluate.py --model three.pt --test-symbols test_symbols.txt --name run1
"""

import os
import math
import argparse
import datetime

import numpy as np
import pandas as pd
import torch

from three_model import THREE  # noqa: F401  反序列化整模型需要类在命名空间

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


HORIZON_NAMES = ['one', 'three', 'seven']  # 1/3/7 季度
NUM_LABELS = 6  # growth(3) + death(3)


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
        model = THREE(input_shape, output_shape)
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
        g, d = model(x)  # [B, 3, 1], [B, 3, 1]
        g = g.reshape(len(buf_x), 3).cpu().numpy()
        d = d.reshape(len(buf_x), 3).cpu().numpy()
        for i, (sym, end, gl, dl) in enumerate(buf_meta):
            row = {'symbol': sym, 'endDate': end}
            for hi, h in enumerate(HORIZON_NAMES):
                row[f'pred_growth_{h}'] = g[i, hi]
                row[f'real_growth_{h}'] = gl[hi]
                row[f'pred_death_{h}'] = d[i, hi]
                row[f'real_death_{h}'] = dl[hi]
                # growth/death（排序分），分母加小保护避免除零
                pd_ = d[i, hi] if abs(d[i, hi]) > 1e-6 else 1e-6
                rd_ = dl[hi] if abs(dl[hi]) > 1e-6 else 1e-6
                row[f'pred_gd_{h}'] = g[i, hi] / pd_
                row[f'real_gd_{h}'] = gl[hi] / rd_
            rows.append(row)
        buf_x.clear()
        buf_meta.clear()

    for path, sym, end in tqdm(files, desc="预测中"):
        try:
            data = pd.read_hdf(path)
        except Exception:
            continue
        x = data.iloc[:, :-NUM_LABELS].values.astype('float32')
        gl = data.iloc[0, -6:-3].values.astype('float32')   # growth 三期限
        dl = data.iloc[0, -3:].values.astype('float32')     # death 三期限
        buf_x.append(x)
        buf_meta.append((sym, end, gl, dl))
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
# 主入口（被 main() 和 train.py 共用）
# --------------------------------------------------------------------------- #
def append_history(history_csv, row, ranking_metric='icir_gd_combined'):
    """把一次评估的汇总行追加进历史 CSV，并标注当前最好的一行。

    - ranking_metric：用来判定「最好」的列名（默认排序分 gd 的 ICIR）。
      想改按 IC 判定就传 'ic_gd_combined'。
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
    if device is None:
        device = select_device()
    os.makedirs(output_dir, exist_ok=True)

    model = load_model(model_path, device, input_shape=[features, seq_len], output_shape=[3, 1])

    if isinstance(test_symbols, str):
        with open(test_symbols) as fp:
            test_symbols = {line.strip() for line in fp if line.strip()}
        print(f"已加载测试股票列表：{len(test_symbols)} 只")

    files = list_test_files(train_folder, test_start, test_end, test_symbols)
    if not files:
        print("没有符合条件的测试样本。")
        return None, None
    ends = [e for _, _, e in files]
    if test_symbols:
        n_sym = len({s for _, s, _ in files})
        print(f"测试集：股票数 = {n_sym}，样本数 = {len(files)}，endDate ∈ [{min(ends)}, {max(ends)}]")
    else:
        print(f"测试集：endDate ∈ [{min(ends)}, {max(ends)}]，样本数 = {len(files)}")

    df = run_predictions(model, files, device, batch_size)
    if df.empty:
        print("预测结果为空。")
        return None, None

    # 合并分：growth/death 两个头的三个期限求和
    for branch in ['growth', 'death', 'gd']:
        df[f'pred_{branch}_combined'] = df[[f'pred_{branch}_{h}' for h in HORIZON_NAMES]].sum(axis=1)
        df[f'real_{branch}_combined'] = df[[f'real_{branch}_{h}' for h in HORIZON_NAMES]].sum(axis=1)

    print(f"\n{'=' * 80}")
    print(f" 样本外评估：{name}")
    print(f"{'=' * 80}")
    print(f"分支/horizon{'':<4}{'IC':>9}{'ICIR':>9}{'t(乐观)':>10}{'n_dates':>9}{'MSE':>12}")

    ic_by_date = {}
    summary = {}  # 保存每个 branch/horizon 的 IC/ICIR 汇总，用于写评估历史
    # growth、death、growth/death（排序分）
    for branch in ['growth', 'death', 'gd']:
        label = {'growth': 'growth(上行)', 'death': 'death(下行)', 'gd': 'growth/death(排序分)'}[branch]
        print(f"\n【{label}】")
        for h in HORIZON_NAMES + ['combined']:
            pred_col, real_col = f'pred_{branch}_{h}', f'real_{branch}_{h}'
            s = summarize(per_date_ic(df, pred_col, real_col, min_stocks))
            mse = ((df[pred_col] - df[real_col]) ** 2).mean()
            tag = "  <- 实际排序用的分数" if (branch == 'gd' and h == 'combined') else ""
            print(f"{branch}/{h:<10}{s['ic']:>9.4f}{s['icir']:>9.3f}{s['t']:>10.2f}"
                  f"{s['n']:>9d}{mse:>12.4f}{tag}")
            key = f'ic_{branch}_{h}' if h != 'combined' else f'ic_{branch}_combined'
            ic_by_date[key] = per_date_ic(df, pred_col, real_col, min_stocks)
            base = f'{branch}_{h}' if h != 'combined' else f'{branch}_combined'
            summary[f'ic_{base}'] = s['ic']
            summary[f'icir_{base}'] = s['icir']

    # 分档回测（growth/death 合并分 = 实际排序分）
    print(f"\n分档回测（gd/combined，{deciles} 档，每档真实 growth/death 均值）：")
    tbl = decile_backtest(df, 'pred_gd_combined', 'real_gd_combined', deciles, min_stocks)
    if tbl is not None:
        for b, v in tbl.items():
            mark = " (最低)" if b == 0 else (" (最高)" if b == deciles - 1 else "")
            print(f"  档 {int(b)}{mark:<6}: {v:>10.4f}")
        spread = tbl.iloc[-1] - tbl.iloc[0]
        print(f"  多空价差 (最高档 - 最低档): {spread:>10.4f}")

    # 落盘
    sample_csv = os.path.join(output_dir, f"{name}_predictions.csv")
    df.to_csv(sample_csv, index=False)
    ic_df = pd.DataFrame(ic_by_date)
    ic_csv = os.path.join(output_dir, f"{name}_ic_by_date.csv")
    ic_df.to_csv(ic_csv)
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
    append_history(history_csv, hist_row, ranking_metric='icir_gd_combined')

    # 配对比较
    if compare_to:
        prev = pd.read_csv(compare_to).set_index('endDate')['ic_gd_combined']
        merged = pd.concat([ic_df['ic_gd_combined'].rename('new'), prev.rename('old')], axis=1).dropna()
        if merged.empty:
            print("\n配对比较失败：两次结果没有共同日期。")
        else:
            diff = merged['new'] - merged['old']
            k = len(diff)
            win = (diff > 0).mean()
            t = diff.mean() / diff.std() * math.sqrt(k) if diff.std() > 0 else float('nan')
            print(f"\n 与 {os.path.basename(compare_to)} 配对比较（gd combined IC）：")
            print(f"  共同日期数 {k}，平均 IC 提升 {diff.mean():+.4f}，胜率 {win:.1%}，配对 t {t:.2f}")

    return ic_df, df


def parse_args():
    p = argparse.ArgumentParser(description="three 模型样本外排序能力评估")
    p.add_argument('--model', required=True)
    p.add_argument('--train-folder', default=None)
    p.add_argument('--test-start', type=int, default=None)
    p.add_argument('--test-end', type=int, default=None)
    p.add_argument('--test-symbols', default=None,
                   help="测试集股票列表文件（每行一个 symbol，如 train.py 生成的 test_symbols.txt）")
    p.add_argument('--name', default='model')
    p.add_argument('--compare-to', default=None)
    p.add_argument('--output-dir', default='eval_results')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--min-stocks', type=int, default=5)
    p.add_argument('--deciles', type=int, default=10)
    p.add_argument('--features', type=int, default=127)
    p.add_argument('--seq-len', type=int, default=31)
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
        compare_to=args.compare_to,
        features=args.features,
        seq_len=args.seq_len,
    )


if __name__ == '__main__':
    main()
