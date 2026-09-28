"""
evaluate.py — one_v2 模型样本外排序能力评估（截面 IC/ICIR）。

one_v2：输入 [889, 31]，输出 close_volume_delta(辅助，忽略) + close_preds(5 个 horizon)。
排序分数 = Σ close_1..close_127（与 get_one_predict.py 的 up_down 一致）。
用法：
  python evaluate.py --model one.pt --test-symbols test_symbols.txt --name run1
"""

import os
import math
import argparse
import datetime

import numpy as np
import pandas as pd
import torch

from one_model import ONE, HORIZONS, AUX_OUTPUT_DAYS  # noqa: F401

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

DAYS_INPUT = 127 * 7
TOMORROW_IDX = DAYS_INPUT
HORIZON_NAMES = [f'close_{h}' for h in HORIZONS]
N_HORIZONS = len(HORIZONS)
STATS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'target_stats.npz')


def select_device():
    if torch.cuda.is_available():
        try:
            (torch.zeros(2, 2, device='cuda') @ torch.zeros(2, 2, device='cuda')).cpu()
            return torch.device('cuda')
        except Exception as e:
            print(f'[警告] CUDA 检测到但 kernel 不可用（{type(e).__name__}），退回 CPU。')
    return torch.device('cpu')


def load_model(model_path, device):
    model = torch.load(model_path, map_location=device, weights_only=False)
    return model.to(device).eval()


def load_stats():
    s = np.load(STATS_PATH)
    return s['close_mean'].astype(np.float32), s['close_std'].astype(np.float32)


def compute_raw_labels(data):
    """5 个 close 标签（原始 log 比值，未标准化）。"""
    close_tomorrow = data['close'].iloc[TOMORROW_IDX]
    labels = []
    for h in HORIZONS:
        w = data['close'].iloc[TOMORROW_IDX + 1: TOMORROW_IDX + 1 + h]
        g = math.log(w.max()) + math.log(w.median()) + math.log(w.min()) - 3.0 * math.log(close_tomorrow)
        labels.append(g)
    return np.array(labels, dtype=np.float32)


def list_test_files(train_folder, symbols=None):
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
        files.append((os.path.join(train_folder, fn), symbol, end))
    return files


@torch.no_grad()
def run_predictions(model, files, device, batch_size, close_mean, close_std):
    rows = []
    buf_x, buf_meta = [], []

    def flush():
        if not buf_x:
            return
        x = torch.tensor(np.stack(buf_x)).float().to(device)
        _, close_preds = model(x)  # (B, 1, 5)
        close_preds = close_preds.reshape(len(buf_x), N_HORIZONS).cpu().numpy()
        # 2026-09-26：模型现在**直接输出真实单位**（标签只除尺度、不减均值；
        # 且 one/train.py 存盘时会写恒等 buffer）-> 这里不能再换算，否则双重换算。
        # ⚠️ 若用旧 .pt（z 空间），需要改回 `close_preds * close_std + close_mean`。
        for i, (sym, end, real) in enumerate(buf_meta):
            row = {'symbol': sym, 'endDate': end}
            for hi, hname in enumerate(HORIZON_NAMES):
                row[f'pred_{hname}'] = close_preds[i, hi]
                row[f'real_{hname}'] = real[hi]
            rows.append(row)
        buf_x.clear()
        buf_meta.clear()

    for path, sym, end in tqdm(files, desc='预测中'):
        try:
            data = pd.read_hdf(path, key='data')
        except Exception:
            continue
        x = data.iloc[:DAYS_INPUT].values.astype('float32')
        real = compute_raw_labels(data)
        buf_x.append(x)
        buf_meta.append((sym, end, real))
        if len(buf_x) >= batch_size:
            flush()
    flush()

    return pd.DataFrame(rows)


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


def append_history(history_csv, row, ranking_metric='icir_combined'):
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


def evaluate_model(model_path, train_folder, test_symbols=None, name='model',
                   output_dir='eval_results', device=None, batch_size=256,
                   min_stocks=5, deciles=10, timestamp=None):
    if device is None:
        device = select_device()
    os.makedirs(output_dir, exist_ok=True)

    model = load_model(model_path, device)
    close_mean, close_std = load_stats()

    if isinstance(test_symbols, str):
        with open(test_symbols) as fp:
            test_symbols = {line.strip() for line in fp if line.strip()}
        print(f'已加载测试股票列表：{len(test_symbols)} 只')

    files = list_test_files(train_folder, test_symbols)
    if not files:
        print('没有符合条件的测试样本。')
        return None, None
    ends = [e for _, _, e in files]
    n_sym = len({s for _, s, _ in files})
    print(f'测试集：股票数 = {n_sym}，样本数 = {len(files)}，endDate ∈ [{min(ends)}, {max(ends)}]')

    df = run_predictions(model, files, device, batch_size, close_mean, close_std)
    if df.empty:
        print('预测结果为空。')
        return None, None

    df['pred_combined'] = df[[f'pred_{h}' for h in HORIZON_NAMES]].sum(axis=1)
    df['real_combined'] = df[[f'real_{h}' for h in HORIZON_NAMES]].sum(axis=1)

    print(f'\n{"=" * 80}')
    print(f' 样本外评估：{name}')
    print(f'{"=" * 80}')
    print(f'{"horizon":<12}{"IC":>9}{"ICIR":>9}{"t":>10}{"n_dates":>9}{"MSE":>12}')

    ic_by_date = {}
    summary = {}
    for h in HORIZON_NAMES + ['combined']:
        pred_col, real_col = f'pred_{h}', f'real_{h}'
        s = summarize(per_date_ic(df, pred_col, real_col, min_stocks))
        mse = ((df[pred_col] - df[real_col]) ** 2).mean()
        tag = "  <- 实际排序分数" if h == 'combined' else ""
        print(f'{h:<12}{s["ic"]:>9.4f}{s["icir"]:>9.3f}{s["t"]:>10.2f}{s["n"]:>9d}{mse:>12.4f}{tag}')
        ic_by_date[h] = per_date_ic(df, pred_col, real_col, min_stocks)
        summary[f'ic_{h}'] = s['ic']
        summary[f'icir_{h}'] = s['icir']

    print(f'\n分档回测（combined，{deciles} 档，每档真实 combined 均值）：')
    tbl = decile_backtest(df, 'pred_combined', 'real_combined', deciles, min_stocks)
    if tbl is not None:
        for b, v in tbl.items():
            mark = " (最低)" if b == 0 else (" (最高)" if b == deciles - 1 else "")
            print(f'  档 {int(b)}{mark:<6}: {v:>10.4f}')
        spread = tbl.iloc[-1] - tbl.iloc[0]
        print(f'  多空价差 (最高档 - 最低档): {spread:>10.4f}')

    sample_csv = os.path.join(output_dir, f'{name}_predictions.csv')
    df.to_csv(sample_csv, index=False)
    ic_df = pd.DataFrame(ic_by_date)
    ic_csv = os.path.join(output_dir, f'{name}_ic_by_date.csv')
    ic_df.to_csv(ic_csv)
    print(f'\n已保存：\n  逐样本预测 -> {sample_csv}\n  逐日期 IC  -> {ic_csv}')

    history_csv = os.path.join(output_dir, 'history.csv')
    hist_row = {
        'timestamp': timestamp or datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'name': name,
        'model': os.path.basename(model_path),
        'n_symbols': n_sym,
        'n_samples': len(df),
    }
    hist_row.update(summary)
    append_history(history_csv, hist_row, ranking_metric='icir_combined')

    return ic_df, df


def parse_args():
    p = argparse.ArgumentParser(description='one_v2 模型样本外排序能力评估')
    p.add_argument('--model', required=True)
    p.add_argument('--train-folder', default=None)
    p.add_argument('--test-symbols', default=None, help='测试股票列表文件（每行一个 symbol）')
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

    if not args.test_symbols:
        print('[警告] 未指定 --test-symbols：正在用【全部】样本评估，这不是样本外结果，只能用于调试！\n')

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
    )


if __name__ == '__main__':
    main()
