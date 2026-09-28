import torch
import os
import random
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

pd.set_option('future.no_silent_downcasting', True)

# ==============================================================================
# 2026-09-25 起，`two` 阶段由 nowcast 那条线训出来的模型承担；
# 2026-09-26 起**文件名按【角色】定**：two 这一档就读 `two.pt`，不再暴露来源。
#   * 模型：`two.pt`（就放在本目录），3 个头。
#     以前叫 `nowcast_u9.pt` —— 语义误导：它是"two 阶段的模型"，不是"nowcast 的 u9"。
#     训练产物仍由 `two/nowcast/train.py` 写在 `nowcast/` 下，`refresh_model.py` 训完复制到这里。
#     ⚠️ 落盘列名仍是硬编码的 gpMed7/revMed7/taMed7（不取自模型头名）——u9 的真实头名是
#        gpMedP3F/revMedP3F/taMedP3F。列名只是下游认的标签，但**名不副实**，别拿它当语义。
#     gpMed7/revMed7/taMed7 = 未来 7 季「毛利/营收/总资产」中位数的前后变化 ÷ 总资产[jc]
#   * 本阶段行为：把候选池按【毛利头】砍掉后一半。动手的是下游
#     `one/get_one_predict.py` 那行 `up_down > up_down.median()`。
#   * 曾试过「量纲换算」up_down = 毛利头 × 总资产/市值（= 预测毛利改善额 ÷ 市值），
#     2026-09-25 用户判定**不行，已回退**。当时的实测留档（干净测试集 44,352 行逐日 IC vs
#     未来收益，全体样本）：原值 1/3/7 季 −0.001/−0.006/−0.040 → 换算后 +0.014/+0.013/+0.006；
#     但副作用是排序头部被推向大资产/低市值的银行保险股，且与 buffett（本身已是价值度量）
#     形成两层价值筛选叠加。**回退后 up_down 恢复为毛利头原值。**
#   * 三个头落盘列名 = nowcast 的头名 `gpMed7/revMed7/taMed7`（旧版是
#     seven/thirty_one/two_hundred_and_twenty_seven，语义已完全不同）。
#     ⚠️ 这两处列名必须与 `one/get_one_predict.py` 里的 drop 列表一致。
#   * 回退两条路：
#     ① 只换模型权重（同架构）—— 把 `_snapshot_20260926/two/nowcast_u6.pt` 复制成 `two.pt`；
#     ② 换回旧实现 —— 跑 `get_two_predict_legacy.py`，它读 `two_legacy.pt`
#        （旧 TWO 架构的权重，2026-09-26 从原 `two.pt` 改名而来；两套架构不通用，别混）。
#   * 特征必须与训练完全一致。2026-09-26 起用 `two_features.build_x`（从
#     two/nowcast/gen_data2.py 逐字抽取、逐位校验过），不再从 nowcast 导入，也不再用本文件里
#     那份 add_technical_factor 复制品（它算的是旧因子集 CURRENT_24，且 idx 分母写错成 380）。
# ==============================================================================
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../one'))
# 2026-09-26：改从 **two 自己** 的 two_features 导入（原先 `from gen_data2 import ...`）。
# 这一步是为了让**生产推理不再依赖 nowcast/ 目录** —— nowcast 是实验目录，用户要清理冗余，
# 删掉它之后 two 必须照样能跑。two_features 里的 build_x 是从 gen_data2 **逐字抽取**的，
# 并由 `two/_verify_vs_nowcast.py` 用真实日线窗口做过**逐位相等**校验（6/6 通过）。
from two_features import build_x, RAW as NC_RAW, WINDOW as NC_WINDOW, DAYS_INPUT as NC_DAYS_INPUT, available_date  # noqa: E402

# def add_technical_factor(data):
#     # 均线
#     data['ma3'] = data['close'].rolling(3).mean()
#     data['ma7'] = data['close'].rolling(7).mean()
#     data['ma31'] = data['close'].rolling(31).mean()
#     # rsi
#     delta = data['close'].diff()
#     gain3 = delta.where(delta > 0, 0).rolling(3).mean()
#     loss3 = -delta.where(delta < 0, 0).rolling(3).mean()
#     data['rsi3'] = (100 - (100 / (1 + (gain3 / (loss3 + 1e-6))))) * 0.01
#     gain7 = delta.where(delta > 0, 0).rolling(7).mean()
#     loss7 = -delta.where(delta < 0, 0).rolling(7).mean()
#     data['rsi7'] = (100 - (100 / (1 + (gain7 / (loss7 + 1e-6))))) * 0.01
#     gain31 = delta.where(delta > 0, 0).rolling(31).mean()
#     loss31 = -delta.where(delta < 0, 0).rolling(31).mean()
#     data['rsi31'] = (100 - (100 / (1 + (gain31 / (loss31 + 1e-6))))) * 0.01
#     # atr
#     data['atr3'] = (data['delta'].rolling(3).mean())*31
#     data['atr7'] = (data['delta'].rolling(7).mean())*31
#     data['atr31'] = (data['delta'].rolling(31).mean())*31
#     # obv
#     obv = delta * data['volume']
#     data['obv3'] = (obv.rolling(3).mean())*31
#     data['obv7'] = (obv.rolling(7).mean())*31
#     data['obv31'] = (obv.rolling(31).mean())*31
#     # corr
#     data['corr3'] = data['volume'].rolling(3).corr(data['close'])
#     data['corr7'] = data['volume'].rolling(7).corr(data['close'])
#     data['corr31'] = data['volume'].rolling(31).corr(data['close'])
#     # curvature
#     data['curvature'] = (data['close'].diff().diff())*31
#     # vma
#     data['vma_3_7'] = data['volume'].rolling(window=3).mean()/data['volume'].rolling(window=7).mean() - 1
#     data['vma_7_31'] = data['volume'].rolling(window=7).mean()/data['volume'].rolling(window=31).mean() - 1
#     # factor
#     data['factor'] = (data['close'].pct_change(3) - data['volume'].rolling(7).std()) 
#     # overnight
#     overnight = data['open']*data['close'] / data['close'].shift(1) - 1
#     data['overnight3'] = overnight.rolling(3).mean()*31
#     data['overnight7'] = overnight.rolling(7).mean()*31
#     data['overnight31'] = overnight.rolling(31).mean()*31
#     # aplha22
#     # 3_7
#     rolling_corr_3 = (data['high']*data['close']).rolling(3).corr(data['volume'])
#     delta_corr_3 = rolling_corr_3.diff(3)
#     std_close_7 = data['close'].rolling(7).std()
#     data['aplha22_3_7'] = -1 * (delta_corr_3 * std_close_7)*31
#     # 7_31
#     rolling_corr_7 = (data['high']*data['close']).rolling(7).corr(data['volume'])
#     delta_corr_7 = rolling_corr_7.diff(7)
#     std_close_31 = data['close'].rolling(31).std()
#     data['aplha22_7_31'] = -1 * (delta_corr_7 * std_close_31)*31
#     return data


def raise_on_empty_merge(merged, left_dates, right_dates, source):
    """按 (symbol, date) inner merge 后为空时报错，避免静默写出空 CSV。

    与 one/get_one_predict.py 里的同名函数一致：日期错位（上游股票池没跟着
    日线数据刷新）会让两边 (symbol, date) 完全不相交，这里直接报出两边日期。
    """
    if not merged.empty:
        return
    left_set, right_set = set(left_dates), set(right_dates)
    raise RuntimeError(
        f"与 {source} 按 (symbol, date) 合并后无任何匹配：\n"
        f"  本次预测日期: {sorted(left_set)[:5]}（共 {len(left_set)} 个）\n"
        f"  {source} 日期 : {sorted(right_set)[:5]}（共 {len(right_set)} 个）\n"
        f"  日期交集: {sorted(left_set & right_set)[:5]}（共 {len(left_set & right_set)} 个）\n"
        f"  请先确认 {source} 是否已随最新的日线数据一起刷新。"
    )


def load_ashare_fundamentals(data_name):
    """A 股每只股票【最新已披露】季度的 (总资产, 毛利, 股本) -> {symbol: (avail, ta, gp, sh)}。

    披露日按 A 股规则推算（`gen_data2.available_date`）；取 asof ≤ 预测日的最后一个季度
    —— 生产上不能用还没披露的报表。
    用途（2026-09-26 用户定案）：第三个估值块 `毛利/市值` 的分子 =
        **最新季毛利 + 模型预测的毛利提升额**，其中 预测提升额 = 模型输出(Δ/总资产) × 总资产。
    """
    frames = []
    for ex in ('SHZ', 'SHH'):
        bal = pd.read_csv(f'{data_name}{ex}/balance_{ex.lower()}.csv',
                          usecols=['symbol', 'endDate', 'totalAssets'])
        inc = pd.read_csv(f'{data_name}{ex}/income_{ex.lower()}.csv',
                          usecols=['symbol', 'endDate', 'grossProfit', 'weightedAverageShsOut'])
        frames.append(bal.merge(inc, on=['symbol', 'endDate'], how='outer'))
    f = pd.concat(frames, ignore_index=True)
    f['symbol'] = f.symbol.astype(str)
    f = f[f.endDate.astype(str).str[4:].isin(['0331', '0630', '0930', '1231'])]
    f = f.drop_duplicates(subset=['symbol', 'endDate'], keep='last')
    f['avail'] = available_date(f.endDate.values)
    f = f.sort_values(['symbol', 'endDate']).reset_index(drop=True)
    out = {}
    for s, g in f.groupby('symbol', sort=False):
        out[s] = (g['avail'].values, g['totalAssets'].values,
                  g['grossProfit'].values, g['weightedAverageShsOut'].values)
    return out


def get_two_candidates(check_days=0, target_date=None):
    # 检查GPU是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 模型文件按【角色】命名：two 这一档读 `two.pt`（不再叫 nowcast_u9.pt 那类暴露来源的名字）。
    # 换用 u9 变体的依据：终点指标（逐日截面 IC vs 未来收益）上的**配对比较** —— 同一批
    # 44,352 行上 u9 至少不差于 u6（1/3 季打平、7 季更好），且多覆盖 1,818 行 / 11 只股票。
    # 详见 `two/nowcast/paired_u6_u9.py`、`two/nowcast/variants.py` 的 u9 段、`refresh_model.py` 的 two 配方。
    # ⚠️ `two.pt` 必须存在，否则本文件会在这里直接抛异常、每日选票全断。
    # 回退（同架构换权重）：cp _snapshot_20260926/two/nowcast_u6.pt two/two.pt
    model_name = os.path.join(os.path.dirname(__file__), 'two.pt')
    model = torch.load(model_name, weights_only=False).to(device)
    # 加载daily数据
    data_name = os.path.join(os.path.dirname(__file__), '../data/')
    daily_shz = pd.read_csv(data_name + 'SHZ/daily_shz.csv')
    daily_shh = pd.read_csv(data_name + 'SHH/daily_shh.csv')
    daily_data = pd.concat([daily_shz, daily_shh], axis=0).reset_index(drop=True)
    # 与训练端一致（gen_data2.load_daily）：只保留 2000 年以后、且按 (symbol, date) 排序。
    # 不这样过滤的话，对历史不足 889 行的股票会用 2000 年前的行凑窗口 -> 特征出分布外。
    daily_data = daily_data[daily_data['symbol'].notna() & (daily_data['date'] >= 20000101)]
    daily_data = daily_data.sort_values(['symbol', 'date']).reset_index(drop=True)
    if target_date is not None:
        daily_data = daily_data[daily_data['date'] <= int(target_date)].reset_index(drop=True)
    zero_data = pd.read_csv(data_name + '../zero/zero_predict.csv')
    zero_data.drop(columns=['three'], inplace=True)
    zero_data.drop(columns=['seven'], inplace=True)
    zero_data.drop(columns=['thirty_one'], inplace=True)
    zero_data = zero_data[zero_data['zero'] > zero_data['zero'].median()].reset_index(drop=True)
    print(zero_data)
    schloss_data = pd.read_csv(data_name + '../schloss/schloss.csv')
    schloss_data.drop(columns=['endDate'], inplace=True)
    print(schloss_data)
    buffett_data = pd.merge(schloss_data, zero_data, on=['symbol'], how='inner').dropna().reset_index(drop=True)
    buffett_data['buffett'] = buffett_data['dcf']*buffett_data['netAssetValuePerShare']/buffett_data['close'] + buffett_data['schloss']
    buffett_data = buffett_data.sort_values('buffett', ascending=False)
    buffett_data = buffett_data.reset_index(drop=True)
    # buffett_data = buffett_data[buffett_data['buffett'] > buffett_data['buffett'].median()].reset_index(drop=True)
    buffett_list = buffett_data['symbol'].to_list()
    # 第三个估值块的分子要用的财务数据：最新【已披露】季度的 总资产/毛利/股本
    print('读财务数据（总资产 / 毛利 / 股本）...', flush=True)
    funda = load_ashare_fundamentals(data_name)
    groups = list(daily_data.groupby('symbol'))
    predict_list = []
    n_skip_funda = 0
    for i in tqdm(range(len(groups))):
        symbol = groups[i][0]
        if symbol not in buffett_list:
            continue
        daily_group = groups[i][1].reset_index(drop=True)
        if check_days != 0:
            daily_group = daily_group.iloc[:-check_days].reset_index(drop=True)
        daily_group = daily_group.iloc[-NC_WINDOW:].reset_index(drop=True)
        group_data_length = len(daily_group)
        if group_data_length != NC_WINDOW:
            continue
        date = daily_group['date'].iloc[-1]

        # --- 特征：完全复用 nowcast 训练时的那一份（归一化 / 因子集 / idx 分母都在里面）---
        daily_input = build_x(daily_group)
        if daily_input is None:        # 窗口末行的 close/volume 非正
            continue
        # build_x 返回 1017 行（889 输入 + 128 辅助），模型只吃前 889 行 —— 训练时也是 xf[:, :DAYS_INPUT]
        daily_input = torch.tensor(daily_input[:NC_DAYS_INPUT]).unsqueeze(0).float().to(device)
        _, scalar = model(daily_input)          # nowcast: (aux, scalar)，scalar = (B, 3 字段, 1 期限)
        heads = scalar.squeeze(0)[:, 0].cpu().detach().numpy()     # [毛利, 营收, 总资产]
        gp, rev, ta = float(heads[0]), float(heads[1]), float(heads[2])
        # 三个头按 nowcast 自己的头名落盘。
        # ⚠️ 改名后必须同步改 `one/get_one_predict.py` 里那行 drop 列表，否则 one 会 KeyError。
        # --- 第三个估值块：毛利/市值，分子 = 最新季毛利 + 模型预测的毛利提升额 ---
        #     预测提升额 = 模型输出(Δ/总资产) × 总资产；总资产取【最新已披露】季度（lag-safe）
        #     财务缺失时该块记 NaN（后面按可用块求平均，不丢股票）
        fp = funda.get(symbol)
        ta_val = gp_last = sh_val = np.nan
        if fp is not None:
            av, tas, gps, shs = fp
            k = int(np.searchsorted(av, int(date), side='right')) - 1
            if k >= 0:
                ta_val, gp_last, sh_val = float(tas[k]), float(gps[k]), float(shs[k])
        mcap = sh_val * float(daily_group['close'].iloc[-1])
        if np.isfinite(mcap) and mcap > 0 and np.isfinite(ta_val) and np.isfinite(gp_last):
            # 模型输出**已经是真实单位**（反标准化焼在模型的 buffer 里，见 one_model.set_output_scale）
            # -> 直接乘总资产就是"预测的毛利提升额（元）"，不需要任何手工换算。
            delta_gp = gp * ta_val
            gp_yield = (gp_last + delta_gp) / mcap
        else:
            gp_yield = np.nan
            n_skip_funda += 1
        up_down = gp          # ← 下游按这列做中位数过滤 = 「按毛利头砍掉后一半」（与估值公式无关）
        predict_result = {'symbol': symbol, 'date': int(date), 'gpMed7': gp, 'revMed7': rev,
                          'taMed7': ta, 'up_down': up_down, 'gp_yield': gp_yield}
        predict_list.append(predict_result)
    print(f'  财务缺失（第三个估值块记 NaN）{n_skip_funda} 只', flush=True)
    predict_data = pd.DataFrame(predict_list)
    predict_dates = predict_data['date'].tolist() if 'date' in predict_data.columns else []
    predict_data = pd.merge(predict_data, buffett_data, on=['symbol', 'date'], how='inner').reset_index(drop=True)
    # 只对原有列做 dropna（gp_yield 允许缺失，不能因此丢股票）
    predict_data = predict_data.dropna(
        subset=[c for c in predict_data.columns if c != 'gp_yield']).reset_index(drop=True)
    raise_on_empty_merge(predict_data, predict_dates, buffett_data['date'].tolist(), 'schloss/schloss.csv')

    # ================= 估值公式（2026-09-26 用户定案：三块，直接相加）=================
    #   ① dcf × B/P        = 前瞻自由现金流收益率（seven 模型输出 × 净资产/股价）
    #   ② 每股运营资本/股价 = 格雷厄姆 NCAV/市值（schloss/get_schloss.py 已按标准口径修正）
    #   ③ 毛利/市值         = (最新季毛利 + 模型预测的毛利提升额) / 市值   ← **×7**
    # 三块的原始量级：① ~0.43 / ② ~0.10 / ③ ~0.052 → ③ 乘 7 后中位 ~0.37，与 ① 的 0.43 可比。
    # **用户 2026-09-26 定 ×7（按"幅度对齐"）**：三块都是"多少公司价值"的同一量纲，
    #   乘法后中位数可比，含义清楚。
    # ⚠️ 已知副作用（读结果时记住）：×7 会放大 ③ 的负尾（小市值亏毛利的公司，③ 最低到 −18.8），
    #   所以在**排序相关性**上 ③ 话语权最大（0.780 vs ① 0.187）。但**中位数口径下 ③ (0.366)
    #   仍小于 ① (0.429)**，且 ×7 版前 10 与"只用③"只重叠 3/10 —— 没有到"独占"。
    #   若要更均衡，可改用 ×2.6（影响力 0.40/0.60/0.50），或**三个块各自去极值**后再说。
    # **不做标准化**（用户要求）：保留各块数值的大小信息（越极端=信号越强）。
    # ⚠️ 缺 ③ 的股票（财务缺失）会少一项，评分偏低；今日 0 只。
    blk = pd.DataFrame(index=predict_data.index)
    blk['①dcf×B/P'] = predict_data['dcf'] * predict_data['netAssetValuePerShare'] / predict_data['close']
    blk['②NCAV/市值'] = predict_data['schloss']
    blk['③毛利/市值'] = predict_data['gp_yield']
    W = pd.Series({'①dcf×B/P': 1.0, '②NCAV/市值': 1.0, '③毛利/市值': 7.0})
    wsum = (blk * W).sum(axis=1)
    predict_data['buffett'] = wsum

    print(f'\n{"="*90}\n  估值公式诊断（{len(predict_data)} 只，直接相加，③×7）\n{"="*90}')
    print(f'  {"块":14s}{"权重":>6s}{"中位":>10s}{"p10":>10s}{"p90":>10s}'
          f'{"min":>10s}{"max":>10s}{"p10~p90跨度":>13s}{"缺失":>6s}')
    for c in blk.columns:
        v = blk[c] * W[c]
        print(f'  {c:14s}{W[c]:>6.1f}{v.median():>+10.3f}{v.quantile(.1):>+10.3f}'
              f'{v.quantile(.9):>+10.3f}{v.min():>+10.3f}{v.max():>+10.3f}'
              f'{v.quantile(.9)-v.quantile(.1):>13.3f}{v.isna().sum():>6d}')
    print(f'  {"buffett（合计）":14s}{"":>6s}{wsum.median():>+12.4f}{wsum.quantile(.1):>+12.4f}'
          f'{wsum.quantile(.9):>+12.4f}')
    print(f'\n  三块之间的秩相关：')
    r = blk.rank(pct=True)
    for i, a in enumerate(blk.columns):
        for b in blk.columns[i + 1:]:
            print(f'    {a} × {b} = {r[a].corr(r[b]):+.3f}')
    print(f'\n  加权后各块对总分的"影响力"（与总分的秩相关，越接近 1 越主导）:')
    for c in blk.columns:
        print(f'    {c:14s}{wsum.rank(pct=True).corr(r[c]):+.3f}')
    # 三块正交时，均势（各块离散度相等）对应每块影响力 ≈ 1/√3 = 0.577
    print(f'\n  ③ 的系数扫描（看三块什么时候均势；三块正交 -> 等离散度时各块 ≈ 0.577）:')
    print(f'    {"③系数":>8s}{"①":>9s}{"②":>9s}{"③":>9s}')
    for k in [1.0, 1.5, 2.0, 2.6, 3.5, 5.0, 7.0]:
        s = blk['①dcf×B/P'] + blk['②NCAV/市值'] + k * blk['③毛利/市值']
        sr = s.rank(pct=True)
        print(f'    {k:>8.1f}' + ''.join(f'{sr.corr(r[c]):>+9.3f}' for c in blk.columns))
    Z = (blk.rank(pct=True) - 0.5).mean(axis=1)      # 仅供对照，不落盘
    # ---- 砍半后池子（= one 实际打分的池子，`up_down > 中位数`）的三块统计 ----
    half = predict_data[predict_data['up_down'] > predict_data['up_down'].median()]
    hb = (blk * W).loc[half.index]
    print(f'\n{"="*126}')
    print(f'  砍半后池子（up_down > 中位数）= one 实际打分的 {len(half)} 只：三块统计')
    print(f'{"="*126}')
    hdr = (f'  {"块":20s}{"n":>5s}{"缺失":>5s}{"均值":>10s}{"标准差":>10s}{"最小":>11s}'
           f'{"p5":>9s}{"p25":>9s}{"中位":>9s}{"p75":>9s}{"p95":>9s}{"最大":>11s}')
    print(hdr)
    for c in blk.columns:
        v = hb[c]
        print(f'  {c + "(×%d)" % W[c]:20s}{len(v):>5d}{v.isna().sum():>5d}{v.mean():>+10.3f}'
              f'{v.std():>10.3f}{v.min():>+11.3f}{v.quantile(.05):>+9.3f}{v.quantile(.25):>+9.3f}'
              f'{v.median():>+9.3f}{v.quantile(.75):>+9.3f}{v.quantile(.95):>+9.3f}{v.max():>+11.3f}')
    t = half['buffett']
    print(f'  {"buffett 合计":20s}{len(t):>5d}{t.isna().sum():>5d}{t.mean():>+10.3f}'
          f'{t.std():>10.3f}{t.min():>+11.3f}{t.quantile(.05):>+9.3f}{t.quantile(.25):>+9.3f}'
          f'{t.median():>+9.3f}{t.quantile(.75):>+9.3f}{t.quantile(.95):>+9.3f}{t.max():>+11.3f}')
    print(f'\n  各块对总分的平均贡献占比（|块均值×权重| / Σ|块均值×权重|）:')
    am = hb.abs().mean()
    for c in blk.columns:
        print(f'    {c:16s}{am[c] / am.sum() * 100:>6.1f}%')
    print(f'  砍半后三块之间的秩相关:')
    hr = hb.rank(pct=True)
    for i, a in enumerate(blk.columns):
        for b in blk.columns[i + 1:]:
            print(f'    {a} × {b} = {hr[a].corr(hr[b]):+.3f}')
    print(f'\n  对照：本版 与「标准化等权相加」版的秩相关 = '
          f'{wsum.rank(pct=True).corr(Z.rank(pct=True)):+.3f}')
    print(f'\n  本版   前 10 : {" ".join(predict_data.nlargest(10, "buffett")["symbol"])}')
    print(f'  只用①前 10 : {" ".join(predict_data.assign(_d=blk["①dcf×B/P"]).nlargest(10, "_d")["symbol"])}')
    print(f'  只用②前 10 : {" ".join(predict_data.assign(_n=blk["②NCAV/市值"]).nlargest(10, "_n")["symbol"])}')
    print(f'  只用③前 10 : {" ".join(predict_data.assign(_g=blk["③毛利/市值"]).nlargest(10, "_g")["symbol"])}')

    predict_data.drop(columns=['gp_yield'], inplace=True)      # 内部列，不落盘（避免级联进 one/buy）
    predict_data = predict_data.sort_values('buffett', ascending=False)
    predict_data = predict_data.reset_index(drop=True)
    print(predict_data)
    two_predict_name = os.path.join(os.path.dirname(__file__), 'two_predict.csv')
    predict_data.to_csv(two_predict_name, index=False)
 

    
if __name__ == "__main__":
    get_two_candidates()
    # get_two_all()