import torch
import os
import random
import pandas as pd
import numpy as np
from tqdm import tqdm

pd.set_option('future.no_silent_downcasting', True)

# 市场 -> 交易所列表 / 产物文件后缀：美股 + A 股两套独立管线，不混横截面
MARKET_EXCHANGES = {'CN': ['SHZ', 'SHH'], 'US': ['AMEX', 'NASDAQ', 'NYSE']}
MARKET_SUFFIX = {'CN': '', 'US': '_us'}


def ncav_per_share(balance, income):
    """每股【净流动资产】（格雷厄姆 NCAV 口径）—— 2026-09-26 修正。

        NCAV = 流动资产 − 【普通股之外的全部索取权】
             = totalCurrentAssets − totalLiabilities − minorityInterest − preferredStock
        schloss = NCAV / 股价

    ⚠️ 为什么显式减 minorityInterest / preferredStock：**实测 A 股会计恒等式**
       `totalLiabilities` **不含**少数股东权益 ——
         (总负债 + 总权益) / 总资产          : 中位 0.9913，只有 50.1% 落在 [0.99,1.01]
         (总负债 + 少数股东权益 + 总权益)/总资产: 中位 1.0000，**97.4% 落在 [0.99,1.01]**
       分量不可忽略：少数股东权益/总资产 中位 0.88%、p90 7.4%，**47.8% 的样本 >1%**；
       优先股仅 3.99% 非零。
    ⚠️ 为什么不用 indicator_*.csv 的 `operatingCapitalPerShare`：那是旧口径
       (流动资产 − 流动负债 − **长期负债**)，漏掉三类：其它非流动负债（递延税、租赁负债…）
       + 少数股东权益 + 优先股；实测会**高估 NCAV**（每股中位 1.15 元 vs 正确值 0.68 元）。
       本函数就地现算，**不改 indicator 的定义**（那个字段另有消费者，且重算 indicator 是 DB 写操作）。
    ⚠️ 顺带绕开一个隐患：`longTermDebt` 在 write_stock_data 的 _FFILL_COLS 里（0 被当作缺失
       前向填充）-> 旧口径会把"已还清债务"的公司填回历史值；改用 totalLiabilities 不受影响。
    """
    # 防御：balance 帧若已含股本列，先去掉，避免 merge 后变成 _x/_y 后缀
    bal = balance.drop(columns=[c for c in ['weightedAverageShsOut'] if c in balance.columns])
    f = bal.merge(income[['symbol', 'endDate', 'weightedAverageShsOut']],
                  on=['symbol', 'endDate'], how='inner')
    # 股本为 0（基金/权证/优先股/SPAC/退市等非普通股）会导致除以 0 = inf；
    # 置成 NaN 让下游 dropna 丢掉（inf 不是 NaN，dropna 拦不住）。
    sh = f['weightedAverageShsOut'].replace(0, np.nan)
    out = f[['symbol', 'endDate']].copy()
    out['ncavPerShare'] = (f['totalCurrentAssets'] - f['totalLiabilities']
                           - f['minorityInterest'].fillna(0)
                           - f['preferredStock'].fillna(0)) / sh
    return out


def refresh_schloss(market='CN', target_date=None):
    data_name = os.path.join(os.path.dirname(__file__), '../data/')
    exchanges = MARKET_EXCHANGES[market]
    daily_data = pd.concat(
        [pd.read_csv(data_name + ex + '/daily_' + ex.lower() + '.csv') for ex in exchanges],
        axis=0).reset_index(drop=True)
    if target_date is not None:
        daily_data = daily_data[daily_data['date'] <= int(target_date)].reset_index(drop=True)
    # 每只股票取自己的最新交易日（市场内也可能有个股停牌晚一天，取全局 max 会丢掉它）
    daily_last = daily_data.sort_values('date').groupby('symbol', sort=False).tail(1).reset_index(drop=True)
    # close=0（退市/数据错误）会导致 schloss 除出 inf，直接剔掉
    daily_last = daily_last[daily_last['close'] > 0]
    daily_last = daily_last[['symbol', 'date', 'close']]


    indicator_data = pd.concat(
        [pd.read_csv(data_name + ex + '/indicator_' + ex.lower() + '.csv') for ex in exchanges],
        axis=0).reset_index(drop=True)
    idx = indicator_data.groupby('symbol')['endDate'].idxmax()
    indicator_last = indicator_data.loc[idx].reset_index(drop=True)
    indicator_last = indicator_last[['symbol', 'endDate', 'operatingCapitalPerShare', 'netAssetValuePerShare']]

    # 用格雷厄姆口径现算每股净流动资产，覆盖 indicator 里那个旧口径的值
    # （列名保持不变 -> schloss.csv 的 schema 不变 -> 下游 two/one 的 CSV 不会多出列）
    bal = pd.concat([pd.read_csv(data_name + ex + '/balance_' + ex.lower() + '.csv') for ex in exchanges],
                    axis=0, ignore_index=True)
    inc = pd.concat([pd.read_csv(data_name + ex + '/income_' + ex.lower() + '.csv',
                                 usecols=['symbol', 'endDate', 'weightedAverageShsOut']) for ex in exchanges],
                    axis=0, ignore_index=True)
    ncav = ncav_per_share(bal, inc)
    indicator_last = indicator_last.merge(ncav, on=['symbol', 'endDate'], how='left')
    n_missing = int(indicator_last['ncavPerShare'].isna().sum())
    indicator_last['operatingCapitalPerShare'] = indicator_last['ncavPerShare']
    indicator_last = indicator_last.drop(columns=['ncavPerShare'])
    print(f'  每股净流动资产（格雷厄姆口径）已重算；因缺财务数据无法算的 {n_missing} 只')

    schloss = pd.merge(daily_last, indicator_last, on='symbol', how='inner')
    schloss = schloss.dropna().reset_index(drop=True)
    schloss['schloss'] = schloss['operatingCapitalPerShare']/schloss['close']
    # A 股只保留代码首位 0/3/6（剔 B 股等）；美股剔除明显非普通股的品种
    schloss['symbol'] = schloss['symbol'].astype(str)
    if market == 'CN':
        schloss = schloss[schloss['symbol'].str.startswith(('0', '3', '6'))].reset_index(drop=True)
    else:
        # 权证 -WT / SPAC unit -UN / 权利 -RI / 优先股 -P*（如 -PA/-PB）—— 类股 -A/-B（BRK-B/AKO-A）不在此列
        bad = schloss['symbol'].str.contains(r'-(WT|UN|RI|P[A-Z])$', regex=True)
        schloss = schloss[~bad].reset_index(drop=True)
    # schloss.drop(columns=['close'], inplace=True)
    schloss = schloss.sort_values('schloss', ascending=False)
    schloss = schloss.reset_index(drop=True)
    print(schloss)
    schloss_name = os.path.join(os.path.dirname(__file__), f'schloss{MARKET_SUFFIX[market]}.csv')
    schloss.to_csv(schloss_name, index=False)
    
if __name__ == "__main__":
    refresh_schloss()
    # get_one_all()