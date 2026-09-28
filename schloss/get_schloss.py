import torch
import os
import random
import pandas as pd
import numpy as np
from tqdm import tqdm

pd.set_option('future.no_silent_downcasting', True)


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
    sh = f['weightedAverageShsOut']
    out = f[['symbol', 'endDate']].copy()
    out['ncavPerShare'] = (f['totalCurrentAssets'] - f['totalLiabilities']
                           - f['minorityInterest'].fillna(0)
                           - f['preferredStock'].fillna(0)) / sh
    return out


def refresh_schloss(target_date=None):
    data_name = os.path.join(os.path.dirname(__file__), '../data/')
    daily_shz = pd.read_csv(data_name + 'SHZ/daily_shz.csv')
    daily_shh = pd.read_csv(data_name + 'SHH/daily_shh.csv')
    daily_data = pd.concat([daily_shz, daily_shh], axis=0).reset_index(drop=True)
    if target_date is not None:
        daily_data = daily_data[daily_data['date'] <= int(target_date)].reset_index(drop=True)
    last_day = daily_data['date'].max()
    daily_last = daily_data[daily_data['date'] == last_day].reset_index(drop=True)
    daily_last = daily_last[['symbol', 'date', 'close']]


    indicator_shz = pd.read_csv(data_name + 'SHZ/indicator_shz.csv')
    indicator_shh = pd.read_csv(data_name + 'SHH/indicator_shh.csv')
    indicator_data = pd.concat([indicator_shz, indicator_shh], axis=0).reset_index(drop=True)
    idx = indicator_data.groupby('symbol')['endDate'].idxmax()
    indicator_last = indicator_data.loc[idx].reset_index(drop=True)
    indicator_last = indicator_last[['symbol', 'endDate', 'operatingCapitalPerShare', 'netAssetValuePerShare']]

    # 用格雷厄姆口径现算每股净流动资产，覆盖 indicator 里那个旧口径的值
    # （列名保持不变 -> schloss.csv 的 schema 不变 -> 下游 two/one 的 CSV 不会多出列）
    bal = pd.concat([pd.read_csv(data_name + 'SHZ/balance_shz.csv'),
                     pd.read_csv(data_name + 'SHH/balance_shh.csv')], axis=0, ignore_index=True)
    inc = pd.concat([pd.read_csv(data_name + 'SHZ/income_shz.csv',
                                 usecols=['symbol', 'endDate', 'weightedAverageShsOut']),
                     pd.read_csv(data_name + 'SHH/income_shh.csv',
                                 usecols=['symbol', 'endDate', 'weightedAverageShsOut'])],
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
    schloss = schloss[schloss['symbol'].str.startswith(('0', '3', '6'))].reset_index(drop=True)
    # schloss.drop(columns=['close'], inplace=True)
    schloss = schloss.sort_values('schloss', ascending=False)
    schloss = schloss.reset_index(drop=True)
    print(schloss)
    schloss_name = os.path.join(os.path.dirname(__file__), 'schloss.csv')
    schloss.to_csv(schloss_name, index=False)
    
if __name__ == "__main__":
    refresh_schloss()
    # get_one_all()