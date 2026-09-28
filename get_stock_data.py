import os
import sys
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import datetime
import csv
from tqdm import tqdm
import random

pd.set_option('future.no_silent_downcasting', True)

# 本机凭据（DSN / API key / 邮箱授权码）在 config_local.py —— **不入库**（见 .gitignore）。
# 2026-09-28 从本文件里抽出来的：原来 DSN 是明文硬编码，推公开仓库会泄漏。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # 无论从哪个 cwd 起，都能 import 到根目录的 config_local
try:
    from config_local import MYSQL_DSN
except ImportError:
    raise SystemExit('缺少 config_local.py —— 本机凭据不入库。请照 config_local.example.py '
                     '在仓库根写一份（cp config_local.example.py config_local.py 后填值）。')

#创建数据库
engine = create_engine(MYSQL_DSN)


def get_exchange_stock_symbol_data(exchange):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(BASE_DIR, 'data')
    data_exchange_folder = os.path.join(data_folder, exchange)
    os.makedirs(data_exchange_folder, exist_ok=True)
    table_name = 'stock_symbol_' + exchange.lower()
    file_name = os.path.join(data_exchange_folder, table_name + '.csv')
    if (os.path.exists(file_name)) == True:
        os.remove(file_name)
    print(table_name + ' ...')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = "SELECT * FROM " + table_name
    symbol_data = pd.read_sql(sql, session.connection())
    symbol_data.to_csv(file_name, index=False,  quoting=csv.QUOTE_NONNUMERIC)


def get_exchange_income_data(exchange):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(BASE_DIR, 'data')
    data_exchange_folder = os.path.join(data_folder, exchange)
    os.makedirs(data_exchange_folder, exist_ok=True)
    table_name = 'income_' + exchange.lower()
    file_name = os.path.join(data_exchange_folder, table_name + '.csv')
    if os.path.exists(file_name):
        os.remove(file_name)
    print(table_name + ' ...')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = "SELECT * FROM " + table_name
    income_data = pd.read_sql(sql, session.connection())
    # 从第三列开始，将object类型转换为float64
    for col in income_data.columns[2:]:
        income_data[col] = pd.to_numeric(income_data[col], errors='coerce')
    income_data.to_csv(file_name, index=False, quoting=csv.QUOTE_NONNUMERIC)

def get_exchange_balance_data(exchange):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(BASE_DIR, 'data')
    data_exchange_folder = os.path.join(data_folder, exchange)
    os.makedirs(data_exchange_folder, exist_ok=True)
    table_name = 'balance_' + exchange.lower()
    file_name = os.path.join(data_exchange_folder, table_name + '.csv')
    if os.path.exists(file_name):
        os.remove(file_name)

    print(table_name + ' ...')
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = "SELECT * FROM " + table_name
    balance_data = pd.read_sql(sql, session.connection())
    # 从第三列开始，将object类型转换为float64
    for col in balance_data.columns[2:]:
        balance_data[col] = pd.to_numeric(balance_data[col], errors='coerce')
    balance_data.to_csv(file_name, index=False, quoting=csv.QUOTE_NONNUMERIC)

def get_exchange_cashflow_data(exchange):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(BASE_DIR, 'data')
    data_exchange_folder = os.path.join(data_folder, exchange)
    os.makedirs(data_exchange_folder, exist_ok=True)
    table_name = 'cashflow_' + exchange.lower()
    file_name = os.path.join(data_exchange_folder, table_name + '.csv')
    if os.path.exists(file_name):
        os.remove(file_name)

    print(table_name + ' ...')
    Session = sessionmaker(bind=engine)
    session = Session()


    sql = "SELECT * FROM " + table_name
    cashflow_data = pd.read_sql(sql, session.connection())
    # 从第三列开始，将object类型转换为float64
    for col in cashflow_data.columns[2:]:
        cashflow_data[col] = pd.to_numeric(cashflow_data[col], errors='coerce')
    cashflow_data.to_csv(file_name, index=False, quoting=csv.QUOTE_NONNUMERIC)


def get_exchange_indicator_data(exchange):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(BASE_DIR, 'data')
    data_exchange_folder = os.path.join(data_folder, exchange)
    os.makedirs(data_exchange_folder, exist_ok=True)
    table_name = 'indicator_' + exchange.lower()
    file_name = os.path.join(data_exchange_folder, table_name + '.csv')
    if os.path.exists(file_name):
        os.remove(file_name)

    print(table_name + ' ...')

    Session = sessionmaker(bind=engine)
    session = Session()

    sql = "SELECT * FROM " + table_name
    indicator_data = pd.read_sql(sql, session.connection())
    # 从第三列开始，将object类型转换为float64
    for col in indicator_data.columns[2:]:
        indicator_data[col] = pd.to_numeric(indicator_data[col], errors='coerce')
    indicator_data.to_csv(file_name, index=False, quoting=csv.QUOTE_NONNUMERIC)

def get_exchange_daily_data(exchange):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(BASE_DIR, 'data')
    data_exchange_folder = os.path.join(data_folder, exchange)
    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(data_exchange_folder, exist_ok=True)
    table_name = 'daily_' + exchange.lower()
    file_name = os.path.join(data_exchange_folder, table_name + '.csv')
    if os.path.exists(file_name):
        os.remove(file_name)

    print(table_name + ' ...')

    Session = sessionmaker(bind=engine)
    session = Session()
    sql = "SELECT * FROM " + table_name
    daily_data = pd.read_sql(sql, session.connection())
    daily_data.to_csv(file_name, index=False, quoting=csv.QUOTE_NONNUMERIC)

def get_mean_std_data(data):
    mean_data = pd.DataFrame()
    std_data = pd.DataFrame()
    endDate_list = sorted(data['endDate'].unique())
    col_names = []
    for i in tqdm(range(1, len(endDate_list))):
        endDate = endDate_list[i]
        filtered_data = data[data['endDate'] <= endDate]
        filtered_data = filtered_data.reset_index()
        filtered_data.drop(columns=['index'], inplace=True)
        #如果这个endDate股票数小于127，则放弃
        groups = list(filtered_data.groupby('symbol'))
        if len(groups) < 127:
            continue
        filtered_data = filtered_data.drop('symbol', axis=1)
        col_names = filtered_data.columns.values
        mean_list = [endDate]
        std_list = [endDate]
        for k in range(1, len(col_names)):
            col_name = col_names[k]
            mean_value = filtered_data[col_name].mean()
            std_value = filtered_data[col_name].std()
            threshold = 7
            # 根据阈值筛选出异常值的索引
            outlier_indices = filtered_data.index[abs(filtered_data[col_name] - mean_value) > threshold * std_value]
            # 剔除包含异常值的行
            filtered_data_cleaned = filtered_data.drop(outlier_indices)
            filtered_data_cleaned = filtered_data_cleaned.reset_index(drop=True)
            # 计算剔除异常值后的均值和方差
            mean_cleaned = filtered_data_cleaned[col_name].mean()
            std_cleaned = filtered_data_cleaned[col_name].std()
            mean_list.append(mean_cleaned)
            std_list.append(std_cleaned)
        mean_dataframe = pd.DataFrame([mean_list])
        std_dataframe = pd.DataFrame([std_list])
        mean_data = pd.concat([mean_data, mean_dataframe])
        std_data = pd.concat([std_data, std_dataframe])
        mean_data = mean_data.reset_index()
        mean_data.drop(columns=['index'], inplace=True)
        std_data = std_data.reset_index()
        std_data.drop(columns=['index'], inplace=True)
    mean_data.columns = col_names
    std_data.columns = col_names
    return mean_data, std_data


def get_exchange_financial_data(exchange):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(BASE_DIR, 'data')
    data_exchange_folder = os.path.join(data_folder, exchange)
    os.makedirs(data_folder, exist_ok=True)
    os.makedirs(data_exchange_folder, exist_ok=True)
    get_exchange_income_data(exchange)
    get_exchange_balance_data(exchange)
    get_exchange_cashflow_data(exchange)
    get_exchange_indicator_data(exchange)

    print('mean&std ...')

    income_data = pd.read_csv(data_exchange_folder + '/income_' + exchange + '.csv')
    balance_data = pd.read_csv(data_exchange_folder + '/balance_' + exchange + '.csv')
    cashflow_data = pd.read_csv(data_exchange_folder + '/cashflow_' + exchange + '.csv')
    #合并财务相关数据
    financial_data = pd.merge(income_data, balance_data, on=['symbol', 'endDate'], how='outer')
    financial_data = pd.merge(financial_data, cashflow_data, on=['symbol', 'endDate'], how='outer')
    print(financial_data)
    financial_data = financial_data.dropna(subset=['symbol', 'endDate'])
    financial_data = financial_data.fillna(0)
    financial_data = financial_data.reset_index(drop=True)
    print(financial_data)
    financial_data.drop_duplicates(subset=['symbol', 'endDate'], keep='first', inplace=True)
    financial_data = financial_data.reset_index(drop=True)
    print(financial_data)
    # 从第三列开始，将object类型转换为float64
    for col in financial_data.columns[2:]:
        financial_data[col] = pd.to_numeric(financial_data[col], errors='coerce')
    #生成对应的均值，方差矩阵
    mean_data, std_data = get_mean_std_data(financial_data)
    mean_data.to_csv(data_exchange_folder + '/mean_' + exchange.lower() + '.csv', index=False)
    std_data.to_csv(data_exchange_folder + '/std_' + exchange.lower() + '.csv', index=False)


def get_exchange_data(exchange):
    get_exchange_financial_data(exchange)
    get_exchange_daily_data(exchange)


def get_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # 遍历 data/ 下全部交易所（已在上游过滤，不再走 train_exchanges.csv 的 num_symbols 阈值）
    data_dir = os.path.join(BASE_DIR, 'data')
    exchange_list = sorted(d for d in os.listdir(data_dir)
                           if os.path.isdir(os.path.join(data_dir, d)))
    failed_list = []
    print(exchange_list)
    for exchange in exchange_list:
        try:
            print(exchange)
            get_exchange_financial_data(exchange)
            get_exchange_daily_data(exchange)
        except:
            failed_list.append(exchange)

    print(failed_list)

def get_indicator_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(BASE_DIR, 'data')
    exchanges = sorted(d for d in os.listdir(data_dir)
                       if os.path.isdir(os.path.join(data_dir, d)))
    failed_list = []
    for exchange in exchanges:
        try:
            print(exchange)
            get_exchange_indicator_data(exchange)
        except:
            failed_list.append(exchange)

    print(failed_list)   


if __name__ == "__main__":
    get_data()







