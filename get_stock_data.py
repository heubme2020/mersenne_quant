import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import datetime
import csv
from tqdm import tqdm
import random

pd.set_option('future.no_silent_downcasting', True)

#创建数据库
engine = create_engine('mysql+pymysql://root:12o34o56o@localhost:3306/stock')


def get_exchange_stock_symbol_data(exchange):
    os.makedirs('data/' + exchange, exist_ok=True)
    table_name = 'stock_symbol_' + exchange.lower()
    file_name = 'data/' + exchange + '/' + table_name + '.csv'
    if (os.path.exists(file_name)) == True:
        os.remove(file_name)
    print(table_name + ' ...')
    # metadata = MetaData()
    Session = sessionmaker(bind=engine)
    session = Session()
    # table = Table(table_name, metadata, autoload=True, autoload_with=engine)
    # row_count = session.query(func.count('*')).select_from(table).scalar()
    # print(table_name + ' row_number:' + str(row_count))

    sql = "SELECT * FROM " + table_name
    symbol_data = pd.read_sql(sql, session.connection())
    symbol_data.to_csv(file_name, index=False,  quoting=csv.QUOTE_NONNUMERIC)


def get_exchange_income_data(exchange):
    os.makedirs('data/' + exchange, exist_ok=True)
    table_name = 'income_' + exchange.lower()
    file_name = 'data/' + exchange + '/' + table_name + '.csv'
    if os.path.exists(file_name):
        os.remove(file_name)
    print(table_name + ' ...')
    # metadata = MetaData()
    Session = sessionmaker(bind=engine)
    session = Session()

    sql = "SELECT * FROM " + table_name
    income_data = pd.read_sql(sql, session.connection())
    # 从第三列开始，将object类型转换为float64
    for col in income_data.columns[2:]:
        income_data[col] = pd.to_numeric(income_data[col], errors='coerce')
    income_data.to_csv(file_name, index=False, quoting=csv.QUOTE_NONNUMERIC)

def get_exchange_balance_data(exchange):
    os.makedirs('data/' + exchange, exist_ok=True)
    table_name = 'balance_' + exchange.lower()
    file_name = 'data/' + exchange + '/' + table_name + '.csv'
    if os.path.exists(file_name):
        os.remove(file_name)

    print(table_name + ' ...')
    # metadata = MetaData()
    Session = sessionmaker(bind=engine)
    session = Session()
    # table = Table(table_name, metadata, autoload=True, autoload_with=engine)
    # row_count = session.query(func.count('*')).select_from(table).scalar()
    # print(table_name + ' row_number:' + str(row_count))

    sql = "SELECT * FROM " + table_name
    balance_data = pd.read_sql(sql, session.connection())
    # 从第三列开始，将object类型转换为float64
    for col in balance_data.columns[2:]:
        balance_data[col] = pd.to_numeric(balance_data[col], errors='coerce')
    balance_data.to_csv(file_name, index=False, quoting=csv.QUOTE_NONNUMERIC)

def get_exchange_cashflow_data(exchange):
    os.makedirs('data/' + exchange, exist_ok=True)
    table_name = 'cashflow_' + exchange.lower()
    file_name = 'data/' + exchange + '/' + table_name + '.csv'
    if os.path.exists(file_name):
        os.remove(file_name)

    print(table_name + ' ...')
    # metadata = MetaData()
    Session = sessionmaker(bind=engine)
    session = Session()


    sql = "SELECT * FROM " + table_name
    cashflow_data = pd.read_sql(sql, session.connection())
    # 从第三列开始，将object类型转换为float64
    for col in cashflow_data.columns[2:]:
        cashflow_data[col] = pd.to_numeric(cashflow_data[col], errors='coerce')
    cashflow_data.to_csv(file_name, index=False, quoting=csv.QUOTE_NONNUMERIC)


def get_exchange_indicator_data(exchange):
    os.makedirs('data/' + exchange, exist_ok=True)
    table_name = 'indicator_' + exchange.lower()
    file_name = 'data/' + exchange + '/' + table_name + '.csv'
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
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/' + exchange, exist_ok=True)
    table_name = 'daily_' + exchange.lower()
    file_name = 'data/' + exchange + '/' + table_name + '.csv'
    if os.path.exists(file_name):
        os.remove(file_name)

    print(table_name + ' ...')

    Session = sessionmaker(bind=engine)
    session = Session()
    sql = "SELECT * FROM " + table_name
    daily_data = pd.read_sql(sql, session.connection())
    daily_data.to_csv(file_name, index=False, quoting=csv.QUOTE_NONNUMERIC)


#求取上个季度的最后一天, 如果已经是季度末，则返回原日期
def get_quarter_end_date(date_str):
    date_str = str(date_str)
    date_check = date_str[-4:]
    if date_check == '1231' or date_check == '0331' or date_check == '0630' or date_check =='0930':
        return date_str
    date = datetime.datetime.strptime(date_str, '%Y%m%d').date()
    quarter_month = ((date.month-1)//3) * 3 + 1
    # 构建季度末日期
    quarter_end_date = datetime.date(date.year, quarter_month, 1) + datetime.timedelta(days=-1)
    # 将日期格式转换为字符串格式
    quarter_end_date_str = quarter_end_date.strftime('%Y%m%d')
    return quarter_end_date_str

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
            # print(outlier_indices)
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
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/' + exchange, exist_ok=True)
    get_exchange_income_data(exchange)
    get_exchange_balance_data(exchange)
    get_exchange_cashflow_data(exchange)
    get_exchange_indicator_data(exchange)

    print('mean&std ...')

    income_data = pd.read_csv('data/' + exchange + '/income_' + exchange + '.csv')
    balance_data = pd.read_csv('data/' + exchange + '/balance_' + exchange + '.csv')
    cashflow_data = pd.read_csv('data/' + exchange + '/cashflow_' + exchange + '.csv')
    # indicator_data = pd.read_csv('data/' + exchange + '/indicator_' + exchange + '.csv')
    #合并财务相关数据
    # financial_data = pd.merge(indicator_data, income_data, on=['symbol', 'endDate'], how='outer')
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
    mean_data.to_csv('data/' + exchange + '/mean_' + exchange.lower() + '.csv', index=False)
    std_data.to_csv('data/' + exchange + '/std_' + exchange.lower() + '.csv', index=False)


def get_exchange_data(exchange):
    get_exchange_financial_data(exchange)
    get_exchange_daily_data(exchange)


def get_data():
    failed_list = []
    exchanges_data = pd.read_csv('train_exchanges.csv', encoding="utf-8")
    exchange_list = exchanges_data['exchange'].tolist()
    # exchange_list = ['SHZ', 'SHH']
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
    failed_list = []
    exchanges_data = pd.read_csv('train_exchanges.csv', encoding="utf-8")
    exchange_list = exchanges_data['exchange'].tolist()
    print(exchange_list)
    for exchange in exchange_list:
        try:
            print(exchange)
            get_exchange_indicator_data(exchange)
        except:
            failed_list.append(exchange)

    print(failed_list)   


if __name__ == "__main__":
    # get_exchange_indicator_data('TSXV')
    get_data()







