import os
import random
import ssl
import certifi
import json
import pandas as pd
from urllib.request import urlopen
from sqlalchemy import create_engine, func, update, delete, MetaData, Table, Column, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import sqlalchemy as sa
import datetime
import csv
import threading
import numpy as np
from tqdm import tqdm
import multiprocessing
from sqlalchemy import text
from get_stock_data import get_exchange_data, get_exchange_daily_data, get_indicator_data
pd.set_option('future.no_silent_downcasting', True)
from sqlalchemy import func, over, and_
from sqlalchemy.sql import label
import time


import calendar
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
token = 'abcdefg'#这里要换成自己financialmodelingprep的token


#创建数据库
engine = create_engine('mysql+pymysql://root:1234567@localhost:3306/stock', pool_size=10, max_overflow=20)

def get_jsonparsed_data(url):
    context = ssl.create_default_context(cafile=certifi.where())
    response = urlopen(url, context=context)
    data = response.read().decode("utf-8")
    return json.loads(data)

# def get_jsonparsed_data(url):
#     response = urlopen(url, cafile=certifi.where())
#     data = response.read().decode("utf-8")
#     return json.loads(data)

def get_exchange_suffix(exchange):
    exchanges_data = pd.read_csv('train_exchanges.csv', encoding='utf-8')
    exchange_data = exchanges_data[exchanges_data['exchange'] == exchange]
    suffix = exchange_data['symbolSuffix'].values[0]
    suffix = suffix.replace('.', '')
    return suffix

def write_symbol_data_into_table(exchange, stock_list):
    #建立连接
    exchange_stock_symbol_table = 'stock_symbol_' + exchange.lower()
    metadata = MetaData()
    stock_table = Table(exchange_stock_symbol_table, metadata,  autoload_with=engine)
    conn = engine.connect()
    company_profile_list = []
    for i in range(len(stock_list)):
        print(exchange_stock_symbol_table + ':' + str(float(i)/(len(stock_list)+1.0)))
        symbol = stock_list[i]
        try:
            url = 'https://financialmodelingprep.com/stable/profile?symbol=' + symbol + '&apikey=' + token
            company_profile = get_jsonparsed_data(url)
        except:
            print(symbol)
            continue
        if len(company_profile) == 1:
            company_profile_list.append(company_profile[0])
    stock_data = pd.DataFrame.from_records(company_profile_list)
    print(stock_data)
    stock_data = stock_data[['symbol', 'companyName', 'currency', 'exchange', 'website', 'city', 'ipoDate']]
    stock_data['companyName'] = stock_data['companyName'].fillna('')
    stock_data['city'] = stock_data['city'].fillna('')
    stock_data['website'] = stock_data['website'].fillna('https://')
    stock_data['website'] = stock_data['website'].str[:200]
    stock_data['ipoDate'] = stock_data['ipoDate'].fillna('1900-01-01')
    stock_data['ipoDate'] = stock_data['ipoDate'].apply(lambda x: x.replace('-', ''))
    stock_data = stock_data.drop_duplicates(subset=['symbol'], keep='first')
    count = 0
    for _, row in stock_data.iterrows():
        data = row.to_dict()
        try:
            _ = conn.execute(stock_table.insert(), data)
            count += 1
            conn.commit()
        except:
            pass
    print('inserted stock_data rows:', count)
    # stock_data.to_sql(exchange_stock_symbol_table, con=engine, if_exists='append', index=False)
    print(stock_data)


def init_exchange_stock_symbol_data(exchange):
    inspector = sa.inspect(engine)
    # 获取表格名称列表
    table_names = inspector.get_table_names()
    exchange_stock_symbol_name = 'stock_symbol_' + exchange.lower()
    if exchange_stock_symbol_name in table_names:
        # 如果表格存在，则输出提示信息
        print(exchange_stock_symbol_name + ' table exists!')
    else:
        # 创建元数据对象
        metadata = MetaData()
        # 创建表对象
        table = Table(exchange_stock_symbol_name, metadata,
                      Column('symbol', String(50), primary_key=True),
                      Column('companyName', String(200)),
                      Column('currency', String(50)),
                      Column('exchange', String(50)),
                      Column('website', String(200)),
                      Column('city', String(50)),
                      Column('ipoDate', String(50)),
                      )
        # 创建表
        metadata.create_all(engine)
        url = 'https://financialmodelingprep.com/stable/financial-statement-symbol-list?apikey=' + token
        financial_statement_symbols = get_jsonparsed_data(url)
        financial_statement_symbols = pd.DataFrame(financial_statement_symbols)
        financial_statement_symbols = financial_statement_symbols['symbol'].tolist()

        # url = 'https://financialmodelingprep.com/stable/company-screener?limit=131071&isActivelyTrading=True&exchange=' + exchange + '&apikey=' + token
        url = 'https://financialmodelingprep.com/stable/company-screener?limit=131071&isEtf=false&isFund=false&exchange=' + exchange + '&apikey=' + token
        exchange_symbols = get_jsonparsed_data(url)
        exchange_symbols = pd.DataFrame(exchange_symbols)
        exchange_symbols = exchange_symbols['symbol'].tolist()
        print(len(exchange_symbols))
        stock_list = list(set(financial_statement_symbols) & set(exchange_symbols))#要有财务数据
        # stock_list = [item for item in stock_list if item.endswith(exchange_reference_dict[exchange][0])]
        print(len(stock_list))
        symbol_num = len(stock_list)
        split_num = int(0.333*symbol_num)
        stock_list0 = stock_list[:split_num]
        stock_list1 = stock_list[split_num:2*split_num]
        stock_list2 = stock_list[2*split_num:]

        t0 = threading.Thread(target=write_symbol_data_into_table, args=(exchange, stock_list0))
        t1 = threading.Thread(target=write_symbol_data_into_table, args=(exchange, stock_list1))
        t2 = threading.Thread(target=write_symbol_data_into_table, args=(exchange, stock_list2))
        t0.start()
        t1.start()
        t2.start()
        t0.join()
        t1.join()
        t2.join()
        print(exchange_stock_symbol_name + ' data inited!')


def write_exchange_stock_symbol_data(exchange):
    inspector = sa.inspect(engine)
    # 获取表格名称列表
    table_names = inspector.get_table_names()
    exchange_stock_symbol_table = 'stock_symbol_' + exchange.lower()
    if exchange_stock_symbol_table in table_names:
        metadata = MetaData()
        #先获得当前table的存的股票列表
        Session = sessionmaker(bind=engine)
        session = Session()
        stock_symbol_sql = "SELECT * FROM " + exchange_stock_symbol_table
        stock_symbol_data_pre = pd.read_sql(stock_symbol_sql, session.connection())
        symbol_list_pre = stock_symbol_data_pre['symbol'].tolist()
        #再获得最新的交易所内股票列表
        url = 'https://financialmodelingprep.com/stable/financial-statement-symbol-list?apikey=' + token
        financial_statement_symbols = get_jsonparsed_data(url)
        financial_statement_symbols = pd.DataFrame(financial_statement_symbols)
        financial_statement_symbols = financial_statement_symbols['symbol'].tolist()
        # url = 'https://financialmodelingprep.com/stable/company-screener?limit=131071&isActivelyTrading=True&exchange=' + exchange + '&apikey=' + token
        url = 'https://financialmodelingprep.com/stable/company-screener?limit=131071&isEtf=false&isFund=false&exchange=' + exchange + '&apikey=' + token
        exchange_symbols = get_jsonparsed_data(url)
        exchange_symbols = pd.DataFrame(exchange_symbols)
        exchange_symbols = exchange_symbols['symbol'].tolist()
        stock_list = list(set(financial_statement_symbols) & set(exchange_symbols))
        append_stock_list = list(set(stock_list) - set(symbol_list_pre))
        if len(append_stock_list) == 0:
            print(exchange, '没有新股要添加')
            return
        print('新添加的股票数: ', len(append_stock_list))
        print('新添加的股票: ', append_stock_list)
        print('准备写入的新股票...')
        write_symbol_data_into_table(exchange, append_stock_list)
    else:
        print(exchange_stock_symbol_table + ' table not exist!')
        print('start initing table...')
        init_exchange_stock_symbol_data(exchange)


#求取季度的最后一天, 如果已经是本季度最后一天，则返回该日期，否则返回上个季度的最后一天
def get_quarter_end_date(date_str):
    date_str = str(date_str)
    date_check = date_str[-4:]
    # print(date_check)
    if date_check == '1231' or date_check == '0331' or date_check == '0630' or date_check == '0930':
        return date_str
    date = datetime.datetime.strptime(date_str, '%Y%m%d').date()
    quarter_month = ((date.month-1)//3) * 3 + 1
    # 构建季度末日期
    quarter_end_date = datetime.date(date.year, quarter_month, 1) + datetime.timedelta(days=-1)
    # 将日期格式转换为字符串格式
    quarter_end_date_str = quarter_end_date.strftime('%Y%m%d')
    return quarter_end_date_str


def init_exchange_income_data(exchange):
    exchange = exchange.lower()
    inspector = sa.inspect(engine)
    # 获取表格名称列表
    table_names = inspector.get_table_names()
    income_exchange_name = 'income_' + exchange
    if income_exchange_name in table_names:
        # 如果表格存在，则输出提示信息
        print(income_exchange_name + ' Table exists')
    else:
        # 创建元数据对象
        metadata = MetaData()
        # 创建表对象
        table = Table(income_exchange_name, metadata,
                      Column('symbol', String(50), primary_key=True),
                      Column('endDate', String(50), primary_key=True),
                      Column('revenue', String(50)),
                      Column('costOfRevenue', String(50)),
                      Column('grossProfit', String(50)),
                      Column('researchAndDevelopmentExpenses', String(50)),
                      Column('generalAndAdministrativeExpenses', String(50)),
                      Column('sellingAndMarketingExpenses', String(50)),
                      Column('sellingGeneralAndAdministrativeExpenses', String(50)),
                      Column('otherExpenses', String(50)),
                      Column('operatingExpenses', String(50)),
                      Column('costAndExpenses', String(50)),
                      Column('netInterestIncome', String(50)),
                      Column('interestIncome', String(50)),
                      Column('interestExpense', String(50)),
                      Column('depreciationAndAmortization', String(50)),
                      Column('ebitda', String(50)),
                      Column('ebit', String(50)),
                      Column('nonOperatingIncomeExcludingInterest', String(50)),
                      Column('operatingIncome', String(50)),
                      Column('totalOtherIncomeExpensesNet', String(50)),
                      Column('incomeBeforeTax', String(50)),
                      Column('incomeTaxExpense', String(50)),
                      Column('netIncomeFromContinuingOperations', String(50)),          
                      Column('netIncomeFromDiscontinuedOperations', String(50)),    
                      Column('otherAdjustmentsToNetIncome', String(50)),               
                      Column('netIncome', String(50)),
                      Column('netIncomeDeductions', String(50)),
                      Column('bottomLineNetIncome', String(50)),
                      Column('eps', String(50)),
                      Column('epsdiluted', String(50)),
                      Column('weightedAverageShsOut', String(50)),
                      Column('weightedAverageShsOutDil', String(50)),
                      )
        # 创建表
        metadata.create_all(engine)
        # 先获得所有exchange内的股票信息
        exchange_stock_symbol_name = 'stock_symbol_' + exchange
        Session = sessionmaker(bind=engine)
        session = Session()
        stock_symbol_sql = "SELECT * FROM " + exchange_stock_symbol_name
        stock_symbol_data = pd.read_sql(stock_symbol_sql, session.connection())
        #再插入income数据
        for i in range(len(stock_symbol_data)):
            print(income_exchange_name + ':' + str(float(i) / len(stock_symbol_data)))
            stock_data = stock_symbol_data.iloc[i]
            symbol = stock_data['symbol']
            try:
                url = 'https://financialmodelingprep.com/stable/income-statement?limit=8191&period=quarter&symbol=' + symbol + '&apikey=' + token
                # url = 'https://financialmodelingprep.com/api/v3/income-statement/'+symbol+'?period=quarter&limit=127&apikey='+token
                fetch_data = get_jsonparsed_data(url)
            except:
                print(symbol)
                continue
            if len(fetch_data) == 0:
                continue
            fetch_data = pd.DataFrame(fetch_data)
            income_data = pd.DataFrame()
            income_data['symbol'] = fetch_data['symbol']
            income_data['endDate'] = fetch_data['date'].apply(lambda x: x.replace('-', ''))
            income_data['endDate'] = income_data['endDate'].apply(get_quarter_end_date)
            income_data['revenue'] = fetch_data['revenue']
            income_data['costOfRevenue'] = fetch_data['costOfRevenue']
            income_data['grossProfit'] = fetch_data['grossProfit']
            income_data['researchAndDevelopmentExpenses'] = fetch_data['researchAndDevelopmentExpenses']
            income_data['generalAndAdministrativeExpenses'] = fetch_data['generalAndAdministrativeExpenses']
            income_data['sellingAndMarketingExpenses'] = fetch_data['sellingAndMarketingExpenses']
            income_data['sellingGeneralAndAdministrativeExpenses'] = fetch_data['sellingGeneralAndAdministrativeExpenses']
            income_data['otherExpenses'] = fetch_data['otherExpenses']
            income_data['operatingExpenses'] = fetch_data['operatingExpenses']
            income_data['costAndExpenses'] = fetch_data['costAndExpenses']
            income_data['netInterestIncome'] = fetch_data['netInterestIncome']
            income_data['interestIncome'] = fetch_data['interestIncome']
            income_data['interestExpense'] = fetch_data['interestExpense']
            income_data['depreciationAndAmortization'] = fetch_data['depreciationAndAmortization']
            income_data['ebitda'] = fetch_data['ebitda']
            income_data['ebit'] = fetch_data['ebit']
            income_data['nonOperatingIncomeExcludingInterest'] = fetch_data['nonOperatingIncomeExcludingInterest']
            income_data['operatingIncome'] = fetch_data['operatingIncome']
            income_data['totalOtherIncomeExpensesNet'] = fetch_data['totalOtherIncomeExpensesNet']
            income_data['incomeBeforeTax'] = fetch_data['incomeBeforeTax']
            income_data['incomeTaxExpense'] = fetch_data['incomeTaxExpense']
            income_data['netIncomeFromContinuingOperations'] = fetch_data['netIncomeFromContinuingOperations']
            income_data['netIncomeFromDiscontinuedOperations'] = fetch_data['netIncomeFromDiscontinuedOperations']
            income_data['otherAdjustmentsToNetIncome'] = fetch_data['otherAdjustmentsToNetIncome']
            income_data['netIncome'] = fetch_data['netIncome']
            income_data['netIncomeDeductions'] = fetch_data['netIncomeDeductions']
            income_data['bottomLineNetIncome'] = fetch_data['bottomLineNetIncome']
            income_data['eps'] = fetch_data['eps']
            income_data['epsDiluted'] = fetch_data['epsDiluted']
            income_data['weightedAverageShsOut'] = fetch_data['weightedAverageShsOut']
            income_data['weightedAverageShsOutDil'] = fetch_data['weightedAverageShsOutDil']

            income_data = income_data.dropna(subset=['symbol', 'endDate'])
            income_data = income_data.fillna(0)
            income_data = income_data.sort_values(by='endDate', ascending=False)
            income_data = income_data.drop_duplicates(subset=['symbol', 'endDate'], keep='first')
            income_data = income_data.reset_index(drop=True)
            income_data.to_sql(income_exchange_name, con=engine, if_exists='append', index=False)
        print(income_exchange_name + ' data inited!')


def write_exchange_income_data(exchange, quarter_num=7):
    exchange = exchange.lower()
    inspector = sa.inspect(engine)
    # 获取表格名称列表
    table_names = inspector.get_table_names()
    income_exchange_name = 'income_' + exchange
    if income_exchange_name in table_names:
        #先获得当前table的数据
        Session = sessionmaker(bind=engine)
        session = Session()
        #再获得所有exchange内的股票信息
        exchange_stock_symbol_name = 'stock_symbol_' + exchange
        stock_symbol_sql = "SELECT * FROM " + exchange_stock_symbol_name
        stock_symbol_data = pd.read_sql(stock_symbol_sql, session.connection())
        #建立连接
        metadata = MetaData()
        income_table = Table(income_exchange_name, metadata, autoload_with=engine)
        conn = engine.connect()
        #开始遍历股票添加数据
        for i in range(len(stock_symbol_data)):
            print(income_exchange_name + ':' + str(float(i) / len(stock_symbol_data)))
            stock_data = stock_symbol_data.iloc[i]
            symbol = stock_data['symbol']
            try:
                url = 'https://financialmodelingprep.com/stable/income-statement?limit=' + str(quarter_num) + '&period=quarter&symbol=' + symbol + '&apikey=' + token
                # url = 'https://financialmodelingprep.com/api/v3/income-statement/' + symbol + '?period=quarter&limit=' + str(quarter_num) + '&apikey=' + token
                fetch_data = get_jsonparsed_data(url)
            except:
                print(symbol)
                continue
            if len(fetch_data) == 0:
                continue
            fetch_data = pd.DataFrame(fetch_data)
            income_data = pd.DataFrame()
            income_data['symbol'] = fetch_data['symbol']
            income_data['endDate'] = fetch_data['date'].apply(lambda x: x.replace('-', ''))
            income_data['endDate'] = income_data['endDate'].apply(get_quarter_end_date)
            income_data['revenue'] = fetch_data['revenue']
            income_data['costOfRevenue'] = fetch_data['costOfRevenue']
            income_data['grossProfit'] = fetch_data['grossProfit']
            income_data['researchAndDevelopmentExpenses'] = fetch_data['researchAndDevelopmentExpenses']
            income_data['generalAndAdministrativeExpenses'] = fetch_data['generalAndAdministrativeExpenses']
            income_data['sellingAndMarketingExpenses'] = fetch_data['sellingAndMarketingExpenses']
            income_data['sellingGeneralAndAdministrativeExpenses'] = fetch_data['sellingGeneralAndAdministrativeExpenses']
            income_data['otherExpenses'] = fetch_data['otherExpenses']
            income_data['operatingExpenses'] = fetch_data['operatingExpenses']
            income_data['costAndExpenses'] = fetch_data['costAndExpenses']
            income_data['netInterestIncome'] = fetch_data['netInterestIncome']
            income_data['interestIncome'] = fetch_data['interestIncome']
            income_data['interestExpense'] = fetch_data['interestExpense']
            income_data['depreciationAndAmortization'] = fetch_data['depreciationAndAmortization']
            income_data['ebitda'] = fetch_data['ebitda']
            income_data['ebit'] = fetch_data['ebit']
            income_data['nonOperatingIncomeExcludingInterest'] = fetch_data['nonOperatingIncomeExcludingInterest']
            income_data['operatingIncome'] = fetch_data['operatingIncome']
            income_data['totalOtherIncomeExpensesNet'] = fetch_data['totalOtherIncomeExpensesNet']
            income_data['incomeBeforeTax'] = fetch_data['incomeBeforeTax']
            income_data['incomeTaxExpense'] = fetch_data['incomeTaxExpense']
            income_data['netIncomeFromContinuingOperations'] = fetch_data['netIncomeFromContinuingOperations']
            income_data['netIncomeFromDiscontinuedOperations'] = fetch_data['netIncomeFromDiscontinuedOperations']
            income_data['otherAdjustmentsToNetIncome'] = fetch_data['otherAdjustmentsToNetIncome']
            income_data['netIncome'] = fetch_data['netIncome']
            income_data['netIncomeDeductions'] = fetch_data['netIncomeDeductions']
            income_data['bottomLineNetIncome'] = fetch_data['bottomLineNetIncome']
            income_data['eps'] = fetch_data['eps']
            income_data['epsDiluted'] = fetch_data['epsDiluted']
            income_data['weightedAverageShsOut'] = fetch_data['weightedAverageShsOut']
            income_data['weightedAverageShsOutDil'] = fetch_data['weightedAverageShsOutDil']

            income_data = income_data.dropna(subset=['symbol', 'endDate'])
            income_data = income_data.fillna(0)
            income_data = income_data.sort_values(by='endDate', ascending=False)
            income_data = income_data.drop_duplicates(subset=['symbol', 'endDate'], keep='first')
            income_data = income_data.reset_index(drop=True)
            count = 0
            for _, row in income_data.iterrows():
                data = row.to_dict()
                try:
                    _ = conn.execute(income_table.insert(), data)
                    count += 1
                    conn.commit()
                except:
                    pass
            print('inserted income_data rows:', count)
    else:
        print(income_exchange_name + ' table not exist!')
        print('start initing table...')
        init_exchange_income_data(exchange)

def init_exchange_balance_data(exchange):
    exchange = exchange.lower()
    inspector = sa.inspect(engine)
    # 获取表格名称列表
    table_names = inspector.get_table_names()
    balance_exchange_name = 'balance_' + exchange
    if balance_exchange_name in table_names:
        # 如果表格存在，则输出提示信息
        print(balance_exchange_name + ' Table exists')
    else:
        # 创建元数据对象
        metadata = MetaData()
        # 创建表对象
        table = Table(balance_exchange_name, metadata,
                      Column('symbol', String(50), primary_key=True),
                      Column('endDate', String(50), primary_key=True),
                      Column('cashAndCashEquivalents', String(50)),
                      Column('shortTermInvestments', String(50)),
                      Column('cashAndShortTermInvestments', String(50)),
                      Column('netReceivables', String(50)),
                      Column('accountsReceivables', String(50)),
                      Column('otherReceivables', String(50)),
                      Column('inventory', String(50)),
                      Column('prepaids', String(50)),
                      Column('otherCurrentAssets', String(50)),
                      Column('totalCurrentAssets', String(50)),
                      Column('propertyPlantEquipmentNet', String(50)),
                      Column('goodwill', String(50)),
                      Column('intangibleAssets', String(50)),
                      Column('goodwillAndIntangibleAssets', String(50)),
                      Column('longTermInvestments', String(50)),
                      Column('taxAssets', String(50)),
                      Column('otherNonCurrentAssets', String(50)),
                      Column('totalNonCurrentAssets', String(50)),
                      Column('otherAssets', String(50)),
                      Column('totalAssets', String(50)),
                      Column('totalPayables', String(50)),
                      Column('accountPayables', String(50)),
                      Column('otherPayables', String(50)),
                      Column('accruedExpenses', String(50)),
                      Column('shortTermDebt', String(50)),
                      Column('capitalLeaseObligationsCurrent', String(50)),
                      Column('taxPayables', String(50)),
                      Column('deferredRevenue', String(50)),
                      Column('otherCurrentLiabilities', String(50)),
                      Column('totalCurrentLiabilities', String(50)),
                      Column('longTermDebt', String(50)),
                      Column('capitalLeaseObligationsNonCurrent', String(50)),
                      Column('deferredRevenueNonCurrent', String(50)),
                      Column('deferredTaxLiabilitiesNonCurrent', String(50)),
                      Column('otherNonCurrentLiabilities', String(50)),
                      Column('totalNonCurrentLiabilities', String(50)),
                      Column('otherLiabilities', String(50)),
                      Column('capitalLeaseObligations', String(50)),
                      Column('totalLiabilities', String(50)),
                      Column('treasuryStock', String(50)),
                      Column('preferredStock', String(50)),
                      Column('commonStock', String(50)),
                      Column('retainedEarnings', String(50)),
                      Column('additionalPaidInCapital', String(50)),
                      Column('accumulatedOtherComprehensiveIncomeLoss', String(50)),
                      Column('otherTotalStockholdersEquity', String(50)),
                      Column('totalStockholdersEquity', String(50)),
                      Column('totalEquity', String(50)),
                      Column('minorityInterest', String(50)),
                      Column('totalLiabilitiesAndTotalEquity', String(50)),
                      Column('totalInvestments', String(50)),
                      Column('totalDebt', String(50)),
                      Column('netDebt', String(50)),
                      )
        # 创建表
        metadata.create_all(engine)
        # 先获得所有exchange内的股票信息
        exchange_stock_symbol_name = 'stock_symbol_' + exchange
        Session = sessionmaker(bind=engine)
        session = Session()
        stock_symbol_sql = "SELECT * FROM " + exchange_stock_symbol_name
        stock_symbol_data = pd.read_sql(stock_symbol_sql, session.connection())
        #再插入balance数据
        for i in range(len(stock_symbol_data)):
            print(balance_exchange_name + ':'  + str(float(i) / len(stock_symbol_data)))
            stock_data = stock_symbol_data.iloc[i]
            symbol = stock_data['symbol']
            try:
                url = 'https://financialmodelingprep.com/stable/balance-sheet-statement?limit=8191&period=quarter&symbol=' + symbol + '&apikey=' + token
                fetch_data = get_jsonparsed_data(url)
            except:
                print(symbol)
                continue
            if len(fetch_data) != 0:
                fetch_data = pd.DataFrame(fetch_data)
                balance_data = pd.DataFrame()
                balance_data['symbol'] = fetch_data['symbol']
                balance_data['endDate'] = fetch_data['date'].apply(lambda x: x.replace('-', ''))
                balance_data['endDate'] = balance_data['endDate'].apply(get_quarter_end_date)
                balance_data['cashAndCashEquivalents'] = fetch_data['cashAndCashEquivalents']
                balance_data['shortTermInvestments'] = fetch_data['shortTermInvestments']
                balance_data['cashAndShortTermInvestments'] = fetch_data['cashAndShortTermInvestments']
                balance_data['netReceivables'] = fetch_data['netReceivables']
                balance_data['accountsReceivables'] = fetch_data['accountsReceivables']
                balance_data['otherReceivables'] = fetch_data['otherReceivables']
                balance_data['inventory'] = fetch_data['inventory']
                balance_data['prepaids'] = fetch_data['prepaids']
                balance_data['otherCurrentAssets'] = fetch_data['otherCurrentAssets']
                balance_data['totalCurrentAssets'] = fetch_data['totalCurrentAssets']
                balance_data['propertyPlantEquipmentNet'] = fetch_data['propertyPlantEquipmentNet']
                balance_data['goodwill'] = fetch_data['goodwill']
                balance_data['intangibleAssets'] = fetch_data['intangibleAssets']
                balance_data['goodwillAndIntangibleAssets'] = fetch_data['goodwillAndIntangibleAssets'] 
                balance_data['longTermInvestments'] = fetch_data['longTermInvestments']
                balance_data['taxAssets'] = fetch_data['taxAssets']
                balance_data['otherNonCurrentAssets'] = fetch_data['otherNonCurrentAssets']
                balance_data['totalNonCurrentAssets'] = fetch_data['totalNonCurrentAssets']
                balance_data['otherAssets'] = fetch_data['otherAssets']
                balance_data['totalAssets'] = fetch_data['totalAssets']
                balance_data['totalPayables'] = fetch_data['totalPayables']
                balance_data['accountPayables'] = fetch_data['accountPayables']
                balance_data['otherPayables'] = fetch_data['otherPayables']
                balance_data['accruedExpenses'] = fetch_data['accruedExpenses']
                balance_data['shortTermDebt'] = fetch_data['shortTermDebt']
                balance_data['capitalLeaseObligationsCurrent'] = fetch_data['capitalLeaseObligationsCurrent']
                balance_data['taxPayables'] = fetch_data['taxPayables']
                balance_data['deferredRevenue'] = fetch_data['deferredRevenue']
                balance_data['otherCurrentLiabilities'] = fetch_data['otherCurrentLiabilities']
                balance_data['totalCurrentLiabilities'] = fetch_data['totalCurrentLiabilities']
                balance_data['longTermDebt'] = fetch_data['longTermDebt']
                balance_data['capitalLeaseObligationsNonCurrent'] = fetch_data['capitalLeaseObligationsNonCurrent']
                balance_data['deferredRevenueNonCurrent'] = fetch_data['deferredRevenueNonCurrent']
                balance_data['deferredTaxLiabilitiesNonCurrent'] = fetch_data['deferredTaxLiabilitiesNonCurrent']
                balance_data['otherNonCurrentLiabilities'] = fetch_data['otherNonCurrentLiabilities']
                balance_data['totalNonCurrentLiabilities'] = fetch_data['totalNonCurrentLiabilities']
                balance_data['otherLiabilities'] = fetch_data['otherLiabilities']
                balance_data['capitalLeaseObligations'] = fetch_data['capitalLeaseObligations']
                balance_data['totalLiabilities'] = fetch_data['totalLiabilities']
                balance_data['treasuryStock'] = fetch_data['treasuryStock']
                balance_data['preferredStock'] = fetch_data['preferredStock']
                balance_data['commonStock'] = fetch_data['commonStock']
                balance_data['retainedEarnings'] = fetch_data['retainedEarnings']
                balance_data['additionalPaidInCapital'] = fetch_data['additionalPaidInCapital']
                balance_data['accumulatedOtherComprehensiveIncomeLoss'] = fetch_data['accumulatedOtherComprehensiveIncomeLoss']
                balance_data['otherTotalStockholdersEquity'] = fetch_data['otherTotalStockholdersEquity']
                balance_data['totalStockholdersEquity'] = fetch_data['totalStockholdersEquity']
                balance_data['totalEquity'] = fetch_data['totalEquity']
                balance_data['minorityInterest'] = fetch_data['minorityInterest']
                balance_data['totalLiabilitiesAndTotalEquity'] = fetch_data['totalLiabilitiesAndTotalEquity']
                balance_data['totalInvestments'] = fetch_data['totalInvestments']
                balance_data['totalDebt'] = fetch_data['totalDebt']
                balance_data['netDebt'] = fetch_data['netDebt'] 

                balance_data = balance_data.dropna(subset=['symbol', 'endDate'])
                balance_data = balance_data.fillna(0)
                balance_data = balance_data.sort_values(by='endDate', ascending=False)
                balance_data = balance_data.drop_duplicates(subset=['symbol', 'endDate'], keep='first')
                balance_data = balance_data.reset_index(drop=True)
                balance_data.to_sql(balance_exchange_name, con=engine, if_exists='append', index=False)
        print(balance_exchange_name + ' data inited!')

def write_exchange_balance_data(exchange, quarter_num=7):
    exchange = exchange.lower()
    inspector = sa.inspect(engine)
    # 获取表格名称列表
    table_names = inspector.get_table_names()
    balance_exchange_name = 'balance_' + exchange
    if balance_exchange_name in table_names:
        #先获得当前table的数据
        Session = sessionmaker(bind=engine)
        session = Session()
        #再获得所有exchange内的股票信息
        exchange_stock_symbol_name = 'stock_symbol_' + exchange
        stock_symbol_sql = "SELECT * FROM " + exchange_stock_symbol_name
        stock_symbol_data = pd.read_sql(stock_symbol_sql, session.connection())
        #建立连接
        metadata = MetaData()
        balance_table = Table(balance_exchange_name, metadata, autoload_with=engine)
        conn = engine.connect()
        #开始遍历股票添加数据
        for i in range(len(stock_symbol_data)):
            print(balance_exchange_name + ':' + str(float(i) / len(stock_symbol_data)))
            stock_data = stock_symbol_data.iloc[i]
            symbol = stock_data['symbol']
            try:
                url = 'https://financialmodelingprep.com/stable/balance-sheet-statement?limit=' + str(quarter_num) + '&period=quarter&symbol=' + symbol + '&apikey=' + token
                fetch_data = get_jsonparsed_data(url)
            except:
                print(symbol)
                continue
            if len(fetch_data) == 0:
                continue
            fetch_data = pd.DataFrame(fetch_data)
            balance_data = pd.DataFrame()
            balance_data['symbol'] = fetch_data['symbol']
            balance_data['endDate'] = fetch_data['date'].apply(lambda x: x.replace('-', ''))
            balance_data['endDate'] = balance_data['endDate'].apply(get_quarter_end_date)
            balance_data['cashAndCashEquivalents'] = fetch_data['cashAndCashEquivalents']
            balance_data['shortTermInvestments'] = fetch_data['shortTermInvestments']
            balance_data['cashAndShortTermInvestments'] = fetch_data['cashAndShortTermInvestments']
            balance_data['netReceivables'] = fetch_data['netReceivables']
            balance_data['accountsReceivables'] = fetch_data['accountsReceivables']
            balance_data['otherReceivables'] = fetch_data['otherReceivables']
            balance_data['inventory'] = fetch_data['inventory']
            balance_data['prepaids'] = fetch_data['prepaids']
            balance_data['otherCurrentAssets'] = fetch_data['otherCurrentAssets']
            balance_data['totalCurrentAssets'] = fetch_data['totalCurrentAssets']
            balance_data['propertyPlantEquipmentNet'] = fetch_data['propertyPlantEquipmentNet']
            balance_data['goodwill'] = fetch_data['goodwill']
            balance_data['intangibleAssets'] = fetch_data['intangibleAssets']
            balance_data['goodwillAndIntangibleAssets'] = fetch_data['goodwillAndIntangibleAssets'] 
            balance_data['longTermInvestments'] = fetch_data['longTermInvestments']
            balance_data['taxAssets'] = fetch_data['taxAssets']
            balance_data['otherNonCurrentAssets'] = fetch_data['otherNonCurrentAssets']
            balance_data['totalNonCurrentAssets'] = fetch_data['totalNonCurrentAssets']
            balance_data['otherAssets'] = fetch_data['otherAssets']
            balance_data['totalAssets'] = fetch_data['totalAssets']
            balance_data['totalPayables'] = fetch_data['totalPayables']
            balance_data['accountPayables'] = fetch_data['accountPayables']
            balance_data['otherPayables'] = fetch_data['otherPayables']
            balance_data['accruedExpenses'] = fetch_data['accruedExpenses']
            balance_data['shortTermDebt'] = fetch_data['shortTermDebt']
            balance_data['capitalLeaseObligationsCurrent'] = fetch_data['capitalLeaseObligationsCurrent']
            balance_data['taxPayables'] = fetch_data['taxPayables']
            balance_data['deferredRevenue'] = fetch_data['deferredRevenue']
            balance_data['otherCurrentLiabilities'] = fetch_data['otherCurrentLiabilities']
            balance_data['totalCurrentLiabilities'] = fetch_data['totalCurrentLiabilities']
            balance_data['longTermDebt'] = fetch_data['longTermDebt']
            balance_data['capitalLeaseObligationsNonCurrent'] = fetch_data['capitalLeaseObligationsNonCurrent']
            balance_data['deferredRevenueNonCurrent'] = fetch_data['deferredRevenueNonCurrent']
            balance_data['deferredTaxLiabilitiesNonCurrent'] = fetch_data['deferredTaxLiabilitiesNonCurrent']
            balance_data['otherNonCurrentLiabilities'] = fetch_data['otherNonCurrentLiabilities']
            balance_data['totalNonCurrentLiabilities'] = fetch_data['totalNonCurrentLiabilities']
            balance_data['otherLiabilities'] = fetch_data['otherLiabilities']
            balance_data['capitalLeaseObligations'] = fetch_data['capitalLeaseObligations']
            balance_data['totalLiabilities'] = fetch_data['totalLiabilities']
            balance_data['treasuryStock'] = fetch_data['treasuryStock']
            balance_data['preferredStock'] = fetch_data['preferredStock']
            balance_data['commonStock'] = fetch_data['commonStock']
            balance_data['retainedEarnings'] = fetch_data['retainedEarnings']
            balance_data['additionalPaidInCapital'] = fetch_data['additionalPaidInCapital']
            balance_data['accumulatedOtherComprehensiveIncomeLoss'] = fetch_data['accumulatedOtherComprehensiveIncomeLoss']
            balance_data['otherTotalStockholdersEquity'] = fetch_data['otherTotalStockholdersEquity']
            balance_data['totalStockholdersEquity'] = fetch_data['totalStockholdersEquity']
            balance_data['totalEquity'] = fetch_data['totalEquity']
            balance_data['minorityInterest'] = fetch_data['minorityInterest']
            balance_data['totalLiabilitiesAndTotalEquity'] = fetch_data['totalLiabilitiesAndTotalEquity']
            balance_data['totalInvestments'] = fetch_data['totalInvestments']
            balance_data['totalDebt'] = fetch_data['totalDebt']
            balance_data['netDebt'] = fetch_data['netDebt'] 

            balance_data = balance_data.dropna(subset=['symbol', 'endDate'])
            balance_data = balance_data.fillna(0)
            balance_data = balance_data.sort_values(by='endDate', ascending=False)
            balance_data = balance_data.drop_duplicates(subset=['symbol', 'endDate'], keep='first')
            balance_data = balance_data.reset_index(drop=True)
            count = 0
            for _, row in balance_data.iterrows():
                data = row.to_dict()
                try:
                    _ = conn.execute(balance_table.insert(), data)
                    conn.commit()
                    count += 1
                except:
                    pass
            print('inserted balance_data rows:', count)
    else:
        print(balance_exchange_name + ' table not exist!')
        print('start initing table...')
        init_exchange_balance_data(exchange)

def init_exchange_cashflow_data(exchange):
    exchange = exchange.lower()
    inspector = sa.inspect(engine)
    # 获取表格名称列表
    table_names = inspector.get_table_names()
    cashflow_exchange_name = 'cashflow_' + exchange
    if cashflow_exchange_name in table_names:
        # 如果表格存在，则输出提示信息
        print(cashflow_exchange_name + ' Table exists')
    else:
        # 创建元数据对象
        metadata = MetaData()
        # 创建表对象
        table = Table(cashflow_exchange_name, metadata,
                      Column('symbol', String(50), primary_key=True),
                      Column('endDate', String(50), primary_key=True),
                      Column('deferredIncomeTax', String(50)),
                      Column('stockBasedCompensation', String(50)),
                      Column('changeInWorkingCapital', String(50)),
                      Column('accountsPayables', String(50)),
                      Column('otherWorkingCapital', String(50)),
                      Column('otherNonCashItems', String(50)),
                      Column('netCashProvidedByOperatingActivities', String(50)),
                      Column('investmentsInPropertyPlantAndEquipment', String(50)),
                      Column('acquisitionsNet', String(50)),
                      Column('purchasesOfInvestments', String(50)),
                      Column('salesMaturitiesOfInvestments', String(50)),
                      Column('otherInvestingActivities', String(50)),
                      Column('netCashProvidedByInvestingActivities', String(50)),
                      Column('netDebtIssuance', String(50)),
                      Column('longTermNetDebtIssuance', String(50)),
                      Column('shortTermNetDebtIssuance', String(50)),
                      Column('netStockIssuance', String(50)),
                      Column('netCommonStockIssuance', String(50)),
                      Column('commonStockIssuance', String(50)),
                      Column('commonStockRepurchased', String(50)),
                      Column('netPreferredStockIssuance', String(50)),
                      Column('netDividendsPaid', String(50)),
                      Column('commonDividendsPaid', String(50)),
                      Column('preferredDividendsPaid', String(50)),
                      Column('otherFinancingActivities', String(50)),
                      Column('netCashProvidedByFinancingActivities', String(50)),
                      Column('effectOfForexChangesOnCash', String(50)),
                      Column('netChangeInCash', String(50)),
                      Column('cashAtEndOfPeriod', String(50)),
                      Column('cashAtBeginningOfPeriod', String(50)),
                      Column('operatingCashFlow', String(50)),
                      Column('capitalExpenditure', String(50)),
                      Column('freeCashFlow', String(50)),
                      Column('incomeTaxesPaid', String(50)),
                      Column('interestPaid', String(50)),
                      )
        # 创建表
        metadata.create_all(engine)
        # 先获得所有exchange内的股票信息
        exchange_stock_symbol_name = 'stock_symbol_' + exchange
        Session = sessionmaker(bind=engine)
        session = Session()
        stock_symbol_sql = "SELECT * FROM " + exchange_stock_symbol_name
        stock_symbol_data = pd.read_sql(stock_symbol_sql, session.connection())
        #再插入cashflow数据
        for i in range(len(stock_symbol_data)):
            print(cashflow_exchange_name + ':' + str(float(i) / len(stock_symbol_data)))
            stock_data = stock_symbol_data.iloc[i]
            symbol = stock_data['symbol']
            try:
                url = 'https://financialmodelingprep.com/stable/cash-flow-statement?limit=8191&period=quarter&symbol=' + symbol + '&apikey=' + token
                fetch_data = get_jsonparsed_data(url)
            except:
                print(symbol)
                continue
            if len(fetch_data) != 0:
                fetch_data = pd.DataFrame(fetch_data)
                cashflow_data = pd.DataFrame()
                cashflow_data['symbol'] = fetch_data['symbol']
                cashflow_data['endDate'] = fetch_data['date'].apply(lambda x: x.replace('-', ''))
                cashflow_data['endDate'] = cashflow_data['endDate'].apply(get_quarter_end_date)
                cashflow_data['deferredIncomeTax'] = fetch_data['deferredIncomeTax']
                cashflow_data['stockBasedCompensation'] = fetch_data['stockBasedCompensation']
                cashflow_data['changeInWorkingCapital'] = fetch_data['changeInWorkingCapital']
                cashflow_data['accountsPayables'] = fetch_data['accountsPayables']
                cashflow_data['otherWorkingCapital'] = fetch_data['otherWorkingCapital']
                cashflow_data['otherNonCashItems'] = fetch_data['otherNonCashItems']
                cashflow_data['netCashProvidedByOperatingActivities'] = fetch_data['netCashProvidedByOperatingActivities']
                cashflow_data['investmentsInPropertyPlantAndEquipment'] = fetch_data['investmentsInPropertyPlantAndEquipment']
                cashflow_data['acquisitionsNet'] = fetch_data['acquisitionsNet']
                cashflow_data['purchasesOfInvestments'] = fetch_data['purchasesOfInvestments']
                cashflow_data['salesMaturitiesOfInvestments'] = fetch_data['salesMaturitiesOfInvestments']
                cashflow_data['otherInvestingActivities'] = fetch_data['otherInvestingActivities']
                cashflow_data['netCashProvidedByInvestingActivities'] = fetch_data['netCashProvidedByInvestingActivities']
                cashflow_data['netDebtIssuance'] = fetch_data['netDebtIssuance']
                cashflow_data['longTermNetDebtIssuance'] = fetch_data['longTermNetDebtIssuance']
                cashflow_data['shortTermNetDebtIssuance'] = fetch_data['shortTermNetDebtIssuance']
                cashflow_data['netStockIssuance'] = fetch_data['netStockIssuance']
                cashflow_data['netCommonStockIssuance'] = fetch_data['netCommonStockIssuance']
                cashflow_data['commonStockIssuance'] = fetch_data['commonStockIssuance']
                cashflow_data['commonStockRepurchased'] = fetch_data['commonStockRepurchased']
                cashflow_data['netPreferredStockIssuance'] = fetch_data['netPreferredStockIssuance']
                cashflow_data['netDividendsPaid'] = fetch_data['netDividendsPaid']
                cashflow_data['commonDividendsPaid'] = fetch_data['commonDividendsPaid']
                cashflow_data['preferredDividendsPaid'] = fetch_data['preferredDividendsPaid']
                cashflow_data['otherFinancingActivities'] = fetch_data['otherFinancingActivities']
                cashflow_data['netCashProvidedByFinancingActivities'] = fetch_data['netCashProvidedByFinancingActivities']
                cashflow_data['effectOfForexChangesOnCash'] = fetch_data['effectOfForexChangesOnCash']
                cashflow_data['netChangeInCash'] = fetch_data['netChangeInCash']
                cashflow_data['cashAtEndOfPeriod'] = fetch_data['cashAtEndOfPeriod']
                cashflow_data['cashAtBeginningOfPeriod'] = fetch_data['cashAtBeginningOfPeriod']
                cashflow_data['operatingCashFlow'] = fetch_data['operatingCashFlow']
                cashflow_data['capitalExpenditure'] = fetch_data['capitalExpenditure']
                cashflow_data['freeCashFlow'] = fetch_data['freeCashFlow']
                cashflow_data['incomeTaxesPaid'] = fetch_data['incomeTaxesPaid']
                cashflow_data['interestPaid'] = fetch_data['interestPaid']

                cashflow_data = cashflow_data.dropna(subset=['symbol', 'endDate'])
                cashflow_data = cashflow_data.fillna(0)
                cashflow_data = cashflow_data.sort_values(by='endDate', ascending=False)
                cashflow_data = cashflow_data.drop_duplicates(subset=['symbol', 'endDate'], keep='first')
                cashflow_data = cashflow_data.reset_index(drop=True)
                cashflow_data.to_sql(cashflow_exchange_name, con=engine, if_exists='append', index=False)
        print(cashflow_exchange_name + ' data inited!')

def write_exchange_cashflow_data(exchange, quarter_num=7):
    exchange = exchange.lower()
    inspector = sa.inspect(engine)
    # 获取表格名称列表
    table_names = inspector.get_table_names()
    cashflow_exchange_name = 'cashflow_' + exchange
    if cashflow_exchange_name in table_names:
        #先获得当前table的数据
        Session = sessionmaker(bind=engine)
        session = Session()
        #再获得所有exchange内的股票信息
        exchange_stock_symbol_name = 'stock_symbol_' + exchange
        stock_symbol_sql = "SELECT * FROM " + exchange_stock_symbol_name
        stock_symbol_data = pd.read_sql(stock_symbol_sql, session.connection())
        #建立连接
        metadata = MetaData()
        cashflow_table = Table(cashflow_exchange_name, metadata, autoload_with=engine)
        conn = engine.connect()
        #开始遍历股票添加数据
        for i in range(len(stock_symbol_data)):
            print(cashflow_exchange_name  + ':' + str(float(i) / len(stock_symbol_data)))
            stock_data = stock_symbol_data.iloc[i]
            symbol = stock_data['symbol']
            try:
                url = 'https://financialmodelingprep.com/stable/cash-flow-statement?limit=' + str(quarter_num) + '&period=quarter&symbol=' + symbol + '&apikey=' + token
                fetch_data = get_jsonparsed_data(url)
            except:
                print(symbol)
                continue
            if len(fetch_data) == 0:
                continue
            fetch_data = pd.DataFrame(fetch_data)
            cashflow_data = pd.DataFrame()
            cashflow_data['symbol'] = fetch_data['symbol']
            cashflow_data['endDate'] = fetch_data['date'].apply(lambda x: x.replace('-', ''))
            cashflow_data['endDate'] = cashflow_data['endDate'].apply(get_quarter_end_date)
            cashflow_data['deferredIncomeTax'] = fetch_data['deferredIncomeTax']
            cashflow_data['stockBasedCompensation'] = fetch_data['stockBasedCompensation']
            cashflow_data['changeInWorkingCapital'] = fetch_data['changeInWorkingCapital']
            cashflow_data['accountsPayables'] = fetch_data['accountsPayables']
            cashflow_data['otherWorkingCapital'] = fetch_data['otherWorkingCapital']
            cashflow_data['otherNonCashItems'] = fetch_data['otherNonCashItems']
            cashflow_data['netCashProvidedByOperatingActivities'] = fetch_data['netCashProvidedByOperatingActivities']
            cashflow_data['investmentsInPropertyPlantAndEquipment'] = fetch_data['investmentsInPropertyPlantAndEquipment']
            cashflow_data['acquisitionsNet'] = fetch_data['acquisitionsNet']
            cashflow_data['purchasesOfInvestments'] = fetch_data['purchasesOfInvestments']
            cashflow_data['salesMaturitiesOfInvestments'] = fetch_data['salesMaturitiesOfInvestments']
            cashflow_data['otherInvestingActivities'] = fetch_data['otherInvestingActivities']
            cashflow_data['netCashProvidedByInvestingActivities'] = fetch_data['netCashProvidedByInvestingActivities']
            cashflow_data['netDebtIssuance'] = fetch_data['netDebtIssuance']
            cashflow_data['longTermNetDebtIssuance'] = fetch_data['longTermNetDebtIssuance']
            cashflow_data['shortTermNetDebtIssuance'] = fetch_data['shortTermNetDebtIssuance']
            cashflow_data['netStockIssuance'] = fetch_data['netStockIssuance']
            cashflow_data['netCommonStockIssuance'] = fetch_data['netCommonStockIssuance']
            cashflow_data['commonStockIssuance'] = fetch_data['commonStockIssuance']
            cashflow_data['commonStockRepurchased'] = fetch_data['commonStockRepurchased']
            cashflow_data['netPreferredStockIssuance'] = fetch_data['netPreferredStockIssuance']
            cashflow_data['netDividendsPaid'] = fetch_data['netDividendsPaid']
            cashflow_data['commonDividendsPaid'] = fetch_data['commonDividendsPaid']
            cashflow_data['preferredDividendsPaid'] = fetch_data['preferredDividendsPaid']
            cashflow_data['otherFinancingActivities'] = fetch_data['otherFinancingActivities']
            cashflow_data['netCashProvidedByFinancingActivities'] = fetch_data['netCashProvidedByFinancingActivities']
            cashflow_data['effectOfForexChangesOnCash'] = fetch_data['effectOfForexChangesOnCash']
            cashflow_data['netChangeInCash'] = fetch_data['netChangeInCash']
            cashflow_data['cashAtEndOfPeriod'] = fetch_data['cashAtEndOfPeriod']
            cashflow_data['cashAtBeginningOfPeriod'] = fetch_data['cashAtBeginningOfPeriod']
            cashflow_data['operatingCashFlow'] = fetch_data['operatingCashFlow']
            cashflow_data['capitalExpenditure'] = fetch_data['capitalExpenditure']
            cashflow_data['freeCashFlow'] = fetch_data['freeCashFlow']
            cashflow_data['incomeTaxesPaid'] = fetch_data['incomeTaxesPaid']
            cashflow_data['interestPaid'] = fetch_data['interestPaid']

            cashflow_data = cashflow_data.dropna(subset=['symbol', 'endDate'])
            cashflow_data = cashflow_data.fillna(0)
            cashflow_data = cashflow_data.sort_values(by='endDate', ascending=False)
            cashflow_data = cashflow_data.drop_duplicates(subset=['symbol', 'endDate'], keep='first')
            cashflow_data = cashflow_data.reset_index(drop=True)
            count = 0
            for _, row in cashflow_data.iterrows():
                data = row.to_dict()
                try:
                    _ = conn.execute(cashflow_table.insert(), data)
                    # print("Inserted cashflow_data rows:", data)
                    conn.commit()
                    count += 1
                except:
                    pass
            print('inserted cashflow_data rows:', count)
    else:
        print(cashflow_exchange_name + ' table not exist!')
        print('start initing table...')
        init_exchange_cashflow_data(exchange)

def write_daily_data_into_table(daily_exchange_name, symbol_list, days, check=False):
    #建立连接
    metadata = MetaData()
    daily_table = Table(daily_exchange_name, metadata, autoload_with=engine)
    conn = engine.connect()
    for symbol in tqdm(symbol_list):
        today = datetime.datetime.today()
        if check:
            today = today - datetime.timedelta(days=3)
        days_ago = today - datetime.timedelta(days=days)
        today = today.strftime("%Y-%m-%d")
        days_ago = days_ago.strftime("%Y-%m-%d")
        try:
            url = 'https://financialmodelingprep.com/stable/historical-price-eod/full?symbol=' + symbol + '&from=' + days_ago +'&to='+ str(today)+'&apikey='+token
            fetch_data = get_jsonparsed_data(url)
        except:
            print(symbol)
            continue
        insert_num = 0
        if len(fetch_data) != 0:
            fetch_data = pd.DataFrame(fetch_data)
            daily_data = pd.DataFrame()
            daily_data['symbol'] = fetch_data['symbol']
            daily_data['date'] = fetch_data['date'] 
            daily_data['date'] = daily_data['date'].apply(lambda x: x.replace('-', ''))
            daily_data['open'] = fetch_data['open']
            daily_data['low'] = fetch_data['low']       
            daily_data['high'] = fetch_data['high']
            daily_data['close'] = fetch_data['close']
            daily_data['volume'] = fetch_data['volume']
            daily_data.drop_duplicates(subset=['symbol', 'date'], keep='first', inplace=True)
            daily_data = daily_data.reset_index(drop=True)
            if check:
                daily_data.to_sql(daily_exchange_name, con=engine, if_exists='append', index=False)
            else:
                for _, row in daily_data.iterrows():
                    data = row.to_dict()
                    try:
                        _ = conn.execute(daily_table.insert(), data)
                        conn.commit()
                        insert_num += 1
                    except:
                        pass
                print('insert_num:' + str(insert_num))



def init_exchange_daily_data(exchange):
    exchange = exchange.lower()
    inspector = sa.inspect(engine)
    # 获取表格名称列表
    table_names = inspector.get_table_names()
    daily_exchange_name = 'daily_' + exchange
    if daily_exchange_name in table_names:
        # 如果表格存在，则输出提示信息
        print(daily_exchange_name + ' Table exists')
    else:
        # 创建元数据对象
        metadata = MetaData()
        # 创建表对象
        table = Table(daily_exchange_name, metadata,
                      Column('symbol', String(50), primary_key=True),
                      Column('date', String(50), primary_key=True),
                      Column('open', String(50)),
                      Column('low', String(50)),
                      Column('high', String(50)),
                      Column('close', String(50)),
                      Column('volume', String(50)),
                      )
        # 创建表
        metadata.create_all(engine)
        # 先获得所有exchange内的股票信息
        exchange_stock_symbol_name = 'stock_symbol_' + exchange
        Session = sessionmaker(bind=engine)
        session = Session()
        stock_symbol_sql = "SELECT * FROM " + exchange_stock_symbol_name
        stock_symbol_data = pd.read_sql(stock_symbol_sql, session.connection())

        # 将stock_symbol_data一分为3，开线程写入daily_exchange_name
        symbol_list = stock_symbol_data['symbol'].tolist()
        split_num = int(len(symbol_list) * 0.333)
        symbol_list_0 = symbol_list[:split_num]
        symbol_list_1 = symbol_list[split_num:2 * split_num]
        symbol_list_2 = symbol_list[2 * split_num:]
        days = 131071
        # 创建线程并启动它们
        thread1 = threading.Thread(target=write_daily_data_into_table, args=(daily_exchange_name, symbol_list_0, days, True))
        thread2 = threading.Thread(target=write_daily_data_into_table, args=(daily_exchange_name, symbol_list_1, days, True))
        thread3 = threading.Thread(target=write_daily_data_into_table, args=(daily_exchange_name, symbol_list_2, days, True))

        thread1.start()
        thread2.start()
        thread3.start()

        thread1.join()
        thread2.join()
        thread3.join()

        print(daily_exchange_name + ' data inited!')


def write_exchange_daily_data(exchange, days=127, check=False):
    # write_exchange_stock_symbol_data(exchange)
    exchange = exchange.lower()
    inspector = sa.inspect(engine)
    # 获取表格名称列表
    table_names = inspector.get_table_names()
    daily_exchange_name = 'daily_' + exchange
    if daily_exchange_name in table_names:
        # 如果表格存在，则输出提示信息
        print(daily_exchange_name + ' Table exists')
        #再获得所有exchange内的股票信息
        exchange_stock_symbol_name = 'stock_symbol_' + exchange
        Session = sessionmaker(bind=engine)
        session = Session()
        stock_symbol_sql = "SELECT * FROM " + exchange_stock_symbol_name
        stock_symbol_data = pd.read_sql(stock_symbol_sql, session.connection())
        stock_symbol_data = stock_symbol_data.sample(frac=1, random_state=None)
        stock_symbol_data = stock_symbol_data.reset_index(drop=True)
        #将stock_symbol_data一分为3，开线程写入daily_exchange_name
        symbol_list = stock_symbol_data['symbol'].tolist()
        split_num = int(len(symbol_list) * 0.333)
        symbol_list_0 = symbol_list[:split_num]
        symbol_list_1 = symbol_list[split_num:2 * split_num]
        symbol_list_2 = symbol_list[2 * split_num:]
        # 创建线程并启动它们
        thread1 = threading.Thread(target=write_daily_data_into_table, args=(daily_exchange_name, symbol_list_0, days, check))
        thread2 = threading.Thread(target=write_daily_data_into_table, args=(daily_exchange_name, symbol_list_1, days, check))
        thread3 = threading.Thread(target=write_daily_data_into_table, args=(daily_exchange_name, symbol_list_2, days, check))

        thread1.start()
        thread2.start()
        thread3.start()

        thread1.join()
        thread2.join()
        thread3.join()
    else:
        print(daily_exchange_name + ' table not exist!')
        print('start initing table...')
        init_exchange_daily_data(exchange)


def init_exchange_indicator_data(exchange):
    inspector = sa.inspect(engine)
    # 获取表格名称列表
    table_names = inspector.get_table_names()
    indicator_exchange_name = 'indicator_' + exchange.lower()
    # 创建元数据对象
    metadata = MetaData()
    if indicator_exchange_name in table_names:
        # 如果表格存在，则输出提示信息
        print(indicator_exchange_name + ' Table exists')
    else:
        # 创建表对象
        table = Table(indicator_exchange_name, metadata,
                      Column('symbol', String(50), primary_key=True),
                      Column('endDate', String(50), primary_key=True),
                      Column('netAssetValuePerShare', String(50)),
                      Column('dcfPerShare', String(50)),
                      Column('dividendPerShare', String(50)),
                      Column('debtRatio', String(50)),
                      Column('debtToEquity', String(50)),
                      Column('interestCoverage', String(50)),
                      Column('cashRatio', String(50)),
                      Column('quickRatio', String(50)),
                      Column('currentRatio', String(50)),
                      Column('inventoryTurnover', String(50)),
                      Column('revenueUnit', String(50)),
                      Column('costOfRevenueUnit', String(50)),
                      Column('grossProfitUnit', String(50)), 
                      Column('operatingExpensesUnit', String(50)), 
                      Column('costAndExpensesUnit', String(50)), 
                      Column('ebitdaUnit', String(50)),
                      Column('operatingIncomeUnit', String(50)),
                      Column('incomeBeforeTaxUnit', String(50)),
                      Column('netReceivablesUnit', String(50)),
                      Column('accountsReceivablesUnit', String(50)),
                      Column('inventoryUnit', String(50)), 
                      Column('totalCurrentAssetsUnit', String(50)), 
                      Column('propertyPlantEquipmentNetUnit', String(50)),
                      Column('longTermInvestmentsUnit', String(50)),
                      Column('totalAssetsUnit', String(50)),
                      Column('totalPayablesUnit', String(50)),
                      Column('accountPayablesUnit', String(50)),
                      Column('shortTermDebtUnit', String(50)),
                      Column('totalCurrentLiabilitiesUnit', String(50)),
                      Column('longTermDebtUnit', String(50)),
                      Column('totalLiabilitiesUnit', String(50)),
                      Column('retainedEarningsUnit', String(50)), 
                      Column('totalEquityUnit', String(50)), 
                      Column('totalLiabilitiesAndTotalEquityUnit', String(50)), 
                      Column('totalInvestmentsUnit', String(50)),
                      Column('totalDebtUnit', String(50)),
                      Column('netDebtUnit', String(50)),
                      Column('netCashProvidedByOperatingActivitiesUnit', String(50)),
                      Column('netCashProvidedByFinancingActivitiesUnit', String(50)),                
                      Column('cashAtEndOfPeriodUnit', String(50)),
                      Column('cashAtBeginningOfPeriodUnit', String(50)),
                      )
        # 创建表
        metadata.create_all(engine)
        print(indicator_exchange_name + ' Table created!')
        #从mysql中读取交易所财务三张表和分红数据
        Session = sessionmaker(bind=engine)
        session = Session()
        income_table_name = 'income_' + exchange
        income_sql = "SELECT * FROM " + income_table_name
        income_data = pd.read_sql(income_sql, session.connection())
        balance_table_name = 'balance_' + exchange
        balance_sql = "SELECT * FROM " + balance_table_name
        balance_data = pd.read_sql(balance_sql, session.connection())
        cashflow_table_name = 'cashflow_' + exchange
        cashflow_sql = "SELECT * FROM " + cashflow_table_name
        cashflow_data = pd.read_sql(cashflow_sql, session.connection())


        #合并三张表和分红表
        merge_data = pd.merge(income_data, balance_data, on=['symbol', 'endDate'], how='outer')
        merge_data = pd.merge(merge_data, cashflow_data, on=['symbol', 'endDate'], how='outer')
        merge_data = merge_data.dropna(subset=['symbol', 'endDate'])
        merge_data = merge_data.fillna(0)
        merge_data = merge_data.reset_index(drop=True)
        print(merge_data)
        merge_data['weightedAverageShsOut'] = merge_data['weightedAverageShsOut'].replace(0.0, pd.NA).ffill().bfill()
        merge_data['totalStockholdersEquity'] = merge_data['totalStockholdersEquity'].replace(0.0, pd.NA).ffill().bfill()
        merge_data['freeCashFlow'] = merge_data['freeCashFlow'].replace(0.0, pd.NA).ffill().bfill()
        merge_data['commonDividendsPaid'] = merge_data['commonDividendsPaid'].replace(0.0, pd.NA).ffill().bfill()
        merge_data['totalAssets'] = merge_data['totalAssets'].replace(0.0, pd.NA).ffill().bfill()
        merge_data['operatingIncome'] = merge_data['operatingIncome'].replace(0.0, pd.NA).ffill().bfill()
        merge_data['totalLiabilities'] = merge_data['totalLiabilities'].replace(0.0, pd.NA).ffill().bfill()
        merge_data['interestExpense'] = merge_data['interestExpense'].replace(0.0, pd.NA).ffill().bfill()
        merge_data['cashAndCashEquivalents'] = merge_data['cashAndCashEquivalents'].replace(0.0, pd.NA).ffill().bfill()
        merge_data['shortTermInvestments'] = merge_data['shortTermInvestments'].replace(0.0, pd.NA).ffill().bfill()
        merge_data['netReceivables'] = merge_data['netReceivables'].replace(0.0, pd.NA).ffill().bfill()
        merge_data['totalCurrentLiabilities'] = merge_data['totalCurrentLiabilities'].replace(0.0, pd.NA).ffill().bfill()
        merge_data['totalCurrentAssets'] = merge_data['totalCurrentAssets'].replace(0.0, pd.NA).ffill().bfill()
        merge_data['costOfRevenue'] = merge_data['costOfRevenue'].replace(0.0, pd.NA).ffill().bfill()
        merge_data['inventory'] = merge_data['inventory'].replace(0.0, pd.NA).ffill().bfill()
        merge_data['revenue'] = merge_data['revenue'].replace(0.0, pd.NA).ffill().bfill()
        merge_data['grossProfit'] = merge_data['grossProfit'].replace(0.0, pd.NA).ffill().bfill()
        merge_data['operatingExpenses'] = merge_data['operatingExpenses'].replace(0.0, pd.NA).ffill().bfill()
        merge_data['costAndExpenses'] = merge_data['costAndExpenses'].replace(0.0, pd.NA).ffill().bfill()
        merge_data['ebitda'] = merge_data['ebitda'].replace(0.0, pd.NA).ffill().bfill()
        merge_data['incomeBeforeTax'] = merge_data['incomeBeforeTax'].replace(0.0, pd.NA).ffill().bfill()
        merge_data['accountsReceivables'] = merge_data['accountsReceivables'].replace(0.0, pd.NA).ffill().bfill()
        merge_data['propertyPlantEquipmentNet'] = merge_data['propertyPlantEquipmentNet'].replace(0.0, pd.NA).ffill().bfill()
        merge_data['longTermInvestments'] = merge_data['longTermInvestments'].replace(0.0, pd.NA).ffill().bfill()
        merge_data['totalPayables'] = merge_data['totalPayables'].replace(0.0, pd.NA).ffill().bfill()
        merge_data['accountPayables'] = merge_data['accountPayables'].replace(0.0, pd.NA).ffill().bfill()
        merge_data['shortTermDebt'] = merge_data['shortTermDebt'].replace(0.0, pd.NA).ffill().bfill()
        merge_data['longTermDebt'] = merge_data['longTermDebt'].replace(0.0, pd.NA).ffill().bfill()
        merge_data['retainedEarnings'] = merge_data['retainedEarnings'].replace(0.0, pd.NA).ffill().bfill()
        merge_data['totalEquity'] = merge_data['totalEquity'].replace(0.0, pd.NA).ffill().bfill()
        merge_data['totalLiabilitiesAndTotalEquity'] = merge_data['totalLiabilitiesAndTotalEquity'].replace(0.0, pd.NA).ffill().bfill()
        merge_data['totalInvestments'] = merge_data['totalInvestments'].replace(0.0, pd.NA).ffill().bfill()
        merge_data['totalDebt'] = merge_data['totalDebt'].replace(0.0, pd.NA).ffill().bfill()
        merge_data['netDebt'] = merge_data['netDebt'].replace(0.0, pd.NA).ffill().bfill()
        merge_data['netCashProvidedByOperatingActivities'] = merge_data['netCashProvidedByOperatingActivities'].replace(0.0, pd.NA).ffill().bfill()
        merge_data['netCashProvidedByFinancingActivities'] = merge_data['netCashProvidedByFinancingActivities'].replace(0.0, pd.NA).ffill().bfill()
        merge_data['cashAtEndOfPeriod'] = merge_data['cashAtEndOfPeriod'].replace(0.0, pd.NA).ffill().bfill()
        merge_data['cashAtBeginningOfPeriod'] = merge_data['cashAtBeginningOfPeriod'].replace(0.0, pd.NA).ffill().bfill()



        indicator_data = pd.DataFrame()
        indicator_data['symbol'] = merge_data['symbol']
        indicator_data['endDate'] = merge_data['endDate']
        indicator_data['netAssetValuePerShare'] = merge_data['totalStockholdersEquity'].astype(float)/merge_data['weightedAverageShsOut'].astype(float)
        indicator_data['dcfPerShare'] = merge_data['freeCashFlow'].astype(float)/merge_data['weightedAverageShsOut'].astype(float)
        indicator_data['dividendPerShare'] = abs(merge_data['commonDividendsPaid'].astype(float)/merge_data['weightedAverageShsOut'].astype(float))
        indicator_data['debtRatio'] = merge_data['totalLiabilities'].astype(float)/merge_data['totalAssets'].astype(float)
        indicator_data['debtToEquity'] = merge_data['totalLiabilities'].astype(float)/merge_data['totalStockholdersEquity'].astype(float)
        indicator_data['interestCoverage'] = merge_data['operatingIncome'].astype(float)/merge_data['interestExpense'].astype(float)
        indicator_data["cashRatio"] = (merge_data["cashAndCashEquivalents"].astype(float) + merge_data["shortTermInvestments"].astype(float)) / merge_data["totalCurrentLiabilities"].astype(float)
        indicator_data["quickRatio"] = (merge_data["cashAndCashEquivalents"].astype(float) + merge_data["shortTermInvestments"].astype(float) 
                                + merge_data["netReceivables"].astype(float)) / merge_data["totalCurrentLiabilities"].astype(float)
        indicator_data['currentRatio'] = merge_data['totalCurrentAssets'].astype(float)/merge_data['totalCurrentLiabilities'].astype(float)
        indicator_data['inventoryTurnover'] = merge_data['costOfRevenue'].astype(float)/merge_data['inventory'].astype(float)
        indicator_data["revenueUnit"] = merge_data["revenue"].astype(float) / merge_data["totalStockholdersEquity"].astype(float)
        indicator_data['costOfRevenueUnit'] = merge_data["costOfRevenue"].astype(float) / merge_data["totalStockholdersEquity"].astype(float)
        indicator_data["grossProfitUnit"] = merge_data["grossProfit"].astype(float) / merge_data["totalStockholdersEquity"].astype(float)
        indicator_data["operatingExpensesUnit"] = merge_data["operatingExpenses"].astype(float) / merge_data["totalStockholdersEquity"].astype(float)
        indicator_data["costAndExpensesUnit"] = merge_data["costAndExpenses"].astype(float) / merge_data["totalStockholdersEquity"].astype(float)
        indicator_data["ebitdaUnit"] = merge_data["ebitda"].astype(float) / merge_data["totalStockholdersEquity"].astype(float)
        indicator_data['operatingIncomeUnit'] = merge_data["operatingIncome"].astype(float) / merge_data["totalStockholdersEquity"].astype(float)
        indicator_data['incomeBeforeTaxUnit'] = merge_data["incomeBeforeTax"].astype(float) / merge_data["totalStockholdersEquity"].astype(float)
        indicator_data['netReceivablesUnit'] = merge_data["netReceivables"].astype(float) / merge_data["totalStockholdersEquity"].astype(float)
        indicator_data['accountsReceivablesUnit'] = merge_data["accountsReceivables"].astype(float) / merge_data["totalStockholdersEquity"].astype(float)
        indicator_data['inventoryUnit'] = merge_data["inventory"].astype(float) / merge_data["totalStockholdersEquity"].astype(float)
        indicator_data['totalCurrentAssetsUnit'] = merge_data["totalCurrentAssets"].astype(float) / merge_data["totalStockholdersEquity"].astype(float)
        indicator_data['propertyPlantEquipmentNetUnit'] = merge_data["propertyPlantEquipmentNet"].astype(float) / merge_data["totalStockholdersEquity"].astype(float)
        indicator_data['longTermInvestmentsUnit'] = merge_data["longTermInvestments"].astype(float) / merge_data["totalStockholdersEquity"].astype(float)
        indicator_data['totalAssetsUnit'] = merge_data["totalAssets"].astype(float) / merge_data["totalStockholdersEquity"].astype(float)
        indicator_data['totalPayablesUnit'] = merge_data["totalPayables"].astype(float) / merge_data["totalStockholdersEquity"].astype(float)
        indicator_data['accountPayablesUnit'] = merge_data["accountPayables"].astype(float) / merge_data["totalStockholdersEquity"].astype(float)
        indicator_data['shortTermDebtUnit'] = merge_data["shortTermDebt"].astype(float) / merge_data["totalStockholdersEquity"].astype(float)
        indicator_data['totalCurrentLiabilitiesUnit'] = merge_data["totalCurrentLiabilities"].astype(float) / merge_data["totalStockholdersEquity"].astype(float)
        indicator_data['longTermDebtUnit'] = merge_data["longTermDebt"].astype(float) / merge_data["totalStockholdersEquity"].astype(float)
        indicator_data['totalLiabilitiesUnit'] = merge_data["totalLiabilities"].astype(float) / merge_data["totalStockholdersEquity"].astype(float)
        indicator_data['retainedEarningsUnit'] = merge_data["retainedEarnings"].astype(float) / merge_data["totalStockholdersEquity"].astype(float)
        indicator_data['totalEquityUnit'] = merge_data["totalEquity"].astype(float) / merge_data["totalStockholdersEquity"].astype(float)
        indicator_data['totalLiabilitiesAndTotalEquityUnit'] = merge_data["totalLiabilitiesAndTotalEquity"].astype(float) / merge_data["totalStockholdersEquity"].astype(float)
        indicator_data['totalInvestmentsUnit'] = merge_data["totalInvestments"].astype(float) / merge_data["totalStockholdersEquity"].astype(float)
        indicator_data['totalDebtUnit'] = merge_data["totalDebt"].astype(float) / merge_data["totalStockholdersEquity"].astype(float)
        indicator_data['netDebtUnit'] = merge_data["netDebt"].astype(float) / merge_data["totalStockholdersEquity"].astype(float)
        indicator_data['netCashProvidedByOperatingActivitiesUnit'] = merge_data["netCashProvidedByOperatingActivities"].astype(float) / merge_data["totalStockholdersEquity"].astype(float)
        indicator_data['netCashProvidedByFinancingActivitiesUnit'] = merge_data["netCashProvidedByFinancingActivities"].astype(float) / merge_data["totalStockholdersEquity"].astype(float)
        indicator_data['cashAtEndOfPeriodUnit'] = merge_data["cashAtEndOfPeriod"].astype(float) / merge_data["totalStockholdersEquity"].astype(float)
        indicator_data['cashAtBeginningOfPeriodUnit'] = merge_data["cashAtBeginningOfPeriod"].astype(float) / merge_data["totalStockholdersEquity"].astype(float)
        
        indicator_data = indicator_data.dropna(subset=['symbol', 'endDate'])
        indicator_data = indicator_data.replace([np.inf, -np.inf], 0)
        indicator_data = indicator_data.fillna(0)
        indicator_data.drop_duplicates(subset=['symbol', 'endDate'], keep='first', inplace=True)
        indicator_data = indicator_data.reset_index(drop=True)
        print(indicator_data)
        indicator_data.to_sql(indicator_exchange_name, con=engine, if_exists='append', index=False)
        print(indicator_exchange_name + ' data inited!')


def write_exchange_indicator_data(exchange):
    #删除indicator那张表
    exchange = exchange.lower()
    inspector = sa.inspect(engine)
    table_names = inspector.get_table_names()
    drop_list = []
    for i in range(len(table_names)):
        table_name = table_names[i]
        if exchange in table_name:
            drop_list.append(table_name)
    metadata = MetaData()
    for i in range(len(drop_list)):
        drop_name = drop_list[i]
        if 'indicator' in drop_name:
            drop_table = Table(drop_name, metadata, autoload_with=engine)
            drop_table.drop(engine)
            print(drop_name + ' deleted!')
    #重新生成
    init_exchange_indicator_data(exchange)


def write_exchange_financial_data(exchange, quater_num=7):
    t1 = threading.Thread(target=write_exchange_income_data, args=(exchange, quater_num))
    t2 = threading.Thread(target=write_exchange_balance_data, args=(exchange, quater_num))
    t3 = threading.Thread(target=write_exchange_cashflow_data, args=(exchange, quater_num))
    t1.start()
    t2.start()
    t3.start()
    t1.join()
    t2.join()
    t3.join()
    write_exchange_indicator_data(exchange)


def repair_exchange_data(exchange):
    write_exchange_stock_symbol_data(exchange)
    write_exchange_financial_data(exchange, 127)
    write_exchange_daily_data(exchange, 8191)


def update_exchange_data(exchange):
    write_exchange_stock_symbol_data(exchange)
    write_exchange_financial_data(exchange, 7)
    write_exchange_daily_data(exchange, 127)
    get_exchange_data(exchange)


def append_exchange_data(exchange):
    write_exchange_daily_data(exchange, 31)
    get_exchange_daily_data(exchange)


def repair_data():
    get_train_exchanges()
    exchanges_data = pd.read_csv('train_exchanges.csv', encoding="utf-8")
    exchange_list = exchanges_data['exchange'].tolist()
    # exchange_list = ['SHZ', 'SHH']
    print(exchange_list)
    failed_list = []
    exchanges = exchange_list
    # random.shuffle(exchanges)
    for exchange in exchanges:
        print(exchange)
        try:
            repair_exchange_data(exchange)
        except:
            failed_list.append(exchange)
    time.sleep(31)
    print(failed_list)


def get_train_exchanges():
    url_symbols = 'https://financialmodelingprep.com/stable/financial-statement-symbol-list?apikey=' + token
    symbol_list = get_jsonparsed_data(url_symbols)
    symbol_list = pd.DataFrame.from_records(symbol_list)
    symbol_list = symbol_list['symbol'].to_list()
    print(symbol_list)
    url_exchanges = 'https://financialmodelingprep.com/stable/available-exchanges?apikey=' + token
    avaliable_exchanges = get_jsonparsed_data(url_exchanges)
    print(avaliable_exchanges)
    train_exchange_list = []
    for i in range(len(avaliable_exchanges)):
        exchange = avaliable_exchanges[i]
        exchange_name = exchange['exchange']
        exchange_symbol_url = 'https://financialmodelingprep.com/stable/company-screener?limit=131071&isEtf=false&isFund=false&exchange=' + exchange_name + '&apikey=' + token
        exchange_symbol_list = get_jsonparsed_data(exchange_symbol_url)
        exchange_symbol_list = pd.DataFrame.from_records(exchange_symbol_list)
        if len(exchange_symbol_list) == 0:
            print(exchange)
            continue
        exchange_symbol_list = exchange_symbol_list['symbol'].tolist()
        exchange_symbol_list = list(set(symbol_list) & set(exchange_symbol_list))#要有财务数据
        if len(exchange_symbol_list) > 127 and len(exchange_symbol_list) < 8191:
            exchange['num_symbols'] = len(exchange_symbol_list)
            print(exchange)
            train_exchange_list.append(exchange)
    train_exchanges = pd.DataFrame(train_exchange_list)
    train_exchanges = train_exchanges.sort_values(by='num_symbols', ascending=False)
    train_exchanges = train_exchanges.iloc[:31] 
    train_exchanges = train_exchanges.reset_index(drop=True)  
    print(train_exchanges) 
    train_exchanges.to_csv('train_exchanges.csv', index=False, encoding="utf-8")

def test_fetch_data():
    symbol = '600617.SS'
    exchange = 'SHH'
    try:
        # https://financialmodelingprep.com/stable/historical-price-eod/full?symbol=AAPL&apikey=GPlAljd6JWEZA7YSBLKgHTZg2QiNpBH4
        # https://financialmodelingprep.com/stable/financial-statement-symbol-list?apikey=GPlAljd6JWEZA7YSBLKgHTZg2QiNpBH4
        today = datetime.datetime.today()
        days_ago = today - datetime.timedelta(days=8191)
        today = today.strftime("%Y-%m-%d")
        days_ago = days_ago.strftime("%Y-%m-%d")
        try:
            url = 'https://financialmodelingprep.com/stable/cash-flow-statement?limit=' + str(7) + '&period=quarter&symbol=' + symbol + '&apikey=' + token
            # url = 'https://financialmodelingprep.com/stable/balance-sheet-statement?limit=' + str(7) + '&period=quarter&symbol=' + symbol + '&apikey=' + token
            # url = 'https://financialmodelingprep.com/stable/historical-price-eod/full?symbol=' + symbol + '&from=' + days_ago +'&to='+ str(today)+'&apikey='+token
        except:
            print(symbol)
        data = get_jsonparsed_data(url)
        print(data)
    except Exception as e:
        print(e)
    data = pd.DataFrame(data)
    data.to_csv('test_'+symbol+'.csv', index=False, encoding="utf-8")
    print(data)


def refresh_indicator():
    exchanges_data = pd.read_csv('train_exchanges.csv', encoding="utf-8")
    exchange_list = exchanges_data['exchange'].tolist()
    print(exchange_list)
    failed_list = []
    exchanges = exchange_list
    # random.shuffle(exchanges)
    for exchange in exchanges:
        print(exchange)
        try:
            write_exchange_indicator_data(exchange)
        except:
            failed_list.append(exchange)
    time.sleep(1)
    print(failed_list)
    get_indicator_data()



if __name__ == "__main__":
    # write_exchange_indicator_data('TSXV')
    # repair_exchange_data('TSXV')
    update_exchange_data('SHZ')
    update_exchange_data('SHH')