import os
import sys
import ssl
import certifi
import json
import math
import random
import time
import datetime
import urllib.error

import numpy as np
import pandas as pd
import sqlalchemy as sa
import multiprocessing as mp

# Baostock 替补数据源（仅 A 股 SHZ/SHH 日线，FMP 挂了才用）
try:
    import baostock as _bs
    _HAS_BAOSTOCK = True
except ImportError:
    _bs = None
    _HAS_BAOSTOCK = False

from tqdm import tqdm
from urllib.request import urlopen
from sqlalchemy import create_engine, MetaData, Table, Column, String, Text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.mysql import insert as mysql_insert

from get_stock_data import get_exchange_data, get_exchange_daily_data, get_indicator_data

pd.set_option('future.no_silent_downcasting', True)

# =========================
# 配置
# =========================
# 凭据（DSN / FMP key）都搬到 config_local.py 了 —— **不入库**（见 .gitignore）。
# 2026-09-28 抽出来的：原来 FMP key 与 MySQL DSN 是明文硬编码，推公开仓库会泄漏。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from config_local import MYSQL_DSN, FMP_TOKEN
except ImportError:
    raise SystemExit('缺少 config_local.py —— 本机凭据不入库。请照 config_local.example.py '
                     '在仓库根写一份（cp config_local.example.py config_local.py 后填值）。')

TOKEN = FMP_TOKEN          # 沿用旧名：下面 fmp() 里拼 URL 用的是 TOKEN
engine = create_engine(
    MYSQL_DSN,
    pool_size=10, max_overflow=20,
    echo=False  # 关闭 SQL 语句输出
)

# =========================
# 工具函数
# =========================
# 全局限速：FMP 额度约 3000 次/分钟（50/s），这里压到 40/s 留 20% 余量，
# 从源头避免触发 429。限速器通过 Pool initializer 注入各 worker 进程，
# 保证的是多个进程的总速率（而非单进程速率）不超过额度。
_RATE_LIMIT_PER_MINUTE = 3000
_RATE_LIMIT_FACTOR = 0.8
_RATE_LIMIT_INTERVAL = 60.0 / (_RATE_LIMIT_PER_MINUTE * _RATE_LIMIT_FACTOR)

_g_rate_lock = None
_g_last_req = None
_g_min_interval = None


def _init_rate_limit(lock, last_req, min_interval):
    global _g_rate_lock, _g_last_req, _g_min_interval
    _g_rate_lock = lock
    _g_last_req = last_req
    _g_min_interval = min_interval


def _wait_for_slot():
    """每次 FMP 请求前调用，跨进程保证总请求速率不超过额度。"""
    if _g_rate_lock is None:
        return
    with _g_rate_lock:
        now = time.time()
        wait = _g_min_interval - (now - _g_last_req.value)
        if wait > 0:
            time.sleep(wait)
        _g_last_req.value = time.time()


def fetch_json(url: str, retries: int = 31, base_delay: float = 2.0, max_delay: float = 60.0):
    """带 429/网络错误重试的 JSON 拉取。遇到限流自动指数退避等待后重试。"""
    _wait_for_slot()
    context = ssl.create_default_context(cafile=certifi.where())
    last_exc = None
    for attempt in range(retries):
        try:
            response = urlopen(url, context=context)
            return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last_exc = e
            # 429 限流：尊重 Retry-After，否则指数退避
            if e.code == 429:
                wait = min(base_delay * (2 ** attempt), max_delay)
                if e.headers:
                    retry_after = e.headers.get('Retry-After')
                    if retry_after:
                        try:
                            wait = max(wait, float(retry_after))
                        except (TypeError, ValueError):
                            pass
                wait += random.uniform(0, 0.5)
                print(f'  [429] rate limited, retry {attempt + 1}/{retries} in {wait:.1f}s ...', flush=True)
                time.sleep(wait)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_exc = e
            wait = min(base_delay * (2 ** attempt), max_delay) + random.uniform(0, 0.5)
            print(f'  [retry] {type(e).__name__}: {e}, retry {attempt + 1}/{retries} in {wait:.1f}s ...', flush=True)
            time.sleep(wait)
            continue
    raise last_exc


def fmp(path: str) -> str:
    """构造 FMP API URL。"""
    sep = '&' if '?' in path else '?'
    return f'https://financialmodelingprep.com/stable/{path}{sep}apikey={TOKEN}'


def get_table_names() -> list:
    return sa.inspect(engine).get_table_names()


def ensure_table(name: str, columns: list) -> Table:
    """如果表不存在则创建，并返回 Table 对象。"""
    metadata = MetaData()
    if name not in get_table_names():
        table = Table(name, metadata, *columns)
        metadata.create_all(engine)
        print(f'[create table] {name}')
    else:
        table = Table(name, metadata, autoload_with=engine)
    return table


def bulk_upsert(df: pd.DataFrame, table_name: str, batch_size: int = 1000) -> int:
    if df.empty:
        return 0

    metadata = MetaData()
    table = Table(table_name, metadata, autoload_with=engine)
    pk_cols = {col.name for col in table.primary_key}
    
    total = 0
    records = df.to_dict(orient='records')
    
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        with engine.begin() as conn:
            stmt = mysql_insert(table).values(batch)
            update_dict = {
                col.name: stmt.inserted[col.name]
                for col in table.columns
                if col.name not in pk_cols
            }
            stmt = stmt.on_duplicate_key_update(**update_dict)
            result = conn.execute(stmt)
            total += result.rowcount

    return total

def check_table_count(table_name: str) -> int:
    """Return row count for a table."""
    if table_name not in get_table_names():
        print(f"[{table_name}] table does not exist")
        return 0
    with engine.connect() as conn:
        count = conn.execute(sa.text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
    print(f"[{table_name}] current rows: {count}")
    return count

def split_chunks(lst: list, n: int) -> list:
    """Split a list into roughly even chunks."""
    size = math.ceil(len(lst) / n) if n else len(lst)
    return [lst[i:i + size] for i in range(0, len(lst), size) if lst[i:i + size]]


def get_symbol_list(exchange: str) -> list:
    with sessionmaker(bind=engine)() as session:
        df = pd.read_sql(
            f"SELECT symbol FROM stock_symbol_{exchange.lower()}",
            session.connection()
        )
    return df['symbol'].tolist()


def run_multiprocess(func, task_args: list):
    n = min(mp.cpu_count(), len(task_args))
    with mp.Manager() as mgr:
        lock = mgr.Lock()
        last_req = mgr.Value('d', 0.0)
        with mp.Pool(
            processes=n,
            initializer=_init_rate_limit,
            initargs=(lock, last_req, _RATE_LIMIT_INTERVAL),
        ) as pool:
            pool.starmap(func, task_args)


def get_quarter_end_date(date_str: str) -> str:
    date_str = str(date_str)
    tail = date_str[-4:]
    if tail in ('1231', '0331', '0630', '0930'):
        return date_str
    date = datetime.datetime.strptime(date_str, '%Y%m%d').date()
    q_month = ((date.month - 1) // 3) * 3 + 1
    q_end = datetime.date(date.year, q_month, 1) - datetime.timedelta(days=1)
    return q_end.strftime('%Y%m%d')


# =========================
# Stock Symbol
# =========================
def _symbol_table_columns(name):
    return [
        Column('symbol',      String(50), primary_key=True),
        Column('companyName', String(200)),
        Column('currency',    String(50)),
        Column('exchange',    String(50)),
        Column('website',     String(200)),
        Column('city',        String(50)),
        Column('ipoDate',     String(50)),
    ]


def _fetch_symbol_profiles(exchange: str, symbol_list: list) -> pd.DataFrame:
    rows = []
    for sym in tqdm(symbol_list, desc=f'[profile] {exchange}'):
        try:
            data = fetch_json(fmp(f'profile?symbol={sym}'))
            if data:
                rows.append(data[0])
        except Exception:
            print(f'  profile fetch failed: {sym}')
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)[['symbol', 'companyName', 'currency', 'exchange', 'website', 'city', 'ipoDate']]
    df['companyName'] = df['companyName'].fillna('')
    df['city']        = df['city'].fillna('')
    df['website']     = df['website'].fillna('https://').str[:200]
    df['ipoDate']     = df['ipoDate'].fillna('1900-01-01').str.replace('-', '', regex=False)
    return df.drop_duplicates('symbol')


def _write_symbol_chunk(exchange: str, symbol_list: list):
    table_name = f'stock_symbol_{exchange.lower()}'
    ensure_table(table_name, _symbol_table_columns(table_name))
    df = _fetch_symbol_profiles(exchange, symbol_list)
    if not df.empty:
        n = bulk_upsert(df, table_name)
        print(f'[{table_name}] upsert {n} rows')


def write_exchange_stock_symbol_data(exchange: str):
    table_name = f'stock_symbol_{exchange.lower()}'
    ensure_table(table_name, _symbol_table_columns(table_name))

    # 已有 symbol
    existing = set(get_symbol_list(exchange)) if table_name in get_table_names() else set()

    # FMP 最新 symbol 列表：有财务数据且属于该交易所
    fs_syms  = set(pd.DataFrame(fetch_json(fmp('financial-statement-symbol-list')))['symbol'])
    ex_syms  = set(pd.DataFrame(fetch_json(fmp(f'company-screener?limit=131071&isEtf=false&isFund=false&exchange={exchange}')))['symbol'])
    new_syms = list((fs_syms & ex_syms) - existing)

    if not new_syms:
        print(f'[{exchange}] no new symbols')
        return

    print(f'[{exchange}] new symbols: {len(new_syms)}, start writing...')
    random.shuffle(new_syms)
    chunks = split_chunks(new_syms, mp.cpu_count())
    run_multiprocess(_write_symbol_chunk, [(exchange, c) for c in chunks])


# =========================
# 通用季度财务数据写入
# =========================
def _build_income_df(raw: pd.DataFrame) -> pd.DataFrame:
    cols = [
        'symbol', 'revenue', 'costOfRevenue', 'grossProfit',
        'researchAndDevelopmentExpenses', 'generalAndAdministrativeExpenses',
        'sellingAndMarketingExpenses', 'sellingGeneralAndAdministrativeExpenses',
        'otherExpenses', 'operatingExpenses', 'costAndExpenses',
        'netInterestIncome', 'interestIncome', 'interestExpense',
        'depreciationAndAmortization', 'ebitda', 'ebit',
        'nonOperatingIncomeExcludingInterest', 'operatingIncome',
        'totalOtherIncomeExpensesNet', 'incomeBeforeTax', 'incomeTaxExpense',
        'netIncomeFromContinuingOperations', 'netIncomeFromDiscontinuedOperations',
        'otherAdjustmentsToNetIncome', 'netIncome', 'netIncomeDeductions',
        'bottomLineNetIncome', 'eps', 'epsDiluted',
        'weightedAverageShsOut', 'weightedAverageShsOutDil',
    ]
    df = raw[cols].copy()
    df['endDate'] = raw['date'].str.replace('-', '', regex=False).apply(get_quarter_end_date)
    # FMP 返回 epsDiluted，数据库字段统一使用 epsdiluted。
    df = df.rename(columns={'epsDiluted': 'epsdiluted'})
    return df


def _build_balance_df(raw: pd.DataFrame) -> pd.DataFrame:
    cols = [
        'symbol', 'cashAndCashEquivalents', 'shortTermInvestments',
        'cashAndShortTermInvestments', 'netReceivables', 'accountsReceivables',
        'otherReceivables', 'inventory', 'prepaids', 'otherCurrentAssets',
        'totalCurrentAssets', 'propertyPlantEquipmentNet', 'goodwill',
        'intangibleAssets', 'goodwillAndIntangibleAssets', 'longTermInvestments',
        'taxAssets', 'otherNonCurrentAssets', 'totalNonCurrentAssets',
        'otherAssets', 'totalAssets', 'totalPayables', 'accountPayables',
        'otherPayables', 'accruedExpenses', 'shortTermDebt',
        'capitalLeaseObligationsCurrent', 'taxPayables', 'deferredRevenue',
        'otherCurrentLiabilities', 'totalCurrentLiabilities', 'longTermDebt',
        'capitalLeaseObligationsNonCurrent', 'deferredRevenueNonCurrent',
        'deferredTaxLiabilitiesNonCurrent', 'otherNonCurrentLiabilities',
        'totalNonCurrentLiabilities', 'otherLiabilities',
        'capitalLeaseObligations', 'totalLiabilities', 'treasuryStock',
        'preferredStock', 'commonStock', 'retainedEarnings',
        'additionalPaidInCapital', 'accumulatedOtherComprehensiveIncomeLoss',
        'otherTotalStockholdersEquity', 'totalStockholdersEquity', 'totalEquity',
        'minorityInterest', 'totalLiabilitiesAndTotalEquity',
        'totalInvestments', 'totalDebt', 'netDebt',
    ]
    df = raw[cols].copy()
    df['endDate'] = raw['date'].str.replace('-', '', regex=False).apply(get_quarter_end_date)
    return df


def _build_cashflow_df(raw: pd.DataFrame) -> pd.DataFrame:
    cols = [
        'symbol', 'deferredIncomeTax', 'stockBasedCompensation',
        'changeInWorkingCapital', 'accountsPayables', 'otherWorkingCapital',
        'otherNonCashItems', 'netCashProvidedByOperatingActivities',
        'investmentsInPropertyPlantAndEquipment', 'acquisitionsNet',
        'purchasesOfInvestments', 'salesMaturitiesOfInvestments',
        'otherInvestingActivities', 'netCashProvidedByInvestingActivities',
        'netDebtIssuance', 'longTermNetDebtIssuance', 'shortTermNetDebtIssuance',
        'netStockIssuance', 'netCommonStockIssuance', 'commonStockIssuance',
        'commonStockRepurchased', 'netPreferredStockIssuance', 'netDividendsPaid',
        'commonDividendsPaid', 'preferredDividendsPaid', 'otherFinancingActivities',
        'netCashProvidedByFinancingActivities', 'effectOfForexChangesOnCash',
        'netChangeInCash', 'cashAtEndOfPeriod', 'cashAtBeginningOfPeriod',
        'operatingCashFlow', 'capitalExpenditure', 'freeCashFlow',
        'incomeTaxesPaid', 'interestPaid',
    ]
    df = raw[cols].copy()
    df['endDate'] = raw['date'].str.replace('-', '', regex=False).apply(get_quarter_end_date)
    return df


# API 路径与 DataFrame 构造函数映射
_FINANCIAL_CONFIG = {
    'income':   ('income-statement',       _build_income_df),
    'balance':  ('balance-sheet-statement', _build_balance_df),
    'cashflow': ('cash-flow-statement',     _build_cashflow_df),
}

# 各财务表建表字段
_INCOME_COLS = lambda name: [
    Column('symbol', String(50), primary_key=True),
    Column('endDate', String(50), primary_key=True),
    *[Column(c, String(32)) for c in [
        'revenue','costOfRevenue','grossProfit','researchAndDevelopmentExpenses',
        'generalAndAdministrativeExpenses','sellingAndMarketingExpenses',
        'sellingGeneralAndAdministrativeExpenses','otherExpenses','operatingExpenses',
        'costAndExpenses','netInterestIncome','interestIncome','interestExpense',
        'depreciationAndAmortization','ebitda','ebit',
        'nonOperatingIncomeExcludingInterest','operatingIncome',
        'totalOtherIncomeExpensesNet','incomeBeforeTax','incomeTaxExpense',
        'netIncomeFromContinuingOperations','netIncomeFromDiscontinuedOperations',
        'otherAdjustmentsToNetIncome','netIncome','netIncomeDeductions',
        'bottomLineNetIncome','eps',
        'epsdiluted',
        'weightedAverageShsOut','weightedAverageShsOutDil',
    ]]
]

_BALANCE_COLS = lambda name: [
    Column('symbol', String(50), primary_key=True),
    Column('endDate', String(50), primary_key=True),
    *[Column(c, String(32)) for c in [
        'cashAndCashEquivalents','shortTermInvestments','cashAndShortTermInvestments',
        'netReceivables','accountsReceivables','otherReceivables','inventory','prepaids',
        'otherCurrentAssets','totalCurrentAssets','propertyPlantEquipmentNet','goodwill',
        'intangibleAssets','goodwillAndIntangibleAssets','longTermInvestments','taxAssets',
        'otherNonCurrentAssets','totalNonCurrentAssets','otherAssets','totalAssets',
        'totalPayables','accountPayables','otherPayables','accruedExpenses','shortTermDebt',
        'capitalLeaseObligationsCurrent','taxPayables','deferredRevenue',
        'otherCurrentLiabilities','totalCurrentLiabilities','longTermDebt',
        'capitalLeaseObligationsNonCurrent','deferredRevenueNonCurrent',
        'deferredTaxLiabilitiesNonCurrent','otherNonCurrentLiabilities',
        'totalNonCurrentLiabilities','otherLiabilities','capitalLeaseObligations',
        'totalLiabilities','treasuryStock','preferredStock','commonStock',
        'retainedEarnings','additionalPaidInCapital',
        'accumulatedOtherComprehensiveIncomeLoss','otherTotalStockholdersEquity',
        'totalStockholdersEquity','totalEquity','minorityInterest',
        'totalLiabilitiesAndTotalEquity','totalInvestments','totalDebt','netDebt',
    ]]
]

_CASHFLOW_COLS = lambda name: [
    Column('symbol', String(50), primary_key=True),
    Column('endDate', String(50), primary_key=True),
    *[Column(c, String(32)) for c in [
        'deferredIncomeTax','stockBasedCompensation','changeInWorkingCapital',
        'accountsPayables','otherWorkingCapital','otherNonCashItems',
        'netCashProvidedByOperatingActivities','investmentsInPropertyPlantAndEquipment',
        'acquisitionsNet','purchasesOfInvestments','salesMaturitiesOfInvestments',
        'otherInvestingActivities','netCashProvidedByInvestingActivities',
        'netDebtIssuance','longTermNetDebtIssuance','shortTermNetDebtIssuance',
        'netStockIssuance','netCommonStockIssuance','commonStockIssuance',
        'commonStockRepurchased','netPreferredStockIssuance','netDividendsPaid',
        'commonDividendsPaid','preferredDividendsPaid','otherFinancingActivities',
        'netCashProvidedByFinancingActivities','effectOfForexChangesOnCash',
        'netChangeInCash','cashAtEndOfPeriod','cashAtBeginningOfPeriod',
        'operatingCashFlow','capitalExpenditure','freeCashFlow',
        'incomeTaxesPaid','interestPaid',
    ]]
]

_TABLE_COLS = {
    'income':   _INCOME_COLS,
    'balance':  _BALANCE_COLS,
    'cashflow': _CASHFLOW_COLS,
}


def _write_financial_chunk(kind: str, table_name: str, symbol_list: list, quarter_num: int):
    """按 symbol 拉取季度财务数据并实时 upsert，单只失败不影响其它 symbol。"""
    api_path, build_fn = _FINANCIAL_CONFIG[kind]
    ensure_table(table_name, _TABLE_COLS[kind](table_name))

    total_upserted = 0
    for sym in tqdm(symbol_list, desc=f'[{kind}] {table_name}'):
        try:
            raw = fetch_json(fmp(f'{api_path}?limit={quarter_num}&period=quarter&symbol={sym}'))
        except Exception:
            print(f'  {kind} fetch failed: {sym}')
            continue
        if not raw:
            continue
        try:
            df = build_fn(pd.DataFrame(raw))
            df = df.dropna(subset=['symbol', 'endDate']).fillna(0)
            df = df.drop_duplicates(['symbol', 'endDate'])
            if df.empty:
                continue
            n = bulk_upsert(df, table_name)
            total_upserted += n
            if n > 0:
                print(f'  [{sym}] upsert {n} rows, total {total_upserted}')
        except Exception as e:
            print(f'  {kind} write failed {sym}: {e}')

    print(f'[{table_name}] chunk done, upsert {total_upserted} rows')


def _write_exchange_financial(kind: str, exchange: str, quarter_num: int):
    exchange = exchange.lower()
    table_name = f'{kind}_{exchange}'
    ensure_table(table_name, _TABLE_COLS[kind](table_name))

    symbol_list = get_symbol_list(exchange)
    random.shuffle(symbol_list)
    chunks = split_chunks(symbol_list, mp.cpu_count())
    args = [(kind, table_name, c, quarter_num) for c in chunks]
    print(f'[{table_name}] {len(symbol_list)} symbols, {len(args)} processes')
    run_multiprocess(_write_financial_chunk, args)
    print(f'[{table_name}] done')


def write_exchange_income_data(exchange: str, quarter_num: int = 31):
    _write_exchange_financial('income', exchange, quarter_num)

def write_exchange_balance_data(exchange: str, quarter_num: int = 31):
    _write_exchange_financial('balance', exchange, quarter_num)

def write_exchange_cashflow_data(exchange: str, quarter_num: int = 31):
    _write_exchange_financial('cashflow', exchange, quarter_num)


# =========================
# Daily 数据
# =========================
_DAILY_COLS = lambda name: [
    Column('symbol', String(50), primary_key=True),
    Column('date',   String(50), primary_key=True),
    Column('open',   String(50)),
    Column('low',    String(50)),
    Column('high',   String(50)),
    Column('close',  String(50)),
    Column('volume', String(50)),
]


# =========================
# Baostock 替补（仅 A 股日线，FMP 挂了才用）
# =========================
_BS_LOGGED_IN = False
_BS_LOGIN_FAILED = False


def _fmp_to_baostock(sym: str):
    """FMP A 股代码 -> Baostock 代码；非 A 股返回 None。"""
    if sym.endswith('.SZ'):
        return 'sz.' + sym[:-3]
    if sym.endswith('.SS'):
        return 'sh.' + sym[:-3]
    return None


def _baostock_login() -> bool:
    """登录 Baostock（每进程一次，惰性）。失败后记住，不再为每只票重试登录。"""
    global _BS_LOGGED_IN, _BS_LOGIN_FAILED
    if not _HAS_BAOSTOCK:
        return False
    if _BS_LOGGED_IN:
        return True
    if _BS_LOGIN_FAILED:
        return False
    try:
        lg = _bs.login()
        if getattr(lg, 'error_code', None) == '0':
            _BS_LOGGED_IN = True
            print('[baostock] login ok', flush=True)
            return True
    except Exception as e:
        print(f'[baostock] login failed: {e}', flush=True)
    _BS_LOGIN_FAILED = True
    print('[baostock] 登录失败，本轮跳过 Baostock 替补（不再重试登录）', flush=True)
    return False


def _fetch_daily_baostock(sym: str, from_dt: str, to_dt: str, retries: int = 3):
    """用 Baostock 拉 A 股日线，返回与 FMP 同构的 DataFrame。

    date 保持 'YYYY-MM-DD'（外层统一去横杠）；open/low/high/close/volume 转数值，
    volume 空值补 0；空结果返回 None。服务器连接失败/限流时退避重试。
    """
    code = _fmp_to_baostock(sym)
    if code is None or not _baostock_login():
        return None
    for attempt in range(retries):
        try:
            rs = _bs.query_history_k_data_plus(
                code, 'date,open,high,low,close,volume',
                start_date=from_dt, end_date=to_dt,
                frequency='d', adjustflag='3',  # 3=不复权，与 FMP historical-price-eod 一致
            )
            if getattr(rs, 'error_code', '1') != '0':
                # Baostock 服务器失败/限流，退避后重试
                if attempt < retries - 1:
                    time.sleep(1.0 + attempt)
                    continue
                return None
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if not rows:
                return None  # 无数据（停牌等）
            df = pd.DataFrame(rows, columns=rs.fields)
            df['symbol'] = sym
            for c in ['open', 'low', 'high', 'close', 'volume']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            df = df.dropna(subset=['open', 'low', 'high', 'close'])
            df['volume'] = df['volume'].fillna(0)
            return df[['symbol', 'date', 'open', 'low', 'high', 'close', 'volume']]
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1.0 + attempt)
                continue
            print(f'[baostock] fetch failed {sym}: {e}', flush=True)
            return None
    return None


def _check_fmp_health() -> bool:
    """快速探测 FMP 是否可用（单次请求 + 8s 超时，不触发长重试）。

    429（限流/带宽打满）或连接类错误都视为「不可用」返回 False；其它 HTTP 错误
    （404/500 等）说明 FMP 可达，返回 True。
    """
    try:
        url = fmp('historical-price-eod/full?symbol=AAPL&from=2026-01-02&to=2026-01-03')
        context = ssl.create_default_context(cafile=certifi.where())
        response = urlopen(url, context=context, timeout=8)
        response.read()
        return True
    except urllib.error.HTTPError as e:
        return e.code != 429  # 429 限流/带宽打满 = 拿不到数据，视为不可用
    except Exception:
        return False


def _write_daily_chunk(table_name: str, symbol_list: list, days: int, fmp_down: bool = False):
    ensure_table(table_name, _DAILY_COLS(table_name))
    today    = datetime.datetime.today()
    from_dt  = (today - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
    to_dt    = today.strftime('%Y-%m-%d')

    all_dfs = []
    for sym in tqdm(symbol_list, desc=f'[daily] {table_name}'):
        df = None
        if not fmp_down:
            try:
                raw = fetch_json(fmp(
                    f'historical-price-eod/full?symbol={sym}&from={from_dt}&to={to_dt}'
                ))
                if raw:
                    df = pd.DataFrame(raw)[['symbol', 'date', 'open', 'low', 'high', 'close', 'volume']]
            except Exception:
                print(f'  daily fetch failed: {sym}')
        # baostock 仅作为 FMP 完全不可用（fmp_down）时的应急方案，只拉日线。
        # FMP 正常时，个别 FMP 缺口的票（如 000733.SZ）直接跳过，不逐票走 baostock。
        if fmp_down and (df is None or df.empty):
            df = _fetch_daily_baostock(sym, from_dt, to_dt)
        if df is None or df.empty:
            continue
        df['date'] = df['date'].str.replace('-', '', regex=False)
        df.drop_duplicates(['symbol', 'date'], inplace=True)
        all_dfs.append(df)

    if not all_dfs:
        return
    merged = pd.concat(all_dfs, ignore_index=True)
    # FMP 偶尔会返回残缺 K 线（部分票的 open/low/high 或 volume 为 null，
    # 如 SAU 的 8120.SR），pandas 转成 NaN 后 pymysql 会直接抛
    # "nan can not be used with MySQL" 搞死整批。这里统一清洗，
    # 与 baostock 分支保持一致：OHLC 缺任一就丢弃该行，volume 缺则补 0。
    for c in ['open', 'low', 'high', 'close', 'volume']:
        merged[c] = pd.to_numeric(merged[c], errors='coerce')
    merged = merged.dropna(subset=['open', 'low', 'high', 'close'])
    merged['volume'] = merged['volume'].fillna(0)
    n = bulk_upsert(merged, table_name)
    print(f'[{table_name}] upsert {n} rows')


def write_exchange_daily_data(exchange: str, days: int = 127):
    exchange   = exchange.lower()
    table_name = f'daily_{exchange}'
    ensure_table(table_name, _DAILY_COLS(table_name))

    symbol_list = get_symbol_list(exchange)
    random.shuffle(symbol_list)

    # 先快速探测 FMP：挂了就跳过 FMP，A 股直接走 Baostock，避免每只票都触发 31 次重试
    fmp_down = not _check_fmp_health()
    if fmp_down:
        print(f'[FMP] 探测失败，本轮 {table_name} 跳过 FMP，直接走 Baostock 替补')

    # Baostock 免费源扛不住高并发，降到少量进程；FMP 正常时用满核
    num_procs = 1 if fmp_down else mp.cpu_count()
    chunks = split_chunks(symbol_list, num_procs)

    args = [(table_name, c, days, fmp_down) for c in chunks]
    print(f'[{table_name}] {len(symbol_list)} symbols, {len(args)} processes')
    run_multiprocess(_write_daily_chunk, args)
    print(f'[{table_name}] done')




# =========================
# Indicator 数据
# =========================
_FFILL_COLS = [
    'weightedAverageShsOut','totalCurrentAssets','totalCurrentLiabilities',
    'longTermDebt','totalStockholdersEquity','freeCashFlow','commonDividendsPaid',
    'totalAssets','operatingIncome','totalLiabilities','interestExpense',
    'cashAndCashEquivalents','shortTermInvestments','netReceivables','costOfRevenue',
    'inventory','revenue','grossProfit','operatingExpenses','costAndExpenses',
    'ebitda','incomeBeforeTax','accountsReceivables','propertyPlantEquipmentNet',
    'longTermInvestments','totalPayables','accountPayables','shortTermDebt',
    'retainedEarnings','totalEquity','totalLiabilitiesAndTotalEquity',
    'totalInvestments','totalDebt','netDebt',
    'netCashProvidedByOperatingActivities','netCashProvidedByFinancingActivities',
    'cashAtEndOfPeriod','cashAtBeginningOfPeriod',
]

_INDICATOR_COLS = lambda name: [
    Column('symbol',  String(50), primary_key=True),
    Column('endDate', String(50), primary_key=True),
    *[Column(c, String(32)) for c in [
        'operatingCapitalPerShare','netAssetValuePerShare','dcfPerShare',
        'dividendPerShare','debtRatio','debtToEquity','interestCoverage',
        'cashRatio','quickRatio','currentRatio','inventoryTurnover',
        'ocfToLiab','goodwillToEquity','accrual',
        'revenueUnit','costOfRevenueUnit','grossProfitUnit','operatingExpensesUnit',
        'costAndExpensesUnit','ebitdaUnit','operatingIncomeUnit','incomeBeforeTaxUnit',
        'netReceivablesUnit','accountsReceivablesUnit','inventoryUnit',
        'totalCurrentAssetsUnit','propertyPlantEquipmentNetUnit',
        'totalAssetsUnit','totalPayablesUnit',
        'accountPayablesUnit','shortTermDebtUnit','totalCurrentLiabilitiesUnit',
        'longTermDebtUnit','retainedEarningsUnit',
        'totalEquityUnit','totalLiabilitiesAndTotalEquityUnit',
        'totalInvestmentsUnit','netDebtUnit',
        'netCashProvidedByOperatingActivitiesUnit',
        'netCashProvidedByFinancingActivitiesUnit',
        'cashAtEndOfPeriodUnit','cashAtBeginningOfPeriodUnit',
    ]]
]


def write_exchange_indicator_data(exchange: str):
    exchange   = exchange.lower()
    table_name = f'indicator_{exchange}'

    # indicator 是全量重算，先删除旧表。
    if table_name in get_table_names():
        meta = MetaData()
        Table(table_name, meta, autoload_with=engine).drop(engine)
        print(f'[drop table] {table_name}')

    ensure_table(table_name, _INDICATOR_COLS(table_name))

    # 读取三张财务表并合并。
    with sessionmaker(bind=engine)() as session:
        conn = session.connection()
        income   = pd.read_sql(f'SELECT * FROM income_{exchange}',   conn)
        balance  = pd.read_sql(f'SELECT * FROM balance_{exchange}',  conn)
        cashflow = pd.read_sql(f'SELECT * FROM cashflow_{exchange}', conn)

    m = (income
         .merge(balance,  on=['symbol', 'endDate'], how='outer')
         .merge(cashflow, on=['symbol', 'endDate'], how='outer')
         .dropna(subset=['symbol', 'endDate'])
         .fillna(0)
         .reset_index(drop=True))

    for col in _FFILL_COLS:
        # 只向前填充，去掉 bfill：避免用未来值回填序列开头的空缺（前视）
        m[col] = m[col].replace(0.0, pd.NA).ffill().fillna(0)

    def f(col): return m[col].astype(float)
    eq = f('totalStockholdersEquity')

    ind = pd.DataFrame({
        'symbol':  m['symbol'],
        'endDate': m['endDate'],
        'operatingCapitalPerShare': (f('totalCurrentAssets') - f('totalCurrentLiabilities') - f('longTermDebt')) / f('weightedAverageShsOut'),
        'netAssetValuePerShare':    f('totalStockholdersEquity') / f('weightedAverageShsOut'),
        'dcfPerShare':              f('freeCashFlow') / f('weightedAverageShsOut'),
        'dividendPerShare':         f('commonDividendsPaid').abs() / f('weightedAverageShsOut'),
        'debtRatio':                f('totalLiabilities') / f('totalAssets'),
        'debtToEquity':             f('totalLiabilities') / f('totalStockholdersEquity'),
        'interestCoverage':         f('operatingIncome') / f('interestExpense'),
        'cashRatio':                (f('cashAndCashEquivalents') + f('shortTermInvestments')) / f('totalCurrentLiabilities'),
        'quickRatio':               (f('cashAndCashEquivalents') + f('shortTermInvestments') + f('netReceivables')) / f('totalCurrentLiabilities'),
        'currentRatio':             f('totalCurrentAssets') / f('totalCurrentLiabilities'),
        'inventoryTurnover':        f('costOfRevenue') / f('inventory'),
        'ocfToLiab':                f('netCashProvidedByOperatingActivities') / f('totalLiabilities'),
        'goodwillToEquity':         f('goodwill') / f('totalStockholdersEquity'),
        'accrual':                  (f('netIncome') - f('netCashProvidedByOperatingActivities')) / f('totalAssets'),
        **{
            f'{raw_col}Unit': f(raw_col) / eq
            for raw_col in [
                'revenue','costOfRevenue','grossProfit','operatingExpenses',
                'costAndExpenses','ebitda','operatingIncome','incomeBeforeTax',
                'netReceivables','accountsReceivables','inventory',
                'totalCurrentAssets','propertyPlantEquipmentNet',
                'totalAssets','totalPayables','accountPayables','shortTermDebt',
                'totalCurrentLiabilities','longTermDebt',
                'retainedEarnings','totalEquity','totalLiabilitiesAndTotalEquity',
                'totalInvestments','netDebt',
                'netCashProvidedByOperatingActivities',
                'netCashProvidedByFinancingActivities',
                'cashAtEndOfPeriod','cashAtBeginningOfPeriod',
            ]
        }
    })

    ind = (ind
           .dropna(subset=['symbol', 'endDate'])
           .replace([np.inf, -np.inf], 0)
           .fillna(0)
           .drop_duplicates(['symbol', 'endDate'])
           .reset_index(drop=True))

    n = bulk_upsert(ind, table_name)
    print(f'[{table_name}] upsert {n} rows, done')


# =========================
# 高层操作
# =========================
def repair_exchange_data(exchange: str):
    write_exchange_stock_symbol_data(exchange)
    write_exchange_income_data(exchange, 127)
    write_exchange_balance_data(exchange, 127)
    write_exchange_cashflow_data(exchange, 127)
    write_exchange_indicator_data(exchange)
    write_exchange_daily_data(exchange, 8191)
    get_exchange_data(exchange)


def update_exchange_data(exchange: str):
    write_exchange_stock_symbol_data(exchange)
    write_exchange_income_data(exchange, 7)
    write_exchange_balance_data(exchange, 7)
    write_exchange_cashflow_data(exchange, 7)
    write_exchange_indicator_data(exchange)
    write_exchange_daily_data(exchange, 127)
    get_exchange_data(exchange)


def append_exchange_data(exchange: str):
    write_exchange_daily_data(exchange, 31)
    get_exchange_daily_data(exchange)




def repair_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # 先自动发现股票数最多的 31 个交易所（有财务数据且 127 < 数量 < 8191），
    # 发现失败时回退到 data/ 目录下已有的交易所。
    try:
        exchange_list = get_train_exchanges()
        print('[discover] top31 exchanges:', exchange_list)
    except Exception as e:
        print(f'[discover] 失败，回退到 data/ 目录: {e}')
        data_dir = os.path.join(BASE_DIR, 'data')
        exchange_list = sorted(d for d in os.listdir(data_dir)
                               if os.path.isdir(os.path.join(data_dir, d)))
        print(exchange_list)
    failed = []
    for exchange in exchange_list:
        print(f'\n=== {exchange} ===')
        try:
            repair_exchange_data(exchange)
        except Exception as e:
            print(f'  failed: {e}')
            failed.append(exchange)
    time.sleep(7)
    print('failed list:', failed)


def refresh_indicator():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(BASE_DIR, 'data')
    exchanges = sorted(d for d in os.listdir(data_dir)
                       if os.path.isdir(os.path.join(data_dir, d)))
    failed = []
    for exchange in exchanges:
        try:
            write_exchange_indicator_data(exchange)
        except Exception as e:
            print(f'  {exchange} failed: {e}')
            failed.append(exchange)
    get_indicator_data()
    print('failed list:', failed)


def get_train_exchanges():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    exchanges_path = os.path.join(BASE_DIR, 'train_exchanges.csv')

    symbol_list = set(pd.DataFrame(fetch_json(fmp('financial-statement-symbol-list')))['symbol'])
    avail_exchanges = fetch_json(fmp('available-exchanges'))

    train_list = []
    for ex in avail_exchanges:
        name = ex['exchange']
        try:
            ex_syms = set(
                pd.DataFrame(fetch_json(fmp(f'company-screener?limit=131071&isEtf=false&isFund=false&exchange={name}')))['symbol']
            )
        except Exception:
            continue
        overlap = symbol_list & ex_syms
        if 127 < len(overlap) < 8191:
            ex['num_symbols'] = len(overlap)
            train_list.append(ex)

    df = (pd.DataFrame(train_list)
          .sort_values('num_symbols', ascending=False)
          .head(31)
          .reset_index(drop=True))
    print(df)
    df.to_csv(exchanges_path, index=False, encoding='utf-8')
    return df['exchange'].tolist()


def delete_today_daily(exchange: str, today=None):
    if today is None:
        today = int(datetime.date.today().strftime('%Y%m%d'))
    table = f'daily_{exchange.lower()}'
    with engine.begin() as conn:
        result = conn.execute(sa.text(f"DELETE FROM {table} WHERE date = :d"), {'d': str(today)})
        print(f'[{table}] deleted {result.rowcount} rows for {today}')

def delete_latest_daily(exchange: str):
    table = f'daily_{exchange.lower()}'
    with engine.begin() as conn:
        latest = conn.execute(sa.text(f"SELECT MAX(date) FROM {table}")).scalar()
        if latest is None:
            print(f'[{table}] empty table, nothing to delete')
            return
        result = conn.execute(sa.text(f"DELETE FROM {table} WHERE date = :d"), {'d': latest})
        print(f'[{table}] deleted {result.rowcount} rows for {latest}')
    get_exchange_daily_data(exchange)


if __name__ == '__main__':
    # repair_news_data()
    repair_data()
    # repair_exchange_data('SAU')
    # delete_latest_daily('SHZ')
    # delete_latest_daily('SHH')



