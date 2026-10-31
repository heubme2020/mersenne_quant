import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
from write_stock_data import update_exchange_data, append_exchange_data
import datetime
import importlib.util
import sys, os
import pandas as pd


base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, base_dir)     # 让 `from config_local import ...` 在任意 cwd 下都能解析

# 本机凭据（邮箱授权码 / 收件人名单）在 config_local.py —— **不入库**（见 .gitignore）
try:
    from config_local import SMTP_SERVER, SMTP_PORT, MAIL_SENDER, MAIL_PASSWORD, MAIL_RECEIVERS
except ImportError:
    raise SystemExit('缺少 config_local.py —— 本机凭据不入库。请照 config_local.example.py '
                     '在仓库根写一份（cp config_local.example.py config_local.py 后填值）。')
sys.path.append(os.path.join(base_dir, "schloss"))
sys.path.append(os.path.join(base_dir, "zero"))
sys.path.append(os.path.join(base_dir, "one"))
sys.path.append(os.path.join(base_dir, "two"))
sys.path.append(os.path.join(base_dir, "three"))
sys.path.append(os.path.join(base_dir, "seven"))

def import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# 市场 -> 产物文件后缀（与各模型脚本里的 MARKET_SUFFIX 一致）
MARKET_SUFFIX = {'CN': '', 'US': '_us'}


def _quarterly_ready(market):
    """该市场的季度基本面产物（seven/three/zero）是否都已生成且非空；没有就自动从头跑。"""
    suffix = MARKET_SUFFIX[market]
    base = os.path.dirname(__file__)
    for rel in (f'seven/seven_predict{suffix}.csv',
                f'three/three_predict{suffix}.csv',
                f'zero/zero_predict{suffix}.csv'):
        p = os.path.join(base, rel)
        if not os.path.exists(p):
            return False
        try:
            if pd.read_csv(p).empty:
                return False
        except Exception:
            return False
    return True


def update_data():
    for ex in ('AMEX', 'NASDAQ', 'NYSE'):
        print(f"更新美股 {ex} 的数据...")
        update_exchange_data(ex)
        print(f"美股 {ex} 的数据更新完成")
    print("更新深交所的数据...")
    update_exchange_data('SHZ')
    print("深交所的数据更新完成")
    print("更新上交所的数据...")
    update_exchange_data('SHH')
    print("上交所的数据更新完成")

def append_data():
    for ex in ('AMEX', 'NASDAQ', 'NYSE'):
        print(f"更新美股 {ex} 的数据...")
        append_exchange_data(ex)
        print(f"美股 {ex} 的数据更新完成")
    print("更新深交所的数据...")
    append_exchange_data('SHZ')
    print("深交所的数据更新完成")
    print("更新上交所的数据...")
    append_exchange_data('SHH')
    print("上交所的数据更新完成")

def refresh_schloss(market='CN', target_date=None):
    # 当前路径
    base_dir = os.path.dirname(__file__)
    mod = import_from_path("refresh_schloss", os.path.join(base_dir, "schloss", "get_schloss.py"))
    print(f"更新schloss股票池（{market}）...")
    mod.refresh_schloss(market, target_date=target_date)
    print(f"更新schloss股票池（{market}）结束")

def refresh_zero_predict(market='CN'):
    # 当前路径
    base_dir = os.path.dirname(__file__)
    mod = import_from_path("refresh_zero", os.path.join(base_dir, "zero", "get_zero_predict.py"))
    print(f"更新zero股票池（{market}）...")
    mod.refresh_zero(market)
    print(f"更新zero股票池（{market}）结束")

def refresh_three_predict(market='CN'):
    # 当前路径
    base_dir = os.path.dirname(__file__)
    mod = import_from_path("refresh_growth_death", os.path.join(base_dir, "three", "get_three_predict.py"))
    print(f"更新three股票池（{market}）...")
    mod.refresh_growth_death(market)
    print(f"更新three股票池（{market}）结束")

def refresh_seven_predict(market='CN'):
    # 当前路径
    base_dir = os.path.dirname(__file__)
    mod = import_from_path("refresh_dcf", os.path.join(base_dir, "seven", "get_seven_predict.py"))
    print(f"更新seven股票池（{market}）...")
    mod.refresh_dcf(market)
    print(f"更新seven股票池（{market}）结束")


def refresh_two_predict(market='CN', target_date=None):
    # 当前路径
    base_dir = os.path.dirname(__file__)
    mod = import_from_path("get_two_candidates", os.path.join(base_dir, "two", "get_two_predict.py"))
    print(f"更新two股票池（{market}）...")
    mod.get_two_candidates(market, target_date=target_date)
    print(f"更新two股票池（{market}）结束")


def refresh_one_predict(market='CN', target_date=None):
    # 当前路径
    base_dir = os.path.dirname(__file__)
    mod = import_from_path("refresh_buy", os.path.join(base_dir, "one", "get_one_predict.py"))
    print(f"更新one股票池（{market}）...")
    mod.refresh_buy(market, target_date=target_date)
    print(f"更新one股票池（{market}）结束")


def refresh(target_date=None):
    if target_date is not None:
        # 指定日期：跳过 API 数据更新，也跳过按季度计算的基本面模型
        # （zero/three/seven 只依赖财务季度，与交易日期无关，直接复用已有结果），
        # 仅按目标日期重算日线相关的股票池（schloss/two/one）。
        for market in ('US', 'CN'):
            refresh_schloss(market, target_date)
            refresh_two_predict(market, target_date)
            refresh_one_predict(market, target_date)
        return
    today = datetime.datetime.now().date()
    weekday = today.weekday()
    if weekday == 5 or weekday == 6:
        # 周末不跑数据也不出候选，静默返回，不发邮件
        return
    if weekday == 4:
        update_data()
    else:
        append_data()
    # # 季度基本面模型：周五全跑；其余日若该市场还没初始化（缺产物）则自动从头跑
    # for market in ('US', 'CN'):
    #     if weekday == 4 or not _quarterly_ready(market):
    #         refresh_seven_predict(market)
    #         refresh_three_predict(market)
    #         refresh_zero_predict(market)
    # 日线相关：每个交易日两市场都跑
    for market in ('US', 'CN'):
        refresh_seven_predict(market)
        refresh_three_predict(market)
        refresh_zero_predict(market)
        refresh_schloss(market)
        refresh_two_predict(market)
        refresh_one_predict(market)


def _send_mail(subject, body):
    # 邮箱配置（发件人/授权码/收件人名单都在 config_local.py —— **不入库**，见 .gitignore。
    # 2026-09-28 抽出来的：原来授权码与 6 个收件人邮箱是明文，推公开仓库会泄漏。）
    smtp_server, port = SMTP_SERVER, SMTP_PORT
    sender, password = MAIL_SENDER, MAIL_PASSWORD
    receiver_list = list(MAIL_RECEIVERS)

    message = MIMEMultipart()
    message["From"] = sender
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))
    message["To"] = ", ".join(receiver_list)
    try:
        with smtplib.SMTP_SSL(smtp_server, port) as server:
            server.login(sender, password)
            server.sendmail(sender, receiver_list, message.as_string())
    except smtplib.SMTPResponseException as e:
        # 忽略掉 (-1, b'\x00\x00\x00') 这种非标准错误
        if e.smtp_code == -1:
            print("邮件已发送成功，但服务器返回了非标准关闭响应。")
        else:
            raise


def _pick_by_buffett(buy_data):
    """按 buffett 分档各取 up_down 最高的一只，返回 [1, 3, 7, 31, 127] 五档的 symbol。

    每档先按 buffett 降序取前 N 只（N 越小越严格），再在档内按 up_down 降序取第一名：
      1 档 = 前 127 / 3 档 = 前 31 / 7 档 = 前 7 / 31 档 = 前 3 / 127 档 = 前 1。
    """
    one = buy_data.sort_values('buffett', ascending=False).iloc[:127]
    one = one.sort_values('up_down', ascending=False)
    one_buy = one['symbol'].iloc[0]

    three = buy_data.sort_values('buffett', ascending=False).iloc[:31]
    three = three.sort_values('up_down', ascending=False)
    three_buy = three['symbol'].iloc[0]

    seven = buy_data.sort_values('buffett', ascending=False).iloc[:7]
    seven = seven.sort_values('up_down', ascending=False)
    seven_buy = seven['symbol'].iloc[0]

    thirty_one = buy_data.sort_values('buffett', ascending=False).iloc[:3]
    thirty_one = thirty_one.sort_values('up_down', ascending=False)
    thirty_one_buy = thirty_one['symbol'].iloc[0]

    one_hundred_and_twenty_seven = buy_data.sort_values('buffett', ascending=False).iloc[:1]
    one_hundred_and_twenty_seven = one_hundred_and_twenty_seven.sort_values('up_down', ascending=False)
    one_hundred_and_twenty_seven_buy = one_hundred_and_twenty_seven['symbol'].iloc[0]

    return [one_buy, three_buy, seven_buy, thirty_one_buy, one_hundred_and_twenty_seven_buy]


def _market_section(label, buy_data):
    if buy_data.empty:
        return f"{label}:\n（无候选）"
    picks = _pick_by_buffett(buy_data)
    return (f"{label}:\n"
            f"1: {picks[0]}\n"
            f"3: {picks[1]}\n"
            f"7: {picks[2]}\n"
            f"31: {picks[3]}\n"
            f"127: {picks[4]}")


def send_candidates(target_date=None):
    if target_date is None:
        today = datetime.datetime.now().date()
        weekday = today.weekday()
        if weekday == 5 or weekday == 6:
            # 周末不跑数据也不出候选，静默返回，不发邮件
            return
        refresh()
    else:
        refresh(target_date)
    base = os.path.dirname(__file__)
    cn_data = pd.read_csv(os.path.join(base, 'buy_predict.csv'))
    us_data = pd.read_csv(os.path.join(base, 'buy_predict_us.csv'))
    if cn_data.empty and us_data.empty:
        raise RuntimeError(
            "buy_predict.csv 与 buy_predict_us.csv 都为空，无法选出候选。通常是 one 股票池没产出"
            "（多为 schloss/two 未随日线数据刷新导致的日期错位，详见 get_one_predict.py 的报错）。"
        )
    # subject 用 A 股日期为主（美股可能滞后一天）
    date = cn_data['date'].iloc[0] if not cn_data.empty else us_data['date'].iloc[0]

    subject = str(date) + ':推的票'
    body = _market_section('America', us_data) + '\n\n' + _market_section('China', cn_data)
    print(body)
    _send_mail(subject, body)


if __name__ == '__main__':
    # refresh_model()
    # refresh_data()
    # get_data()
    target_date = None
    if len(sys.argv) > 1:
        raw = sys.argv[1]
        if not (raw.isdigit() and len(raw) == 8):
            print(f"日期格式错误: {raw}，应为 8 位数字，例如 20260820")
            sys.exit(1)
        target_date = raw
    send_candidates(target_date)
