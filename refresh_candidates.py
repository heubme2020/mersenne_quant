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

def update_data():
    print("更新深交所的数据...")
    update_exchange_data('SHZ')
    print("深交所的数据更新完成")
    print("更新上交所的数据...")
    update_exchange_data('SHH')
    print("上交所的数据更新完成")

def append_data():
    print("更新深交所的数据...")
    append_exchange_data('SHZ')
    print("深交所的数据更新完成")
    print("更新上交所的数据...")
    append_exchange_data('SHH')
    print("上交所的数据更新完成")

def refresh_schloss(target_date=None):
    # 当前路径
    base_dir = os.path.dirname(__file__)
    refresh_schloss = import_from_path("refresh_schloss", os.path.join(base_dir, "schloss", "get_schloss.py"))
    print("更新schloss股票池...")
    refresh_schloss.refresh_schloss(target_date)
    print("更新schloss股票池结束")

def refresh_zero_predict():
    # 当前路径
    base_dir = os.path.dirname(__file__)
    refresh_zero = import_from_path("refresh_zero", os.path.join(base_dir, "zero", "get_zero_predict.py"))
    print("更新zero股票池...")
    refresh_zero.refresh_zero()
    print("更新zero股票池结束")

def refresh_three_predict():
    # 当前路径
    base_dir = os.path.dirname(__file__)
    refresh_growth_death = import_from_path("refresh_growth_death", os.path.join(base_dir, "three", "get_three_predict.py"))
    print("更新three股票池...")
    refresh_growth_death.refresh_growth_death()
    print("更新three股票池结束")

def refresh_seven_predict():
    # 当前路径
    base_dir = os.path.dirname(__file__)
    refresh_dcf = import_from_path("refresh_dcf", os.path.join(base_dir, "seven", "get_seven_predict.py"))
    print("更新seven股票池...")
    refresh_dcf.refresh_dcf()
    print("更新seven股票池结束")


def refresh_two_predict(target_date=None):
    # 当前路径
    base_dir = os.path.dirname(__file__)
    get_two_candidates = import_from_path("get_two_candidates", os.path.join(base_dir, "two", "get_two_predict.py"))
    print("更新two股票池...")
    get_two_candidates.get_two_candidates(target_date=target_date)
    print("更新two股票池结束")


def refresh_one_predict(target_date=None):
    # 当前路径
    base_dir = os.path.dirname(__file__)
    refresh_buy = import_from_path("refresh_buy", os.path.join(base_dir, "one", "get_one_predict.py"))
    print("更新one股票池...")
    refresh_buy.refresh_buy(target_date=target_date)
    print("更新one股票池结束")


def refresh(target_date=None):
    if target_date is not None:
        # 指定日期：跳过 API 数据更新，也跳过按季度计算的基本面模型
        # （zero/three/seven 只依赖财务季度，与交易日期无关，直接复用已有结果），
        # 仅按目标日期重算日线相关的股票池（schloss/two/one）。
        refresh_schloss(target_date)
        refresh_two_predict(target_date)
        refresh_one_predict(target_date)
        return
    today = datetime.datetime.now().date()
    weekday = today.weekday()
    if weekday == 4:
        update_data()
        refresh_seven_predict()
        refresh_three_predict()
        refresh_zero_predict()
        refresh_schloss()
        refresh_two_predict()
        refresh_one_predict()
    elif weekday == 5 or weekday == 6:
        pass
        # update_data()
        # refresh_seven_predict()
        # refresh_three_predict()
        # refresh_zero_predict()
        # refresh_schloss()
        # refresh_two_predict()
        # refresh_one_predict()
    else:
        append_data()
        refresh_schloss()
        refresh_two_predict()
        refresh_one_predict()
        # update_data()
        # refresh_zero_predict()
        # refresh_three_predict()
        # refresh_seven_predict()
        # refresh_one_predict()


def send_candidates(target_date=None):
    if target_date is None:
        today = datetime.datetime.now().date()
        weekday = today.weekday()
        if weekday == 5 or weekday == 6:
            # return
            refresh()
        else:
            # return
            refresh()
    else:
        refresh(target_date)
    buy_name = os.path.join(os.path.dirname(__file__), 'buy_predict.csv')
    buy_data = pd.read_csv(buy_name)
    if buy_data.empty:
        raise RuntimeError(
            "buy_predict.csv 为空，无法选出候选。通常是 one 股票池没产出（多为 "
            "schloss/two 未随日线数据刷新导致的日期错位，详见 get_one_predict.py 的报错）。"
        )
    date = buy_data['date'].iloc[0]
    one_buy_data = buy_data.sort_values('buffett', ascending=False).reset_index(drop=True)
    # one_buy_data = one_buy_data.iloc[:127]
    one_buy_data = one_buy_data.sort_values('up_down', ascending=False).reset_index(drop=True)
    print(one_buy_data)
    one_buy = one_buy_data['symbol'].iloc[0]
    print(one_buy)
    three_buy_data = buy_data.sort_values('buffett', ascending=False).reset_index(drop=True)
    three_buy_data = three_buy_data.iloc[:127]
    three_buy_data = three_buy_data.sort_values('up_down', ascending=False).reset_index(drop=True)
    print(three_buy_data)
    three_buy = three_buy_data['symbol'].iloc[0]
    print(three_buy)
    seven_buy_data = buy_data.sort_values('buffett', ascending=False).reset_index(drop=True)
    seven_buy_data = seven_buy_data.iloc[:31]
    seven_buy_data = seven_buy_data.sort_values('up_down', ascending=False).reset_index(drop=True)
    print(seven_buy_data)
    seven_buy = seven_buy_data['symbol'].iloc[0]
    print(seven_buy)
    thirty_one_buy_data = buy_data.sort_values('buffett', ascending=False).reset_index(drop=True)
    thirty_one_buy_data = thirty_one_buy_data.iloc[:7]
    thirty_one_buy_data = thirty_one_buy_data.sort_values('up_down', ascending=False).reset_index(drop=True)
    print(thirty_one_buy_data)
    thirty_one_buy = thirty_one_buy_data['symbol'].iloc[0]
    print(thirty_one_buy)
    one_hundred_and_twenty_seven_buy_data = buy_data.sort_values('buffett', ascending=False).reset_index(drop=True)
    one_hundred_and_twenty_seven_buy_data = one_hundred_and_twenty_seven_buy_data.iloc[:3]
    one_hundred_and_twenty_seven_buy_data = one_hundred_and_twenty_seven_buy_data.sort_values('up_down', ascending=False).reset_index(drop=True)
    print(one_hundred_and_twenty_seven_buy_data)
    one_hundred_and_twenty_seven_buy = one_hundred_and_twenty_seven_buy_data['symbol'].iloc[0]
    print(one_hundred_and_twenty_seven_buy)

    # 邮箱配置（发件人/授权码/收件人名单都在 config_local.py —— **不入库**，见 .gitignore。
    # 2026-09-28 抽出来的：原来授权码与 6 个收件人邮箱是明文，推公开仓库会泄漏。）
    smtp_server, port = SMTP_SERVER, SMTP_PORT
    sender, password = MAIL_SENDER, MAIL_PASSWORD
    receiver_list = list(MAIL_RECEIVERS)

    # 构建邮件
    message = MIMEMultipart()
    message["From"] = sender
    message["Subject"] = str(date) + ':推的票'

    body = ("1: " + one_buy + '\n' + "3: " + three_buy + '\n' + "7: " + seven_buy + '\n' +
             "31: " + thirty_one_buy + '\n' + '127: ' + one_hundred_and_twenty_seven_buy)
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
