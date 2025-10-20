import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from write_stock_data import update_exchange_data, append_exchange_data
import datetime
from write_stock_data import update_exchange_data
import importlib.util
import sys, os
import numpy as np
import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir, "zero"))
sys.path.append(os.path.join(base_dir, "one"))
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
    refresh_dcf = import_from_path("refresh_dcf", os.path.join(base_dir, "seven", "get_three_predict.py"))
    print("更新seven股票池...")
    refresh_dcf.refresh_dcf()
    print("更新seven股票池结束")

def refresh_one_predict():
    # 当前路径
    base_dir = os.path.dirname(__file__)
    refresh_buy = import_from_path("refresh_buy", os.path.join(base_dir, "one", "get_one_predict.py"))
    print("更新one股票池...")
    refresh_buy.refresh_buy()
    print("更新one股票池结束")


def refresh():
    today = datetime.datetime.now().date()
    weekday = today.weekday()
    if weekday == 4:
        update_data()
        refresh_zero_predict()
        refresh_three_predict()
        refresh_seven_predict()
        refresh_one_predict()
    elif weekday == 5 or weekday == 6:
        pass
    else:
        append_data()
        refresh_one_predict()


def send_candidates():
    today = datetime.datetime.now().date()
    weekday = today.weekday()
    if weekday == 5 or weekday == 6:
        return
    else:
        refresh()
    buy_name = os.path.join(os.path.dirname(__file__), 'buy_predict.csv')
    buy_data = pd.read_csv(buy_name)
    date = buy_data['date'].iloc[0]
    one_buy_data = buy_data.sort_values('up_down', ascending=False).reset_index(drop=True)
    print(one_buy_data)
    one_buy = one_buy_data['symbol'].iloc[0]
    print(one_buy)
    three_buy_data = buy_data.sort_values('value', ascending=False).reset_index(drop=True)
    three_buy_data = three_buy_data.iloc[:127]
    three_buy_data = three_buy_data.sort_values('up_down', ascending=False).reset_index(drop=True)
    print(three_buy_data)
    three_buy = three_buy_data['symbol'].iloc[0]
    print(three_buy)
    seven_buy_data = buy_data.sort_values('value', ascending=False).reset_index(drop=True)
    seven_buy_data = seven_buy_data.iloc[:31]
    seven_buy_data = seven_buy_data.sort_values('up_down', ascending=False).reset_index(drop=True)
    print(seven_buy_data)
    seven_buy = seven_buy_data['symbol'].iloc[0]
    print(seven_buy)
    thirty_one_buy_data = buy_data.sort_values('value', ascending=False).reset_index(drop=True)
    thirty_one_buy_data = thirty_one_buy_data.iloc[:7]
    thirty_one_buy_data = thirty_one_buy_data.sort_values('up_down', ascending=False).reset_index(drop=True)
    print(thirty_one_buy_data)
    thirty_one_buy = thirty_one_buy_data['symbol'].iloc[0]
    print(thirty_one_buy)
    one_hundred_and_twenty_seven_buy_data = buy_data.sort_values('value', ascending=False).reset_index(drop=True)
    one_hundred_and_twenty_seven_buy_data = one_hundred_and_twenty_seven_buy_data.iloc[:3]
    print(one_hundred_and_twenty_seven_buy_data)
    one_hundred_and_twenty_seven_buy = one_hundred_and_twenty_seven_buy_data['symbol'].to_list()
    print(one_hundred_and_twenty_seven_buy)

    # 邮箱配置
    smtp_server = "smtp.qq.com"
    port = 465   # SSL 端口
    sender = "abcdefg@qq.com" #写入你的邮箱
    password = ""   # 注意是授权码，不是QQ密码
    receiver = "abcdefg@qq.com"

    # 构建邮件
    message = MIMEMultipart()
    message["From"] = sender
    message["To"] = receiver
    message["Subject"] = str(date) + ':推荐的票'

    body = ("1: " + one_buy + '\n' + "3: " + three_buy + '\n' + "7: " + seven_buy + '\n' +
             "31: " + thirty_one_buy + '\n' + '127: ' + one_hundred_and_twenty_seven_buy[0] + ', ' + 
             one_hundred_and_twenty_seven_buy[1] + ', ' + one_hundred_and_twenty_seven_buy[2])
    message.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP_SSL(smtp_server, port) as server:
            server.login(sender, password)
            server.sendmail(sender, [receiver], message.as_string())
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
    send_candidates()
