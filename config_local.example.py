"""本机私密配置的**模板**（这份入库，真值那份 `config_local.py` 不入库）。

    cp config_local.example.py config_local.py   # 然后填上你自己的值

为什么要这么绕：原来这些凭据是明文硬编码在 `get_stock_data.py` /
`write_stock_data.py` / `refresh_candidates.py` 里，2026-09-28 准备推 GitHub 时
抽出来的 —— 公开仓库里的密钥会被爬虫在几小时内扫走。
"""
# ---- 数据库（MySQL）----
# 形如 mysql+pymysql://<user>:<password>@<host>:<port>/<database>
MYSQL_DSN = 'mysql+pymysql://user:password@localhost:3306/stock'

# ---- 数据源 API ----
# financialmodelingprep.com 的 API key（write_stock_data.fmp() 用）
FMP_TOKEN = 'YOUR_FMP_API_KEY'
# 另一个服务 token（现役代码未使用，留着备用）
SINKING_TOKEN = 'YOUR_SINKING_TOKEN'

# ---- 邮件（refresh_candidates.py 推送选票用）----
SMTP_SERVER = 'smtp.qq.com'
SMTP_PORT = 465                     # SSL 端口
MAIL_SENDER = 'you@example.com'
MAIL_PASSWORD = 'YOUR_SMTP_AUTH_CODE'   # 注意是授权码，不是邮箱登录密码
MAIL_RECEIVERS = ['you@example.com']    # 收件人列表
