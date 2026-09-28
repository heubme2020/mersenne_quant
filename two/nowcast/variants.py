"""头 1 / 头 2 的构造方式对照实验（2026-09-23）。

要回答两个单变量问题：

  头 1：`毛利/总资产`（level_flow，现在） vs `毛利增长`（growth，自己比自己）
  头 2：`营收增长`（growth，自相对，现在） vs `营收/总资产`（level_flow）

每个 variant 只改一个头，其余不动，所以是干净的单变量对照。
三套标签共用同一份 X（特征不变），标签数组以 `_y{suffix}.npy` / `_b` / `_r` 后缀共存于
同一个 chunk 目录，由 `data.Store(suffix=...)` 选择 —— 不必复制 19 GB 的特征。

  suffix=''    基线（就是已训好的那版：毛利/总资产 · 营收增长 · Δ总资产/总资产）
  suffix='v1'  头 1 换成 毛利增长
  suffix='v2'  头 2 换成 营收/总资产
"""

BASE = {
    'grossProfit':   ('income',  'grossProfit', 'level_flow'),
    'revenueGrowth': ('income',  'revenue',     'growth'),
    'assetGrowth':   ('balance', 'totalAssets', 'level_stock'),
}

VARIANTS = {
    # 头 1：毛利/总资产  ->  毛利增长（自己比自己）
    'v1': {
        'gpGrowth':      ('income',  'grossProfit', 'growth'),
        'revenueGrowth': ('income',  'revenue',     'growth'),
        'assetGrowth':   ('balance', 'totalAssets', 'level_stock'),
    },
    # 头 2：营收增长  ->  营收/总资产
    'v2': {
        'grossProfit':   ('income',  'grossProfit', 'level_flow'),
        'revenue_TA':    ('income',  'revenue',     'level_flow'),
        'assetGrowth':   ('balance', 'totalAssets', 'level_stock'),
    },
    # 头 3：总资产增长  ->  EBIT 增长（自己比自己）
    # 注意：EBIT 有 17.6% 的公司为负，自相对分母 ≤0 会丢样本（覆盖率要在日志里看）。
    'v3': {
        'grossProfit':   ('income',  'grossProfit', 'level_flow'),
        'revenueGrowth': ('income',  'revenue',     'growth'),
        'ebitGrowth':    ('income',  'ebit',        'growth'),
    },
    # v4 = v3 的「分母换成总资产」版，用来把 v3 的**样本损失**和**信号本身**分开：
    #   v3 vs v4 只差分母 -> 差异归因于「自相对 vs 比总资产」
    #   v4 vs base 只差头 3 的科目 -> 差异归因于「EBIT vs 总资产」
    'v4': {
        'grossProfit':   ('income',  'grossProfit', 'level_flow'),
        'revenueGrowth': ('income',  'revenue',     'growth'),
        'ebit_TA':       ('income',  'ebit',        'level_flow'),
    },
    # v5 = 「三个头全部自相对」+ **不丢负分母样本**（分母取 |·|，符号正确）
    #   vs v3：只差头 3 的分母处理 -> 隔离「丢不丢负分母」这个选择
    #   头 1/2 的科目恒正，growth_signed 与 growth 只差常数 1，rank 等价 -> 只有头 3 真的变了
    # ===== u1（用户 2026-09-23 指定）=====
    # 三头都用同一个「年化同比变化率」形式，输入仍是 one 的 889×31。
    # suffix 'u1' 下若要跑全球数据，需要重新生成 X（A 股 X 可复用）。
    'u1': {
        'gpDelta':  ('income',  'grossProfit', 'yoy_delta'),
        'revDelta': ('income',  'revenue',     'yoy_delta'),
        'taDelta':  ('balance', 'totalAssets', 'yoy_delta'),
    },
    # ===== u2 / u3：只换【分母】，隔离「哪种归一化最好」（2026-09-24 用户要求）=====
    #   分子统一 = (ΣX[j+1..j+h] − h·X[j-3])
    #   u1(A) = 分子 / (h·X[j-3])   自相对（现值，靠稳健尺度+裁剪兜极值）
    #   u2(B) = 分子 / 市值          恒正且大 -> 无分母爆炸；但市值把价格带进标签
    #   u3(C) = 分子 / 总资产[j-3]   恒正稳定 -> 无分母爆炸，且【不带价格】
    #           ⚠️ 分母**不乘 h**（与 u1 的 h·X[j-3] 不同）—— 实测标签 = (ΣX−h·X[j-3])/TA[j-3]，
    #           所以 u3 的标签量纲随 h 线性放大（z-score 时按头各自标准化，不影响训练）。
    #           ⚠️ u2/u3 的 j = 最后【已披露】季度，比锚点晚 1~2 季 -> 窗口开头约 0.6 个季度
    #           在锚点日已【结束】（未披露）。实测 53~57% 的行至少含 1 个这样的季度。
    #           u6/u7 用 jc（最后【已结束】季）-> 该比例是 0。**两者不能直接比 IC**。
    # 三套标签共用同一份 X、同一批样本行（行集由 u1 的 ok 掩码决定），所以对比是干净的。
    # 注意：u2/u3 的标签不在 labels.py 里算 —— u2 要用锚点日的收盘价，只能在实际生成时算，
    # 见 add_bc_labels.py。
    'u2': {
        'gpDmcap':  ('income',  'grossProfit', 'dgp_placeholder'),
        'revDmcap': ('income',  'revenue',     'dgp_placeholder'),
        'taDmcap':  ('balance', 'totalAssets', 'dgp_placeholder'),
    },
    'u3': {
        'gpDta':  ('income',  'grossProfit', 'dgp_placeholder'),
        'revDta': ('income',  'revenue',     'dgp_placeholder'),
        'taDta':  ('balance', 'totalAssets', 'dgp_placeholder'),
    },
    # u4(D)：分子分母都用「最后【已结束】的季度 jc」的 X[jc] 与 总资产[jc]（用户 2026-09-24 指定）。
    #   label = ( ΣX[jc+1..jc+h] − h·X[jc] ) / ( h·总资产[jc] )
    #   最新鲜（基准只差约 1 个月），但 (a) 失去季节对齐（jc 与未来首季差 1 季而非 4 季），
    #   (b) 锚点日 jc 的报告可能还没披露（A 股 Q2→8/31、Q4→次年 4/30）→ 有前视。
    #   由 add_bc_labels.py 写入（要用日线锚点日做 asof，labels.py 里算不了）。
    'u4': {
        'gpDjc':  ('income',  'grossProfit', 'dgp_placeholder'),
        'revDjc': ('income',  'revenue',     'dgp_placeholder'),
        'taDjc':  ('balance', 'totalAssets', 'dgp_placeholder'),
    },
    # u5(E)：j = 【锚点日所在的季度】；label = ( ΣX[j..j+h-1] − h·X[j-3] ) / ( h·总资产[j-3] )
    #   锚点 7/21 -> 基准 2023Q4(12/31)、窗口从 2024Q3(9/30) 起（用户 2026-09-24 指定）。
    #   无前视（X[j-3] 早已披露）；代价是失去季节对齐（差 3 季而非 4 季）。
    'u5': {
        'gpDjc3':  ('income',  'grossProfit', 'dgp_placeholder'),
        'revDjc3': ('income',  'revenue',     'dgp_placeholder'),
        'taDjc3':  ('balance', 'totalAssets', 'dgp_placeholder'),
    },
    # u6（用户 2026-09-25 指定，最简版）：3 个头，每个头只有一个值
    #   label = ( median(X[jc+1..jc+7]) − median(X[jc-6..jc]) ) / 总资产[jc]
    #   jc = 锚点日前【最后一个已结束】的季度（7/21 -> 2024Q2/6-30）
    #   中位数抗单季异常；分母是单季总资产 -> 恒正不爆。
    #   ⚠️ jc 的报告可能尚未披露（7/21 时 Q2 要到 8/31）-> 有前视。
    'u6': {
        'gpMed7':  ('income',  'grossProfit', 'med7'),
        'revMed7': ('income',  'revenue',     'med7'),
        'taMed7':  ('balance', 'totalAssets', 'med7'),
    },
    # u7（用户 2026-09-25 指定）：u6 的 3 个期限版 —— 3 字段 × h∈{1,3,7} = 9 个头。
    #   label_h = ( median(X[jc+1..jc+h]) − median(X[jc-h+1..jc]) ) / 总资产[jc]
    #   h=1 = 「未来一季度 − 刚过去的一季度」；h=3 = 3 季中位数之差；h=7 = u6。
    #   目的：把 u6 扩到和 u3 相同的头数（9），让 u3 / u6 / u7 在同等头数下可比。
    #   ⚠️ 与 u6 同样有 jc 前视（jc 的报告在锚点日可能未披露）；u3 没有。
    'u7': {
        'gpMed':  ('income',  'grossProfit', 'med'),
        'revMed': ('income',  'revenue',     'med'),
        'taMed':  ('balance', 'totalAssets', 'med'),
    },
    # u8（用户 2026-09-25 指定）：把「同比去季节」（u3 的强项）和「中位数抗异常」（u6 的强项）
    # 合起来。3 头 × 1 期限（h=7），**无量纲，不再除以任何东西**：
    #   毛利头：median over q∈[jc+1..jc+7] of ( GP_q/TA_q − GP_{q−4}/TA_{q−4} )
    #   营收头：median over q∈[jc+1..jc+7] of ( Rev_q/TA_q − Rev_{q−4}/TA_{q−4} )
    #   总资产：median over q∈[jc+1..jc+7] of ( TA_q − TA_{q−4} ) / TA_{q−4}   ← 教科书 asset growth 的中位版
    # 依据：u7 的短期限崩了（gpMed1 逐日 IC 仅 0.048）而 u3 各期限都平（同比对齐救了它），
    # 说明「同比对齐」是短期限的关键、「中位数」是长期限的关键。u8 两者都占。
    # 另：同比差分后历史要求降到 jc−3（u6 要 jc−6）→ 可用样本比 u6 多。
    'u8': {
        'gpYoy':  ('income',  'grossProfit', 'yoymed'),
        'revYoy': ('income',  'revenue',     'yoymed'),
        'taYoy':  ('balance', 'totalAssets', 'yoymed'),
    },
    # u9（用户 2026-09-26 指定）：u6 把【过去窗口】从 7 季改成 3 季，前向窗口仍是 7 季。
    #   label = ( median(X[jc+1..jc+7]) − median(X[jc−2..jc]) ) / 总资产[jc]
    # 生成：`--suffix u9 --horizons 7 --past-win 3`
    # 【结论（2026-09-26 修订）：验证 IC 明显更差 —— 但**终点指标不差**，已选为线上变体】
    # ⚠️ 此处曾写「明显更差，已弃用」—— **那个结论下早了，已作废**。
    #   原因：下面这一段数字全是【验证 IC vs 财务量】口径（衡量"能不能预测财务量"），
    #   而 two 的用处是**排序选股**，该看【逐日截面 IC vs 未来收益】—— 见本段末尾「终点指标」。
    #   又一个「预测得准 ≠ 预测值有区分度」的实例。
    # 验证 IC 口径，2 轮筛选：ep0/ep1 验证 IC = 0.2634 / 0.2746
    #   （u6 同口径 = 0.3394 / 0.3526，差 −0.076，两个 epoch 一致 -> 远超噪声）
    #   * 分头看：逐日 IC  gp **−43%**（0.2609→0.1496）/ rev −27% / **ta 仅 −7%**
    #   * 机制确认：过去窗口只覆盖 3 个日历季（u6 是 4 = 完整四季）-> 季节偏差抵不掉；
    #     而**总资产季节性最弱**，所以它受害最小（正好是跌幅最小的那个头）。
    #   * 用户的假设（缩短过去窗口可少依赖"已涨上来的趋势"）**方向对但幅度小**：
    #     标签层 corr(标签, mom889) 0.170→0.110（−35%），但**模型层只传导了 1/3**：
    #     corr(模型预测, mom889) A股 0.520→0.455（−12.5%）—— 因为模型会放大动量信号。
    #   * 唯一确定的好处：样本 +5%（历史要求从 14 季降到 10 季：584,078 vs 555,633 行）
    # ⚠️ 离线诊断（SHZ 3000 行）的两条，事后被证实：
    #   1. **median of 3 就是中间那个值本身**，几乎没有抗异常能力 -> 过去基准的单季敏感度
    #      从 0.039 涨到 0.112（脆 2.9 倍，信号尺度不变）
    #   2. 过去窗口只覆盖 3 个不同日历季 -> 季节偏差抵不掉（上面已实测确认）
    # ── 终点指标（2026-09-26 配对比较，`two/nowcast/paired_u6_u9.py`）──
    #   在同一批 44,352 行 / 386 锚点日上配对（u6 的测试集是 u9 的**真子集**）：
    #     超出差(u9−u6)  h=1 +0.0002 (t=0.07) 打平 | h=3 +0.0020 (t=0.86) 打平
    #                    h=7 +0.0068 (t=2.87) u9 更好
    #   u9 还多覆盖 1,818 行 / 11 只股票 -> **two 最终采用 u9**（见 retrain_all.py 的 two 配方）。
    #   保留意见：两者都没跑赢各自的免费基线（超出仍为负），"模型没加价值"对两者都成立；
    #     且 h=7 的 t 受重叠样本影响偏乐观。
    'u9': {
        'gpMedP3F':  ('income',  'grossProfit', 'med'),
        'revMedP3F': ('income',  'revenue',     'med'),
        'taMedP3F':  ('balance', 'totalAssets', 'med'),
    },
    'v5': {
        'gpGrowth':      ('income',  'grossProfit', 'growth_signed'),
        'revGrowth':     ('income',  'revenue',     'growth_signed'),
        'ebitGrowth':    ('income',  'ebit',        'growth_signed'),
    },
}


# 每个 variant 的期限集合（u6/u8/u9 只有一个期限 7）
_HORIZONS_OVERRIDE = {'u6': [7], 'u8': [7], 'u9': [7]}


def horizons_for(suffix):
    return list(_HORIZONS_OVERRIDE.get(suffix, [1, 3, 7]))


def fields_for(suffix):
    """suffix -> FIELDS 字典（顺序即展平顺序，与 model 的头一一对应）。"""
    if not suffix:
        return dict(BASE)
    if suffix not in VARIANTS:
        raise KeyError(f'未知 variant {suffix!r}，可选 {[""] + list(VARIANTS)}')
    return dict(VARIANTS[suffix])


def label_desc(suffix):
    """人读的一句话描述，用于日志/文档。"""
    f = fields_for(suffix)
    return ' · '.join(f'{k}({v[2]})' for k, v in f.items())
