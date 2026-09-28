"""nowcast 模型：复用 one 的架构，标量头改成 (B, 字段, 期限) = (B, 3, 3)。

    输入 (B, 889, 31) -> 编码 -> aux 头 (B, 128, 3) + 标量头 (B, 3, 3)
    标量头轴序: [:, 字段, 期限]，
    字段 ∈ {毛利/总资产, 营收增长, Δ总资产/总资产}，期限 ∈ {1, 3, 7} 季

**字段名与顺序直接从 `labels.py` 导入**，不再各写一份：y 数组是「字段优先」展平的
（`for f in FIELDS for h in HORIZONS`），而 train.py 用 `y[:, f*N_HORIZONS + h]` 索引、
FLAT_NAMES 给头和日志命名 —— 三处错位过一次就会静默地把头和标签配错。

注意：这**不是**结构性改动 —— `RegressionHead` 内部就是
`Linear(d_model, f*h)` + `.view(B, out_seq, out_features)`，所以 (B,3,3) 与
(B,1,9) 参数、表达力完全相同，只是索引方式不同。真正能加归纳偏置的是
「因子化头」或「嵌套期限约束」，那些留到第二轮消融。
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'one'))
sys.path.insert(0, HERE)

from one_model import ONE as _ONE, RegressionHead      # noqa: E402
from labels import FIELDS as LABEL_FIELDS, HORIZONS as _H, flat_names  # noqa: E402

FIELDS = list(LABEL_FIELDS)
HORIZONS = list(_H)
N_FIELDS, N_HORIZONS = len(FIELDS), len(HORIZONS)
N_HEADS = N_FIELDS * N_HORIZONS
# 展平顺序与标签数组一致（字段优先），供评估/日志用
FLAT_NAMES = flat_names()


class Nowcast(_ONE):
    """与 one.ONE 完全同构，只把标量头换成 (B, 字段, 期限)。"""

    def __init__(self, input_shape=(31, 127 * 7), d_model=128, num_heads=8,
                 num_layers=4, dropout=0.1, aux_out_len=128, aux_out_dim=3,
                 n_fields=None, n_horizons=None):
        # n_fields/n_horizons 默认取 labels（基线 3×3）；变体（如 u6 只有 3 个头）可覆盖
        nf = N_FIELDS if n_fields is None else n_fields
        nh = N_HORIZONS if n_horizons is None else n_horizons
        self.n_fields, self.n_horizons, self.n_heads = nf, nh, nf * nh
        super().__init__(input_shape, d_model=d_model, num_heads=num_heads,
                         num_layers=num_layers, dropout=dropout,
                         aux_out_len=aux_out_len, aux_out_dim=aux_out_dim,
                         n_horizons=self.n_heads)
        # RegressionHead(d_model, [out_features, out_seq]) -> view(B, out_seq, out_features)
        # 想要 (B, 字段, 期限)：out_seq=字段, out_features=期限
        self.close_head = RegressionHead(d_model, [nh, nf], dropout)

    def forward(self, x):
        aux, scalar = super().forward(x)
        return aux, scalar.view(scalar.size(0), self.n_fields, self.n_horizons)
