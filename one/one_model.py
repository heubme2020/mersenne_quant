import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# 输出定义：
#   - 辅助头 close_volume_delta：未来 AUX_OUTPUT_DAYS 天的 [close, volume, delta] 序列（轻量池化+位置解码）
#   - close 标量头：5 个时间窗口（1/3/7/31/127），相对明天收盘价
AUX_OUTPUT_DAYS = 128
HORIZONS = [1, 3, 7, 31, 127]


class MultiHeadAttention(nn.Module):
    """标准多头注意力：d_model 均分到各头。"""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def _split_heads(self, x):
        b, s, _ = x.shape
        return x.reshape(b, s, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        b, _, s, _ = x.shape
        return x.permute(0, 2, 1, 3).contiguous().reshape(b, s, self.d_model)

    def forward(self, x):
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        return self.out_proj(self._merge_heads(out))


class TransformerBlock(nn.Module):
    """pre-norm Transformer block：attention + FFN。"""

    def __init__(self, d_model, num_heads, ffn_mult=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers=4, ffn_mult=4, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, ffn_mult, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class AttentionPool(nn.Module):
    """可学习 query 对序列做注意力池化，(B, S, D) -> (B, D)。"""

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.query = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(d_model)

    def forward(self, x):
        batch_size = x.size(0)
        query = self.query.expand(batch_size, -1, -1)
        scores = torch.bmm(query, self.key(x).transpose(1, 2)) / self.scale
        weights = self.dropout(torch.softmax(scores, dim=-1))
        return torch.bmm(weights, self.value(x)).squeeze(1)


class RegressionHead(nn.Module):
    """标量头：AttentionPool + MLP -> (B, out_seq, out_feat)。"""

    def __init__(self, d_model, output_shape, dropout=0.1):
        super().__init__()
        out_features, out_seq = output_shape
        self.out_feature_num = out_features
        self.out_seq_length = out_seq
        self.pool = AttentionPool(d_model, dropout)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, out_features * out_seq),
        )

    def forward(self, x):
        output = self.net(self.pool(x))
        return output.view(output.size(0), self.out_seq_length, self.out_feature_num)


class SequenceHead(nn.Module):
    """
    轻量序列头：AttentionPool 压成 (B, d_model)，加可学习位置 embedding 后 MLP 逐日解码。
    （辅助正则，不需要完整 transformer decoder）
    """

    def __init__(self, d_model, out_len, out_dim, dropout=0.1):
        super().__init__()
        self.out_len = out_len
        self.out_dim = out_dim
        self.pool = AttentionPool(d_model, dropout)
        self.pos = nn.Parameter(torch.empty(1, out_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, x):
        pooled = self.pool(x).unsqueeze(1)   # (B, 1, d_model)
        h = pooled + self.pos                 # (B, out_len, d_model)
        return self.net(h)                    # (B, out_len, out_dim)


class ONE(nn.Module):
    """
    three 风格重构：
        输入 : (B, S=127*7, D=31)
        输出 : close_volume_delta (B, 128, 3)   —— 辅助序列头
               close_preds        (B, 1, 5)    —— close 标量头（1/3/7/31/127）
    """

    def __init__(self, input_shape, d_model=128, num_heads=8, num_layers=4,
                 dropout=0.1, aux_out_len=AUX_OUTPUT_DAYS, aux_out_dim=3,
                 n_horizons=len(HORIZONS)):
        super().__init__()
        in_features, seq_len = input_shape

        self.input_proj = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        self.encoder = Encoder(d_model, num_heads, num_layers, ffn_mult=4, dropout=dropout)
        self.close_head = RegressionHead(d_model, [n_horizons, 1], dropout)
        self.aux_head = SequenceHead(d_model, aux_out_len, aux_out_dim, dropout)

        # 输出端的反标准化（2026-09-26 加）：训练时标签被 (Y − median)/scale 标准化过，
        # 所以模型输出是 z 空间的值。把这一对常数**作为 buffer 存进模型**，forward 里自动还原
        # -> 任何调用方拿到的都是【真实单位】的输出，不可能忘（以前常数放在推理脚本/硬编码里，
        #    漏掉就会静默错一个量级 —— nowcast 的"毛利/市值"曾因此被放大 76 倍）。
        # 默认是恒等（scale=1, mean=0），所以老模型不受影响；训练后用 set_output_scale() 写入。
        # 展平顺序与标签一致（标签数组是「字段优先」展平的）。
        self.register_buffer('out_scale', torch.ones(1))
        self.register_buffer('out_mean', torch.zeros(1))

    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.pos_embedding[:, :x.size(1), :]
        enc = self.encoder(x)
        close_preds = self.close_head(enc)          # (B, 1, 5)
        close_volume_delta = self.aux_head(enc)     # (B, 128, 3)
        # z 空间 -> 真实单位（getattr 兼容没有这两个 buffer 的旧 .pt）
        sc = getattr(self, 'out_scale', None)
        if sc is not None and sc.numel() == close_preds.shape[1:].numel():
            sc = sc.to(device=close_preds.device, dtype=close_preds.dtype)
            mn = self.out_mean.to(device=close_preds.device, dtype=close_preds.dtype)
            close_preds = (close_preds.reshape(close_preds.size(0), -1) * sc
                           + mn).reshape(close_preds.shape)
        return close_volume_delta, close_preds


def set_output_scale(model, mean, scale):
    """把训练时的标签统计量写进模型的 buffer（之后 forward 直接输出真实单位）。

    mean/scale 是**展平顺序**的一维序列（与标签数组、`labels.flat_names` 同序）。
    """
    m = torch.as_tensor(mean, dtype=torch.float32)
    s = torch.as_tensor(scale, dtype=torch.float32)
    # ⚠️ 必须确保它们是 **buffer**：旧 .pt 的 _buffers 里没有这两个名字，
    #    直接 `model.out_scale = s` 会退化成【普通属性】，.to(device) 不会搬它 -> 设备不一致。
    for name, v in (('out_mean', m), ('out_scale', s)):
        if name in model._buffers:
            model._buffers[name] = v
        else:
            if hasattr(model, name):        # 上一版误设成的普通属性 -> 先删
                delattr(model, name)
            model.register_buffer(name, v)
    return model


if __name__ == "__main__":
    in_feat, seq_len = 31, 127 * 7
    model = ONE([in_feat, seq_len])
    x = torch.randn(2, seq_len, in_feat)

    cvd, close = model(x)
    print("input  :", tuple(x.shape))
    print("cvd    :", tuple(cvd.shape))
    print("close  :", tuple(close.shape))

    n_params = sum(p.numel() for p in model.parameters())
    print("params : %.2fM" % (n_params / 1e6))
