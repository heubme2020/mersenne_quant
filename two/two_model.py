"""
two_model.py — 改进版 TWO 模型

主要改动（对照 train.py / get_two_predict.py / gen_train_data.py）：
  ① Input Projection：先把原始 31 维特征映射到 d_attn 维（解决 31 是质数无法整除多头问题）
  ② Positional Encoding：正弦位置编码，让模型感知时间顺序
  ③ Pre-norm Block：LayerNorm → MHA → 残差；LayerNorm → FFN → 残差（标准 Transformer 结构）
  ④ 标准 MultiHeadAttention：head_dim = d_attn / num_heads，参数量大幅减少
  ⑤ GainDecoder 换成注意力池化 + MLP，替代原来 27559 → 31 的巨型展平线性层
  ⑥ Dropout 补充在各处
  ⑦ 输出命名与 train.py 对齐：seven_gain / thirty_one_gain / one_twenty_seven_gain
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 基础组件
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    正弦/余弦位置编码。
    为 Transformer 提供时间位置信息（序列长度 889 天）。
    """
    def __init__(self, d_model: int, max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)

        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        # 处理 d_model 为奇数的边界情况
        pe[:, 1::2] = torch.cos(pos * div) if d_model % 2 == 0 \
                      else torch.cos(pos * div[:-1])

        self.register_buffer('pe', pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D)
        return self.drop(x + self.pe[:, :x.size(1)])


class MultiHeadAttention(nn.Module):
    """
    标准缩放点积多头自注意力。

    与原版的区别：
      原版把 D 当 per-head 维度（W_Q: D → D*H），总维度会展开到 D*H，
      导致 block_0（62 头）的 W_Q 参数量 = 31×(31×62) = 59K（仅一层）。
      新版：d_model 是总维度，head_dim = d_model / num_heads，标准实现。
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, \
            f"d_model({d_model}) 必须能被 num_heads({num_heads}) 整除"
        self.h     = num_heads
        self.d     = d_model // num_heads        # per-head dim
        self.scale = math.sqrt(self.d)

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

    def _split(self, t: torch.Tensor) -> torch.Tensor:
        """(B, S, d_model) → (B, H, S, head_dim)"""
        B, S, _ = t.shape
        return t.reshape(B, S, self.h, self.d).permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        Q = self._split(self.W_Q(x))
        K = self._split(self.W_K(x))
        V = self._split(self.W_V(x))

        scores  = torch.matmul(Q, K.transpose(-1, -2)) / self.scale  # (B, H, S, S)
        weights = self.attn_drop(F.softmax(scores, dim=-1))
        out     = torch.matmul(weights, V)                            # (B, H, S, head_dim)
        out     = out.permute(0, 2, 1, 3).contiguous().reshape(B, S, D)
        return self.W_O(out)


class Block(nn.Module):
    """
    Pre-norm Transformer Block（比 post-norm 训练更稳定）：

        x ──► LayerNorm ──► MHA ──► + (残差) ──► x'
        x' ─► LayerNorm ──► FFN ──► + (残差) ──► 输出

    原版 Block 只有 Attention，没有 FFN 子层，这里补全。
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))   # 注意力子层 + 残差
        x = x + self.ffn(self.norm2(x))    # FFN 子层 + 残差
        return x


class Encoder(nn.Module):
    """
    多层 Block 堆叠 + 最终 LayerNorm。

    原版 Encoder 在 block_1 之后将残差加回原始 x（即两次跳过整个 block），
    现改为正常的逐层前向传播。
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int = 3,
        ffn_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [Block(d_model, num_heads, ffn_mult, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# ─────────────────────────────────────────────────────────────────────────────
# 输出头
# ─────────────────────────────────────────────────────────────────────────────

class ForecastDecoder(nn.Module):
    """
    序列 → 序列头：预测未来 127 天的 close / volume / delta。

    用 AdaptiveAvgPool 把序列从 889 压缩到 out_seq=127，
    再用线性层从 d_attn 投影到 out_features=3。
    """
    def __init__(
        self,
        d_model: int,
        out_seq: int,
        out_features: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(out_seq)
        self.proj = nn.Linear(d_model, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D)
        out = self.pool(x.permute(0, 2, 1)).permute(0, 2, 1)   # (B, out_seq, D)
        return self.norm(self.drop(self.proj(out)))              # (B, out_seq, out_features)


class GainDecoder(nn.Module):
    """
    注意力池化头：预测标量收益（7日 / 31日 / 127日）。

    原版 Decoder 先把 (B, 889, 31) flatten 成 (B, 27559) 再接线性层，
    三个增益头合计 ~255 万参数，且丢失了所有时序结构。

    新版改用可学习 Query 对编码器序列做注意力池化：
      query (1, 1, D) ──► 与 key(S, D) 做点积 ──► softmax ──► 加权求和 value
    只需 O(D²) 参数，同时保留了"哪些时间步更重要"的归纳偏置。
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.query = nn.Parameter(torch.empty(1, 1, d_model).normal_(std=0.02))
        self.key   = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(d_model)
        self.mlp   = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D)
        B   = x.size(0)
        q   = self.query.expand(B, -1, -1)                              # (B, 1, D)
        attn = F.softmax(
            torch.bmm(q, self.key(x).transpose(1, 2)) / self.scale,    # (B, 1, S)
            dim=-1,
        )
        pooled = torch.bmm(attn, self.value(x)).squeeze(1)             # (B, D)
        return self.mlp(pooled).unsqueeze(1)                            # (B, 1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 主模型
# ─────────────────────────────────────────────────────────────────────────────

class TWO(nn.Module):
    """
    TWO — Time-series Weighted Outcome model（A 股价格预测）

    网络结构：
        输入 (B, seq_len=889, d_in=31)
          │
          ▼
        Linear(d_in → d_attn) + LayerNorm         # ① Input Projection
          │
          ▼
        PositionalEncoding(d_attn)                 # ② 时间位置信息
          │
          ▼
        Encoder: num_layers × Block                # ③/④ 标准 Transformer
          │
          ├──► ForecastDecoder  → (B, 127, 3)      # close / volume / delta
          ├──► GainDecoder(7d)  → (B, 1, 1)        # 7 日收益预测
          ├──► GainDecoder(31d) → (B, 1, 1)        # 31 日收益预测
          └──► GainDecoder(127d)→ (B, 1, 1)        # 127 日收益预测

    参数：
        input_shape  : [d_in, seq_len]   e.g. [31, 889]
        output_shape : [d_out, out_seq]  e.g. [3, 127]
        d_attn       : 注意力内部维度（默认 64，需能被 num_heads 整除）
        num_heads    : 注意力头数（默认 8）
        num_layers   : Block 层数（默认 3）
        dropout      : Dropout 概率（默认 0.1）
    """
    def __init__(
        self,
        input_shape,
        output_shape,
        d_attn: int = 64,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        d_in    = input_shape[0]   # 31  (特征数)
        seq_len = input_shape[1]   # 889 (时间步)
        d_out   = output_shape[0]  # 3   (预测特征数: close/volume/delta)
        out_seq = output_shape[1]  # 127 (预测天数)

        assert d_attn % num_heads == 0, \
            f"d_attn={d_attn} 必须能被 num_heads={num_heads} 整除"

        # ① 输入投影：31 维原始特征 → d_attn 维，解决 31 是质数无法整除多头的问题
        self.input_proj = nn.Sequential(
            nn.Linear(d_in, d_attn),
            nn.LayerNorm(d_attn),
        )

        # ② 位置编码：让模型知道每一天在 889 天窗口中的相对位置
        self.pos_enc = PositionalEncoding(d_attn, max_len=seq_len + 16, dropout=dropout)

        # ③/④ Transformer 编码器（标准 Pre-norm Block，含 FFN）
        self.encoder = Encoder(d_attn, num_heads, num_layers, ffn_mult=4, dropout=dropout)

        # 输出头
        self.decoder_forecast         = ForecastDecoder(d_attn, out_seq, d_out, dropout)
        self.decoder_seven            = GainDecoder(d_attn, dropout)
        self.decoder_thirty_one       = GainDecoder(d_attn, dropout)
        self.decoder_one_twenty_seven = GainDecoder(d_attn, dropout)

    def forward(self, x: torch.Tensor):
        """
        x: (B, seq_len, d_in)  e.g. (B, 889, 31)

        返回（与 train.py 变量名对应）：
            close_volume_delta    : (B, 127, 3)   供 MSE Loss
            seven_gain            : (B, 1, 1)     7 日收益
            thirty_one_gain       : (B, 1, 1)     31 日收益
            one_twenty_seven_gain : (B, 1, 1)     127 日收益
        """
        x   = self.input_proj(x)   # (B, S, d_attn)
        x   = self.pos_enc(x)
        enc = self.encoder(x)

        close_volume_delta    = self.decoder_forecast(enc)
        seven_gain            = self.decoder_seven(enc)
        thirty_one_gain       = self.decoder_thirty_one(enc)
        one_twenty_seven_gain = self.decoder_one_twenty_seven(enc)

        return close_volume_delta, seven_gain, thirty_one_gain, one_twenty_seven_gain


# 向后兼容别名（原有代码 import ONE 不受影响）
ONE = TWO


# ─────────────────────────────────────────────────────────────────────────────
# 快速验证
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = TWO([31, 127 * 7], [3, 127]).to(device)

    B = 4
    x = torch.randn(B, 127 * 7, 31).to(device)   # (B, 889, 31)

    forecast, s7, s31, s127 = model(x)

    print(f"输入维度              : {x.shape}")
    print(f"Forecast (close/vol/δ): {forecast.shape}")   # (4, 127, 3)
    print(f"7 日收益              : {s7.shape}")          # (4, 1, 1)
    print(f"31 日收益             : {s31.shape}")         # (4, 1, 1)
    print(f"127 日收益            : {s127.shape}")        # (4, 1, 1)

    total = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量             : {total:,}")

    # 对比各模块参数
    for name, m in model.named_children():
        n = sum(p.numel() for p in m.parameters())
        print(f"  {name:<35}: {n:>10,}")








