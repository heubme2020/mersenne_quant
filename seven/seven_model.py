import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# 三个输出分支（顺序即输出 tensor 的第 1 维：0=fcf, 1=dividend, 2=netasset）
BRANCHES = ['fcf', 'dividend', 'netasset']
# 三个预测期限（未来季度数），顺序即输出 tensor 的第 2 维：0=3Q, 1=7Q, 2=31Q
HORIZONS = [3, 7, 31]
HORIZON_NAMES = ['three', 'seven', 'thirty_one']


def label_columns():
    """训练样本 .h5 中标签列的顺序（branch-major，共 9 列）。
    按 (len(BRANCHES), len(HORIZONS)) reshape 后即 [branch, horizon]，与模型输出一致。"""
    return [f'{b}_{h}' for b in BRANCHES for h in HORIZON_NAMES]


class MultiHeadAttention(nn.Module):
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
    """一个分支的回归头：AttentionPool + MLP -> 输出 num_horizons 个期限的预测值。"""
    def __init__(self, d_model, num_horizons, dropout=0.1):
        super().__init__()
        self.num_horizons = num_horizons
        self.pool = AttentionPool(d_model, dropout)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_horizons),
        )

    def forward(self, x):
        output = self.net(self.pool(x))
        return output.view(output.size(0), self.num_horizons)


class SEVEN(nn.Module):
    def __init__(
        self,
        input_shape,
        num_horizons=3,
        d_model=128,
        num_heads=8,
        num_layers=4,
        dropout=0.1,
    ):
        super().__init__()
        in_features, seq_len = input_shape

        self.input_shape = input_shape
        self.num_horizons = num_horizons
        self.num_branches = len(BRANCHES)

        self.input_proj = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        self.encoder = Encoder(d_model, num_heads, num_layers, ffn_mult=4, dropout=dropout)
        # 三个分支各一个回归头，每个头输出 num_horizons 个期限
        self.fcf_head = RegressionHead(d_model, num_horizons, dropout)
        self.dividend_head = RegressionHead(d_model, num_horizons, dropout)
        self.netasset_head = RegressionHead(d_model, num_horizons, dropout)

    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.pos_embedding[:, : x.size(1), :]
        encoder = self.encoder(x)
        fcf = self.fcf_head(encoder)           # [batch, num_horizons]
        dividend = self.dividend_head(encoder) # [batch, num_horizons]
        netasset = self.netasset_head(encoder) # [batch, num_horizons]
        # 输出 [batch, num_branches, num_horizons]
        #   第 1 维 = branch（0=fcf, 1=dividend, 2=netasset）
        #   第 2 维 = horizon（0=3Q, 1=7Q, 2=31Q）
        return torch.stack([fcf, dividend, netasset], dim=1)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SEVEN([127, 31]).to(device)

    x = torch.randn(4, 31, 127).to(device)
    out = model(x)

    print("input shape :", tuple(x.shape))
    print("output shape:", tuple(out.shape), "(branch × horizon)")
    print("branches    :", BRANCHES, "->", HORIZON_NAMES)
    print("label cols  :", label_columns())
    print("params      :", sum(p.numel() for p in model.parameters()))
