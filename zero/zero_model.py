import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, d_model, num_heads, num_layers=3, ffn_mult=4, dropout=0.1):
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


class ZERO(nn.Module):
    def __init__(
        self,
        input_shape,
        output_shape,
        d_model=32,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
    ):
        super().__init__()
        in_features, seq_len = input_shape

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.input_proj = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        self.encoder = Encoder(d_model, num_heads, num_layers, ffn_mult=4, dropout=dropout)
        self.head = RegressionHead(d_model, output_shape, dropout)

    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.pos_embedding[:, : x.size(1), :]
        x = self.encoder(x)
        return self.head(x)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ZERO([7, 3], [1, 1]).to(device)

    x = torch.randn(4, 3, 7).to(device)
    y = model(x)

    print("input shape :", tuple(x.shape))
    print("output shape:", tuple(y.shape))
    print("params      :", sum(p.numel() for p in model.parameters()))
