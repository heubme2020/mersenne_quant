import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import shap
import pandas as pd


class MultiHeadAttention(nn.Module):
    def __init__(self, D, H):
        super(MultiHeadAttention, self).__init__()
        self.H = H  # number of heads
        self.D = D  # feature num

        self.W_Q = nn.Linear(self.D, self.D * self.H)
        self.W_K = nn.Linear(self.D, self.D * self.H)
        self.W_V = nn.Linear(self.D, self.D * self.H)
        self.W_O = nn.Linear(self.D * self.H, self.D)


    def concat_heads(self, x):
        '''(B, H, S, D) => (B, S, D*H)'''
        B, H, S, D = x.shape
        x = x.permute((0, 2, 1, 3)).contiguous()  # (B, S, H, D)
        x = x.reshape((B, S, H * D))  # (B, S, D*H)
        return x

    def split_heads(self, x):
        '''(B, S, D*H) => (B, H, S, D)'''
        B, S, D_H = x.shape
        x = x.reshape(B, S, self.H, self.D)  # (B, S, H, D)
        x = x.permute((0, 2, 1, 3))  # (B, H, S, D)
        return x

    def forward(self, x):
        Q = self.W_Q(x)  # (B, S, D)
        K = self.W_K(x)  # (B, S, D)
        V = self.W_V(x)  # (B, S, D)
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        attention_scores = torch.matmul(Q, K.transpose(-1, -2))  # (B,H,S,S)
        attention_scores = attention_scores / math.sqrt(self.D)
        attention_weights = nn.Softmax(dim=-1)(attention_scores)
        output = torch.matmul(attention_weights, V)  # (B, H, S, D)
        output = self.concat_heads(output)  # (B, S, D*H)
        output = self.W_O(output)
        return output

class Block(nn.Module):
    def __init__(self, d_model, num_heads):
        super(Block, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.gelu = nn.GELU()
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        attention_output = self.attention(x)
        output = x + attention_output
        output = self.linear(output)
        output = self.norm(output)
        output = self.gelu(output)
        output = self.out(output)
        return output


class Encoder(nn.Module):
    def __init__(self, d_model, num_heads):
        super(Encoder, self).__init__()
        self.block_0 = Block(d_model, num_heads*2)
        self.block_1 = Block(d_model, num_heads)
        self.block_2 = Block(d_model, num_heads)

    def forward(self, x):
        output = self.block_0(x)
        output = x + output
        output = self.block_1(output)
        output = x + output
        output = self.block_2(output)
        return output


class Decoder(nn.Module):
    ##input_shape, output_shape这里[feature_num, seq_length]
    def __init__(self, input_shape, output_shape, device=None):
        super(Decoder, self).__init__()
        self.device = device
        self.in_feature_num = input_shape[0]
        self.in_seq_length = input_shape[1]
        self.out_feature_num = output_shape[0]
        self.out_seq_length = output_shape[1]

        self.linear = nn.Linear(input_shape[0]*input_shape[1], input_shape[0]*output_shape[1])
        self.norm = nn.LayerNorm(input_shape[0]*output_shape[1])
        self.gelu = nn.GELU()
        self.linear_out = nn.Linear(input_shape[0]*output_shape[1], output_shape[0]*output_shape[1])

    def forward(self, x):
        batch_size = x.shape[0]
        # print(x.shape)
        # print(self.in_feature_num*self.in_seq_length)
        decoder_output = x.reshape(batch_size, self.in_feature_num*self.in_seq_length)
        decoder_output = self.linear(decoder_output)
        decoder_output = self.norm(decoder_output)
        decoder_output = self.gelu(decoder_output)
        output = self.linear_out(decoder_output)
        output = output.view(batch_size, self.out_seq_length, self.out_feature_num)
        return output


class SEVEN(nn.Module):
    def __init__(self, input_shape,  output_shape):
        super(SEVEN, self).__init__()
        self.encoder = Encoder(input_shape[0], 31)
        self.decoder_one = Decoder(input_shape, output_shape)
        self.decoder_three = Decoder(input_shape, output_shape)
        self.decoder_seven = Decoder(input_shape, output_shape)
    def forward(self, x):
        encoder = self.encoder(x)
        dcf_three = self.decoder_one(encoder)
        dcf_seven = self.decoder_three(encoder)
        dcf_thirty_one = self.decoder_seven(encoder)
        return dcf_three, dcf_seven, dcf_thirty_one


if __name__ == '__main__':
    model = torch.load('one.pt').to('cuda')
    X0 = torch.randn(4, 256, 150).to('cuda')
    X1 = torch.randn(4, 256, 150).to('cuda')
    prob, decoder0, decoder1 = model(X0, X1)
    # 使用 SHAP 解释模型
    print(X0.shape)
    print(prob.shape)
    # explainer = shap.DeepExplainer(model, X_tensor)
    # shap_values = explainer.shap_values(X_tensor)

    # # 可视化特征重要性
    # shap_values_np = np.array(shap_values)
    # shap.summary_plot(shap_values_np, X)
    #
    # # 解释单个样本
    # shap.initjs()
    # shap.force_plot(explainer.expected_value[0], shap_values_np[0], X.iloc[0])
    #
    # # 计算并输出特征重要性
    # feature_importance = np.abs(shap_values_np).mean(axis=0)
    # importance_df = pd.DataFrame({
    #     'feature': X.columns,
    #     'importance': feature_importance
    # }).sort_values(by='importance', ascending=False)
    #
    # print(importance_df)










