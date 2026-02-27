import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        # 支持 (Batch, context_length, Dim)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 使用 transpose 处理批次数据的转置
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))

        # 缩放因子计算
        d_k = K.size(-1)
        attn_weights = torch.softmax(attn_scores / (d_k ** 0.5), dim=-1)

        return torch.matmul(attn_weights, V)


class CasualAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()

        self.d_out = d_out
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, context_length, d_in = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1))
        attn_scores.masked_fill_(
            self.mask.bool()[:context_length, :context_length], -torch.inf
        )
        d_k = K.size(-1)
        attn_weights = torch.softmax(attn_scores / (d_k ** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)

        return torch.matmul(attn_weights, V)


class MultiheadAttention(nn.Module):
    """ 
    多头注意力机制实现
    """

    def __init__(self, d_in, d_out, num_heads, context_length, dropout, qkv_bias=False):
        super().__init__()

        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        # 一般 d_in = d_out = d_model
        self.d_out = d_out
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_out, d_out)
        self.num_heads = num_heads
        self.head_dim = self.d_out // self.num_heads
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, seq_len, d_in = x.shape

        # 输入点乘投影矩阵得到 Query, Key, Value
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 将 Query, Key, Value 分割成多个头（num_heads 个头，每个头包含 head_dim 个维度）
        # 这里没有一次性完成
        Q = Q.view(b, seq_len, self.num_heads, self.head_dim)
        K = K.view(b, seq_len, self.num_heads, self.head_dim)
        V = V.view(b, seq_len, self.num_heads, self.head_dim)

        # 转置得到分割后的矩阵
        # (batch_size, num_heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        d_k = K.size(-1)    # 用于计算缩放因子
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))
        # 添加因果掩码, 确保当前位置的注意力只能关注到前面的位置
        # 值为 True 的位置会被填充为 -inf
        attn_scores.masked_fill_(
            self.mask.bool()[:seq_len, :seq_len], -torch.inf
        )
        attn_weights = torch.softmax(attn_scores / (d_k ** 0.5), dim=-1)

        # (batch_size, num_heads, seq_len, seq_len)
        attn_weights = self.dropout(attn_weights)

        # (batch_size, num_heads, seq_len, head_dim) -> (batch_size, seq_len, num_heads, head_dim)
        context_vec = torch.matmul(attn_weights, V).transpose(1, 2)

        # (b, seq_len, self.d_out)
        context_vec = context_vec.contiguous().view(b, seq_len, self.d_out)
        # 合并后的输出投影
        context_vec = self.out_proj(context_vec)

        return context_vec
