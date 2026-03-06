"""
GPT-2 模型实现
"""
import math
import torch
import torch.nn as nn

from src.model.base import BaseLanguageModel
from src.model.gpt2.attention import MultiheadAttention


class GELU(nn.Module):
    """GELU 激活函数"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            math.sqrt(2.0 / math.pi) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class LayerNorm(nn.Module):
    """层归一化"""

    def __init__(self, emb_dim) -> None:
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return norm_x * self.scale + self.shift


class FeedForward(nn.Module):
    """前馈网络"""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    """Transformer 块"""

    def __init__(self, cfg) -> None:
        super().__init__()

        self.attn = MultiheadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            context_length=cfg["context_length"],
            dropout=cfg["drop_rate"],
            qkv_bias=False
        )
        self.layer_norm1 = LayerNorm(cfg["emb_dim"])
        self.layer_norm2 = LayerNorm(cfg["emb_dim"])
        self.ffn = FeedForward(cfg)
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        x_ = x
        x = self.layer_norm1(x)
        x = self.attn(x)
        x = self.drop_shortcut(x)
        x += x_

        x__ = x
        x = self.layer_norm2(x)
        x = self.ffn(x)
        x = self.drop_shortcut(x)
        x += x__

        return x


class GPT2(BaseLanguageModel):
    """GPT-2 语言模型"""

    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    简单的文本生成函数
    Args:
        model: 语言模型
        idx: 输入 token IDs，形状为 (batch, seq_len)
        max_new_tokens: 最大生成 token 数
        context_size: 上下文窗口大小
    Returns:
        生成的 token IDs
    """
    for _ in range(max_new_tokens):
        # 防止超出模型上下文窗口
        idx_context = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_context)

        # (batch, seq_len, vocab_size) -> (batch, vocab_size)
        logits = logits[:, -1, :]

        # (batch, vocab_size)
        probas = torch.softmax(logits, dim=-1)

        # (batch, 1)
        idx_next = torch.argmax(probas, dim=1, keepdim=True)

        # (batch, seq_len + 1)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """
    高级文本生成函数，支持 temperature 和 top-k 采样
    Args:
        model: 语言模型
        idx: 输入 token IDs
        max_new_tokens: 最大生成 token 数
        context_size: 上下文窗口大小
        temperature: 温度参数，0表示贪婪解码
        top_k: top-k 采样参数
        eos_id: 结束符 ID，遇到则停止生成
    Returns:
        生成的 token IDs
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # 应用 top-k 过滤
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # 应用温度缩放
        if temperature > 0.0:
            logits = logits / temperature
            logits = logits - logits.max(dim=-1, keepdim=True).values
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx
