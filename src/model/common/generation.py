"""
通用文本生成函数
适用于所有语言模型
"""
import torch


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    简单的文本生成函数（贪婪解码）
    Args:
        model: 语言模型（任何 BaseLanguageModel 子类）
        idx: 输入 token IDs，形状为 (batch, seq_len)
        max_new_tokens: 最大生成 token 数
        context_size: 上下文窗口大小
    Returns:
        生成的 token IDs，形状为 (batch, seq_len + max_new_tokens)
    """
    for _ in range(max_new_tokens):
        # 防止超出模型上下文窗口
        idx_context = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_context)

        # 只取最后一个时间步的 logits
        logits = logits[:, -1, :]

        # 贪婪解码：选择概率最高的 token
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=1, keepdim=True)

        # 拼接到序列
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """
    高级文本生成函数，支持 temperature 和 top-k 采样
    Args:
        model: 语言模型（任何 BaseLanguageModel 子类）
        idx: 输入 token IDs，形状为 (batch, seq_len)
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
            # 数值稳定性处理
            logits = logits - logits.max(dim=-1, keepdim=True).values
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # 检查是否遇到结束符
        if eos_id is not None and (idx_next == eos_id).all():
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx
