"""
# ==============================================================================
# RoPE implementation summary
#
#
# There are two common styles to implement RoPE, which are
# mathematically equivalent;
# they mainly differ in how the rotation matrix pairs dimensions.
#
# 1) Split-halves style (this repo, Hugging Face Transformers):
#
#   For hidden dim d = 8 (example):
#
#       [ x0   x1   x2   x3   x4   x5   x6   x7 ]
#         │    │    │    │    │    │    │    │
#         ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼
#        cos  cos  cos  cos  sin  sin  sin  sin
#
#   Rotation matrix:
#
#       [ cosθ   -sinθ    0      0   ... ]
#       [ sinθ    cosθ    0      0   ... ]
#       [  0       0    cosθ   -sinθ ... ]
#       [  0       0    sinθ    cosθ ... ]
#        ...
#
#   Here, the embedding dims are split into two halves and then
#   each one is rotated in blocks.
#
#
# 2) Interleaved (even/odd) style (original paper, Llama repo):
#
#   For hidden dim d = 8 (example):
#
#       [ x0   x1   x2   x3   x4   x5   x6   x7 ]
#         │    │    │    │    │    │    │    │
#         ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼
#        cos  sin  cos  sin  cos  sin  cos  sin
#
#   Rotation matrix:
#       [ cosθ  -sinθ    0      0   ... 
"""
import torch


def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):
    """
    计算 RoPE 参数
    Args:
        head_dim: 注意力头维度
        theta_base: RoPE theta 基数
        context_length: 上下文长度
        dtype: 数据类型
    Returns:
        cos, sin: 旋转位置编码的余弦和正弦值
    """
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # 计算逆频率
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))

    # 生成位置索引
    positions = torch.arange(context_length, dtype=dtype)

    # 计算角度
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)  # Shape: (context_length, head_dim // 2)

    # 扩展角度以匹配 head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # 预计算正弦和余弦
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(x, cos, sin):
    """
    应用 RoPE 旋转位置编码
    Args:
        x: 输入张量，形状为 (batch_size, num_heads, seq_len, head_dim)
        cos: 余弦值
        sin: 正弦值
    Returns:
        旋转后的张量
    """
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # 将 x 分成前半部分和后半部分
    x1 = x[..., : head_dim // 2]  # 前半部分
    x2 = x[..., head_dim // 2:]  # 后半部分

    # 调整 sin 和 cos 的形状
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # 应用旋转变换
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    # 应用 cos 和 sin 旋转后可以使用低精度
    return x_rotated.to(dtype=x.dtype)
