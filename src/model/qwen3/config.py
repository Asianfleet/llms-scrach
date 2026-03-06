"""
Qwen3 模型配置
"""
from typing import Dict, Any
import torch

# 0.6 billion parameters
QWEN3_CONFIG_06_B = {
    "vocab_size": 151_936,
    "context_length": 40_960,
    "emb_dim": 1024,
    "n_heads": 16,
    "n_layers": 28,
    "hidden_dim": 3072,
    "head_dim": 128,
    "qk_norm": True,
    "n_kv_groups": 8,
    "rope_base": 1_000_000.0,
    "dtype": torch.bfloat16,
}

# 1.7 billion parameters
QWEN3_CONFIG_1_7B = {
    "vocab_size": 151_936,
    "context_length": 40_960,
    "emb_dim": 2048,
    "n_heads": 16,
    "n_layers": 28,
    "hidden_dim": 6144,
    "head_dim": 128,
    "qk_norm": True,
    "n_kv_groups": 8,
    "rope_base": 1_000_000.0,
    "dtype": torch.bfloat16,
}

# 4 billion parameters
QWEN3_CONFIG_4B = {
    "vocab_size": 151_936,
    "context_length": 40_960,
    "emb_dim": 2560,
    "n_heads": 32,
    "n_layers": 36,
    "hidden_dim": 9728,
    "head_dim": 128,
    "qk_norm": True,
    "n_kv_groups": 8,
    "rope_base": 1_000_000.0,
    "dtype": torch.bfloat16,
}

# 8 billion parameters
QWEN3_CONFIG_8B = {
    "vocab_size": 151_936,
    "context_length": 40_960,
    "emb_dim": 4096,
    "n_heads": 32,
    "n_layers": 36,
    "hidden_dim": 12288,
    "head_dim": 128,
    "qk_norm": True,
    "n_kv_groups": 8,
    "rope_base": 1_000_000.0,
    "dtype": torch.bfloat16,
}

# 14 billion parameters
QWEN3_CONFIG_14B = {
    "vocab_size": 151_936,
    "context_length": 40_960,
    "emb_dim": 5120,
    "n_heads": 40,
    "n_layers": 40,
    "hidden_dim": 17408,
    "head_dim": 128,
    "qk_norm": True,
    "n_kv_groups": 8,
    "rope_base": 1_000_000.0,
    "dtype": torch.bfloat16,
}

# 32 billion parameters
QWEN3_CONFIG_32B = {
    "vocab_size": 151_936,
    "context_length": 40_960,
    "emb_dim": 5120,
    "n_heads": 64,
    "n_layers": 64,
    "hidden_dim": 25600,
    "head_dim": 128,
    "qk_norm": True,
    "n_kv_groups": 8,
    "rope_base": 1_000_000.0,
    "dtype": torch.bfloat16,
}

# Mixture of Experts Model
QWEN3_CONFIG_30B_A3B = {
    "vocab_size": 151_936,
    "context_length": 262_144,
    "emb_dim": 2048,
    "n_heads": 32,
    "n_layers": 48,
    "head_dim": 128,
    "qk_norm": True,
    "n_kv_groups": 4,
    "rope_base": 10_000_000.0,
    "dtype": torch.bfloat16,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 768,
}

# 配置映射字典
QWEN3_MODEL_CONFIGS = {
    "0.6B": QWEN3_CONFIG_06_B,
    "1.7B": QWEN3_CONFIG_1_7B,
    "4B": QWEN3_CONFIG_4B,
    "8B": QWEN3_CONFIG_8B,
    "14B": QWEN3_CONFIG_14B,
    "32B": QWEN3_CONFIG_32B,
    "30B-A3B": QWEN3_CONFIG_30B_A3B,
}


def build_qwen3_config(model_size: str) -> Dict[str, Any]:
    """
    构建 Qwen3 模型配置
    Args:
        model_size: 模型规格，如 "0.6B", "1.7B", "8B", "30B-A3B"
    Returns:
        配置字典
    """
    if model_size not in QWEN3_MODEL_CONFIGS:
        raise ValueError(f"不支持的 Qwen3 模型规格: {model_size}。"
                        f"可用规格: {list(QWEN3_MODEL_CONFIGS.keys())}")
    return dict(QWEN3_MODEL_CONFIGS[model_size])
