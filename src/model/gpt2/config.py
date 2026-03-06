"""
GPT-2 模型配置
"""
from typing import Dict, Any

# 基础配置
GPT2_BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
}

# 不同规格的配置
GPT2_MODEL_CONFIGS = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}


def build_gpt2_config(model_name: str) -> Dict[str, Any]:
    """
    构建指定 GPT2 规格的模型配置
    Args:
        model_name: 模型名称，如 "gpt2-small (124M)"
    Returns:
        完整配置字典
    """
    if model_name not in GPT2_MODEL_CONFIGS:
        raise ValueError(f"Unsupported model name: {model_name}")

    cfg = dict(GPT2_BASE_CONFIG)
    cfg.update(GPT2_MODEL_CONFIGS[model_name])
    return cfg
