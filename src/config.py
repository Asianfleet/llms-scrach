# 统一配置，避免在多个模块中重复定义

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,  # 与训练保持一致
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

GPT2_BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True,
}

GPT2_MODEL_CONFIGS = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}


def build_gpt2_config(model_name: str) -> dict:
    """构建指定 GPT2 规格的模型配置。"""
    if model_name not in GPT2_MODEL_CONFIGS:
        raise ValueError(f"Unsupported model name: {model_name}")

    cfg = dict(GPT2_BASE_CONFIG)
    cfg.update(GPT2_MODEL_CONFIGS[model_name])
    return cfg
