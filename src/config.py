# 统一配置，避免在多个模块中重复定义

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,  # 与训练保持一致
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}
