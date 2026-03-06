"""
统一配置系统
支持多种模型架构：GPT-2, Qwen3
"""
from typing import Dict, Any, Optional

# ============== GPT-2 配置 ==============

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


def build_gpt2_config(model_name: str) -> Dict[str, Any]:
    """构建指定 GPT2 规格的模型配置"""
    if model_name not in GPT2_MODEL_CONFIGS:
        raise ValueError(f"Unsupported model name: {model_name}")

    cfg = dict(GPT2_BASE_CONFIG)
    cfg.update(GPT2_MODEL_CONFIGS[model_name])
    return cfg


# ============== Qwen3 配置 ==============

import torch

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
    """构建 Qwen3 模型配置"""
    if model_size not in QWEN3_MODEL_CONFIGS:
        raise ValueError(f"不支持的 Qwen3 模型规格: {model_size}。"
                        f"可用规格: {list(QWEN3_MODEL_CONFIGS.keys())}")
    return dict(QWEN3_MODEL_CONFIGS[model_size])


# ============== 统一配置接口 ==============

MODEL_CONFIGS = {
    "gpt2": GPT2_MODEL_CONFIGS,
    "qwen3": QWEN3_MODEL_CONFIGS,
}


def build_config(model_type: str, model_size: str) -> Dict[str, Any]:
    """
    统一配置构建接口
    Args:
        model_type: 模型类型，如 "gpt2", "qwen3"
        model_size: 模型规格，如 "124M", "0.6B"
    Returns:
        配置字典
    """
    if model_type == "gpt2":
        # GPT-2 使用完整名称，如 "gpt2-small (124M)"
        full_name = f"gpt2-small ({model_size})" if "gpt2" not in model_size else model_size
        return build_gpt2_config(full_name)
    elif model_type == "qwen3":
        return build_qwen3_config(model_size)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


def get_model_info(model_type: str) -> Dict[str, Any]:
    """
    获取模型类型的信息
    Args:
        model_type: 模型类型
    Returns:
        模型信息字典
    """
    if model_type == "gpt2":
        return {
            "name": "GPT-2",
            "description": "OpenAI GPT-2 模型",
            "available_sizes": list(GPT2_MODEL_CONFIGS.keys()),
            "vocab_size": 50257,
        }
    elif model_type == "qwen3":
        return {
            "name": "Qwen3",
            "description": "阿里巴巴 Qwen3 模型",
            "available_sizes": list(QWEN3_MODEL_CONFIGS.keys()),
            "vocab_size": 151_936,
        }
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


def list_available_models() -> Dict[str, list]:
    """列出所有可用的模型及其规格"""
    return {
        "gpt2": list(GPT2_MODEL_CONFIGS.keys()),
        "qwen3": list(QWEN3_MODEL_CONFIGS.keys()),
    }
