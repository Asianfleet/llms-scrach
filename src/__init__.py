"""
LLMs-from-scratch 项目
支持多种模型架构的预训练和微调
"""
from src.config import (
    build_config,
    list_available_models,
    get_model_info,
    GPT_CONFIG_124M,
    GPT2_MODEL_CONFIGS,
    QWEN3_MODEL_CONFIGS,
)

__version__ = "0.2.0"

__all__ = [
    "build_config",
    "list_available_models",
    "get_model_info",
    "GPT_CONFIG_124M",
    "GPT2_MODEL_CONFIGS",
    "QWEN3_MODEL_CONFIGS",
]
