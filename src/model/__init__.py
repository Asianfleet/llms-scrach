"""
模型模块
支持多种语言模型架构：GPT-2, Qwen3

使用示例:
    # 方式1：使用工厂函数（推荐）
    from src.model import create_model
    model, config = create_model("gpt2", "124M")

    # 方式2：使用模型特定模块
    from src.model.gpt2 import create_pretrained_gpt2
    model, config = create_pretrained_gpt2("gpt2-small (124M)")

    # 方式3：使用注册表
    from src.model.registry import create_model, list_available_models
    print(list_available_models())
    model, config = create_model("qwen3", "0.6B", repo_id="Qwen/Qwen3-0.6B")
"""
import torch
from typing import Optional, Tuple, Dict, Any

# 导入基础类
from src.model.base import BaseLanguageModel, ModelInfo

# 导入注册表
from src.model.registry import (
    register_model,
    get_model_info as get_registered_model_info,
    list_available_models,
    create_model,
    build_config,
    is_model_registered,
)

# 导入 GPT-2 模块
from src.model.gpt2 import (
    GPT2,
    GPT2_MODEL_CONFIGS,
    build_gpt2_config,
    generate_text_simple,
    generate,
    create_pretrained_gpt2,
    load_gpt2_from_checkpoint,
)

# 导入 Qwen3 模块
from src.model.qwen3 import (
    Qwen3Model,
    Qwen3Tokenizer,
    QWEN3_MODEL_CONFIGS,
    build_qwen3_config,
    create_pretrained_qwen3,
    create_qwen3_for_training,
)

# 导入通用组件
from src.model.common import RMSNorm, LayerNorm


# ============== 模型注册 ==============

def _register_all_models():
    """注册所有模型到注册表"""

    # 注册 GPT-2
    if not is_model_registered("gpt2"):
        register_model("gpt2", ModelInfo(
            name="gpt2",
            description="OpenAI GPT-2 模型",
            model_class=GPT2,
            config_builder=lambda size: build_gpt2_config(f"gpt2-small ({size})" if "gpt2" not in size else size),
            factory=lambda size, **kwargs: create_pretrained_gpt2(
                f"gpt2-small ({size})" if "gpt2" not in size else size,
                **kwargs
            ),
        ))

    # 注册 Qwen3
    if not is_model_registered("qwen3"):
        register_model("qwen3", ModelInfo(
            name="qwen3",
            description="阿里巴巴 Qwen3 模型",
            model_class=Qwen3Model,
            config_builder=build_qwen3_config,
            factory=lambda size, **kwargs: create_pretrained_qwen3(size, **kwargs),
            tokenizer_class=Qwen3Tokenizer,
        ))


# 初始化注册
_register_all_models()


# ============== 便捷工厂函数 ==============

def create_model_for_training(
    model_type: str,
    model_size: str,
    device: torch.device,
    pretrained: bool = False,
    **kwargs
) -> Tuple[BaseLanguageModel, Dict[str, Any]]:
    """
    创建用于训练的模型
    Args:
        model_type: 模型类型，如 "gpt2", "qwen3"
        model_size: 模型规格，如 "124M", "0.6B"
        device: 计算设备
        pretrained: 是否加载预训练权重
        **kwargs: 额外参数
    Returns:
        model: 模型实例
        config: 配置字典
    """
    if model_type == "gpt2":
        if pretrained:
            model, config = create_pretrained_gpt2(f"gpt2-small ({model_size})", **kwargs)
        else:
            from src.model.gpt2.factory import create_gpt2_for_training
            model, config = create_gpt2_for_training(f"gpt2-small ({model_size})", device)
    elif model_type == "qwen3":
        if pretrained:
            model, config = create_pretrained_qwen3(model_size, **kwargs)
        else:
            model, config = create_qwen3_for_training(model_size, device)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    if not pretrained:
        model.to(device)

    return model, config


def load_model_from_checkpoint(
    model_type: str,
    model_size: str,
    checkpoint_path: str,
    map_location: str = "cpu"
) -> Tuple[BaseLanguageModel, Dict[str, Any], Dict[str, Any]]:
    """
    从检查点加载模型
    Args:
        model_type: 模型类型
        model_size: 模型规格
        checkpoint_path: 检查点路径
        map_location: 设备映射
    Returns:
        model: 模型实例
        config: 配置字典
        checkpoint: 完整检查点数据
    """
    if model_type == "gpt2":
        return load_gpt2_from_checkpoint(f"gpt2-small ({model_size})", checkpoint_path, map_location)
    elif model_type == "qwen3":
        from src.model.qwen3.factory import load_qwen3_from_checkpoint
        return load_qwen3_from_checkpoint(model_size, checkpoint_path, map_location)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


# ============== 导出列表 ==============

__all__ = [
    # 基础类
    "BaseLanguageModel",
    "ModelInfo",

    # 注册表函数
    "create_model",
    "build_config",
    "list_available_models",
    "get_registered_model_info",
    "is_model_registered",

    # 便捷工厂函数
    "create_model_for_training",
    "load_model_from_checkpoint",

    # GPT-2
    "GPT2",
    "GPT2_MODEL_CONFIGS",
    "build_gpt2_config",
    "create_pretrained_gpt2",
    "load_gpt2_from_checkpoint",

    # Qwen3
    "Qwen3Model",
    "Qwen3Tokenizer",
    "QWEN3_MODEL_CONFIGS",
    "build_qwen3_config",
    "create_pretrained_qwen3",
    "create_qwen3_for_training",

    # 通用组件
    "generate_text_simple",
    "generate",
    "RMSNorm",
    "LayerNorm",
]
