"""
Qwen3 模型工厂
"""
import os
import json
import torch

from src.model.qwen3.config import build_qwen3_config, QWEN3_MODEL_CONFIGS
from src.model.qwen3.model import Qwen3Model
from src.model.qwen3.weights import load_weights_into_qwen


def create_pretrained_qwen3(model_size: str, repo_id: str = None, local_dir: str = "weights/qwen3", **kwargs):
    """
    下载并加载预训练 Qwen3 权重
    Args:
        model_size: 模型规格，如 "0.6B", "1.7B", "8B"
        repo_id: HuggingFace 仓库 ID，如 "Qwen/Qwen3-0.6B"
        local_dir: 权重保存目录
        **kwargs: 额外的参数
    Returns:
        model: Qwen3Model 实例
        config: 模型配置字典
    """
    from src.model.qwen3.utils import download_from_huggingface_from_snapshots

    config = build_qwen3_config(model_size)
    model = Qwen3Model(config)

    if repo_id:
        params = download_from_huggingface_from_snapshots(repo_id, local_dir)
        load_weights_into_qwen(model, config, params)

    return model, config


def load_qwen3_from_checkpoint(model_size: str, checkpoint_path: str, map_location="cpu"):
    """
    从 checkpoint 恢复 Qwen3 模型
    Args:
        model_size: 模型规格
        checkpoint_path: 检查点文件路径
        map_location: 设备映射
    Returns:
        model: Qwen3Model 实例
        config: 模型配置字典
        checkpoint: 完整检查点数据
    """
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=map_location)
    config = build_qwen3_config(model_size)
    model = Qwen3Model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, config, checkpoint


def create_qwen3_for_training(model_size: str, device: torch.device):
    """
    创建用于训练的 Qwen3 模型（随机初始化）
    Args:
        model_size: 模型规格
        device: 计算设备
    Returns:
        model: 初始化后的 Qwen3Model
        config: 模型配置字典
    """
    config = build_qwen3_config(model_size)
    model = Qwen3Model(config)
    model.to(device)
    return model, config
