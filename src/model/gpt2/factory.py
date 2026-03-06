"""
GPT-2 模型工厂
"""
import torch

from src.model.gpt2.config import build_gpt2_config
from src.model.gpt2.model import GPT2
from src.model.gpt2.weights import load_weights_into_gpt


def create_pretrained_gpt2(model_name: str, models_dir="weights/gpt2"):
    """
    下载并加载预训练 GPT2 权重
    Args:
        model_name: 模型名称，如 "gpt2-small (124M)"
        models_dir: 权重保存目录
    Returns:
        model: GPT2 模型实例
        config: 模型配置字典
    """
    # 延迟导入避免循环依赖
    from src.model.gpt_download import download_and_load_gpt2

    model_size = model_name.split(" ")[-1].lstrip("(").rstrip(")")
    _, params = download_and_load_gpt2(model_size=model_size, models_dir=models_dir)

    config = build_gpt2_config(model_name)
    model = GPT2(config)
    load_weights_into_gpt(model, params)
    return model, config


def load_gpt2_from_checkpoint(model_name: str, checkpoint_path: str, map_location="cpu"):
    """
    从 checkpoint 恢复 GPT2 模型
    Args:
        model_name: 模型名称
        checkpoint_path: 检查点文件路径
        map_location: 设备映射
    Returns:
        model: GPT2 模型实例
        config: 模型配置字典
        checkpoint: 完整检查点数据
    """
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=map_location)
    config = build_gpt2_config(model_name)
    model = GPT2(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, config, checkpoint


def create_gpt2_for_training(model_name: str, device: torch.device):
    """
    创建用于训练的 GPT-2 模型（随机初始化）
    Args:
        model_name: 模型名称
        device: 计算设备
    Returns:
        model: 初始化后的 GPT2 模型
        config: 模型配置字典
    """
    config = build_gpt2_config(model_name)
    model = GPT2(config)
    model.to(device)
    return model, config
