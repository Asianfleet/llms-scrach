import torch

from src.config import build_gpt2_config
from src.model.gpt2 import GPT2
from src.model.gpt_download import download_and_load_gpt2
from src.training.train import load_weights_into_gpt


def create_pretrained_gpt2(model_name: str, models_dir="weights/gpt2"):
    """下载并加载预训练 GPT2 权重。"""
    model_size = model_name.split(" ")[-1].lstrip("(").rstrip(")")
    _, params = download_and_load_gpt2(model_size=model_size, models_dir=models_dir)

    config = build_gpt2_config(model_name)
    model = GPT2(config)
    load_weights_into_gpt(model, params)
    return model, config


def load_gpt2_from_checkpoint(model_name: str, checkpoint_path: str, map_location="cpu"):
    """从 checkpoint 恢复 GPT2 模型。"""
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=map_location)
    config = build_gpt2_config(model_name)
    model = GPT2(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, config, checkpoint
