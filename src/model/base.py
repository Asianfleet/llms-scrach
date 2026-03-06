"""
基础模型抽象类，定义所有语言模型的通用接口
"""
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn


class BaseLanguageModel(nn.Module, ABC):
    """
    所有语言模型的基类，定义通用接口
    子类必须实现 forward 方法
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            in_idx: 输入 token IDs，形状为 (batch_size, seq_len)
        Returns:
            logits: 输出 logits，形状为 (batch_size, seq_len, vocab_size)
        """
        pass

    def get_context_length(self) -> int:
        """获取模型上下文长度"""
        return self.cfg.get("context_length", 1024)

    def get_vocab_size(self) -> int:
        """获取词汇表大小"""
        return self.cfg.get("vocab_size", 50257)

    def get_device(self) -> torch.device:
        """获取模型所在设备"""
        return next(self.parameters()).device


class ModelInfo:
    """模型元数据信息"""

    def __init__(
        self,
        name: str,
        description: str,
        model_class: type,
        config_builder: callable,
        factory: callable,
        tokenizer_class: Optional[type] = None,
    ):
        self.name = name
        self.description = description
        self.model_class = model_class
        self.config_builder = config_builder
        self.factory = factory
        self.tokenizer_class = tokenizer_class
