"""
模型注册表，管理所有可用的模型架构
"""
from typing import Dict, Optional, Callable, Any
from src.model.base import ModelInfo


class ModelRegistry:
    """
    模型注册表，用于注册和查找模型架构
    使用单例模式确保全局唯一
    """

    _instance = None
    _models: Dict[str, ModelInfo] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def register(self, model_id: str, model_info: ModelInfo) -> None:
        """
        注册新模型
        Args:
            model_id: 模型唯一标识符，如 "gpt2", "qwen3"
            model_info: 模型信息对象
        """
        self._models[model_id] = model_info

    def get(self, model_id: str) -> Optional[ModelInfo]:
        """
        获取模型信息
        Args:
            model_id: 模型唯一标识符
        Returns:
            ModelInfo 或 None
        """
        return self._models.get(model_id)

    def list_models(self) -> Dict[str, str]:
        """
        列出所有注册的模型
        Returns:
            {model_id: description} 字典
        """
        return {k: v.description for k, v in self._models.items()}

    def is_registered(self, model_id: str) -> bool:
        """检查模型是否已注册"""
        return model_id in self._models

    def create_model(self, model_id: str, model_size: str, **kwargs):
        """
        创建模型实例
        Args:
            model_id: 模型类型标识符
            model_size: 模型规格，如 "124M", "1.7B"
            **kwargs: 额外的创建参数
        Returns:
            模型实例
        """
        model_info = self.get(model_id)
        if model_info is None:
            raise ValueError(f"未注册的模型类型: {model_id}")
        return model_info.factory(model_size, **kwargs)

    def build_config(self, model_id: str, model_size: str) -> Dict[str, Any]:
        """
        构建模型配置
        Args:
            model_id: 模型类型标识符
            model_size: 模型规格
        Returns:
            配置字典
        """
        model_info = self.get(model_id)
        if model_info is None:
            raise ValueError(f"未注册的模型类型: {model_id}")
        return model_info.config_builder(model_size)


# 全局注册表实例
_registry = ModelRegistry()


def register_model(model_id: str, model_info: ModelInfo) -> None:
    """全局注册模型函数"""
    _registry.register(model_id, model_info)


def get_model_info(model_id: str) -> Optional[ModelInfo]:
    """全局获取模型信息函数"""
    return _registry.get(model_id)


def list_available_models() -> Dict[str, str]:
    """列出所有可用模型"""
    return _registry.list_models()


def create_model(model_id: str, model_size: str, **kwargs):
    """全局创建模型函数"""
    return _registry.create_model(model_id, model_size, **kwargs)


def build_config(model_id: str, model_size: str) -> Dict[str, Any]:
    """全局构建配置函数"""
    return _registry.build_config(model_id, model_size)


def is_model_registered(model_id: str) -> bool:
    """检查模型是否已注册"""
    return _registry.is_registered(model_id)
