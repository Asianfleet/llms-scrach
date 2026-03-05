import torch


def get_default_device() -> torch.device:
    """返回当前环境推荐的计算设备。"""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        major, minor = map(int, torch.__version__.split(".")[:2])
        if (major, minor) >= (2, 9):
            return torch.device("mps")

    return torch.device("cpu")
