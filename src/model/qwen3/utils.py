"""
Qwen3 工具函数
"""
import os
import json
import requests
from pathlib import Path


def download_from_huggingface(repo_id, filename, local_dir, revision="main"):
    """
    从 HuggingFace 下载单个文件
    Args:
        repo_id: 仓库 ID
        filename: 文件名
        local_dir: 本地保存目录
        revision: 版本分支
    Returns:
        文件本地路径
    """
    base_url = "https://huggingface.co"
    url = f"{base_url}/{repo_id}/resolve/{revision}/{filename}"
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    dest_path = os.path.join(local_dir, filename)

    if os.path.exists(dest_path):
        print(f"File already exists: {dest_path}")
    else:
        print(f"Downloading {url} to {dest_path}...")
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    return dest_path


def download_from_huggingface_from_snapshots(repo_id, local_dir):
    """
    从 HuggingFace 下载完整模型（支持多文件分片）
    Args:
        repo_id: 仓库 ID
        local_dir: 本地保存目录
    Returns:
        权重字典
    """
    try:
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
    except ImportError:
        raise ImportError("需要安装 huggingface_hub 和 safetensors: pip install huggingface_hub safetensors")

    repo_dir = snapshot_download(repo_id=repo_id, local_dir=local_dir)

    index_path = os.path.join(repo_dir, "model.safetensors.index.json")
    single_file_path = os.path.join(repo_dir, "model.safetensors")

    if os.path.exists(index_path):
        # 多文件分片模型
        with open(index_path, "r") as f:
            index = json.load(f)

        weights_dict = {}
        for filename in set(index["weight_map"].values()):
            shard_path = os.path.join(repo_dir, filename)
            shard = load_file(shard_path)
            weights_dict.update(shard)
    elif os.path.exists(single_file_path):
        # 单文件模型
        weights_dict = load_file(single_file_path)
    else:
        raise FileNotFoundError("No model.safetensors or model.safetensors.index.json found.")

    return weights_dict
