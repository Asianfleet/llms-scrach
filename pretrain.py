"""
语言模型预训练入口
支持多种模型架构：GPT-2, Qwen3

使用方法:
    # GPT-2 预训练（默认）
    python pretrain.py

    # 指定模型类型和规格
    python pretrain.py --model_type gpt2 --model_size 124M

    # Qwen3 预训练
    python pretrain.py --model_type qwen3 --model_size 0.6B
"""
import os
import sys
import argparse
import requests
import torch

# 根据模型类型选择 tokenizer
import tiktoken

from src.config import build_config, GPT_CONFIG_124M
from src.data import create_dataloader_v1
from src.model import create_model_for_training, list_available_models
from src.training import train_model_simple
from src.utils import get_default_device

# 获取训练文本
FILE_PATH = "data/the-verdict.txt"
URL = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

# 训练参数
TRAIN_RATIO = 0.90  # 90% 作为训练集，10% 作为验证集
BATCH_SIZE = 2
NUM_EPOCHS = 10
LEARNING_RATE = 0.0004
WEIGHT_DECAY = 0.1
EVAL_FREQ = 5
EVAL_ITER = 5


def download_data():
    """下载训练数据"""
    if not os.path.exists(FILE_PATH):
        print(f"Downloading data from {URL}...")
        os.makedirs(os.path.dirname(FILE_PATH), exist_ok=True)
        response = requests.get(URL, timeout=30)
        response.raise_for_status()
        with open(FILE_PATH, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"Data saved to {FILE_PATH}")
    else:
        print(f"Using existing data: {FILE_PATH}")

    with open(FILE_PATH, "r", encoding="utf-8") as f:
        return f.read()


def get_tokenizer(model_type: str):
    """根据模型类型获取 tokenizer"""
    if model_type == "gpt2":
        return tiktoken.get_encoding("gpt2")
    elif model_type == "qwen3":
        # Qwen3 使用自定义 tokenizer，这里返回一个包装函数
        # 实际使用时需要提供 tokenizer.json 文件
        try:
            from src.model.qwen3 import Qwen3Tokenizer
            # 尝试加载本地 tokenizer
            if os.path.exists("weights/qwen3/tokenizer.json"):
                return Qwen3Tokenizer("weights/qwen3/tokenizer.json", apply_chat_template=False)
            else:
                print("Warning: Qwen3 tokenizer.json not found, falling back to GPT-2 tokenizer")
                return tiktoken.get_encoding("gpt2")
        except Exception as e:
            print(f"Warning: Failed to load Qwen3 tokenizer: {e}")
            print("Falling back to GPT-2 tokenizer")
            return tiktoken.get_encoding("gpt2")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def main():
    parser = argparse.ArgumentParser(description="语言模型预训练")
    parser.add_argument(
        "--model_type",
        type=str,
        default="gpt2",
        choices=["gpt2", "qwen3"],
        help="模型类型: gpt2 或 qwen3"
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="124M",
        help="模型规格，如 gpt2: 124M, 355M; qwen3: 0.6B, 1.7B, 4B, 8B"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help=f"批量大小 (默认: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=NUM_EPOCHS,
        help=f"训练轮数 (默认: {NUM_EPOCHS})"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=LEARNING_RATE,
        help=f"学习率 (默认: {LEARNING_RATE})"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="weights/model_and_optimizer.pth",
        help="检查点保存路径"
    )
    parser.add_argument(
        "--list_models",
        action="store_true",
        help="列出所有可用的模型和规格"
    )

    args = parser.parse_args()

    # 列出可用模型
    if args.list_models:
        print("\n可用模型和规格:")
        models = list_available_models()
        for model_type, sizes in models.items():
            print(f"\n  {model_type}:")
            for size in sizes:
                print(f"    - {size}")
        print()
        return

    # 打印配置
    print(f"\n{'='*60}")
    print(f"模型类型: {args.model_type}")
    print(f"模型规格: {args.model_size}")
    print(f"批量大小: {args.batch_size}")
    print(f"训练轮数: {args.num_epochs}")
    print(f"学习率: {args.learning_rate}")
    print(f"{'='*60}\n")

    # 下载数据
    text_data = download_data()

    # 划分数据集
    split_idx = int(TRAIN_RATIO * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    # 设备选择
    device = get_default_device()
    print(f"使用设备: {device}")

    # 初始化
    torch.manual_seed(123)

    # 获取 tokenizer
    tokenizer = get_tokenizer(args.model_type)

    # 获取配置
    config = build_config(args.model_type, args.model_size)
    context_length = config.get("context_length", GPT_CONFIG_124M["context_length"])

    # 创建数据加载器
    train_loader = create_dataloader_v1(
        train_data,
        batch_size=args.batch_size,
        max_length=context_length,
        stride=context_length,
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader_v1(
        val_data,
        batch_size=args.batch_size,
        max_length=context_length,
        stride=context_length,
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    # 统计信息
    if hasattr(tokenizer, 'encode_batch'):
        # Qwen3 风格 tokenizer
        total_tokens = len(tokenizer.encode(text_data))
    else:
        # GPT-2 风格 tokenizer
        total_tokens = len(tokenizer.encode(text_data))

    print(f"字符数: {len(text_data)}")
    print(f"Token 数: {total_tokens}")

    train_tokens = sum(input_batch.numel() for input_batch, _ in train_loader)
    val_tokens = sum(input_batch.numel() for input_batch, _ in val_loader)
    print(f"训练 tokens: {train_tokens}")
    print(f"验证 tokens: {val_tokens}")
    print(f"总 tokens: {train_tokens + val_tokens}\n")

    # 创建模型
    torch.manual_seed(123)
    model, config = create_model_for_training(
        model_type=args.model_type,
        model_size=args.model_size,
        device=device,
        pretrained=False
    )

    print(f"模型参数数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"模型上下文长度: {config.get('context_length', 1024)}\n")

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=WEIGHT_DECAY
    )

    # 训练
    print("开始训练...\n")
    train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=args.num_epochs,
        eval_freq=EVAL_FREQ,
        eval_iter=EVAL_ITER,
        start_context="Every effort moves you",
        tokenizer=tokenizer,
        checkpoint_path=args.checkpoint_path,
    )

    print(f"\n训练完成！检查点保存至: {args.checkpoint_path}")


if __name__ == "__main__":
    main()
