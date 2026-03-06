"""
语言模型微调入口
支持多种模型架构：GPT-2, Qwen3

使用方法:
    # GPT-2 微调（默认）
    python finetune.py

    # 指定模型类型和规格
    python finetune.py --model_type gpt2 --model_size 355M

    # Qwen3 微调
    python finetune.py --model_type qwen3 --model_size 0.6B --pretrained
"""
import os
import sys
import argparse
import torch
import tiktoken

from src.data import (
    create_instruction_dataloaders,
    format_instruction_input,
    split_instruction_data,
)
from src.model import (
    create_model_for_training,
    load_model_from_checkpoint,
    list_available_models,
)
from src.training import train_model_simple
from src.utils import download_and_load_json, get_default_device

# 默认配置
DEFAULT_CONFIG = {
    "gpt2": {
        "model_size": "355M",
        "batch_size": 8,
        "learning_rate": 0.00005,
        "pretrained": True,
    },
    "qwen3": {
        "model_size": "0.6B",
        "batch_size": 4,
        "learning_rate": 0.0001,
        "pretrained": False,  # Qwen3 需要明确指定下载预训练权重
    }
}

DATA_FILE_PATH = "data/instruction-data.json"
DATA_URL = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)


def get_tokenizer(model_type: str):
    """根据模型类型获取 tokenizer"""
    if model_type == "gpt2":
        return tiktoken.get_encoding("gpt2")
    elif model_type == "qwen3":
        try:
            from src.model.qwen3 import Qwen3Tokenizer
            if os.path.exists("weights/qwen3/tokenizer.json"):
                return Qwen3Tokenizer("weights/qwen3/tokenizer.json", apply_chat_template=True)
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
    parser = argparse.ArgumentParser(description="语言模型微调")
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
        default=None,
        help="模型规格，如 gpt2: 124M, 355M; qwen3: 0.6B, 1.7B"
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="是否加载预训练权重"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="HuggingFace 仓库 ID（用于 Qwen3 下载预训练权重）"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="从检查点恢复模型进行微调"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="批量大小"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=2,
        help="训练轮数 (默认: 2)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="学习率"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="weights/model_and_optimizer_finetune.pth",
        help="输出检查点路径"
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

    # 使用默认配置
    defaults = DEFAULT_CONFIG.get(args.model_type, DEFAULT_CONFIG["gpt2"])
    model_size = args.model_size or defaults["model_size"]
    batch_size = args.batch_size or defaults["batch_size"]
    learning_rate = args.learning_rate or defaults["learning_rate"]
    pretrained = args.pretrained or defaults["pretrained"]

    # 打印配置
    print(f"\n{'='*60}")
    print(f"模型类型: {args.model_type}")
    print(f"模型规格: {model_size}")
    print(f"使用预训练权重: {pretrained}")
    if args.repo_id:
        print(f"HuggingFace 仓库: {args.repo_id}")
    if args.checkpoint_path:
        print(f"检查点路径: {args.checkpoint_path}")
    print(f"批量大小: {batch_size}")
    print(f"训练轮数: {args.num_epochs}")
    print(f"学习率: {learning_rate}")
    print(f"{'='*60}\n")

    # 加载数据
    data = download_and_load_json(DATA_FILE_PATH, DATA_URL)
    print(f"数据条目数: {len(data)}\n")

    train_data, test_data, val_data = split_instruction_data(data)

    # 设备选择
    device = get_default_device()
    print(f"使用设备: {device}\n")

    # 获取 tokenizer
    tokenizer = get_tokenizer(args.model_type)

    # 创建数据加载器
    train_loader, val_loader, _ = create_instruction_dataloaders(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
        num_workers=0,
        allowed_max_length=1024,
    )

    # 加载或创建模型
    if args.checkpoint_path:
        print(f"从检查点恢复: {args.checkpoint_path}")
        model, config, checkpoint = load_model_from_checkpoint(
            model_type=args.model_type,
            model_size=model_size,
            checkpoint_path=args.checkpoint_path,
            map_location=str(device)
        )
    else:
        print(f"创建模型: {args.model_type} ({model_size})")

        # 创建模型时的额外参数
        kwargs = {}
        if args.model_type == "qwen3" and pretrained and args.repo_id:
            kwargs["repo_id"] = args.repo_id
            kwargs["local_dir"] = f"weights/qwen3/{args.repo_id.split('/')[-1]}"

        model, config = create_model_for_training(
            model_type=args.model_type,
            model_size=model_size,
            device=device,
            pretrained=pretrained,
            **kwargs
        )

    model.to(device)
    print(f"模型参数数: {sum(p.numel() for p in model.parameters()):,}\n")

    # 优化器
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.1
    )

    # 准备起始上下文
    if args.model_type == "gpt2":
        start_context = format_instruction_input(val_data[0])
    else:
        # Qwen3 使用自定义格式
        start_context = val_data[0]["instruction"]

    # 训练
    print("开始微调...\n")
    train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=args.num_epochs,
        eval_freq=5,
        eval_iter=5,
        start_context=start_context,
        tokenizer=tokenizer,
        checkpoint_path=args.output_path,
    )

    print(f"\n微调完成！检查点保存至: {args.output_path}")


if __name__ == "__main__":
    main()
