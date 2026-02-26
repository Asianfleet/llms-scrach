"""
GPT2 预训练入口
整合数据下载、数据划分、DataLoader 创建、模型训练
"""
import os
import requests
import tiktoken
import torch

from src.config import GPT_CONFIG_124M
from src.data import create_dataloader_v1
from src.model import GPT2
from src.training import train_model_simple

# 获取训练文本
FILE_PATH = "data/the-verdict.txt"
URL = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

if not os.path.exists(FILE_PATH):
    response = requests.get(URL, timeout=30)
    response.raise_for_status()
    text_data = response.text
    with open(FILE_PATH, "w", encoding="utf-8") as f:
        f.write(text_data)
else:
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        text_data = f.read()

# 90% 作为训练集，10% 作为验证集
TRAIN_RATIO = 0.90
split_idx = int(TRAIN_RATIO * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

# 设备选择
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    major, minor = map(int, torch.__version__.split(".")[:2])
    if (major, minor) >= (2, 9):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
else:
    device = torch.device("cpu")

print(f"Using {device} device.")

# 初始化
torch.manual_seed(123)
tokenizer = tiktoken.get_encoding("gpt2")

train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

# 统计信息
total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("Characters:", total_characters)
print("Tokens:", total_tokens)

# Sanity check
if total_tokens * TRAIN_RATIO < GPT_CONFIG_124M["context_length"]:
    print("Not enough tokens for the training loader. "
          "Try to lower the `GPT_CONFIG_124M['context_length']` or "
          "increase the `training_ratio`")

if total_tokens * (1 - TRAIN_RATIO) < GPT_CONFIG_124M["context_length"]:
    print("Not enough tokens for the validation loader. "
          "Try to lower the `GPT_CONFIG_124M['context_length']` or "
          "decrease the `training_ratio`")

train_tokens = sum(input_batch.numel() for input_batch, _ in train_loader)
val_tokens = sum(input_batch.numel() for input_batch, _ in val_loader)
print("Training tokens:", train_tokens)
print("Validation tokens:", val_tokens)
print("All tokens:", train_tokens + val_tokens)

# 模型与优化器
torch.manual_seed(123)
model = GPT2(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

# 训练
num_epochs = 10
train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)
