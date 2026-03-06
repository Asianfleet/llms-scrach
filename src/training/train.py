"""
训练模块
支持多种模型架构的训练
"""
import torch
import torch.nn.functional as F
import numpy as np

from src.model.common.generation import generate_text_simple
from src.model.base import BaseLanguageModel


def text2token_ids(text, tokenizer):
    """将文本转换为 token 编号列表"""
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids2text(token_ids, tokenizer):
    """将 token 编号列表转换为文本"""
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    """计算单个批量损失"""
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """计算模型在整个数据集的平均损失"""
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """评估模型在训练集和验证集的损失"""
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def get_context_size(model):
    """
    获取模型的上下文大小
    兼容 GPT-2 和 Qwen3 等模型架构
    """
    if hasattr(model, 'cfg'):
        # Qwen3 等模型
        return model.cfg.get('context_length', 1024)
    elif hasattr(model, 'pos_emb'):
        # GPT-2
        return model.pos_emb.weight.shape[0]
    else:
        # 默认
        return 1024


def generate_and_print_sample(model, tokenizer, device, start_context):
    """生成并打印样本"""
    model.eval()
    context_size = get_context_size(model)

    # 处理 tokenizer（兼容 GPT-2 tiktoken 和 Qwen3 自定义 tokenizer）
    if hasattr(tokenizer, 'encode'):
        # 标准 encode 方法（Qwen3Tokenizer）
        if isinstance(start_context, str):
            encoded_text = start_context
        else:
            encoded_text = start_context
        encoded = torch.tensor([tokenizer.encode(encoded_text)]).to(device)
    else:
        # tiktoken（GPT-2）
        encoded = text2token_ids(start_context, tokenizer).to(device)

    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )

    # 解码
    if hasattr(tokenizer, 'decode'):
        decoded_text = tokenizer.decode(token_ids.squeeze(0).tolist())
    else:
        decoded_text = token_ids2text(token_ids, tokenizer)

    print(decoded_text.replace("\n", " "))  # 紧凑打印格式
    model.train()


# ============== 权重加载（保持向后兼容）=============

def load_weights_into_gpt(gpt, params):
    """
    将预训练权重加载到 GPT-2 模型
    此函数保留用于向后兼容，新代码应使用 src.model.gpt2.weights.load_weights_into_gpt
    """
    from src.model.gpt2.weights import load_weights_into_gpt as _load_weights
    _load_weights(gpt, params)


# ============== 训练函数 ==============

def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
    checkpoint_path="weights/model_and_optimizer.pth",
):
    """
    简单的模型训练函数
    Args:
        model: 语言模型（BaseLanguageModel 子类）
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 优化器
        device: 计算设备
        num_epochs: 训练轮数
        eval_freq: 评估频率（步数）
        eval_iter: 每次评估的批次数
        start_context: 用于生成样本的起始文本
        tokenizer: 分词器
        checkpoint_path: 检查点保存路径
    Returns:
        train_losses, val_losses, track_tokens_seen
    """
    # 初始化损失列表和已见 token 数量
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen = input_batch.numel()
            global_step += 1

            # 每 eval_freq 步评估模型
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # 每轮结束后生成样本
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    # 保存检查点
    if checkpoint_path:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            checkpoint_path,
        )

    return train_losses, val_losses, track_tokens_seen


def train_model_with_scheduler(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
    checkpoint_path="weights/model_and_optimizer.pth",
):
    """
    带学习率调度器的训练函数
    """
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            scheduler.step()

            tokens_seen = input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}, "
                      f"LR {scheduler.get_last_lr()[0]:.6f}")

        generate_and_print_sample(model, tokenizer, device, start_context)

    if checkpoint_path:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            checkpoint_path,
        )

    return train_losses, val_losses, track_tokens_seen
