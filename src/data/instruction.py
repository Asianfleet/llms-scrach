from functools import partial

import torch
from torch.utils.data import DataLoader, Dataset


def format_instruction_input(entry: dict) -> str:
    """将一条指令样本格式化为模型输入前缀。"""
    instruction_text = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text


def split_instruction_data(data: list, train_ratio=0.85, test_ratio=0.1):
    """按固定比例切分指令数据集。"""
    train_portion = int(len(data) * train_ratio)
    test_portion = int(len(data) * test_ratio)
    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]
    return train_data, test_data, val_data


class InstructionDataset(Dataset):
    """用于指令微调的数据集类。"""

    def __init__(self, data: list, tokenizer) -> None:
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_instruction_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu",
):
    """将可变长 token 序列整理为训练批次。"""
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))

        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


def create_instruction_dataloaders(
    train_data: list,
    val_data: list,
    test_data: list,
    tokenizer,
    device,
    batch_size=8,
    num_workers=0,
    allowed_max_length=1024,
):
    """创建训练/验证/测试 DataLoader。"""
    collate_fn = partial(
        custom_collate_fn,
        device=device,
        allowed_max_length=allowed_max_length,
    )

    train_loader = DataLoader(
        InstructionDataset(train_data, tokenizer),
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        InstructionDataset(val_data, tokenizer),
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        InstructionDataset(test_data, tokenizer),
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader
