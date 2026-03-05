import torch
import tiktoken
from src.data import (
    create_instruction_dataloaders,
    format_instruction_input,
    split_instruction_data,
)
from src.model import create_pretrained_gpt2
from src.training import train_model_simple
from src.utils import download_and_load_json, get_default_device

CHOOSE_MODEL = "gpt2-medium (355M)"
DATA_FILE_PATH = "data/instruction-data.json"
DATA_URL = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)


def main():
    data = download_and_load_json(DATA_FILE_PATH, DATA_URL)
    print("Number of entries:", len(data))

    train_data, test_data, val_data = split_instruction_data(data)

    device = get_default_device()
    print("Device:", device)

    tokenizer = tiktoken.get_encoding("gpt2")
    train_loader, val_loader, _ = create_instruction_dataloaders(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        tokenizer=tokenizer,
        device=device,
        batch_size=8,
        num_workers=0,
        allowed_max_length=1024,
    )

    model, _ = create_pretrained_gpt2(CHOOSE_MODEL, models_dir="weights/gpt2")
    model.to(device)

    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)
    train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=2,
        eval_freq=5,
        eval_iter=5,
        start_context=format_instruction_input(val_data[0]),
        tokenizer=tokenizer,
        checkpoint_path="weights/model_and_optimizer_finetune.pth",
    )


if __name__ == "__main__":
    main()