import json
import torch
import tiktoken
from tqdm import tqdm

from src.data import format_instruction_input, split_instruction_data
from src.model import generate, load_gpt2_from_checkpoint
from src.training import text2token_ids, token_ids2text
from src.utils import download_and_load_json, get_default_device

CHOOSE_MODEL = "gpt2-medium (355M)"
DATA_FILE_PATH = "data/instruction-data.json"
DATA_URL = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)
FINETUNE_CHECKPOINT = "weights/model_and_optimizer_finetune.pth"
OUTPUT_PATH = "data/instruction-data-with-response.json"


def main():
    device = get_default_device()

    model, config, _ = load_gpt2_from_checkpoint(
        model_name=CHOOSE_MODEL,
        checkpoint_path=FINETUNE_CHECKPOINT,
        map_location=device,
    )
    model.to(device)

    data = download_and_load_json(DATA_FILE_PATH, DATA_URL)
    _, test_data, _ = split_instruction_data(data)
    tokenizer = tiktoken.get_encoding("gpt2")

    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
        input_text = format_instruction_input(entry)
        token_ids = generate(
            model=model,
            idx=text2token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=config["context_length"],
            eos_id=50256,
        )
        generated_text = token_ids2text(token_ids, tokenizer)
        response_text = generated_text[len(input_text):].replace("### Response:", "").strip()
        test_data[i]["model_response"] = response_text

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(test_data, file, indent=4)


if __name__ == "__main__":
    main()