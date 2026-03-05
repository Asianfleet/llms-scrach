import json

from src.data import format_instruction_input, split_instruction_data
from src.evaluation import create_qwen_client, run_qwen_judge
from src.utils import download_and_load_json

DATA_WITH_RESPONSE_PATH = "data/instruction-data-with-response.json"
DATA_WITH_RESPONSE_URL = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data-with-response.json"
)


def load_api_key(config_path="config.json") -> str:
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    return config["OPENAI_API_KEY"]


def build_judge_prompt(entry: dict) -> str:
    return (
        f"Given the input `{format_instruction_input(entry)}` "
        f"and correct output `{entry['output']}`, "
        f"score the model response `{entry['model_response']}` "
        "on a scale from 0 to 100, where 100 is the best score. "
    )


def main():
    api_key = load_api_key()
    client = create_qwen_client(api_key)

    data = download_and_load_json(DATA_WITH_RESPONSE_PATH, DATA_WITH_RESPONSE_URL)
    _, test_data, _ = split_instruction_data(data)

    for entry in test_data[:3]:
        prompt = build_judge_prompt(entry)
        print("\nDataset response:")
        print(">>", entry["output"])
        print("\nModel response:")
        print(">>", entry["model_response"])
        print("\nScore:")
        print(">>", run_qwen_judge(prompt, client))
        print("\n-------------------------")


if __name__ == "__main__":
    main()