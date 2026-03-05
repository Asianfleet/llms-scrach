import json
import os

import requests


def download_and_load_json(file_path: str, url: str) -> list | dict:
    """若本地不存在则下载 JSON 文件，并返回解析结果。"""
    if not os.path.exists(file_path):
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(response.text)

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)
