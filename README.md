# LLMs from Scratch - GPT2 预训练

从零实现 GPT2 预训练，用于学习大语言模型的核心组件与训练流程。代码主要来自 [Build a Large Language Model (From Scratch)](https://amzn.to/4fqvn0D) 一书。

## 项目结构

```
llms-scrach/
├── src/
│   ├── __init__.py
│   ├── config.py             # 统一配置
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py        # 数据集与 DataLoader
│   ├── model/
│   │   ├── __init__.py
│   │   ├── attention.py      # 自注意力机制
│   │   └── gpt2.py           # GPT2 模型
│   └── training/
│       ├── __init__.py
│       └── train.py          # 训练循环、评估
├── data/
│   └── the-verdict.txt       # 训练数据
├── main.py                   # 入口
├── requirements.txt
└── README.md
```

## 安装

```bash
pip install -r requirements.txt
```

## 运行

```bash
python main.py
```

程序会自动下载训练数据（the-verdict.txt），划分训练/验证集，并启动 GPT2-124M 的预训练。
