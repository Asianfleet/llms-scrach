import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):

    def __init__(self, txt: str, tokenizer, max_length, stride) -> None:
        self.input_ids = []
        self.target_ids = []

        # 对输入文本进行分词, 得到 token 编号列表
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        assert len(token_ids) > max_length, "Number of tokenized inputs must at least be equal to max_length+1"

        # 滑动窗口, stride 控制步长
        for i in range(0, len(token_ids) - max_length, stride):
            # 截取长度为 max_length 的文本作为输入
            input_chunk = token_ids[i:i + max_length]
            # 输入右移一位作为输出目标
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]


def create_dataloader_v1(
    txt, 
    batch_size=4, 
    max_length=256,
    stride=128, 
    shuffle=True, 
    drop_last=True,
    num_workers=0
):
    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader

if __name__ == "__main__":
    txt = "Hello, world! This is a test text."
    dataloader = create_dataloader_v1(txt, batch_size=2, max_length=2, stride=1)
    for input_ids, target_ids in dataloader:
        print(input_ids)
        print(target_ids)
        break
