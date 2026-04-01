from torch.utils.data import Dataset
from typing import List, Callable
from src.datasets.schemas import Sample
from src.prompting.grpo_prompt_builder import encode_grpo_prompt
from src.config.schemas import PromptConfig
from src.datasets.json_dataset import load_dataset

class GRPODataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_path: str,
        prompt_fn: Callable,
        max_prompt_length: int = 512,
    ):
        self.samples = load_dataset(data_path)
        self.tokenizer = tokenizer
        self.prompt_fn = prompt_fn
        self.max_prompt_length = max_prompt_length
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = encode_grpo_prompt(
            self.samples[idx],
            tokenizer=self.tokenizer,
            prompt_fn=self.prompt_fn,
            max_prompt_length=self.max_prompt_length,
        )
        data["sample"] = self.samples[idx]  # 直接把原始 Sample 对象也放在这里，后续 reward_fn 评估时会用到
        return data