from torch.utils.data import Dataset
from typing import List, Callable
from src.datasets.schemas import Sample
from src.prompting.sft_prompt_builder import encode_sft_example
from src.config.schemas import PromptConfig
from src.datasets.json_dataset import load_dataset

class SFTDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_path: str,
        prompt_fn: Callable,
        max_length: int = 512,
    ):
        self.samples = load_dataset(data_path)
        self.tokenizer = tokenizer
        self.prompt_fn = prompt_fn
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return encode_sft_example(
            self.samples[idx],
            tokenizer=self.tokenizer,
            prompt_fn=self.prompt_fn,
            max_length=self.max_length,
        )