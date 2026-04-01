import os
import sys
import json
from pathlib import Path
from transformers import AutoTokenizer
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer

from src.datasets.json_dataset import VeracityJsonDataset, load_dataset
from src.config.registry import build_prompt_fn
from src.config.loader import load_config

import src.prompting.prompts

# 配置
DATA_SPLITS = ["train", "val", "test"]
MAX_LENGTH = 256  # 替换为你的最大长度

cfg = load_config("configs/exp/qwen_grpo_v1.yaml")

DATA_DIR_FILES = [cfg.data.train_path, cfg.data.val_path, cfg.data.test_path]
MODEL_NAME_OR_PATH = cfg.model.name_or_path

def _load_dataset(split) -> VeracityJsonDataset:
    file_path = DATA_DIR_FILES[DATA_SPLITS.index(split)]
    return load_dataset(file_path)

def check_prompt_lengths():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
    max_len = 0
    max_lan_id = None
    total_len = 0
    prompt_count = 0
    for split in DATA_SPLITS:
        print(f"Checking {split} set...")
        split_dataset = _load_dataset(split)
        for idx, sample in enumerate(split_dataset):
            full_prompt = prompt_fn(sample, cfg.prompt)  # 构建完整prompt
            input_ids = tokenizer(full_prompt, truncation=False, add_special_tokens=False)["input_ids"]
            if len(input_ids) > MAX_LENGTH:
                print(f"[{split}] idx={idx} 超长: {len(input_ids)} tokens, id={sample.id}")

            if len(input_ids) > max_len:
                max_len = len(input_ids)
                max_lan_id = sample.id
            total_len += len(input_ids)
            prompt_count += 1

    print(f"Maximum prompt length: {max_len}")
    print(f"Sample ID with maximum length: {max_lan_id}")
    print(f"Average prompt length: {total_len / prompt_count if prompt_count > 0 else 0}")

def check_prompt():
    import random
    prompt_fn = build_prompt_fn(cfg.prompt)
    tokenizer: Qwen2Tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
    for split in DATA_SPLITS:
        print(f"Checking {split} set...")
        split_dataset = _load_dataset(split)
        sample = random.choice(split_dataset)
        system_prompt, user_prompt, assistant_prompt = prompt_fn(sample)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        full_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        print("Prompt:")
        print(full_prompt)
        input_ids = tokenizer(full_prompt, truncation=False, add_special_tokens=False)["input_ids"]
        print(f"Tokenized length: {len(input_ids)} tokens")

        print("Assistant Prompt:")
        print(assistant_prompt)
        assistant_input_ids = tokenizer(assistant_prompt, truncation=False, add_special_tokens=False)["input_ids"]
        print(f"Assistant prompt tokenized length: {len(assistant_input_ids)} tokens")

if __name__ == "__main__":
    # python -m src.datasets.data_check
    # check_prompt_lengths()
    check_prompt()