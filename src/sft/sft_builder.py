import torch
from typing import List, Dict, Any

from src.prompting.prompt_builder import build_prompt
from src.prompting.target_builder import build_sft_target
from src.rl.masks import pad_1d_tensors

def build_sft_batch(tokenizer, batch_samples: List[Dict[str, Any]], device: torch.device):
    input_tensors = []
    label_tensors = []

    for sample in batch_samples:
        prompt = build_prompt(sample)
        target = build_sft_target(sample)

        prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
        full_ids = tokenizer(prompt + "\n" + target, return_tensors="pt", add_special_tokens=True)["input_ids"][0]

        labels = full_ids.clone()
        prompt_len = prompt_ids.size(0)

        # prompt 部分不计入 loss
        labels[:prompt_len] = -100

        input_tensors.append(full_ids.to(device))
        label_tensors.append(labels.to(device))

    input_ids, attention_mask = pad_1d_tensors(input_tensors, pad_value=tokenizer.pad_token_id)
    labels, _ = pad_1d_tensors(label_tensors, pad_value=-100)
    return input_ids, attention_mask, labels