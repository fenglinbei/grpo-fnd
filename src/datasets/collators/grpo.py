import torch
from typing import List, Dict, Any

class GRPOPromptCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(len(f["input_ids"]) for f in features)

        batch_input_ids = []
        batch_attention_mask = []

        for f in features:
            pad_len = max_len - len(f["input_ids"])
            batch_input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            batch_attention_mask.append(f["attention_mask"] + [0] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "sample_ids": [f["sample_id"] for f in features],
            "prompt_texts": [f["prompt_text"] for f in features],
            "gold_labels": [f["gold_label"] for f in features],
            "gold_explanations": [f["gold_explanation"] for f in features],
        }