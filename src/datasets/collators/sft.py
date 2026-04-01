import torch
from typing import List, Dict, Any

class SFTCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        features = [f for f in features if not f.get("drop", False)]
        if len(features) == 0:
            raise ValueError("All features were dropped in this batch.")

        max_len = max(len(f["input_ids"]) for f in features)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for f in features:
            pad_len = max_len - len(f["input_ids"])

            batch_input_ids.append(
                f["input_ids"] + [self.pad_token_id] * pad_len
            )
            batch_attention_mask.append(
                f["attention_mask"] + [0] * pad_len
            )
            batch_labels.append(
                f["labels"] + [-100] * pad_len
            )

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }