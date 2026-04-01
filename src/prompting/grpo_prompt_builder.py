from typing import Dict, Any, Callable, List

from click import prompt
from src.datasets.schemas import Sample

def build_grpo_messages(sample: Sample, prompt_fn: Callable) -> List[Dict[str, str]]:
    system_prompt, user_prompt, _ = prompt_fn(sample)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

def encode_grpo_prompt(
    sample: Sample,
    tokenizer,
    prompt_fn: Callable,
    max_prompt_length: int,
) -> Dict[str, Any]:
    messages = build_grpo_messages(sample, prompt_fn)

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # 关键：GRPO 只构造到“等待 assistant 生成”的位置
        enable_thinking=False,
    )

    enc = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_prompt_length,
    )

    return {
        "sample_id": sample.id,
        "prompt_text": prompt_text,
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "gold_label": sample.label,
        "gold_explanation": sample.explanation,
    }
