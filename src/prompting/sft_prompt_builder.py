from typing import Dict, Any, Callable, List

from click import prompt
from src.datasets.schemas import Sample

def build_sft_messages(sample: Sample, prompt_fn: Callable) -> List[Dict[str, str]]:
    system_prompt, user_prompt, assistant_prompt = prompt_fn(sample)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {
            "role": "assistant",
            "content": assistant_prompt,
        },
    ]


def encode_sft_example(
    sample: Sample,
    tokenizer,
    prompt_fn: Callable,
    max_length: int,
) -> Dict[str, Any]:
    messages_full = build_sft_messages(sample, prompt_fn)
    messages_prompt = messages_full[:-1]  # 去掉最后 assistant

    # 完整文本：用于 input_ids / labels
    full_text = tokenizer.apply_chat_template(
        messages_full,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,   # 关键：统一关闭 thinking
    )

    # prompt 文本：用于确定 labels 中哪些 token 需要 mask
    prompt_text = tokenizer.apply_chat_template(
        messages_prompt,
        tokenize=False,
        add_generation_prompt=True,   # 关键：此时要补 assistant 开头
        enable_thinking=False,
    )

    full_enc = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    prompt_enc = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )

    input_ids = full_enc["input_ids"]
    attention_mask = full_enc["attention_mask"]

    labels = input_ids.copy()
    prompt_len = min(len(prompt_enc["input_ids"]), len(labels))
    labels[:prompt_len] = [-100] * prompt_len

    # 如果整个样本被截断后，assistant 区域完全没了，这个样本就不该用于训练
    if all(x == -100 for x in labels):
        return {
            "drop": True,
            "sample_id": sample.id,
        }

    return {
        "drop": False,
        "sample_id": sample.id,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "full_text": full_text,      # 调试时很有用
        "prompt_text": prompt_text,  # 调试时很有用
    }

if __name__ == "__main__":
    # python -m src.prompting.sft_prompt_builder
    from transformers import AutoTokenizer
    from src.config.registry import build_prompt_fn
    from src.datasets.json_dataset import VeracityJsonDataset
    from src.config.loader import load_config
    import src.prompting.prompts  # 注册默认 prompt

    
    cfg = load_config("configs/exp/qwen_grpo_v1.yaml")
    prompt_fn = build_prompt_fn(cfg.prompt)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name_or_path)

    train_dataset = VeracityJsonDataset(cfg.data.train_path)
    val_dataset = VeracityJsonDataset(cfg.data.val_path)
    test_dataset = VeracityJsonDataset(cfg.data.test_path)

    print("Checking train dataset...")
    # print(encode_sft_example(train_dataset[0], tokenizer, prompt_fn, max_length=512))

    total_count = 0
    drop_count = 0
    for sample in val_dataset:
        enc = encode_sft_example(sample, tokenizer, prompt_fn, max_length=512)
        total_count += 1
        if enc["drop"]:
            drop_count += 1
            print(f"Sample {enc['sample_id']} is dropped due to excessive length.")
        else:
            print(f"Sample {enc['sample_id']} is encoded successfully with input_ids length {len(enc['input_ids'])}.")
    print(f"Total samples: {total_count}, Dropped samples: {drop_count}")