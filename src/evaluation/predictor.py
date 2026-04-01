import torch
from typing import Dict, Any, Optional

from src.datasets.schemas import Sample
from src.prompting.grpo_prompt_builder import build_grpo_messages
from src.prompting.output_paser import parse_model_output

@torch.no_grad()
def predict_label(
    model, 
    tokenizer, 
    prompt_fn,
    sample: Sample, 
    device: torch.device, 
    max_prompt_length: int = 1024, 
    max_new_tokens: int = 128) -> tuple[Optional[str], Optional[str]]:

    messages = build_grpo_messages(sample, prompt_fn)

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    model_inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_length,
        add_special_tokens=False,
    ).to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,   # 分类任务先用确定性解码更稳
    )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    parsed = parse_model_output(output_text)
    return parsed['explanation'], parsed['label']