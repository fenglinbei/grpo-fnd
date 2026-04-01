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

@torch.inference_mode()
def predict_label_batch(
    model,
    tokenizer,
    prompt_fn,
    samples: list[Sample],
    device: torch.device,
    max_prompt_length: int = 1024,
    max_new_tokens: int = 64,   # 建议先缩短
):
    model.eval()

    # LLM batch generation 推荐 left padding
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = [prompt_fn(sample) for sample in samples]

    model_inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
        add_special_tokens=False,
    ).to(device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,          # eval 阶段建议确定性解码
        num_beams=1,
        use_cache=True,           # 明确打开 cache
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # 对 batched generate，直接切掉公共输入前缀长度即可
    input_len = model_inputs["input_ids"].shape[1]
    gen_only_ids = generated_ids[:, input_len:]

    outputs = tokenizer.batch_decode(gen_only_ids, skip_special_tokens=True)

    parsed = [parse_model_output(x) for x in outputs]
    pred_explanations = [x[0] for x in parsed]
    pred_labels = [x[1] for x in parsed]
    return pred_explanations, pred_labels