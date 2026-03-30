import torch
from typing import Dict, Any

from src.prompting.prompt_builder import build_prompt
from src.prompting.output_paser import parse_answer_label

@torch.no_grad()
def predict_label(model, tokenizer, sample: Dict[str, Any], device: torch.device, max_new_tokens: int = 128):
    model.eval()
    prompt = build_prompt(sample)
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    pred_label = parse_answer_label(text)
    return pred_label, text