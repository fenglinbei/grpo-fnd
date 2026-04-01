import torch

from typing import List, Dict, Any
from loguru import logger
from src.rl.masks import pad_1d_tensors

@torch.no_grad()
def rollout_group(
    model,
    tokenizer,
    prompt_fn,
    batch_samples: Dict[str, List[Any]],
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
):
    """
    用“当前 batch 开始更新之前”的 policy 采样一组输出，作为 old policy rollout。
    返回:
      seq_batch: [N, T]
      attn_batch: [N, T]
      prompt_lens: List[int], len=N
      flat_texts: List[str], len=N
    其中 N = B * G
    """
    all_seqs = []
    prompt_lens = []
    flat_texts = []

    for i in range(len(batch_samples["sample_ids"])):
        input_ids = batch_samples["input_ids"][i].to(device)
        attention_mask = batch_samples["attention_mask"][i].to(device)
        prompt_len = len(input_ids)

        # 对单个 prompt 一次性采样 group_size 个输出
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=group_size,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )  # [G, T_i]

        for j in range(outputs.size(0)):
            seq = outputs[j]
            all_seqs.append(seq)
            prompt_lens.append(prompt_len)
            flat_texts.append(tokenizer.decode(seq, skip_special_tokens=True))

    seq_batch, attn_batch = pad_1d_tensors(all_seqs, pad_value=tokenizer.pad_token_id)
    return seq_batch, attn_batch, prompt_lens, flat_texts