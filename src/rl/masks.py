import torch
from typing import List

def pad_1d_tensors(tensors: List[torch.Tensor], pad_value: int) -> (torch.Tensor, torch.Tensor):
    """
    返回:
      padded: [N, T]
      attention_mask: [N, T]
    """
    max_len = max(t.size(0) for t in tensors)
    batch = []
    mask = []
    for t in tensors:
        pad_len = max_len - t.size(0)
        if pad_len > 0:
            padded = torch.cat([t, torch.full((pad_len,), pad_value, dtype=t.dtype, device=t.device)], dim=0)
            attn = torch.cat([torch.ones_like(t), torch.zeros((pad_len,), dtype=t.dtype, device=t.device)], dim=0)
        else:
            padded = t
            attn = torch.ones_like(t)
        batch.append(padded)
        mask.append(attn)
    return torch.stack(batch, dim=0), torch.stack(mask, dim=0)

def build_generation_mask(attention_mask: torch.Tensor, prompt_lens: List[int]) -> torch.Tensor:
    """
    attention_mask: [N, T]
    返回:
      gen_mask: [N, T-1]
    只保留生成部分 token 对应的位置
    """
    N, T = attention_mask.size()
    gen_mask = torch.zeros((N, T - 1), dtype=torch.float32, device=attention_mask.device)

    for i in range(N):
        full_len = int(attention_mask[i].sum().item())
        prompt_len = int(prompt_lens[i])

        # target 序列索引对应 input_ids[:, 1:]
        # 生成 token 是 input_ids[prompt_len : full_len]
        # 因此在 token_logprobs 中对应 [prompt_len - 1 : full_len - 1]
        start = max(0, prompt_len - 1)
        end = max(start, full_len - 1)
        if end > start:
            gen_mask[i, start:end] = 1.0

    return gen_mask