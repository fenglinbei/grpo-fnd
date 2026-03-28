import torch
from typing import Optional

def grpo_loss(
    current_logprobs: torch.Tensor,    # [N, T-1]
    old_logprobs: torch.Tensor,        # [N, T-1]
    advantages: torch.Tensor,          # [N]
    gen_mask: torch.Tensor,            # [N, T-1]
    clip_eps: float,
    ref_logprobs: Optional[torch.Tensor] = None,  # [N, T-1]
    kl_beta: float = 0.0,
):
    """
    一个最小可运行版 GRPO loss:
    - 序列级 advantage
    - token 级 ratio / clipping
    - 可选的 sample-based KL proxy
    """
    log_ratio = current_logprobs - old_logprobs
    ratio = torch.exp(log_ratio)  # [N, T-1]

    adv = advantages.unsqueeze(-1)  # [N, 1]

    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    policy_obj = torch.minimum(unclipped, clipped)

    if ref_logprobs is not None and kl_beta > 0.0:
        # 最小实现里的 sample-based KL proxy
        kl_proxy = current_logprobs - ref_logprobs
        token_obj = policy_obj - kl_beta * kl_proxy
    else:
        token_obj = policy_obj

    denom = gen_mask.sum().clamp_min(1.0)
    objective = (token_obj * gen_mask).sum() / denom
    loss = -objective
    return loss