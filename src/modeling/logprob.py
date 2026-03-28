import torch
import torch.nn.functional as F

def gather_token_logprobs(model, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """
    input_ids: [N, T]
    attention_mask: [N, T]
    返回:
      token_logprobs: [N, T-1]
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]            # [N, T-1, V]
    target = input_ids[:, 1:]                     # [N, T-1]

    log_probs = F.log_softmax(logits, dim=-1)
    token_logprobs = torch.gather(log_probs, dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    return token_logprobs