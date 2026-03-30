import torch
from tqdm import tqdm
from src.rl.rollout import rollout_group
from src.reward.advantage import compute_group_advantages
from src.modeling.logprob import gather_token_logprobs
from src.rl.grpo_loss import grpo_loss
from src.rl.masks import build_generation_mask

def train_grpo_epoch(
    model,
    reward_fn,
    prompt_fn,
    ref_model,
    tokenizer,
    dataloader,
    optimizer,
    scheduler,
    device,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    clip_eps: float,
    kl_beta: float,
    num_update_epochs: int,
):
    model.train()
    total_loss = 0.0
    total_reward = 0.0
    total_steps = 0

    pbar = tqdm(dataloader, desc="GRPO")
    for batch_samples in pbar:
        batch_size = len(batch_samples)

        # -------------------------
        # A. rollout（old policy）
        # -------------------------
        model.eval()
        with torch.no_grad():
            seq_batch, attn_batch, prompt_lens, flat_texts = rollout_group(
                model=model,
                tokenizer=tokenizer,
                prompt_fn=prompt_fn,
                batch_samples=batch_samples,
                group_size=group_size,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                device=device,
            )

            old_logprobs = gather_token_logprobs(model, seq_batch, attn_batch)
            if ref_model is not None and kl_beta > 0.0:
                ref_logprobs = gather_token_logprobs(ref_model, seq_batch, attn_batch)
            else:
                ref_logprobs = None

        # -------------------------
        # B. rewards / advantages
        # -------------------------
        rewards = torch.zeros((batch_size, group_size), dtype=torch.float32, device=device)
        idx = 0
        for b in range(batch_size):
            for g in range(group_size):
                rewards[b, g] = reward_fn(flat_texts[idx], batch_samples[b], tokenizer)
                idx += 1

        advantages = compute_group_advantages(rewards)  # [B, G]

        flat_advantages = advantages.reshape(-1)        # [N]
        gen_mask = build_generation_mask(attn_batch, prompt_lens)  # [N, T-1]

        # -------------------------
        # C. 多轮更新当前 policy
        # -------------------------
        model.train()
        last_loss = None
        for _ in range(num_update_epochs):
            current_logprobs = gather_token_logprobs(model, seq_batch, attn_batch)

            loss = grpo_loss(
                current_logprobs=current_logprobs,
                old_logprobs=old_logprobs,
                advantages=flat_advantages,
                gen_mask=gen_mask,
                clip_eps=clip_eps,
                ref_logprobs=ref_logprobs,
                kl_beta=kl_beta,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            last_loss = loss

        step_reward = rewards.mean().item()
        step_loss = float(last_loss.item()) if last_loss is not None else 0.0

        total_loss += step_loss
        total_reward += step_reward
        total_steps += 1

        pbar.set_postfix({
            "loss": f"{step_loss:.4f}",
            "reward": f"{step_reward:.4f}",
        })

    return {
        "loss": total_loss / max(1, total_steps),
        "reward": total_reward / max(1, total_steps),
    }
