import torch
from tqdm import tqdm
from loguru import logger

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
    global_step: int = 0,
    on_step_end=None,
):
    model.train()
    total_loss = 0.0
    total_reward = 0.0
    total_batches = 0
    total_optimizer_updates = 0

    pbar = tqdm(dataloader, desc="GRPO", dynamic_ncols=True)
    for batch_samples in pbar:
        batch_size = len(batch_samples["sample_ids"])
        # A. rollout（old policy）
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

        # B. rewards / advantages
        rewards = torch.zeros(
            (batch_size, group_size),
            dtype=torch.float32,
            device=device,
        )

        idx = 0
        for b in range(batch_size):
            for g in range(group_size):
                rewards[b, g] = reward_fn(flat_texts[idx], batch_samples["sample"][b], tokenizer)
                idx += 1

        advantages = compute_group_advantages(rewards)
        flat_advantages = advantages.reshape(-1)
        gen_mask = build_generation_mask(attn_batch, prompt_lens)

        step_reward = float(rewards.mean().item())
        step_reward_std = float(rewards.std(unbiased=False).item())
        adv_mean = float(advantages.mean().item())
        adv_std = float(advantages.std(unbiased=False).item())

        # C. 多轮更新当前 policy
        model.train()
        last_loss = None

        for inner_update_idx in range(num_update_epochs):
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

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_norm = float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            global_step += 1
            total_optimizer_updates += 1
            last_loss = loss.detach()

            current_lr = float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else 0.0

            if on_step_end is not None:
                on_step_end(
                    global_step=global_step,
                    stage="grpo",
                    model=model,
                    tokenizer=tokenizer,
                    train_metrics={
                        "loss": float(last_loss.item()),
                        "reward": step_reward,
                        "reward_std": step_reward_std,
                        "adv_mean": adv_mean,
                        "adv_std": adv_std,
                        "lr": current_lr,
                        "grad_norm": grad_norm,
                        "batch_size": batch_size,
                        "inner_update": inner_update_idx + 1,
                        "num_update_epochs": num_update_epochs,
                        "optimizer_updates": total_optimizer_updates,
                    },
                )
                model.train()

        step_loss = float(last_loss.item()) if last_loss is not None else 0.0

        total_loss += step_loss
        total_reward += step_reward
        total_batches += 1

        pbar.set_postfix(
            {
                "loss": f"{step_loss:.4f}",
                "reward": f"{step_reward:.4f}",
                "gs": global_step,
            }
        )

    return {
        "loss": total_loss / max(1, total_batches),
        "reward": total_reward / max(1, total_batches),
        "global_step": global_step,
        "num_batches": total_batches,
        "num_optimizer_updates": total_optimizer_updates,
    }