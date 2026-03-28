import os
import re
import json
import math
import copy
import random
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================
# 6. Rollout
# =========================
@torch.no_grad()
def rollout_group(
    model,
    tokenizer,
    batch_samples: List[Dict[str, Any]],
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

    for sample in batch_samples:
        prompt = build_prompt(sample)
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        prompt_len = int(input_ids.size(1))

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


# =========================
# 7. GRPO loss
# =========================
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


# =========================
# 8. SFT warmup
# =========================
def build_sft_batch(tokenizer, batch_samples: List[Dict[str, Any]], device: torch.device):
    input_tensors = []
    label_tensors = []

    for sample in batch_samples:
        prompt = build_prompt(sample)
        target = build_sft_target(sample)

        prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
        full_ids = tokenizer(prompt + "\n" + target, return_tensors="pt", add_special_tokens=True)["input_ids"][0]

        labels = full_ids.clone()
        prompt_len = prompt_ids.size(0)

        # prompt 部分不计入 loss
        labels[:prompt_len] = -100

        input_tensors.append(full_ids.to(device))
        label_tensors.append(labels.to(device))

    input_ids, attention_mask = pad_1d_tensors(input_tensors, pad_value=tokenizer.pad_token_id)
    labels, _ = pad_1d_tensors(label_tensors, pad_value=-100)
    return input_ids, attention_mask, labels


def train_sft_epoch(model, tokenizer, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    total_steps = 0

    pbar = tqdm(dataloader, desc="SFT")
    for batch_samples in pbar:
        input_ids, attention_mask, labels = build_sft_batch(tokenizer, batch_samples, device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        total_steps += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / max(1, total_steps)


# =========================
# 9. 评估
# =========================
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


@torch.no_grad()
def evaluate(model, tokenizer, dataset: Dataset, device: torch.device, max_new_tokens: int = 128):
    correct = 0
    total = 0

    for i in tqdm(range(len(dataset)), desc="Eval"):
        sample = dataset[i]
        pred_label, _ = predict_label(model, tokenizer, sample, device, max_new_tokens=max_new_tokens)
        gold_label = ID2LABEL[sample["label"]]

        if pred_label == gold_label:
            correct += 1
        total += 1

    acc = correct / max(1, total)
    return {"accuracy": acc}


# =========================
# 10. GRPO 训练 epoch
# =========================
def train_grpo_epoch(
    model,
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
                rewards[b, g] = compute_reward(flat_texts[idx], batch_samples[b])
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


# =========================
# 11. 主函数
# =========================
def main():
    parser = argparse.ArgumentParser()

    # 路径
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # 模型
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--use_ref_model", action="store_true")

    # 通用训练
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=2)

    # SFT warmup
    parser.add_argument("--sft_epochs", type=int, default=1)

    # GRPO
    parser.add_argument("--grpo_epochs", type=int, default=1)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--num_update_epochs", type=int, default=2)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--kl_beta", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=128)

    # 设备
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    ensure_dir(args.output_dir)
    set_seed(args.seed)
    device = torch.device(args.device)

    # -------------------------
    # 数据
    # -------------------------
    train_dataset = VeracityJsonDataset(args.train_path)
    val_dataset = VeracityJsonDataset(args.val_path)
    test_dataset = VeracityJsonDataset(args.test_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=basic_collate_fn,
    )

    # -------------------------
    # tokenizer / model
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.to(device)

    ref_model = None
    if args.use_ref_model and args.kl_beta > 0.0:
        ref_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
        ref_model.to(device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_train_steps = (
        len(train_loader) * max(1, args.sft_epochs + args.grpo_epochs * args.num_update_epochs)
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_train_steps // 10),
        num_training_steps=max(10, total_train_steps),
    )

    # -------------------------
    # 先做 SFT warmup
    # -------------------------
    if args.sft_epochs > 0:
        print("\n===== SFT Warmup =====")
        for epoch in range(args.sft_epochs):
            sft_loss = train_sft_epoch(
                model=model,
                tokenizer=tokenizer,
                dataloader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
            )
            val_metrics = evaluate(model, tokenizer, val_dataset, device)
            print(f"[SFT] epoch={epoch+1} loss={sft_loss:.4f} val_acc={val_metrics['accuracy']:.4f}")

    # -------------------------
    # 再做 GRPO
    # -------------------------
    print("\n===== GRPO Training =====")
    best_val_acc = -1.0
    best_ckpt_dir = os.path.join(args.output_dir, "best_checkpoint")

    for epoch in range(args.grpo_epochs):
        train_metrics = train_grpo_epoch(
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            group_size=args.group_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            clip_eps=args.clip_eps,
            kl_beta=args.kl_beta,
            num_update_epochs=args.num_update_epochs,
        )

        val_metrics = evaluate(model, tokenizer, val_dataset, device)
        print(
            f"[GRPO] epoch={epoch+1} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_reward={train_metrics['reward']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            ensure_dir(best_ckpt_dir)
            model.save_pretrained(best_ckpt_dir)
            tokenizer.save_pretrained(best_ckpt_dir)
            print(f"Saved best model to: {best_ckpt_dir}")

    # -------------------------
    # 测试
    # -------------------------
    print("\n===== Final Test =====")
    if os.path.exists(best_ckpt_dir):
        best_model = AutoModelForCausalLM.from_pretrained(best_ckpt_dir).to(device)
        test_metrics = evaluate(best_model, tokenizer, test_dataset, device)
    else:
        test_metrics = evaluate(model, tokenizer, test_dataset, device)

    print(f"Test accuracy = {test_metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()