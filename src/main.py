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
# 10. GRPO 训练 epoch
# =========================


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