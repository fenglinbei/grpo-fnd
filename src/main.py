import os
import sys
import json
import time
import argparse
import random

import torch
from loguru import logger
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

from src.config.loader import load_config, save_resolved_config
from src.config.registry import register_prompt, register_reward
from src.datasets.json_dataset import VeracityJsonDataset
from src.datasets.collators.base import basic_collate_fn
from src.training.train_sft import train_sft_epoch
from src.training.train_grpo import train_grpo_epoch
from src.evaluation.evaluator import evaluate

import src.prompts
import src.reward.reward_fn


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(output_dir: str, log_filename: str = "train.log"):
    os.makedirs(output_dir, exist_ok=True)

    logger.remove()

    # 控制台日志
    logger.add(
        sys.stderr,
        level="INFO",
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level:<8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )

    # 文件日志
    logger.add(
        os.path.join(output_dir, log_filename),
        level="INFO",
        encoding="utf-8",
        enqueue=True,
        backtrace=False,
        diagnose=False,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | "
            "{name}:{function}:{line} - {message}"
        ),
    )


def resolve_torch_dtype(dtype_str: str):
    if dtype_str == "auto":
        return None

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unsupported torch_dtype: {dtype_str}")
    return mapping[dtype_str]


def format_eval_metrics(metrics: dict) -> str:
    return (
        f"acc={metrics['accuracy']:.4f}, "
        f"macro_p={metrics['macro_precision']:.4f}, "
        f"macro_r={metrics['macro_recall']:.4f}, "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--opts", nargs="*", default=[])
    args = parser.parse_args()

    # -------------------------
    # 配置
    # -------------------------
    cfg = load_config(args.config, args.opts)
    os.makedirs(cfg.output_dir, exist_ok=True)
    setup_logger(cfg.output_dir)

    logger.info("Loaded config from {}", args.config)
    if args.opts:
        logger.info("Override opts: {}", args.opts)

    if cfg.logging.save_resolved_config:
        resolved_path = os.path.join(cfg.output_dir, "resolved.yaml")
        save_resolved_config(cfg, resolved_path)
        logger.info("Saved resolved config to {}", resolved_path)

    if cfg.logging.print_config:
        logger.info("Resolved config:\n{}", json.dumps(cfg.model_dump(), indent=2, ensure_ascii=False))

    # -------------------------
    # 基础设置
    # -------------------------
    set_seed(cfg.runtime.seed)
    logger.info("Set random seed to {}", cfg.runtime.seed)

    device = torch.device(cfg.runtime.device if torch.cuda.is_available() else "cpu")
    logger.info("Using device: {}", device)

    batch_size = cfg.train.batch_size
    torch_dtype = resolve_torch_dtype(cfg.model.torch_dtype)
    logger.info("Batch size: {}", batch_size)
    logger.info("Torch dtype: {}", cfg.model.torch_dtype)

    # -------------------------
    # 数据
    # -------------------------
    logger.info("Loading datasets...")
    train_dataset = VeracityJsonDataset(cfg.data.train_path)
    val_dataset = VeracityJsonDataset(cfg.data.val_path)
    test_dataset = VeracityJsonDataset(cfg.data.test_path)

    logger.info(
        "Datasets loaded: train={}, val={}, test={}",
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        collate_fn=basic_collate_fn,
    )
    logger.info(
        "Train DataLoader ready: steps_per_epoch={}, num_workers={}",
        len(train_loader),
        cfg.data.num_workers,
    )

    # -------------------------
    # tokenizer / model
    # -------------------------
    logger.info("Loading tokenizer and model...")
    tokenizer_name = cfg.model.tokenizer_name_or_path or cfg.model.name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=cfg.model.use_fast_tokenizer,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.warning("Tokenizer has no pad_token. Fallback to eos_token as pad_token.")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name_or_path,
        trust_remote_code=cfg.model.trust_remote_code,
        torch_dtype=torch_dtype,
    ).to(device)
    logger.info("Policy model loaded from {}", cfg.model.name_or_path)

    ref_model = None
    if cfg.grpo.enabled and cfg.grpo.use_ref_model and cfg.grpo.kl_beta > 0.0:
        logger.info("Loading reference model for KL regularization...")
        ref_model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name_or_path,
            trust_remote_code=cfg.model.trust_remote_code,
            torch_dtype=torch_dtype,
        ).to(device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        logger.info("Reference model loaded and frozen.")
    else:
        logger.info("Reference model disabled.")

    # -------------------------
    # optimizer / scheduler
    # -------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        betas=tuple(cfg.optimizer.betas),
        eps=cfg.optimizer.eps,
        weight_decay=cfg.optimizer.weight_decay,
    )

    sft_epochs = cfg.sft.epochs if cfg.sft.enabled else 0
    grpo_epochs = cfg.grpo.epochs if cfg.grpo.enabled else 0
    grpo_inner_updates = cfg.grpo.num_update_epochs if cfg.grpo.enabled else 0

    total_train_steps = len(train_loader) * max(
        1, sft_epochs + grpo_epochs * max(1, grpo_inner_updates)
    )
    warmup_steps = int(total_train_steps * cfg.scheduler.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(1, total_train_steps),
    )

    logger.info(
        "Optimizer and scheduler ready: lr={}, warmup_steps={}, total_train_steps={}",
        cfg.optimizer.lr,
        warmup_steps,
        max(1, total_train_steps),
    )

    # -------------------------
    # SFT warmup
    # -------------------------
    if cfg.sft.enabled and cfg.sft.epochs > 0:
        logger.info("===== SFT Warmup starts: epochs={} =====", cfg.sft.epochs)

        sft_epoch_bar = tqdm(
            range(cfg.sft.epochs),
            desc="SFT Warmup",
            dynamic_ncols=True,
            leave=True,
        )

        for epoch in sft_epoch_bar:
            epoch_id = epoch + 1
            logger.info("Starting SFT epoch {}/{}", epoch_id, cfg.sft.epochs)

            start_time = time.time()
            sft_loss = train_sft_epoch(
                model=model,
                tokenizer=tokenizer,
                dataloader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
            )
            elapsed = time.time() - start_time

            logger.info(
                "Finished SFT epoch {}/{} | loss={:.4f} | time={:.2f}s",
                epoch_id,
                cfg.sft.epochs,
                sft_loss,
                elapsed,
            )

            logger.info("Running validation after SFT epoch {}/{}...", epoch_id, cfg.sft.epochs)
            val_metrics = evaluate(
                model=model,
                tokenizer=tokenizer,
                dataset=val_dataset,
                device=device,
                max_new_tokens=cfg.eval.max_new_tokens,
            )

            metric_str = format_eval_metrics(val_metrics)
            logger.info("SFT epoch {}/{} validation: {}", epoch_id, cfg.sft.epochs, metric_str)

            sft_epoch_bar.set_postfix(
                loss=f"{sft_loss:.4f}",
                val_acc=f"{val_metrics['accuracy']:.4f}",
                val_f1=f"{val_metrics['macro_f1']:.4f}",
            )

    else:
        logger.info("SFT warmup disabled.")

    # -------------------------
    # GRPO 训练
    # -------------------------
    best_val_acc = -1.0
    best_ckpt_dir = os.path.join(cfg.output_dir, "best_checkpoint")
    last_ckpt_dir = os.path.join(cfg.output_dir, "last_checkpoint")
    reward_fn = register_reward(cfg.reward.name)

    if cfg.grpo.enabled and cfg.grpo.epochs > 0:
        logger.info("===== GRPO Training starts: epochs={} =====", cfg.grpo.epochs)

        grpo_epoch_bar = tqdm(
            range(cfg.grpo.epochs),
            desc="GRPO Training",
            dynamic_ncols=True,
            leave=True,
        )

        for epoch in grpo_epoch_bar:
            epoch_id = epoch + 1
            logger.info("Starting GRPO epoch {}/{}", epoch_id, cfg.grpo.epochs)

            start_time = time.time()
            train_metrics = train_grpo_epoch(
                model=model,
                reward_fn=reward_fn,
                ref_model=ref_model,
                tokenizer=tokenizer,
                dataloader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                group_size=cfg.grpo.group_size,
                max_new_tokens=cfg.grpo.max_new_tokens,
                temperature=cfg.grpo.temperature,
                top_p=cfg.grpo.top_p,
                clip_eps=cfg.grpo.clip_eps,
                kl_beta=cfg.grpo.kl_beta,
                num_update_epochs=cfg.grpo.num_update_epochs,
            )
            elapsed = time.time() - start_time

            logger.info(
                "Finished GRPO epoch {}/{} | loss={:.4f} | reward={:.4f} | time={:.2f}s",
                epoch_id,
                cfg.grpo.epochs,
                train_metrics["loss"],
                train_metrics["reward"],
                elapsed,
            )

            logger.info("Running validation after GRPO epoch {}/{}...", epoch_id, cfg.grpo.epochs)
            val_metrics = evaluate(
                model=model,
                tokenizer=tokenizer,
                dataset=val_dataset,
                device=device,
                max_new_tokens=cfg.eval.max_new_tokens,
            )

            logger.info(
                "GRPO epoch {}/{} validation: {}",
                epoch_id,
                cfg.grpo.epochs,
                format_eval_metrics(val_metrics),
            )

            grpo_epoch_bar.set_postfix(
                loss=f"{train_metrics['loss']:.4f}",
                reward=f"{train_metrics['reward']:.4f}",
                val_acc=f"{val_metrics['accuracy']:.4f}",
                val_f1=f"{val_metrics['macro_f1']:.4f}",
            )

            if cfg.logging.save_best and val_metrics["accuracy"] > best_val_acc:
                old_best = best_val_acc
                best_val_acc = val_metrics["accuracy"]

                os.makedirs(best_ckpt_dir, exist_ok=True)
                model.save_pretrained(best_ckpt_dir)
                tokenizer.save_pretrained(best_ckpt_dir)

                logger.info(
                    "New best checkpoint saved to {} | val_acc: {:.4f} -> {:.4f}",
                    best_ckpt_dir,
                    old_best,
                    best_val_acc,
                )
    else:
        logger.info("GRPO training disabled.")

    # -------------------------
    # 保存最后一个 checkpoint
    # -------------------------
    if cfg.logging.save_last:
        os.makedirs(last_ckpt_dir, exist_ok=True)
        model.save_pretrained(last_ckpt_dir)
        tokenizer.save_pretrained(last_ckpt_dir)
        logger.info("Saved last checkpoint to {}", last_ckpt_dir)

    # -------------------------
    # Final Test
    # -------------------------
    logger.info("===== Final Test starts =====")
    eval_model = model

    if cfg.logging.save_best and os.path.exists(best_ckpt_dir):
        logger.info("Loading best checkpoint for final test from {}", best_ckpt_dir)
        eval_model = AutoModelForCausalLM.from_pretrained(
            best_ckpt_dir,
            trust_remote_code=cfg.model.trust_remote_code,
            torch_dtype=torch_dtype,
        ).to(device)

    test_metrics = evaluate(
        model=eval_model,
        tokenizer=tokenizer,
        dataset=test_dataset,
        device=device,
        max_new_tokens=cfg.eval.max_new_tokens,
    )

    logger.info(
        "Final test summary | acc={:.4f}, macro_p={:.4f}, macro_r={:.4f}, macro_f1={:.4f}, "
        "weighted_p={:.4f}, weighted_r={:.4f}, weighted_f1={:.4f}, invalid_pred_rate={:.4f}",
        test_metrics["accuracy"],
        test_metrics["macro_precision"],
        test_metrics["macro_recall"],
        test_metrics["macro_f1"],
        test_metrics["weighted_precision"],
        test_metrics["weighted_recall"],
        test_metrics["weighted_f1"],
        test_metrics["invalid_pred_rate"],
    )

    logger.info("Per-class metrics:")
    for label_name, m in test_metrics["per_class"].items():
        logger.info(
            "{} | P={:.4f} R={:.4f} F1={:.4f} Support={}",
            f"{label_name:>12}",
            m["precision"],
            m["recall"],
            m["f1"],
            m["support"],
        )


if __name__ == "__main__":
    main()