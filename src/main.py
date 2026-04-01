import os
import sys
import json
import math
import time
import argparse
import random
import swanlab

import torch
from loguru import logger
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

from src.config.schemas import ExperimentConfig
from src.config.loader import load_config, save_resolved_config
from src.config.registry import build_prompt_fn, build_reward_fn
from src.datasets.json_dataset import VeracityJsonDataset
from src.datasets.sft_datasets import SFTDataset
from src.datasets.grpo_datasets import GRPODataset
from src.datasets.collators.sft import SFTCollator
from src.datasets.collators.grpo import GRPOPromptCollator
from src.training.train_sft import train_sft_epoch
from src.training.train_grpo import train_grpo_epoch
from src.evaluation.evaluator import evaluate

import src.prompting.prompts
import src.reward.reward_fn


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(output_dir: str, log_filename: str = "train.log", log_level: str = "INFO"):
    os.makedirs(output_dir, exist_ok=True)

    logger.remove()

    logger.add(
        sys.stderr,
        level=log_level,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level:<8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )

    logger.add(
        os.path.join(output_dir, log_filename),
        level=log_level,
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

def is_main_process() -> bool:
    if not torch.distributed.is_available():
        return True
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def to_float(v):
    if isinstance(v, torch.Tensor):
        if v.numel() == 1:
            return float(v.item())
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return None


def flatten_scalar_metrics(prefix: str, metrics: dict) -> dict:
    out = {}
    for k, v in metrics.items():
        fv = to_float(v)
        if fv is not None:
            out[f"{prefix}/{k}"] = fv
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--opts", nargs="*", default=[])
    args = parser.parse_args()

    swanlab_run = None

    try:
        # -------------------------
        # 配置
        # -------------------------
        cfg: ExperimentConfig = load_config(args.config, args.opts)
        os.makedirs(cfg.output_dir, exist_ok=True)
        setup_logger(cfg.output_dir, log_level=cfg.logging.level)

        
        swanlab_enabled = cfg.logging.swanlab_enabled

        logger.info("Loaded config from {}", args.config)
        if args.opts:
            logger.info("Override opts: {}", args.opts)

        if cfg.logging.save_resolved_config:
            resolved_path = os.path.join(cfg.output_dir, "resolved.yaml")
            save_resolved_config(cfg, resolved_path)
            logger.info("Saved resolved config to {}", resolved_path)

        if cfg.logging.print_config:
            logger.info(
                "Resolved config:\n{}",
                json.dumps(cfg.model_dump(), indent=2, ensure_ascii=False),
            )

        # -------------------------
        # 基础设置
        # -------------------------
        set_seed(cfg.runtime.seed)
        logger.info("Set random seed to {}", cfg.runtime.seed)

        device = torch.device(cfg.runtime.device if torch.cuda.is_available() else "cpu")
        logger.info("Using device: {}", device)

        sft_batch_size = cfg.sft.batch_size
        torch_dtype = resolve_torch_dtype(cfg.model.torch_dtype)
        logger.info("SFT Batch size: {}", sft_batch_size)

        logger.info("Torch dtype: {}", cfg.model.torch_dtype)

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
        logger.debug("Model dtype: {}", model.dtype)

        model.config.use_cache = cfg.model.use_cache
        if cfg.model.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        # -------------------------
        # 数据
        # -------------------------
        base_train_dataset = VeracityJsonDataset(cfg.data.train_path)
        base_val_dataset = VeracityJsonDataset(cfg.data.val_path)
        base_test_dataset = VeracityJsonDataset(cfg.data.test_path)
        
        prompt_fn = build_prompt_fn(cfg.prompt)

        sft_train_loader = None
        if cfg.sft.enabled and cfg.sft.epochs > 0:
            logger.info("Loading SFT dataset and dataloader...")

            sft_train_dataset = SFTDataset(tokenizer, cfg.data.train_path, prompt_fn, max_length=cfg.sft.max_length)
            sft_val_dataset = SFTDataset(tokenizer, cfg.data.val_path, prompt_fn, max_length=cfg.sft.max_length)
            sft_test_dataset = SFTDataset(tokenizer, cfg.data.test_path, prompt_fn, max_length=cfg.sft.max_length)

            logger.info(
                "Datasets loaded: train={}, val={}, test={}",
                len(sft_train_dataset),
                len(sft_val_dataset),
                len(sft_test_dataset),
            )

            sft_collate_fn = SFTCollator(tokenizer)

            sft_train_loader = DataLoader(
                sft_train_dataset,
                batch_size=sft_batch_size,
                shuffle=True,
                num_workers=cfg.data.num_workers,
                collate_fn=sft_collate_fn,
            )

            # sft_val_loader = DataLoader(
            #     sft_val_dataset,
            #     batch_size=sft_batch_size,
            #     shuffle=False,
            #     num_workers=cfg.data.num_workers,
            #     collate_fn=sft_collate_fn,
            # )

            # sft_test_loader = DataLoader(
            #     sft_test_dataset,
            #     batch_size=sft_batch_size,
            #     shuffle=False,
            #     num_workers=cfg.data.num_workers,
            #     collate_fn=sft_collate_fn,
            # )

            logger.info(
                "SFT Train DataLoader ready: steps_per_epoch={}, num_workers={}",
                len(sft_train_loader),
                cfg.data.num_workers,
            )
        else:
            logger.info("SFT warmup disabled, skip SFT dataset and dataloader.")

        grpo_train_loader = None
        if cfg.grpo.enabled and cfg.grpo.epochs > 0:
            logger.info("Loading GRPO dataset and dataloader...")

            grpo_train_dataset = GRPODataset(tokenizer, cfg.data.train_path, prompt_fn, max_prompt_length=cfg.grpo.max_prompt_length)
            grpo_val_dataset = GRPODataset(tokenizer, cfg.data.val_path, prompt_fn, max_prompt_length=cfg.grpo.max_prompt_length)
            grpo_test_dataset = GRPODataset(tokenizer, cfg.data.test_path, prompt_fn, max_prompt_length=cfg.grpo.max_prompt_length)

            logger.info(
                "Datasets loaded: train={}, val={}, test={}",
                len(grpo_train_dataset),
                len(grpo_val_dataset),
                len(grpo_test_dataset),
            )

            grpo_collate_fn = GRPOPromptCollator(tokenizer)

            grpo_train_loader = DataLoader(
                grpo_train_dataset,
                batch_size=cfg.grpo.group_size,
                shuffle=True,
                num_workers=cfg.data.num_workers,
                collate_fn=grpo_collate_fn,
            )

            # grpo_val_loader = DataLoader(
            #     grpo_val_dataset,
            #     batch_size=cfg.grpo.group_size,
            #     shuffle=False,
            #     num_workers=cfg.data.num_workers,
            #     collate_fn=grpo_collate_fn,
            # )

            # grpo_test_loader = DataLoader(
            #     grpo_test_dataset,
            #     batch_size=cfg.grpo.group_size,
            #     shuffle=False,
            #     num_workers=cfg.data.num_workers,
            #     collate_fn=grpo_collate_fn,
            # )

            logger.info(
                "GRPO Train DataLoader ready: steps_per_epoch={}, num_workers={}",
                len(grpo_train_loader),
                cfg.data.num_workers,
            )
        

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

        grad_accum_steps = max(1, cfg.sft.grad_accum_steps)

        sft_epochs = cfg.sft.epochs if cfg.sft.enabled else 0
        grpo_epochs = cfg.grpo.epochs if cfg.grpo.enabled else 0
        grpo_inner_updates = cfg.grpo.num_update_epochs if cfg.grpo.enabled else 0

        sft_steps_per_epoch = math.ceil(len(sft_train_loader if sft_train_loader else []) / grad_accum_steps)
        sft_train_steps = sft_steps_per_epoch * sft_epochs

        grpo_train_steps = len(grpo_train_loader if grpo_train_loader else []) * grpo_epochs * max(1, grpo_inner_updates)
        total_train_steps = max(1, sft_train_steps + grpo_train_steps)
        warmup_steps = int(total_train_steps * cfg.scheduler.warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_train_steps,
        )

        logger.info(
            "Optimizer and scheduler ready: lr={}, warmup_steps={}, total_train_steps={}",
            cfg.optimizer.lr,
            warmup_steps,
            total_train_steps,
        )

        # -------------------------
        # SwanLab
        # -------------------------
        if swanlab_enabled:

            if is_main_process():
                swanlab_tags = cfg.logging.swanlab_tags or []
                swanlab_group = cfg.logging.swanlab_group or ""
                swanlab_experiment_name = cfg.logging.swanlab_experiment_name or ""

                swanlab_description = cfg.logging.swanlab_description or ""

                init_kwargs = dict(
                    project=cfg.logging.swanlab_project,
                    experiment_name=swanlab_experiment_name,
                    config=cfg.model_dump(),
                    logdir=os.path.join(cfg.output_dir, "swanlab"),
                )
                if swanlab_description:
                    init_kwargs["description"] = swanlab_description
                if swanlab_tags:
                    init_kwargs["tags"] = swanlab_tags
                if swanlab_group:
                    init_kwargs["group"] = swanlab_group

                swanlab_run = swanlab.init(**init_kwargs)
                logger.info("SwanLab initialized.")

                swanlab.log(
                    {
                        "meta/train_size": float(len(base_train_dataset)),
                        "meta/val_size": float(len(base_val_dataset)),
                        "meta/test_size": float(len(base_test_dataset)),
                        "meta/steps_per_epoch": float(len(grpo_train_loader)),
                        "meta/total_train_steps": float(total_train_steps),
                        "meta/warmup_steps": float(warmup_steps),
                    },
                    step=0,
                )
            else:
                logger.info("Non-main process detected, skip SwanLab init.")

        # -------------------------
        # 评估 / checkpoint 状态
        # -------------------------
        best_ckpt_dir = os.path.join(cfg.output_dir, "best_checkpoint")
        last_ckpt_dir = os.path.join(cfg.output_dir, "last_checkpoint")

        best_metric_name = getattr(cfg.eval, "save_best_metric", "accuracy")
        every_n_steps = getattr(cfg.eval, "every_n_steps", 0)
        eval_on_epoch_end = getattr(cfg.eval, "eval_on_epoch_end", True)

        state = {
            "best_metric_value": float("-inf"),
            "last_eval_step": -1,
            "last_eval_stage": None,
        }

        def save_best_checkpoint_if_needed(val_metrics: dict, stage: str, global_step: int, reason: str):
            if not cfg.logging.save_best:
                return

            if best_metric_name not in val_metrics:
                raise KeyError(
                    f"save_best_metric='{best_metric_name}' not found in validation metrics. "
                    f"Available keys: {list(val_metrics.keys())}"
                )

            metric_value = float(val_metrics[best_metric_name])
            old_best = state["best_metric_value"]

            if metric_value > old_best:
                state["best_metric_value"] = metric_value
                os.makedirs(best_ckpt_dir, exist_ok=True)
                model.save_pretrained(best_ckpt_dir)
                tokenizer.save_pretrained(best_ckpt_dir)

                logger.info(
                    "New best checkpoint saved to {} | metric={} | {:.4f} -> {:.4f} | stage={} | step={} | reason={}",
                    best_ckpt_dir,
                    best_metric_name,
                    old_best,
                    metric_value,
                    stage,
                    global_step,
                    reason,
                )

                if swanlab_run is not None and is_main_process():
                    swanlab.log(
                        {
                            f"best/{best_metric_name}": metric_value,
                        },
                        step=global_step,
                    )

        def run_validation(stage: str, global_step: int, reason: str, quick_eval: bool = False):
            if (
                reason == "epoch_end"
                and state["last_eval_step"] == global_step
                and state["last_eval_stage"] == stage
            ):
                logger.info(
                    "Skip duplicated epoch-end validation at step {} ({})",
                    global_step,
                    stage,
                )
                return None

            logger.info(
                "Running validation | stage={} | step={} | reason={}",
                stage,
                global_step,
                reason,
            )

            was_training = model.training
            model.eval()

            val_metrics = evaluate(
                model=model,
                tokenizer=tokenizer,
                prompt_fn=prompt_fn,
                dataset=base_val_dataset,
                device=device,
                max_prompt_length=cfg.eval.max_prompt_length,
                max_new_tokens=cfg.eval.max_new_tokens,
                batch_size=cfg.eval.batch_size,
                quick_eval=quick_eval,
            )

            if was_training:
                model.train()

            logger.info(
                "Validation done | stage={} | step={} | reason={} | {}",
                stage,
                global_step,
                reason,
                format_eval_metrics(val_metrics),
            )

            if swanlab_run is not None and is_main_process():
                val_log = flatten_scalar_metrics(f"val/{stage}", val_metrics)
                val_log["val/global_step"] = float(global_step)
                swanlab.log(val_log, step=global_step)

            state["last_eval_step"] = global_step
            state["last_eval_stage"] = stage

            save_best_checkpoint_if_needed(
                val_metrics=val_metrics,
                stage=stage,
                global_step=global_step,
                reason=reason,
            )
            return val_metrics

        def on_step_end(global_step: int, stage: str, model, tokenizer, train_metrics: dict):
            # 先记录训练指标
            if swanlab_run is not None and is_main_process():
                train_log = flatten_scalar_metrics(f"train/{stage}", train_metrics)
                train_log["train/global_step"] = float(global_step)
                if optimizer.param_groups:
                    train_log["train/lr"] = float(optimizer.param_groups[0]["lr"])
                swanlab.log(train_log, step=global_step)

            # 再按设定触发验证
            if every_n_steps is None or every_n_steps <= 0:
                return

            if global_step % every_n_steps != 0:
                return

            logger.info(
                "Step-triggered validation hit | stage={} | global_step={} | train_metrics={}",
                stage,
                global_step,
                train_metrics,
            )

            run_validation(
                stage=stage,
                global_step=global_step,
                reason="step",
                quick_eval=True,
            )

        # -------------------------
        # SFT warmup
        # -------------------------
        global_step = 0

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
                train_metrics = train_sft_epoch(
                    model=model,
                    tokenizer=tokenizer,
                    dataloader=sft_train_loader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    grad_accum_steps=cfg.sft.grad_accum_steps,
                    global_step=global_step,
                    on_step_end=on_step_end,
                )
                elapsed = time.time() - start_time
                global_step = train_metrics["global_step"]

                logger.info(
                    "Finished SFT epoch {}/{} | loss={:.4f} | global_step={} | time={:.2f}s",
                    epoch_id,
                    cfg.sft.epochs,
                    train_metrics["loss"],
                    global_step,
                    elapsed,
                )

                if swanlab_run is not None and is_main_process():
                    swanlab.log(
                        {
                            "epoch/sft": float(epoch_id),
                            "epoch/sft_loss": float(train_metrics["loss"]),
                            "epoch/sft_time_sec": float(elapsed),
                        },
                        step=global_step,
                    )

                val_metrics = None
                if eval_on_epoch_end:
                    val_metrics = run_validation(
                        stage="sft",
                        global_step=global_step,
                        reason="epoch_end",
                    )

                postfix = {
                    "loss": f"{train_metrics['loss']:.4f}",
                    "gs": global_step,
                }
                if val_metrics is not None:
                    postfix["val_acc"] = f"{val_metrics['accuracy']:.4f}"
                    postfix["val_f1"] = f"{val_metrics['macro_f1']:.4f}"

                sft_epoch_bar.set_postfix(postfix)

        else:
            logger.info("SFT warmup disabled.")

        # -------------------------
        # GRPO 训练
        # -------------------------
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

        reward_fn = build_reward_fn(cfg.reward)

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
                    prompt_fn=prompt_fn,
                    ref_model=ref_model,
                    tokenizer=tokenizer,
                    dataloader=grpo_train_loader,
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
                    global_step=global_step,
                    on_step_end=on_step_end,
                )
                elapsed = time.time() - start_time
                global_step = train_metrics["global_step"]

                logger.info(
                    "Finished GRPO epoch {}/{} | loss={:.4f} | reward={:.4f} | global_step={} | time={:.2f}s",
                    epoch_id,
                    cfg.grpo.epochs,
                    train_metrics["loss"],
                    train_metrics["reward"],
                    global_step,
                    elapsed,
                )

                if swanlab_run is not None and is_main_process():
                    swanlab.log(
                        {
                            "epoch/grpo": float(epoch_id),
                            "epoch/grpo_loss": float(train_metrics["loss"]),
                            "epoch/grpo_reward": float(train_metrics["reward"]),
                            "epoch/grpo_time_sec": float(elapsed),
                        },
                        step=global_step,
                    )

                val_metrics = None
                if eval_on_epoch_end:
                    val_metrics = run_validation(
                        stage="grpo",
                        global_step=global_step,
                        reason="epoch_end",
                    )

                postfix = {
                    "loss": f"{train_metrics['loss']:.4f}",
                    "reward": f"{train_metrics['reward']:.4f}",
                    "gs": global_step,
                }
                if val_metrics is not None:
                    postfix["val_acc"] = f"{val_metrics['accuracy']:.4f}"
                    postfix["val_f1"] = f"{val_metrics['macro_f1']:.4f}"

                grpo_epoch_bar.set_postfix(postfix)

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
            prompt_fn=prompt_fn,
            dataset=base_test_dataset,
            device=device,
            max_prompt_length=cfg.eval.max_prompt_length,
            max_new_tokens=cfg.eval.max_new_tokens,
            batch_size=cfg.eval.batch_size,
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

        if swanlab_run is not None and is_main_process():
            swanlab.log(flatten_scalar_metrics("test", test_metrics), step=global_step)
            for label_name, m in test_metrics.get("per_class", {}).items():
                swanlab.log(
                    {
                        f"test/per_class/{label_name}/precision": float(m["precision"]),
                        f"test/per_class/{label_name}/recall": float(m["recall"]),
                        f"test/per_class/{label_name}/f1": float(m["f1"]),
                        f"test/per_class/{label_name}/support": float(m["support"]),
                    },
                    step=global_step,
                )

    finally:
        if swanlab_run is not None and is_main_process():
            try:
                swanlab.finish()
                logger.info("SwanLab finished.")
            except Exception as e:
                logger.warning(f"[WARN] swanlab.finish() failed: {e}")

if __name__ == "__main__":
    main()