from __future__ import annotations

import random
from typing import Optional, Callable, Tuple

import torch
from loguru import logger
from tqdm import tqdm

from src.datasets.json_dataset import VeracityJsonDataset
from src.datasets.schemas import LABEL2ID, ID2LABEL, Sample
from src.evaluation.metrics import compute_classification_metrics
from src.evaluation.predictor import predict_label_batch
from src.evaluation.evaluator_live_vllm_sync import LiveVLLMSyncedEvaluator


@torch.inference_mode()
def evaluate(
    model,
    tokenizer,
    prompt_fn,
    dataset: VeracityJsonDataset,
    device: torch.device,
    max_prompt_length: int = 1024,
    max_new_tokens: int = 64,
    batch_size: int = 8,
    quick_eval: bool = False,
    quick_eval_samples: int = 256,
    quick_eval_mode: str = "first_k",
    show_results: bool = True,
    show_results_num: int = 5,
    global_step: int = -1,
    live_vllm_evaluator: Optional[LiveVLLMSyncedEvaluator] = None,
    use_live_vllm_sync: bool = False,
):
    # ---------------------------
    # live vLLM sync 分支
    # ---------------------------
    if use_live_vllm_sync and live_vllm_evaluator is not None:
        return live_vllm_evaluator.evaluate(
            model=model,
            tokenizer=tokenizer,
            prompt_fn=prompt_fn,
            dataset=dataset,
            global_step=global_step,
            max_prompt_length=max_prompt_length,
            max_new_tokens=max_new_tokens,
            batch_size=batch_size,
            quick_eval=quick_eval,
            quick_eval_samples=quick_eval_samples,
            quick_eval_mode=quick_eval_mode,
            show_results=show_results,
            show_results_num=show_results_num,
        )

    # ---------------------------
    # 原 HF 分支：保留你现有逻辑
    # ---------------------------
    model.eval()

    pred_ids = []
    gold_ids = []

    eval_num = len(dataset)
    if quick_eval:
        eval_num = min(eval_num, quick_eval_samples)
        sub_dataset = []
        if quick_eval_mode == "random":
            indices = random.sample(range(len(dataset)), eval_num)
            for idx in indices:
                sub_dataset.append(dataset[idx])
        elif quick_eval_mode == "first_k":
            for idx in range(eval_num):
                sub_dataset.append(dataset[idx])
        dataset = sub_dataset

    all_prompts = []
    all_raw_outputs = []
    all_pred_explanations = []
    all_gold_explanations = []
    all_pred_labels = []
    all_gold_labels = []
    all_sample_ids = []

    for start in tqdm(range(0, eval_num, batch_size), desc="Eval(HF)"):
        batch_samples = [dataset[i] for i in range(start, min(start + batch_size, eval_num))]

        pred_explanations, pred_labels, raw_output, prompts = predict_label_batch(
            model=model,
            tokenizer=tokenizer,
            prompt_fn=prompt_fn,
            samples=batch_samples,
            device=device,
            max_prompt_length=max_prompt_length,
            max_new_tokens=max_new_tokens,
        )

        for sample, pred_label in zip(batch_samples, pred_labels):
            gold_id = int(sample.label)
            gold_ids.append(gold_id)

            if pred_label is None:
                pred_ids.append(-1)
            else:
                pred_ids.append(LABEL2ID[pred_label])

        if show_results:
            all_prompts.extend(prompts)
            all_raw_outputs.extend(raw_output)
            all_pred_explanations.extend(pred_explanations)
            all_gold_explanations.extend([sample.explanation for sample in batch_samples])
            all_pred_labels.extend(pred_labels)
            all_gold_labels.extend([ID2LABEL[int(sample.label)] for sample in batch_samples])
            all_sample_ids.extend([sample.id for sample in batch_samples])

    if show_results and len(pred_ids) > 0:
        logger.info("Show some evaluation results:")
        show_n = min(show_results_num, len(pred_ids))
        indices = random.sample(range(len(pred_ids)), show_n)
        for i in indices:
            logger.info("======Sample {}: ID: {}======", i, all_sample_ids[i])
            logger.info("Prompt: {}", all_prompts[i])
            logger.info("Raw Output: {}", all_raw_outputs[i])
            logger.info("Predicted: {}, Gold: {}", all_pred_labels[i], all_gold_labels[i])
            logger.info("Predicted Explanation: {}", all_pred_explanations[i])
            logger.info("Gold Explanation: {}", all_gold_explanations[i])

    metrics = compute_classification_metrics(
        pred_ids=pred_ids,
        gold_ids=gold_ids,
        num_classes=len(ID2LABEL),
    )
    return metrics