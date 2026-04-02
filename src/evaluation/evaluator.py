import torch
import random
from tqdm import tqdm
from loguru import logger
from src.datasets.json_dataset import VeracityJsonDataset
from src.evaluation.predictor import predict_label, predict_label_batch
from src.evaluation.metrics import compute_classification_metrics
from src.datasets.schemas import LABEL2ID, ID2LABEL, Sample

@torch.no_grad()
def _evaluate(
    model, 
    tokenizer, 
    prompt_fn,
    dataset: VeracityJsonDataset, 
    device: torch.device, 
    max_prompt_length: int = 1024,
    max_new_tokens: int = 128):

    model.eval()

    pred_ids = []
    gold_ids = []

    for i in tqdm(range(len(dataset)), desc="Eval"):
        sample: Sample = dataset[i]
        pred_explanation, pred_label = predict_label(
            model=model,
            tokenizer=tokenizer,
            prompt_fn=prompt_fn,
            sample=sample,
            device=device,
            max_prompt_length=max_prompt_length,
            max_new_tokens=max_new_tokens,
        )

        gold_id = int(sample.label)
        gold_ids.append(gold_id)

        if pred_label is None:
            pred_ids.append(-1)
        else:
            pred_ids.append(LABEL2ID[pred_label])

    metrics = compute_classification_metrics(
        pred_ids=pred_ids,
        gold_ids=gold_ids,
        num_classes=len(ID2LABEL),
    )
    return metrics


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
    quick_eval_mode: str = "first_k",  # "random" or "first_k" 
    show_results: bool = True,
    show_results_num: int = 5,
):
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

    for start in tqdm(range(0, eval_num, batch_size), desc="Eval"):
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

    if show_results:
        logger.info("Show some evaluation results:")
        show_results_num = min(show_results_num, len(pred_ids))
        # 随机选择一些结果展示
        indices = random.sample(range(len(pred_ids)), show_results_num)
        for i in indices:
            pred_label = all_pred_labels[i] if all_pred_labels[i] != -1 else "None"
            gold_label = all_gold_labels[i]
            pred_explanation = all_pred_explanations[i] if all_pred_explanations[i] is not None else "None"
            gold_explanation = all_gold_explanations[i] if all_gold_explanations[i] is not None else "None"
            logger.info(f"======Sample {i}: ID: {all_sample_ids[i]}======")
            logger.info(f"Prompt: {all_prompts[i]}")
            logger.info(f"Raw Output: {all_raw_outputs[i]}")
            logger.info(f"Predicted: {all_pred_labels[i]}, Gold: {gold_label}")
            logger.info(f"Predicted Explanation: {pred_explanation}")
            logger.info(f"Gold Explanation: {gold_explanation}")

    metrics = compute_classification_metrics(
        pred_ids=pred_ids,
        gold_ids=gold_ids,
        num_classes=len(ID2LABEL),
    )
    return metrics