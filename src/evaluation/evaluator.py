import torch
from tqdm import tqdm
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
    quick_eval_samples: int = 64,
):
    model.eval()

    pred_ids = []
    gold_ids = []

    eval_num = len(dataset)
    if quick_eval:
        eval_num = min(eval_num, quick_eval_samples)

    for start in tqdm(range(0, eval_num, batch_size), desc="Eval"):
        batch_samples = [dataset[i] for i in range(start, min(start + batch_size, eval_num))]

        pred_explanations, pred_labels = predict_label_batch(
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

    metrics = compute_classification_metrics(
        pred_ids=pred_ids,
        gold_ids=gold_ids,
        num_classes=len(ID2LABEL),
    )
    return metrics