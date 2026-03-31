import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from src.evaluation.predictor import predict_label
from src.evaluation.metrics import compute_classification_metrics
from src.datasets.schemas import LABEL2ID, ID2LABEL, Sample

@torch.no_grad()
def evaluate(model, tokenizer, dataset: Dataset, device: torch.device, max_new_tokens: int = 128):
    model.eval()

    pred_ids = []
    gold_ids = []

    for i in tqdm(range(len(dataset)), desc="Eval"):
        sample: Sample = dataset[i]
        pred_label, _ = predict_label(
            model=model,
            tokenizer=tokenizer,
            sample=sample,
            device=device,
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