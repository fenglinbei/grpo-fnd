import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from evaluation.predictor import predict_label
from datasets.schemas import ID2LABEL

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