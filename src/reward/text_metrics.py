from transformers import PreTrainedTokenizer

def f1_overlap(pred: str, gold: str, tokenizer: PreTrainedTokenizer) -> float:
    pred_tokens = tokenizer.tokenize(pred)
    gold_tokens = tokenizer.tokenize(gold)
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    pred_count = {}
    gold_count = {}
    for t in pred_tokens:
        pred_count[t] = pred_count.get(t, 0) + 1
    for t in gold_tokens:
        gold_count[t] = gold_count.get(t, 0) + 1

    overlap = 0
    for t, c in pred_count.items():
        overlap += min(c, gold_count.get(t, 0))

    if overlap == 0:
        return 0.0

    precision = overlap / max(1, len(pred_tokens))
    recall = overlap / max(1, len(gold_tokens))
    return 2 * precision * recall / max(1e-8, precision + recall)
