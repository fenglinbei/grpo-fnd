from src.datasets.schemas import ID2LABEL

def compute_classification_metrics(pred_ids, gold_ids, num_classes: int):
    """
    pred_ids: List[int]，预测标签 id；无效预测用 -1
    gold_ids: List[int]，真实标签 id
    """
    assert len(pred_ids) == len(gold_ids)
    total = len(gold_ids)

    # accuracy
    correct = sum(int(p == g) for p, g in zip(pred_ids, gold_ids))
    accuracy = correct / max(1, total)

    per_class = {}
    macro_p, macro_r, macro_f1 = 0.0, 0.0, 0.0
    weighted_p, weighted_r, weighted_f1 = 0.0, 0.0, 0.0

    total_support = 0

    for c in range(num_classes):
        tp = sum(1 for p, g in zip(pred_ids, gold_ids) if p == c and g == c)
        fp = sum(1 for p, g in zip(pred_ids, gold_ids) if p == c and g != c)
        fn = sum(1 for p, g in zip(pred_ids, gold_ids) if p != c and g == c)
        support = sum(1 for g in gold_ids if g == c)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        per_class[ID2LABEL[c]] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

        macro_p += precision
        macro_r += recall
        macro_f1 += f1

        weighted_p += precision * support
        weighted_r += recall * support
        weighted_f1 += f1 * support
        total_support += support

    macro_p /= num_classes
    macro_r /= num_classes
    macro_f1 /= num_classes

    weighted_p /= max(1, total_support)
    weighted_r /= max(1, total_support)
    weighted_f1 /= max(1, total_support)

    invalid_pred_count = sum(1 for p in pred_ids if p == -1)
    invalid_pred_rate = invalid_pred_count / max(1, total)

    return {
        "accuracy": accuracy,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_p,
        "weighted_recall": weighted_r,
        "weighted_f1": weighted_f1,
        "invalid_pred_count": invalid_pred_count,
        "invalid_pred_rate": invalid_pred_rate,
        "per_class": per_class,
    }