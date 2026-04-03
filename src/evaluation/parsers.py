from __future__ import annotations

import re
from typing import Optional, Tuple

from src.datasets.schemas import LABEL2ID


def default_parse_factcheck_output(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    默认解析策略：
    1. 优先解析 <answer>...</answer> 内的标签
    2. 否则解析类似 `Label: TRUE`
    3. explanation 则尽量从 <think> / <explanation> 抽取，失败时退回全文
    """
    if text is None:
        return None, None

    raw = text.strip()
    if not raw:
        return None, None

    label = None
    explanation = None

    # explanation
    think_match = re.search(r"<think>\s*(.*?)\s*</think>", raw, flags=re.DOTALL | re.IGNORECASE)
    exp_match = re.search(r"<explanation>\s*(.*?)\s*</explanation>", raw, flags=re.DOTALL | re.IGNORECASE)

    if think_match:
        explanation = think_match.group(1).strip()
    elif exp_match:
        explanation = exp_match.group(1).strip()

    # label
    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", raw, flags=re.DOTALL | re.IGNORECASE)
    answer_text = answer_match.group(1).strip() if answer_match else raw

    label_patterns = [
        r"\b(PANTS_FIRE|FALSE|BARELY_TRUE|HALF_TRUE|MOSTLY_TRUE|TRUE)\b",
        r"Label\s*:\s*(PANTS_FIRE|FALSE|BARELY_TRUE|HALF_TRUE|MOSTLY_TRUE|TRUE)",
        r"Verdict\s*:\s*(PANTS_FIRE|FALSE|BARELY_TRUE|HALF_TRUE|MOSTLY_TRUE|TRUE)",
    ]

    for pattern in label_patterns:
        m = re.search(pattern, answer_text, flags=re.IGNORECASE)
        if m:
            candidate = m.group(1).upper()
            if candidate in LABEL2ID:
                label = candidate
                break

    if explanation is None:
        explanation = raw[:2000].strip()

    return explanation, label