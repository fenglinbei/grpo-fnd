# 从模型输出中解析内容，包括<think>...</think>、<answer>...</answer>以及label名称等

import re

from typing import Optional

from src.datasets.schemas import LABEL2ID



def parse_answer_label(text: str) -> Optional[str]:
    """
    优先从 <answer>...</answer> 中提取；
    如果没有，再在全文里找标签字符串。
    """
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.IGNORECASE | re.DOTALL)
    if m:
        cand = normalize_text(m.group(1)).upper()
        cand = cand.replace("-", "_").replace(" ", "_")
        if cand in LABEL2ID:
            return cand

    upper_text = text.upper().replace("-", "_")
    for label_name in LABEL2ID.keys():
        if label_name in upper_text:
            return label_name
    return None

def parse_think_text(text: str) -> str:
    m = re.search(r"<think>\s*(.*?)\s*</think>", text, re.IGNORECASE | re.DOTALL)
    if m:
        return normalize_text(m.group(1))
    return ""

def has_valid_format(text: str) -> bool:
    return (
        re.search(r"<think>.*?</think>", text, re.IGNORECASE | re.DOTALL) is not None
        and re.search(r"<answer>.*?</answer>", text, re.IGNORECASE | re.DOTALL) is not None
    )