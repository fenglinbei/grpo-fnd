# 从模型输出中解析内容，包括<think>...</think>、<answer>...</answer>以及label名称等

import re

from typing import Optional, Dict, Union

from src.datasets.schemas import LABEL2ID
from src.datasets.sent_matcher import normalize_text


ANSWER_RE = re.compile(
    r"<answer>\s*(PANTS_FIRE|FALSE|BARELY_TRUE|HALF_TRUE|MOSTLY_TRUE|TRUE)\s*</answer>",
    re.IGNORECASE | re.DOTALL,
)

EXPLANATION_RE = re.compile(
    r"<explanation>\s*(.*?)\s*</explanation>",
    re.IGNORECASE | re.DOTALL,
)

def parse_model_output(text: str) -> Dict[str, Union[Optional[str], bool]]:
    explanation_match = EXPLANATION_RE.search(text)
    answer_match = ANSWER_RE.search(text)

    explanation = explanation_match.group(1).strip() if explanation_match else None
    label = answer_match.group(1).strip().upper() if answer_match else None

    return {
        "explanation": explanation,
        "label": label,
        "format_ok": (explanation is not None and label is not None),
    }