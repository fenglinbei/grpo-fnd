# 用于构造最终给模型的instruction prompt，输入为claim、evidence，输出一段构建完成后的prompt

from typing import Dict, Any

from click import prompt
from src.datasets.schemas import Sample
from src.prompting.prompts import USER_PROMPT_V1

def build_prompt(sample: Sample) -> str:
    evidence_text = ""
    if sample.evidence:
        evidence_lines = [f"{i+1}. {ev}" for i, ev in enumerate(sample.evidence)]
        evidence_text = "\n".join(evidence_lines)
    else:
        evidence_text = "No evidence provided."

    prompt = USER_PROMPT_V1.format(claim=sample.claim, evidence_text=evidence_text)
    
    return prompt