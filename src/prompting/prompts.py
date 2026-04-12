from src.datasets.schemas import Sample
from src.config.registry import register_prompt
from src.config.schemas import PromptConfig


@register_prompt("default_veracity_prompt")
def build_default_veracity_prompt(sample: Sample, prompt_cfg: PromptConfig) -> tuple[str, str, str]:
    evidence = sample.evidence
    top_k = prompt_cfg.extras["top_k_evidence"]
    if evidence:
        evidence_text = "\n".join([f"{i+1}. {ev}" for i, ev in enumerate(evidence[:top_k])])
    else:
        evidence_text = "No evidence provided."

    system_prompt = """You are a fact-checking assistant.

Given a claim and its evidence, do the following:
1. Write a concise justification based only on the provided evidence.
2. Predict exactly one label from:
PANTS_FIRE, FALSE, BARELY_TRUE, HALF_TRUE, MOSTLY_TRUE, TRUE

Return exactly in this format:

<explanation>
brief justification
</explanation>
<answer>
ONE_LABEL
</answer>

Do not output any extra text.
"""

    user_prompt = f"""Claim:
{sample.claim}

Evidence:
{evidence_text}
"""
    
    assistant_prompt = f"""<explanation>
{sample.explanation}
</explanation>
<answer>
{sample.label.name}
</answer>"""
    
    return system_prompt, user_prompt, assistant_prompt

@register_prompt("label_first_veracity_prompt")
def build_label_first_veracity_prompt(sample: Sample, prompt_cfg: PromptConfig) -> tuple[str, str, str]:
    evidence = sample.evidence
    top_k = prompt_cfg.extras["top_k_evidence"]
    if evidence:
        evidence_text = "\n".join([f"{i+1}. {ev}" for i, ev in enumerate(evidence[:top_k])])
    else:
        evidence_text = "No evidence provided."

    system_prompt = """You are a fact-checking assistant.

Given a claim and its evidence, do the following:
1. Predict exactly one label from:
PANTS_FIRE, FALSE, BARELY_TRUE, HALF_TRUE, MOSTLY_TRUE, TRUE
2. Write a concise justification based only on the provided evidence.


Return exactly in this format:

<answer>
ONE_LABEL
</answer>
<explanation>
brief justification
</explanation>

Do not output any extra text.
"""

    user_prompt = f"""Claim:
{sample.claim}

Evidence:
{evidence_text}
"""
    
    assistant_prompt = f"""<answer>
{sample.label.name}
</answer>
<explanation>
{sample.explanation}
</explanation>"""
    
    return system_prompt, user_prompt, assistant_prompt

if __name__ == "__main__":
    # 这里简单测试一下 prompt 的输出格式
    from src.datasets.json_dataset import VeracityJsonDataset

    dataset = VeracityJsonDataset(dataset_path="data/processed/LIAR-RAW/test.json")
    sample = dataset[0]
    system, user, assistant = build_default_veracity_prompt(sample, PromptConfig(extras={"top_k_evidence": 5}))

    print("=== System Prompt ===")
    print(repr(system))
    print("\n=== User Prompt ===")
    print(repr(user))
    print("\n=== Assistant Prompt ===")
    print(repr(assistant))