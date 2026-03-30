from src.datasets.schemas import Sample
from src.config.registry import register_prompt


@register_prompt("default_veracity_prompt")
def build_default_veracity_prompt(sample: Sample, prompt_cfg):
    evidence = sample.evidence
    if evidence:
        evidence_text = "\n".join([f"{i+1}. {ev}" for i, ev in enumerate(evidence)])
    else:
        evidence_text = "No evidence provided."

    return f"""You are a fact-checking assistant.

Your task is to classify the truthfulness of the claim into exactly one of:
PANTS_FIRE, FALSE, BARELY_TRUE, HALF_TRUE, MOSTLY_TRUE, TRUE

Please reason first, then answer in the following format exactly:

<think>
your reasoning here
</think>
<answer>
ONE_LABEL
</answer>

Claim:
{sample["claim"]}

Evidence:
{evidence_text}

Now provide your reasoning and final label.
"""