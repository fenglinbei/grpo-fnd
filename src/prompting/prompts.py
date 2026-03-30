USER_PROMPT_V1 = """You are a fact-checking assistant.

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
{claim}

Evidence:
{evidence_text}

Now provide your reasoning and final label.
"""