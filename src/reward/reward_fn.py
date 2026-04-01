from transformers import PreTrainedTokenizer
from typing import Dict, Optional, Any

from src.datasets.schemas import ID2LABEL, Sample
from src.prompting.output_paser import parse_model_output
from src.config.registry import register_reward
from src.config.schemas import RewardConfig

@register_reward("basic_veracity_reward")
def basic_veracity_reward(
    generated_text: str, 
    sample: Sample, 
    tokenizer: PreTrainedTokenizer, 
    reward_cfg: RewardConfig) -> Dict[str, Any]:

    def compute_format_reward(parsed: Dict[str, Optional[str]]) -> float:
        return 1.0 if parsed["format_ok"] else 0.0
    
    def compute_label_reward(parsed: Dict[str, Optional[str]], gold_label: str) -> float:
        pred = parsed["label"]
        if pred is None:
            return 0.0
        return 1.0 if pred == gold_label else 0.0
    
    def compute_explanation_reward(parsed: Dict[str, Optional[str]]) -> float:
        explanation = parsed["explanation"]
        if explanation is None:
            return 0.0

        n_words = len(explanation.split())
        if n_words < 20:
            return 0.2
        if n_words > 100:
            return 0.3
        return 1.0

    parsed = parse_model_output(generated_text)

    r_format = compute_format_reward(parsed)
    r_label = compute_label_reward(parsed, ID2LABEL[sample.label])
    r_explanation = compute_explanation_reward(parsed)

    total = reward_cfg.format_correct * r_format + reward_cfg.label_correct * r_label + reward_cfg.explanation_length * r_explanation

    return {
        "reward": total,
        "r_format": r_format,
        "r_label": r_label,
        "r_explanation": r_explanation,
        "parsed_label": parsed["label"],
        "parsed_explanation": parsed["explanation"],
    }

