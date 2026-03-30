# 输入：

# generated_text
# parsed_output
# sample

# 输出：

# 总 reward
# reward 各子项，便于日志记录

#总 reward 可能包含：

# label correctness
# format reward
# explanation overlap
# evidence mention reward
# invalid output penalty
from transformers import PreTrainedTokenizer

from src.datasets.schemas import ID2LABEL, Sample
from src.prompting.output_paser import parse_answer_label, parse_think_text, has_valid_format
from src.prompting.target_builder import build_sft_target
from src.reward.text_metrics import f1_overlap
from src.config.registry import register_reward

@register_reward("basic_veracity_reward")
def basic_veracity_reward(generated_text: str, sample: Sample, tokenizer: PreTrainedTokenizer, reward_cfg):
    reward = 0.0

    pred_label = parse_answer_label(generated_text)
    gold_label = ID2LABEL[int(sample.label)]
    think_text = parse_think_text(generated_text)

    if pred_label is None:
        reward += reward_cfg.invalid_output_penalty
    elif pred_label == gold_label:
        reward += reward_cfg.label_correct

    if has_valid_format(generated_text):
        reward += reward_cfg.format_correct

    gold_expl = sample.explanation
    if gold_expl:
        reward += reward_cfg.explanation_overlap * f1_overlap(think_text, gold_expl, tokenizer)

    return float(reward)