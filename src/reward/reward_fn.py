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

from datasets.schemas import ID2LABEL, Sample
from prompting.output_paser import parse_answer_label, parse_think_text, has_valid_format
from prompting.target_builder import build_sft_target
from reward.text_metrics import f1_overlap

def compute_reward(generated_text: str, sample: Sample, tokenizer: PreTrainedTokenizer) -> float:
    """
    一个简单但可运行的规则奖励：
    - 标签正确：+1.0
    - 格式正确：+0.2
    - reasoning 与 gold explanation 有一定重叠：最多 +0.2
    - 输出空或标签无法解析：-0.2
    """
    reward = 0.0

    pred_label = parse_answer_label(generated_text)
    gold_label = ID2LABEL[sample.label]
    think_text = parse_think_text(generated_text)

    if pred_label is None:
        reward -= 0.2
    else:
        if pred_label == gold_label:
            reward += 1.0

    if has_valid_format(generated_text):
        reward += 0.2

    if sample["explanation"]:
        overlap = f1_overlap(think_text, sample.explanation, tokenizer)
        reward += 0.2 * overlap

    return float(reward)