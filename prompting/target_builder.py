# 用于构建统一的输出内容，输入explanation、label，输出<think>...</think>
# <answer>LABEL</answer> 格式的target

from datasets.schemas import ID2LABEL, Sample
from datasets.sent_matcher import normalize_text



def build_sft_target(sample: Sample) -> str:
    label_name = ID2LABEL[sample.label]
    explanation = normalize_text(sample.explanation)
    target = f"""<think>
{explanation}
</think>
<answer>
{label_name}
</answer>"""
    return target