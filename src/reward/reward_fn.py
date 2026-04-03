import re
from transformers import PreTrainedTokenizer
from typing import Dict, Optional, Any, Set, List

from src.datasets.schemas import ID2LABEL, LABEL2ID, Sample
from src.evaluation.parsers import default_parse_factcheck_output
from src.config.registry import register_reward
from src.config.schemas import RewardConfig


STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "and", "or", "but", "if", "then", "than", "that", "this", "those", "these",
    "to", "of", "in", "on", "at", "for", "from", "with", "by", "as", "about",
    "into", "after", "before", "over", "under", "between", "during",
    "it", "its", "they", "them", "their", "he", "she", "his", "her",
    "we", "you", "i", "me", "my", "our", "your",
    "do", "does", "did", "done", "have", "has", "had",
    "can", "could", "may", "might", "will", "would", "should",
    "not", "no", "yes",
    "said", "says", "claim", "claims", "claimed",
}

def _normalize_label(label: Optional[str]) -> Optional[str]:
    if label is None:
        return None
    label = label.strip().upper()
    # 防御性归一化，避免 parser 有时给出空格/连字符差异
    label = label.replace("-", "_").replace(" ", "_")
    if label in LABEL2ID:
        return label
    return None


def _tokenize_text(text: str) -> List[str]:
    # 英文场景下足够实用；只取字母数字和下划线
    tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
    return tokens


def _content_token_set(text: str, min_len: int = 3) -> Set[str]:
    tokens = _tokenize_text(text)
    return {
        t for t in tokens
        if len(t) >= min_len and t not in STOPWORDS
    }


def _bucket_of(label_id: int) -> str:
    # 三段式 bucket
    # false-side: PANTS_FIRE / FALSE / BARELY_TRUE
    # middle: HALF_TRUE
    # true-side: MOSTLY_TRUE / TRUE
    if label_id in {0, 1, 2}:
        return "false_side"
    if label_id == 3:
        return "middle"
    return "true_side"


def _ordinal_similarity(pred_id: int, gold_id: int) -> float:
    # 线性序距离，范围 [0, 1]
    # dist=0 -> 1.0, dist=5 -> 0.0
    dist = abs(pred_id - gold_id)
    return max(0.0, 1.0 - dist / 5.0)


def _evidence_grounding_score(
    explanation: Optional[str],
    evidence_list: Optional[List[str]],
) -> float:
    """
    返回 [0, 1]。
    一个轻量 lexical grounding：
    - explanation 与 evidence 的内容词有一定重合 -> 给弱奖励
    - 不要求很高，避免把 reward 变成“复制 evidence”
    """
    if not explanation:
        return 0.0
    if not evidence_list:
        return 0.0

    exp_tokens = _content_token_set(explanation)
    if not exp_tokens:
        return 0.0

    ev_tokens: Set[str] = set()
    for ev in evidence_list:
        ev_tokens |= _content_token_set(ev)

    if not ev_tokens:
        return 0.0

    overlap = exp_tokens & ev_tokens

    # 用 explanation 侧 recall 更合适：
    # “解释里有多少内容词能在 evidence 中找到依据”
    ratio = len(overlap) / max(1, len(exp_tokens))

    # 做一个缓和映射，避免过度鼓励直接复制 evidence
    # 例如 overlap 0.20 左右就已经能拿到不错的小奖励
    if ratio >= 0.30:
        return 1.0
    if ratio >= 0.20:
        return 0.8
    if ratio >= 0.10:
        return 0.5
    if ratio >= 0.05:
        return 0.2
    return 0.0


def _explanation_length_score(explanation: Optional[str]) -> float:
    """
    只做弱约束，不再像原版那样把“长度合适”当大正奖励。
    返回 [0, 1]，建议总权重很小。
    """
    if not explanation:
        return 0.0

    n_words = len(_tokenize_text(explanation))

    # 过短：一般没法形成有效 justification
    if n_words < 8:
        return 0.0
    if n_words < 16:
        return 0.5

    # 比较合适
    if n_words <= 80:
        return 1.0

    # 稍长仍可接受
    if n_words <= 140:
        return 0.7

    # 过长，视为啰嗦/可能在 exploit verbosity
    if n_words <= 220:
        return 0.3

    return 0.0



@register_reward("basic_veracity_reward")
def basic_veracity_reward(
    generated_text: str, 
    sample: Sample, 
    tokenizer: PreTrainedTokenizer, 
    reward_cfg: RewardConfig) -> Dict[str, Any]:

    def compute_format_reward(pred_explanation: Optional[str], pred_label: Optional[str]) -> float:
        # Simplified format reward computation
        return 1.0 if pred_explanation is not None and pred_label is not None else 0.0

    def compute_label_reward(pred_label: Optional[str], gold_label: str) -> float:
        if pred_label is None:
            return 0.0
        return 1.0 if pred_label == gold_label else 0.0
    
    def compute_explanation_reward(pred_explanation: Optional[str], gold_explanation: Optional[str]) -> float:
        if pred_explanation is None:
            return 0.0

        n_words = len(pred_explanation.split())
        if n_words < 20:
            return 0.2
        if n_words > 100:
            return 0.3
        return 1.0

    pred_explanation, pred_label = default_parse_factcheck_output(generated_text)

    r_format = compute_format_reward(pred_explanation=pred_explanation, pred_label=pred_label)
    r_label = compute_label_reward(pred_label=pred_label, gold_label=ID2LABEL[sample.label])
    r_explanation = compute_explanation_reward(pred_explanation=pred_explanation, gold_explanation=sample.explanation)

    total = reward_cfg.format_correct * r_format + reward_cfg.label_correct * r_label + reward_cfg.explanation_length * r_explanation

    return {
        "reward": total,
        "r_format": r_format,
        "r_label": r_label,
        "r_explanation": r_explanation,
        "pred_label": pred_label,
        "parsed_explanation": pred_explanation,
    }

@register_reward("veracity_reward_v2")
def veracity_reward_v2(
    generated_text: str,
    sample: Sample,
    tokenizer: PreTrainedTokenizer,
    reward_cfg: RewardConfig,
) -> Dict[str, Any]:
    """
    exact match 主导 + ordinal/bucket shaping 辅助 + evidence-grounding 弱约束

    建议理解：
    - r_exact: 主奖励，和最终 macro-F1 对齐
    - r_ord / r_bucket: 小幅 shaping，给 GRPO 更密的学习信号
    - r_ground: 解释是否和 evidence 有依据
    - r_len: 很弱的长度约束，防止 explanation 太短或纯灌水
    - invalid penalty: label 解析失败或格式坏掉时明确惩罚
    """
    pred_explanation, pred_label = default_parse_factcheck_output(generated_text)

    gold_id = int(sample.label)
    gold_label = ID2LABEL[gold_id]

    format_ok = bool(pred_explanation is not None and pred_label is not None)

    extras = reward_cfg.extras or {}

    # ------- 默认权重（可被 extras 覆盖） -------
    w_exact = float(extras.get("w_exact", 1.0))
    w_ord = float(extras.get("w_ord", 0.15))
    w_bucket = float(extras.get("w_bucket", 0.10))
    w_ground = float(extras.get("w_ground", 0.10))
    w_len = float(extras.get("w_len", 0.05))

    invalid_penalty = float(extras.get("invalid_penalty", -1.0))
    bad_format_penalty = float(extras.get("bad_format_penalty", -0.2))
    wrong_label_penalty = float(extras.get("wrong_label_penalty", 0.0))

    # ------- invalid / parse fail -------
    if pred_label is None:
        return {
            "reward": invalid_penalty,
            "r_exact": 0.0,
            "r_ord": 0.0,
            "r_bucket": 0.0,
            "r_ground": 0.0,
            "r_len": 0.0,
            "penalty_invalid": 1.0,
            "penalty_bad_format": 0.0 if format_ok else 1.0,
            "pred_label": None,
            "gold_label": gold_label,
            "pred_id": None,
            "gold_id": gold_id,
            "label_distance": None,
            "format_ok": format_ok,
        }

    pred_id = LABEL2ID[pred_label]
    dist = abs(pred_id - gold_id)

    r_exact = 1.0 if pred_id == gold_id else 0.0
    r_ord = _ordinal_similarity(pred_id, gold_id)
    r_bucket = 1.0 if _bucket_of(pred_id) == _bucket_of(gold_id) else 0.0
    r_ground = _evidence_grounding_score(pred_explanation, sample.evidence)
    r_len = _explanation_length_score(pred_explanation)

    reward = (
        w_exact * r_exact
        + w_ord * r_ord
        + w_bucket * r_bucket
        + w_ground * r_ground
        + w_len * r_len
    )

    # 错 label 可选小惩罚。默认 0，先别过重。
    if pred_id != gold_id:
        reward += wrong_label_penalty

    # 格式坏掉但还能 parse 出 label 的情况，给一个轻惩罚
    if not format_ok:
        reward += bad_format_penalty

    return {
        "reward": float(reward),

        # reward 分解
        "r_exact": float(r_exact),
        "r_ord": float(r_ord),
        "r_bucket": float(r_bucket),
        "r_ground": float(r_ground),
        "r_len": float(r_len),

        # penalty 分解
        "penalty_invalid": 0.0,
        "penalty_bad_format": 0.0 if format_ok else 1.0,

        # 调试字段
        "pred_label": pred_label,
        "gold_label": gold_label,
        "pred_id": pred_id,
        "gold_id": gold_id,
        "label_distance": dist,
        "format_ok": format_ok,
        "pred_explanation": pred_explanation,
    }




