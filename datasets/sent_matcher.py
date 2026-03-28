import re
import unicodedata
from difflib import SequenceMatcher
from typing import List, Dict, Union, Optional


# -------------------------
# 文本规范化 / 清洗
# -------------------------

def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", normalize_text(text))


def clean_artifact_prefix(text: str) -> str:
    """
    清理一些明显的脏前缀，比如:
    jsonhalf-true...
    FALSE ...
    half-true ...
    也可以按你的数据继续扩展。
    """
    if not text:
        return ""

    text = text.strip()

    # 去掉可能黏在前面的 json / label 残片
    text = re.sub(
        r'^(json)?\s*(pants-fire|half-true|mostly-true|barely-true|false|true)\s*[:\-]?\s*',
        '',
        text,
        flags=re.IGNORECASE
    )

    # 去掉开头多余的引号/标点残片
    text = re.sub(r'^[\s"\',.;:()\[\]{}]+', '', text)

    return text.strip()


def is_bad_sentence(text: str) -> bool:
    """
    粗略过滤明显无效的句子
    """
    if not text:
        return True

    t = clean_artifact_prefix(text)
    if not t:
        return True

    # 太短
    if len(t.split()) < 4 and len(t) < 20:
        return True

    # 几乎全是符号
    alnum_count = sum(ch.isalnum() for ch in t)
    if alnum_count == 0:
        return True

    return False


# -------------------------
# 原文切句
# -------------------------

def split_raw_text_to_sentences(raw_text: Union[str, List[str]]) -> List[str]:
    if isinstance(raw_text, list):
        return [s.strip() for s in raw_text if str(s).strip()]

    raw_text = str(raw_text).strip()
    if not raw_text:
        return []

    # 先按换行切
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines

    # 再按句末切
    parts = re.split(r'(?<=[。！？!?\.])\s+', raw_text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts if parts else [raw_text]


# -------------------------
# 匹配打分
# -------------------------

def compute_match_score(tok_sent: str, raw_sent: str) -> float:
    tok_norm = normalize_text(tok_sent)
    raw_norm = normalize_text(raw_sent)

    tok_compact = compact_text(tok_sent)
    raw_compact = compact_text(raw_sent)

    if tok_compact == raw_compact:
        return 1.0

    if tok_compact and raw_compact:
        if tok_compact in raw_compact or raw_compact in tok_compact:
            shorter = min(len(tok_compact), len(raw_compact))
            longer = max(len(tok_compact), len(raw_compact))
            return 0.95 * (shorter / longer)

    ratio_norm = SequenceMatcher(None, tok_norm, raw_norm).ratio()
    ratio_compact = SequenceMatcher(None, tok_compact, raw_compact).ratio()
    return max(ratio_norm, ratio_compact)


# -------------------------
# 对齐：tokenized -> raw
# -------------------------

def align_tokenized_to_raw_with_meta(
    raw_text: Union[str, List[str]],
    tokenized: List[Dict],
    look_ahead: int = 8,
    min_score: float = 0.45,
) -> List[Dict]:
    raw_sents = split_raw_text_to_sentences(raw_text)
    if not raw_sents:
        return []

    results = []
    cursor = 0

    for item in tokenized:
        tok_sent = item.get("sent", "") or ""
        is_evidence = item.get("is_evidence", 0)

        if not tok_sent.strip():
            results.append({
                "tokenized_sent": "",
                "raw_sent": "",
                "raw_idx": None,
                "match_score": 0.0,
                "is_evidence": is_evidence,
            })
            continue

        # 局部窗口，保持顺序
        start = max(0, cursor)
        end = min(len(raw_sents), cursor + look_ahead)
        candidate_indices = list(range(start, end))

        # 如果局部找不到，再全局找
        if not candidate_indices:
            candidate_indices = list(range(len(raw_sents)))

        best_idx = None
        best_score = -1.0

        for idx in candidate_indices:
            score = compute_match_score(tok_sent, raw_sents[idx])
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_score < min_score:
            for idx, raw_sent in enumerate(raw_sents):
                score = compute_match_score(tok_sent, raw_sent)
                if score > best_score:
                    best_score = score
                    best_idx = idx

        matched_raw = raw_sents[best_idx] if best_idx is not None else ""

        results.append({
            "tokenized_sent": tok_sent,
            "raw_sent": matched_raw,
            "raw_idx": best_idx,
            "match_score": float(best_score),
            "is_evidence": is_evidence,
        })

        if best_idx is not None:
            # 完全匹配则推进到下一句；部分匹配则停留当前句
            if compact_text(tok_sent) == compact_text(matched_raw):
                cursor = best_idx + 1
            else:
                cursor = best_idx

    return results


# -------------------------
# 最终 evidence 提取
# -------------------------

def extract_evidence_from_report(
    content: Union[str, List[str]],
    tokenized: List[Dict],
    align_score_threshold: float = 0.60,
    fallback_to_tokenized: bool = True,
) -> Dict:
    """
    返回:
    {
        "evidence": [...],
        "aligned_items": [...],
        "empty_reason": ...
    }
    """
    aligned_items = align_tokenized_to_raw_with_meta(content, tokenized)

    evidence_candidates = [x for x in aligned_items if x["is_evidence"] == 1]

    if not evidence_candidates:
        return {
            "evidence": [],
            "aligned_items": aligned_items,
            "empty_reason": "no_positive_is_evidence"
        }

    # 同一个 raw_idx 可能被多个证据片段重复命中
    # 这里对每个 raw_idx 只保留 match_score 最高的一条
    best_by_raw_idx = {}
    tokenized_fallback_pool = []

    for item in evidence_candidates:
        raw_idx = item["raw_idx"]
        score = item["match_score"]

        # 记录 tokenized fallback 候选
        tokenized_fallback_pool.append(item)

        if raw_idx is None:
            continue

        if (raw_idx not in best_by_raw_idx) or (score > best_by_raw_idx[raw_idx]["match_score"]):
            best_by_raw_idx[raw_idx] = item

    final_evidence = []
    seen_norm = set()

    # 先用 raw 对齐结果
    for raw_idx in sorted(best_by_raw_idx.keys()):
        item = best_by_raw_idx[raw_idx]
        text = item["raw_sent"]

        # 分数太低就先不收
        if item["match_score"] < align_score_threshold:
            continue

        text = clean_artifact_prefix(text)
        if is_bad_sentence(text):
            continue

        norm = compact_text(text)
        if norm and norm not in seen_norm:
            seen_norm.add(norm)
            final_evidence.append(text)

    # 如果 raw 一个都没留下，可选地退回 tokenized 句子
    if not final_evidence and fallback_to_tokenized:
        for item in tokenized_fallback_pool:
            text = clean_artifact_prefix(item["tokenized_sent"])
            if is_bad_sentence(text):
                continue

            norm = compact_text(text)
            if norm and norm not in seen_norm:
                seen_norm.add(norm)
                final_evidence.append(text)

    empty_reason = None
    if not final_evidence:
        positive_cnt = sum(x["is_evidence"] == 1 for x in aligned_items)
        if positive_cnt == 0:
            empty_reason = "no_positive_is_evidence"
        else:
            empty_reason = "all_positive_filtered_out"

    return {
        "evidence": final_evidence,
        "aligned_items": aligned_items,
        "empty_reason": empty_reason
    }