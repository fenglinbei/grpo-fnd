from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, Any

import torch
from loguru import logger
from tqdm import tqdm

from src.datasets.json_dataset import VeracityJsonDataset
from src.datasets.schemas import LABEL2ID, ID2LABEL, Sample
from src.evaluation.metrics import compute_classification_metrics
from src.inference.backends.base import GenerationBackend, GenerationRequest


ParseOutputFn = Callable[[str], Tuple[Optional[str], Optional[str]]]
BuildPromptFn = Callable[[Sample], str]


@dataclass
class EvalGenerationConfig:
    max_prompt_length: int = 1024  # 保留接口，当前 raw prompt 模式下不主动截断
    max_new_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0
    stop: Optional[List[str]] = None


def _select_eval_subset(
    dataset: Sequence[Sample],
    quick_eval: bool,
    quick_eval_samples: int,
    quick_eval_mode: str,
) -> List[Sample]:
    if not quick_eval:
        return list(dataset)

    eval_num = min(len(dataset), quick_eval_samples)
    if quick_eval_mode == "random":
        indices = random.sample(range(len(dataset)), eval_num)
        return [dataset[i] for i in indices]

    if quick_eval_mode == "first_k":
        return [dataset[i] for i in range(eval_num)]

    raise ValueError(f"Unsupported quick_eval_mode: {quick_eval_mode}")


def _truncate_prompt_text(prompt: str, max_prompt_length: int) -> str:
    """
    vLLM 的 LLM.generate 会自动分词。
    这里不做 tokenizer 级精确截断，只做一个占位接口，方便你后续改成：
    1. tokenizer.apply_chat_template 后按 token 截断
    2. 统一由 prompt builder 控制长度
    """
    return prompt


def _run_vllm_batch_generation(
    backend: GenerationBackend,
    prompts: List[str],
    gen_cfg: EvalGenerationConfig,
) -> List[str]:
    req = GenerationRequest(
        prompts=prompts,
        max_new_tokens=gen_cfg.max_new_tokens,
        temperature=gen_cfg.temperature,
        top_p=gen_cfg.top_p,
        top_k=gen_cfg.top_k,
        repetition_penalty=gen_cfg.repetition_penalty,
        n=1,
        stop=gen_cfg.stop,
    )
    outputs = backend.generate(req)

    texts: List[str] = []
    for out in outputs:
        if not out.candidates:
            texts.append("")
        else:
            texts.append(out.candidates[0].text)
    return texts


@torch.inference_mode()
def evaluate_vllm(
    backend: GenerationBackend,
    prompt_fn: BuildPromptFn,
    parse_output_fn: ParseOutputFn,
    dataset: VeracityJsonDataset,
    batch_size: int = 8,
    generation_config: Optional[EvalGenerationConfig] = None,
    quick_eval: bool = False,
    quick_eval_samples: int = 256,
    quick_eval_mode: str = "first_k",  # "random" or "first_k"
    show_results: bool = True,
    show_results_num: int = 5,
) -> dict:
    """
    纯 vLLM 评估版本：
    - 不依赖 HF model.generate
    - backend 可替换
    - parser 可替换
    """
    if generation_config is None:
        generation_config = EvalGenerationConfig()

    # 选子集
    sub_dataset = _select_eval_subset(
        dataset=dataset,
        quick_eval=quick_eval,
        quick_eval_samples=quick_eval_samples,
        quick_eval_mode=quick_eval_mode,
    )

    pred_ids: List[int] = []
    gold_ids: List[int] = []

    all_prompts: List[str] = []
    all_raw_outputs: List[str] = []
    all_pred_explanations: List[Optional[str]] = []
    all_gold_explanations: List[Optional[str]] = []
    all_pred_labels: List[Optional[str]] = []
    all_gold_labels: List[str] = []
    all_sample_ids: List[Any] = []

    total_num = len(sub_dataset)
    if total_num == 0:
        raise ValueError("Empty dataset for evaluation.")

    for start in tqdm(range(0, total_num, batch_size), desc="Eval(vLLM)"):
        batch_samples = sub_dataset[start:start + batch_size]

        prompts = [
            _truncate_prompt_text(prompt_fn(sample), generation_config.max_prompt_length)
            for sample in batch_samples
        ]

        raw_outputs = _run_vllm_batch_generation(
            backend=backend,
            prompts=prompts,
            gen_cfg=generation_config,
        )

        pred_explanations: List[Optional[str]] = []
        pred_labels: List[Optional[str]] = []

        for raw_text in raw_outputs:
            pred_explanation, pred_label = parse_output_fn(raw_text)
            pred_explanations.append(pred_explanation)
            pred_labels.append(pred_label)

        for sample, pred_label in zip(batch_samples, pred_labels):
            gold_id = int(sample.label)
            gold_ids.append(gold_id)

            if pred_label is None or pred_label not in LABEL2ID:
                pred_ids.append(-1)
            else:
                pred_ids.append(LABEL2ID[pred_label])

        if show_results:
            all_prompts.extend(prompts)
            all_raw_outputs.extend(raw_outputs)
            all_pred_explanations.extend(pred_explanations)
            all_gold_explanations.extend([sample.explanation for sample in batch_samples])
            all_pred_labels.extend(pred_labels)
            all_gold_labels.extend([ID2LABEL[int(sample.label)] for sample in batch_samples])
            all_sample_ids.extend([sample.id for sample in batch_samples])

    if show_results:
        logger.info("Show some vLLM evaluation results:")
        show_n = min(show_results_num, len(pred_ids))
        indices = random.sample(range(len(pred_ids)), show_n)

        for i in indices:
            logger.info("====== Sample {} | ID: {} ======", i, all_sample_ids[i])
            logger.info("Prompt: {}", all_prompts[i])
            logger.info("Raw Output: {}", all_raw_outputs[i])
            logger.info("Predicted: {}, Gold: {}", all_pred_labels[i], all_gold_labels[i])
            logger.info(
                "Predicted Explanation: {}",
                all_pred_explanations[i] if all_pred_explanations[i] is not None else "None",
            )
            logger.info(
                "Gold Explanation: {}",
                all_gold_explanations[i] if all_gold_explanations[i] is not None else "None",
            )

    metrics = compute_classification_metrics(
        pred_ids=pred_ids,
        gold_ids=gold_ids,
        num_classes=len(ID2LABEL),
    )
    return metrics