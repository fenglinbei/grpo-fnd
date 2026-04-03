from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Tuple

import torch
from loguru import logger
from tqdm import tqdm

from src.datasets.json_dataset import VeracityJsonDataset
from src.datasets.schemas import LABEL2ID, ID2LABEL, Sample
from src.evaluation.metrics import compute_classification_metrics
from src.evaluation.live_vllm_generator import LiveVLLMGenerator
from src.evaluation.live_vllm_sync_controller import LiveVLLMWeightSyncController
from src.evaluation.live_vllm_sync_config import LiveVLLMSyncEvalConfig


ParseOutputFn = Callable[[str], Tuple[Optional[str], Optional[str]]]
PromptFn = Callable[[Sample], str]


@dataclass
class LiveEvalRuntime:
    sync_controller: LiveVLLMWeightSyncController
    generator: LiveVLLMGenerator
    parse_output_fn: ParseOutputFn
    cfg: LiveVLLMSyncEvalConfig


class LiveVLLMSyncedEvaluator:
    def __init__(self, runtime: LiveEvalRuntime):
        self.runtime = runtime

    @staticmethod
    def _select_subset(
        dataset: Sequence[Sample],
        quick_eval: bool,
        quick_eval_samples: int,
        quick_eval_mode: str,
    ) -> list[Sample]:
        if not quick_eval:
            return list(dataset)

        eval_num = min(len(dataset), quick_eval_samples)
        if quick_eval_mode == "random":
            indices = random.sample(range(len(dataset)), eval_num)
            return [dataset[i] for i in indices]
        if quick_eval_mode == "first_k":
            return [dataset[i] for i in range(eval_num)]
        raise ValueError(f"Unsupported quick_eval_mode: {quick_eval_mode}")

    def evaluate(
        self,
        model,
        tokenizer,
        prompt_fn: PromptFn,
        dataset: VeracityJsonDataset,
        global_step: int,
        max_prompt_length: int = 1024,   # 这里先保留接口
        max_new_tokens: int = 64,
        batch_size: int = 8,
        quick_eval: bool = False,
        quick_eval_samples: int = 256,
        quick_eval_mode: str = "first_k",
        show_results: bool = True,
        show_results_num: int = 5,
        force_sync: bool = False,
    ) -> dict:
        model.eval()

        # 1. 评估前同步当前 live model -> vLLM
        self.runtime.sync_controller.sync_from_model(
            model=model,
            global_step=global_step,
            force=force_sync,
        )

        # 2. 选择子集
        sub_dataset = self._select_subset(
            dataset=dataset,
            quick_eval=quick_eval,
            quick_eval_samples=quick_eval_samples,
            quick_eval_mode=quick_eval_mode,
        )

        pred_ids = []
        gold_ids = []

        all_prompts = []
        all_raw_outputs = []
        all_pred_explanations = []
        all_gold_explanations = []
        all_pred_labels = []
        all_gold_labels = []
        all_sample_ids = []

        total_num = len(sub_dataset)
        for start in tqdm(range(0, total_num, batch_size), desc="Eval(vLLM-sync)"):
            batch_samples = sub_dataset[start:start + batch_size]
            prompts = [prompt_fn(sample) for sample in batch_samples]

            gen = self.runtime.generator.generate_batch(
                prompts=prompts,
                max_new_tokens=max_new_tokens,
            )
            raw_outputs = gen.outputs

            pred_explanations = []
            pred_labels = []

            for raw_text in raw_outputs:
                pred_explanation, pred_label = self.runtime.parse_output_fn(raw_text)
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

        if show_results and len(pred_ids) > 0:
            logger.info("Show some live vLLM synced evaluation results:")
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