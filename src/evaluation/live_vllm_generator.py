from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI
from loguru import logger

from src.evaluation.live_vllm_sync_config import LiveVLLMSyncEvalConfig


@dataclass
class BatchGenResult:
    prompts: list[str]
    outputs: list[str]


class LiveVLLMGenerator:
    def __init__(self, cfg: LiveVLLMSyncEvalConfig):
        self.cfg = cfg
        self.client = OpenAI(
            base_url=f"{cfg.base_url.rstrip('/')}/v1",
            api_key=cfg.api_key,
        )

    def _generate_one(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop: Optional[list[str]],
    ) -> str:
        resp = self.client.completions.create(
            model=self.cfg.served_model_name,
            prompt=prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )
        return resp.choices[0].text

    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[list[str]] = None,
    ) -> BatchGenResult:
        if temperature is None:
            temperature = self.cfg.temperature
        if top_p is None:
            top_p = self.cfg.top_p
        if stop is None:
            stop = self.cfg.stop

        outputs = [""] * len(prompts)

        with ThreadPoolExecutor(max_workers=self.cfg.generation_concurrency) as ex:
            future_to_idx = {
                ex.submit(
                    self._generate_one,
                    prompt,
                    max_new_tokens,
                    temperature,
                    top_p,
                    stop,
                ): idx
                for idx, prompt in enumerate(prompts)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                outputs[idx] = future.result()

        return BatchGenResult(prompts=prompts, outputs=outputs)