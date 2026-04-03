from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from loguru import logger

from src.inference.backends.base import (
    CandidateOutput,
    GenerationBackend,
    GenerationOutput,
    GenerationRequest,
)


@dataclass
class VLLMEngineConfig:
    model_name_or_path: str
    tokenizer_name_or_path: Optional[str] = None
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85
    max_model_len: Optional[int] = None
    dtype: str = "auto"  # "auto" / "float16" / "bfloat16" / "float32"
    trust_remote_code: bool = True
    seed: int = 42
    enforce_eager: bool = False
    disable_log_stats: bool = True
    swap_space: float = 4.0
    cpu_offload_gb: float = 0.0
    enable_prefix_caching: bool = True
    generation_config: str = "vllm"  # 避免被 HF generation_config 覆盖默认采样行为


class VLLMGenerationBackend(GenerationBackend):
    """
    程序内 vLLM backend。
    只建议初始化一次，然后在整个训练/评估生命周期里复用。
    """

    def __init__(self, cfg: VLLMEngineConfig):
        self.cfg = cfg
        self._llm = None

    def _lazy_init(self):
        if self._llm is not None:
            return

        try:
            from vllm import LLM
        except ImportError as e:
            raise ImportError(
                "vLLM is not installed. Please install it first, e.g. `pip install vllm`."
            ) from e

        logger.info(
            "Initializing vLLM engine | model={} | tp={} | gpu_mem_util={} | max_model_len={}",
            self.cfg.model_name_or_path,
            self.cfg.tensor_parallel_size,
            self.cfg.gpu_memory_utilization,
            self.cfg.max_model_len,
        )

        llm_kwargs = dict(
            model=self.cfg.model_name_or_path,
            tokenizer=self.cfg.tokenizer_name_or_path or self.cfg.model_name_or_path,
            tensor_parallel_size=self.cfg.tensor_parallel_size,
            gpu_memory_utilization=self.cfg.gpu_memory_utilization,
            trust_remote_code=self.cfg.trust_remote_code,
            seed=self.cfg.seed,
            enforce_eager=self.cfg.enforce_eager,
            disable_log_stats=self.cfg.disable_log_stats,
            swap_space=self.cfg.swap_space,
            cpu_offload_gb=self.cfg.cpu_offload_gb,
            enable_prefix_caching=self.cfg.enable_prefix_caching,
            generation_config=self.cfg.generation_config,
        )

        # 可选字段
        if self.cfg.max_model_len is not None:
            llm_kwargs["max_model_len"] = self.cfg.max_model_len
        if self.cfg.dtype is not None:
            llm_kwargs["dtype"] = self.cfg.dtype

        self._llm = LLM(**llm_kwargs)

    def generate(self, request: GenerationRequest) -> List[GenerationOutput]:
        self._lazy_init()

        try:
            from vllm import SamplingParams
        except ImportError as e:
            raise ImportError(
                "vLLM is not installed. Please install it first, e.g. `pip install vllm`."
            ) from e

        stop = request.stop if request.stop else None

        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            max_tokens=request.max_new_tokens,
            n=request.n,
            stop=stop,
        )

        raw_outputs = self._llm.generate(
            request.prompts,
            sampling_params,
            use_tqdm=False,
        )

        results: List[GenerationOutput] = []
        for out in raw_outputs:
            candidates = []
            for cand in out.outputs:
                finish_reason = getattr(cand, "finish_reason", None)
                candidates.append(
                    CandidateOutput(
                        text=cand.text,
                        finish_reason=finish_reason,
                    )
                )
            results.append(
                GenerationOutput(
                    prompt=out.prompt,
                    candidates=candidates,
                    raw=out,
                )
            )
        return results