from __future__ import annotations

from typing import Any, Optional

from src.inference.backends.base import GenerationBackend
from src.inference.backends.vllm import (
    VLLMEngineConfig,
    VLLMGenerationBackend,
)


def build_generation_backend(cfg: Any) -> Optional[GenerationBackend]:
    """
    允许你从 ExperimentConfig 或 dict 里构建 backend。
    约定:
      cfg.eval.backend: "vllm" / "none"
      cfg.vllm: vllm engine config
    """
    backend_name = getattr(cfg.eval, "backend", "none")
    if backend_name in (None, "", "none", "hf"):
        return None

    if backend_name == "vllm":
        vcfg = cfg.vllm
        engine_cfg = VLLMEngineConfig(
            model_name_or_path=vcfg.model_name_or_path or cfg.model.name_or_path,
            tokenizer_name_or_path=vcfg.tokenizer_name_or_path or cfg.model.tokenizer_name_or_path,
            tensor_parallel_size=vcfg.tensor_parallel_size,
            gpu_memory_utilization=vcfg.gpu_memory_utilization,
            max_model_len=vcfg.max_model_len,
            dtype=vcfg.dtype,
            trust_remote_code=vcfg.trust_remote_code,
            seed=vcfg.seed,
            enforce_eager=vcfg.enforce_eager,
            disable_log_stats=vcfg.disable_log_stats,
            swap_space=vcfg.swap_space,
            cpu_offload_gb=vcfg.cpu_offload_gb,
            enable_prefix_caching=vcfg.enable_prefix_caching,
            generation_config=vcfg.generation_config,
        )
        return VLLMGenerationBackend(engine_cfg)

    raise ValueError(f"Unsupported eval backend: {backend_name}")