from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class LiveVLLMSyncEvalConfig:
    # ---- backend mode ----
    enabled: bool = False
    backend: Literal["hf", "live_vllm_sync"] = "hf"

    # ---- vLLM server ----
    base_url: str = "http://127.0.0.1:8000"
    served_model_name: str = "live-policy"
    api_key: str = "EMPTY"   # vLLM OpenAI-compatible server 默认不校验真实 key

    # ---- weight sync ----
    sync_backend: Literal["nccl", "ipc"] = "nccl"
    sync_policy: Literal["always", "if_step_changed", "never"] = "if_step_changed"
    packed: bool = True
    request_timeout_init_s: int = 60
    request_timeout_update_s: int = 300
    request_timeout_generate_s: int = 300

    # ---- generation ----
    generation_concurrency: int = 8
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 64
    stop: Optional[list[str]] = None

    # ---- routing policy ----
    use_for_step_eval: bool = False
    use_for_epoch_eval: bool = True
    use_for_final_test: bool = True

    # ---- logging / debug ----
    verbose: bool = True