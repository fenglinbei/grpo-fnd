from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Any


@dataclass
class GenerationRequest:
    prompts: List[str]
    max_new_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0
    n: int = 1
    stop: Optional[List[str]] = None


@dataclass
class CandidateOutput:
    text: str
    finish_reason: Optional[str] = None


@dataclass
class GenerationOutput:
    prompt: str
    candidates: List[CandidateOutput] = field(default_factory=list)
    raw: Optional[Any] = None


class GenerationBackend(Protocol):
    """
    统一生成后端接口。
    你可以实现 vLLM / HF generate / OpenAI-compatible server 三种后端，
    evaluator 不需要改。
    """

    def generate(self, request: GenerationRequest) -> List[GenerationOutput]:
        ...