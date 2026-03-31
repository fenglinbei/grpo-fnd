from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field


class StrictBaseModel(BaseModel):
    class Config:
        extra = "forbid"            # 防止拼错字段名
        validate_assignment = True  # 运行时赋值也校验


class DataConfig(StrictBaseModel):
    train_path: str
    val_path: str
    test_path: str
    num_workers: int = 0
    max_evidence: Optional[int] = None


class ModelConfig(StrictBaseModel):
    name_or_path: str
    tokenizer_name_or_path: Optional[str] = None
    trust_remote_code: bool = False
    use_fast_tokenizer: bool = True
    torch_dtype: str = "auto"      # auto / float16 / bfloat16 / float32
    use_cache: bool = False
    gradient_checkpointing: bool = True


class OptimizerConfig(StrictBaseModel):
    name: str = "adamw"
    lr: float = 1e-5
    weight_decay: float = 0.01
    betas: List[float] = Field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    max_grad_norm: float = 1.0


class SchedulerConfig(StrictBaseModel):
    name: str = "linear"
    warmup_ratio: float = 0.1


class RuntimeConfig(StrictBaseModel):
    seed: int = 42
    device: str = "cuda"
    gradient_accumulation_steps: int = 1
    fp16: bool = False
    bf16: bool = False


class PromptConfig(StrictBaseModel):
    name: str = "default_veracity_prompt"
    require_format: bool = True
    extras: Dict[str, Any] = Field(default_factory=dict)


class RewardConfig(StrictBaseModel):
    name: str = "basic_veracity_reward"
    label_correct: float = 1.0
    format_correct: float = 0.2
    explanation_overlap: float = 0.2
    invalid_output_penalty: float = -0.2
    extras: Dict[str, Any] = Field(default_factory=dict)


class SFTConfig(StrictBaseModel):
    enabled: bool = True
    epochs: int = 1
    max_length: int = 512


class GRPOConfig(StrictBaseModel):
    enabled: bool = True
    epochs: int = 1
    group_size: int = 4
    num_update_epochs: int = 2
    clip_eps: float = 0.2
    kl_beta: float = 0.0
    temperature: float = 1.0
    top_p: float = 0.95
    max_new_tokens: int = 128
    use_ref_model: bool = False
    extras: Dict[str, Any] = Field(default_factory=dict)


class TrainConfig(StrictBaseModel):
    batch_size: int = 2
    grad_accum_steps: int = 1


class EvalConfig(StrictBaseModel):
    max_new_tokens: int = 128
    do_sample: bool = False
    every_n_steps: int = 0
    eval_on_epoch_end: bool = True
    save_best_metric: Literal[
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
    ] = "macro_f1"


class LoggingConfig(StrictBaseModel):
    output_root: str = "outputs"
    save_best: bool = True
    save_last: bool = True
    save_resolved_config: bool = True
    print_config: bool = True
    level: str = "INFO"
    swanlab_enabled: bool = False
    swanlab_project: Optional[str] = None
    swanlab_tags: Optional[List[str]] = None
    swanlab_group: Optional[str] = None
    swanlab_experiment_name: Optional[str] = None
    swanlab_description: Optional[str] = None


class ExperimentConfig(StrictBaseModel):
    exp_name: str = "debug"
    task_name: str = "veracity_grpo"

    data: DataConfig
    model: ModelConfig
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    sft: SFTConfig = Field(default_factory=SFTConfig)
    grpo: GRPOConfig = Field(default_factory=GRPOConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # 临时实验字段：当某个想法还没稳定时，先放这里
    experimental: Dict[str, Any] = Field(default_factory=dict)

    @property
    def output_dir(self) -> str:
        return f"{self.logging.output_root}/{self.exp_name}"