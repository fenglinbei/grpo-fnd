from typing import Callable, Dict

PROMPT_REGISTRY: Dict[str, Callable] = {}
REWARD_REGISTRY: Dict[str, Callable] = {}


def register_prompt(name: str):
    def deco(fn: Callable):
        if name in PROMPT_REGISTRY:
            raise ValueError(f"Prompt '{name}' already registered.")
        PROMPT_REGISTRY[name] = fn
        return fn
    return deco


def register_reward(name: str):
    def deco(fn: Callable):
        if name in REWARD_REGISTRY:
            raise ValueError(f"Reward '{name}' already registered.")
        REWARD_REGISTRY[name] = fn
        return fn
    return deco


def build_prompt_fn(prompt_cfg):
    if prompt_cfg.name not in PROMPT_REGISTRY:
        raise KeyError(f"Unknown prompt builder: {prompt_cfg.name}")
    return PROMPT_REGISTRY[prompt_cfg.name]


def build_reward_fn(reward_cfg):
    if reward_cfg.name not in REWARD_REGISTRY:
        raise KeyError(f"Unknown reward fn: {reward_cfg.name}")
    return REWARD_REGISTRY[reward_cfg.name]