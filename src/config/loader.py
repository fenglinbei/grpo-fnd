import os
import json
from copy import deepcopy
from typing import Any, Dict, List, Optional

import yaml

from src.config.schemas import ExperimentConfig


def model_dump_compat(model) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_update(result[k], v)
        else:
            result[k] = deepcopy(v)
    return result


def resolve_base_config(path: str) -> Dict[str, Any]:
    cfg = load_yaml(path)
    base_entry = cfg.pop("_base_", None)

    if base_entry is None:
        return cfg

    if isinstance(base_entry, str):
        base_paths = [base_entry]
    elif isinstance(base_entry, list):
        base_paths = base_entry
    else:
        raise ValueError(f"_base_ must be str or list, got {type(base_entry)}")

    merged_base = {}
    for rel_path in base_paths:
        abs_base = os.path.normpath(os.path.join(os.path.dirname(path), rel_path))
        base_cfg = resolve_base_config(abs_base)
        merged_base = deep_update(merged_base, base_cfg)

    return deep_update(merged_base, cfg)


def parse_value(raw: str) -> Any:
    """
    尽量把字符串解析成 bool/int/float/list/dict/None
    """
    try:
        return yaml.safe_load(raw)
    except Exception:
        return raw


def set_by_dotted_key(cfg: Dict[str, Any], dotted_key: str, value: Any):
    keys = dotted_key.split(".")
    cur = cfg
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def apply_overrides(cfg: Dict[str, Any], overrides: Optional[List[str]]) -> Dict[str, Any]:
    if not overrides:
        return cfg

    result = deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be KEY=VALUE, got: {item}")
        key, raw_value = item.split("=", 1)
        value = parse_value(raw_value)
        set_by_dotted_key(result, key, value)
    return result


def load_config(config_path: str, overrides: Optional[List[str]] = None) -> ExperimentConfig:
    raw = resolve_base_config(config_path)
    raw = apply_overrides(raw, overrides)
    cfg = ExperimentConfig(**raw)
    return cfg


def save_resolved_config(cfg: ExperimentConfig, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    raw = model_dump_compat(cfg)
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f, sort_keys=False, allow_unicode=True)


def save_resolved_json(cfg: ExperimentConfig, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    raw = model_dump_compat(cfg)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)