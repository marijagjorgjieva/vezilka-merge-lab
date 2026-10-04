"""Load and validate MergeKit configurations without contacting model hosts."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs" / "merge"
SOURCE_REGISTRY = ROOT / "configs" / "model_sources.json"
FULL_TASKS = "arc_challenge,arc_easy,boolq,hellaswag,openbookqa,piqa,winogrande"


def load_registry() -> dict[str, Any]:
    return json.loads(SOURCE_REGISTRY.read_text(encoding="utf-8"))


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def model_refs(config: dict[str, Any]) -> list[str]:
    refs = [config["base_model"]]
    for item in config["models"]:
        refs.append(item["model"] if isinstance(item, dict) else item)
    return list(dict.fromkeys(refs))


def validate_config(config: dict[str, Any]) -> None:
    required = {"merge_method", "base_model", "models", "parameters", "tokenizer_source", "dtype"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing MergeKit keys: {', '.join(sorted(missing))}")
    if not isinstance(config["base_model"], str) or not config["base_model"].strip():
        raise ValueError("base_model must be a nonempty model reference")
    if not isinstance(config["models"], list) or not config["models"]:
        raise ValueError("models must be a nonempty list")
    if not isinstance(config["parameters"], dict):
        raise ValueError("parameters must be a mapping")
    if config["tokenizer_source"] != "base":
        raise ValueError("tokenizer_source must be base")
    if config["dtype"] != "bfloat16":
        raise ValueError("dtype must be bfloat16")
    for ref in model_refs(config):
        if not isinstance(ref, str) or not ref.strip():
            raise ValueError("Each model must have a nonempty reference")
        if "+" in ref:
            raise ValueError("LoRA adapter references are not supported")
    if config["merge_method"] == "slerp":
        if len(config["models"]) != 2:
            raise ValueError("SLERP requires exactly two models")
        entries = [item["model"] if isinstance(item, dict) else item for item in config["models"]]
        if config["base_model"] not in entries:
            raise ValueError("SLERP models must include the base_model")
        t_values = config["parameters"].get("t")
        if not isinstance(t_values, list):
            raise ValueError("SLERP parameters.t must be a list")
        protected = any(
            isinstance(item, dict)
            and item.get("filter") == "embed_tokens"
            and item.get("value") == 0.0
            for item in t_values
        )
        if not protected:
            raise ValueError("SLERP must protect embed_tokens at 0.0")


def custom_slerp(base: str, donor: str, t: float) -> dict[str, Any]:
    if not 0.0 <= t <= 1.0:
        raise ValueError("--t must be between 0 and 1")
    config = {
        "merge_method": "slerp",
        "base_model": base,
        "models": [{"model": base}, {"model": donor}],
        "parameters": {
            "t": [
                {"filter": "embed_tokens", "value": 0.0},
                {"filter": "lm_head", "value": 0.0},
                {"value": t},
            ]
        },
        "tokenizer_source": "base",
        "chat_template": "auto",
        "dtype": "bfloat16",
    }
    validate_config(config)
    return config


def override_slerp_models(
    source: dict[str, Any], base: str | None, donor: str | None
) -> dict[str, Any]:
    if source["merge_method"] != "slerp":
        raise ValueError("Model overrides are supported for SLERP YAMLs; edit a copy of multi-donor YAMLs")
    result = copy.deepcopy(source)
    old_base = result["base_model"]
    old_donor_items = [
        item for item in result["models"]
        if (item["model"] if isinstance(item, dict) else item) != old_base
    ]
    if len(old_donor_items) != 1:
        raise ValueError("Expected exactly one donor in the SLERP YAML")
    old_donor = old_donor_items[0]["model"] if isinstance(old_donor_items[0], dict) else old_donor_items[0]
    if base:
        result["base_model"] = base
    changed = []
    for item in result["models"]:
        old_ref = item["model"] if isinstance(item, dict) else item
        new_ref = (base or old_base) if old_ref == old_base else (donor or old_donor)
        changed.append({**item, "model": new_ref} if isinstance(item, dict) else {"model": new_ref})
    result["models"] = changed
    validate_config(result)
    return result


def config_digest(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def is_bundled(path: Path | None, overridden: bool) -> bool:
    return path is not None and not overridden and path.resolve().is_relative_to(CONFIG_DIR)
