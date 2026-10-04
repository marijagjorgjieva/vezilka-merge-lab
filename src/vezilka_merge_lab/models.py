"""Download, prepare, and check Gemma 3 text checkpoints."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

from safetensors import safe_open

from . import extraction
from .config import load_registry, model_refs


ARCH_FIELDS = (
    "model_type",
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
)


def _download(repo: str, revision: str | None) -> Path:
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=repo,
            revision=revision,
            token=os.environ.get("HF_TOKEN") or None,
            allow_patterns=["*.json", "*.jinja", "*.model", "*.safetensors"],
            max_workers=2,
        )
    )


def _extract_known(source: dict[str, str], prepared_root: Path) -> dict[str, Any]:
    raw = _download(source["source"], source["revision"])
    target = prepared_root / f"{source['prepared_name']}-{raw.name[:12]}"
    if not (target / "model.safetensors.index.json").is_file():
        if target.exists():
            raise ValueError(f"Incomplete prepared model at {target}; move it aside before retrying")
        target.mkdir(parents=True)
        text_config = extraction.text_config_from_multimodal(raw / "config.json")
        (target / "config.json").write_text(
            json.dumps(text_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        copied = extraction.copy_tokenizer_files(raw, target)
        count, size = extraction.extract_tensors(raw, target)
        (target / "extraction.json").write_text(
            json.dumps(
                {
                    "source": source["source"],
                    "revision": raw.name,
                    "tensor_count": count,
                    "tensor_bytes": size,
                    "tokenizer_files": copied,
                },
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
    return {
        "repo": source["source"],
        "revision": raw.name,
        "path": str(target.resolve()),
        "extracted_text_only": True,
    }


def prepare_reference(ref: str, prepared_root: Path, base_dir: Path | None = None) -> dict[str, Any]:
    registry = load_registry()["models"]
    alias = ref.removeprefix("./")
    if alias in registry and "source" in registry[alias]:
        result = _extract_known(registry[alias], prepared_root)
        result["requested"] = ref
        return result

    local = Path(ref).expanduser()
    if local.is_dir():
        return {"requested": ref, "repo": None, "revision": None, "path": str(local.resolve()), "extracted_text_only": False}
    if base_dir and not local.is_absolute() and (base_dir / local).is_dir():
        return {
            "requested": ref, "repo": None, "revision": None,
            "path": str((base_dir / local).resolve()), "extracted_text_only": False,
        }
    if ref.startswith("./") or ref.startswith("/"):
        raise FileNotFoundError(f"Local model directory does not exist: {ref}")

    repo, explicit_revision = ref.rsplit("@", 1) if "@" in ref else (ref, None)
    if "/" not in repo:
        raise ValueError(f"Expected a local directory or Hugging Face organization/model ID: {ref}")
    for known in registry.values():
        if known.get("source") == repo:
            extraction_spec = {**known, "revision": explicit_revision or known["revision"]}
            result = _extract_known(extraction_spec, prepared_root)
            result["requested"] = ref
            return result
    revision = explicit_revision or registry.get(repo, {}).get("revision")
    snapshot = _download(repo, revision)
    return {
        "requested": ref,
        "repo": repo,
        "revision": snapshot.name,
        "path": str(snapshot.resolve()),
        "extracted_text_only": False,
    }


def prepare_config(
    config: dict[str, Any], prepared_root: Path, base_dir: Path | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prepared_root.mkdir(parents=True, exist_ok=True)
    sources = {ref: prepare_reference(ref, prepared_root, base_dir) for ref in model_refs(config)}
    resolved = copy.deepcopy(config)
    resolved["base_model"] = sources[config["base_model"]]["path"]
    for index, item in enumerate(resolved["models"]):
        original = config["models"][index]
        ref = original["model"] if isinstance(original, dict) else original
        if isinstance(item, dict):
            item["model"] = sources[ref]["path"]
        else:
            resolved["models"][index] = {"model": sources[ref]["path"]}
    return resolved, list(sources.values())


def _tensor_shapes(model_dir: Path) -> dict[str, tuple[int, ...]]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index["weight_map"]
        shards = sorted(set(weight_map.values()))
    else:
        shards = [item.name for item in model_dir.glob("*.safetensors")]
        weight_map = None
    if not shards:
        raise ValueError(f"No safetensors weights found in {model_dir}")
    shapes: dict[str, tuple[int, ...]] = {}
    for shard in shards:
        path = model_dir / shard
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint shard: {path}")
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                shapes[name] = tuple(handle.get_slice(name).get_shape())
    if weight_map is not None and set(shapes) != set(weight_map):
        raise ValueError(f"Safetensors index does not match checkpoint shards in {model_dir}")
    return shapes


def check_compatibility(sources: list[dict[str, Any]]) -> dict[str, Any]:
    configurations = []
    shapes = []
    for source in sources:
        model_dir = Path(source["path"])
        config_path = model_dir / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing model config: {config_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("model_type") != "gemma3_text" or "Gemma3ForCausalLM" not in config.get("architectures", []):
            raise ValueError(f"{source['requested']} is not a Gemma 3 text-only CausalLM checkpoint")
        if not (model_dir / "tokenizer.json").is_file() and not (model_dir / "tokenizer.model").is_file():
            raise ValueError(f"Missing tokenizer files in {model_dir}")
        configurations.append(config)
        shapes.append(_tensor_shapes(model_dir))

    anchor_config = configurations[0]
    anchor_shapes = shapes[0]
    for source, config, tensor_map in zip(sources[1:], configurations[1:], shapes[1:]):
        differences = [field for field in ARCH_FIELDS if config.get(field) != anchor_config.get(field)]
        if differences:
            raise ValueError(f"{source['requested']} has incompatible architecture fields: {', '.join(differences)}")
        if tensor_map != anchor_shapes:
            missing = sorted(set(anchor_shapes) - set(tensor_map))
            extra = sorted(set(tensor_map) - set(anchor_shapes))
            changed = sorted(name for name in set(anchor_shapes) & set(tensor_map) if anchor_shapes[name] != tensor_map[name])
            raise ValueError(
                f"{source['requested']} has incompatible tensors: "
                f"{len(missing)} missing, {len(extra)} extra, {len(changed)} shape mismatches; "
                f"examples: {(missing + extra + changed)[:5]}"
            )
    return {
        "model_type": "gemma3_text",
        "tensor_count": len(anchor_shapes),
        "architecture": {field: anchor_config.get(field) for field in ARCH_FIELDS},
        "checked_sources": [source["requested"] for source in sources],
    }
