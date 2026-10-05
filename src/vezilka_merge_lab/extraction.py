"""Extract Gemma 3 language-model tensors from a known multimodal donor."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


TOKENIZER_FILES = [
    "added_tokens.json",
    "chat_template.jinja",
    "generation_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
]


def text_config_from_multimodal(config_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    text_config = dict(config["text_config"])
    text_config["architectures"] = ["Gemma3ForCausalLM"]
    text_config["model_type"] = "gemma3_text"
    if "torch_dtype" not in text_config and "dtype" in text_config:
        text_config["torch_dtype"] = text_config["dtype"]
    if "dtype" not in text_config and "torch_dtype" in text_config:
        text_config["dtype"] = text_config["torch_dtype"]
    text_config.setdefault("tie_word_embeddings", True)
    text_config["transformers_version"] = config.get("transformers_version", text_config.get("transformers_version"))
    return {key: value for key, value in text_config.items() if value is not None}


def shard_files(source_dir: Path) -> list[Path]:
    index_path = source_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        names = sorted(set(index["weight_map"].values()))
        return [source_dir / name for name in names]
    return sorted(source_dir.glob("*.safetensors"))


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def extract_tensors(source_dir: Path, output_dir: Path) -> tuple[int, int]:
    weight_map: dict[str, str] = {}
    total_size = 0
    tensor_count = 0
    shards = shard_files(source_dir)

    for idx, source_shard in enumerate(shards, start=1):
        output_name = f"model-{idx:05d}-of-{len(shards):05d}.safetensors"
        output_shard = output_dir / output_name
        tensors: dict[str, torch.Tensor] = {}

        with safe_open(source_shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if not key.startswith("language_model."):
                    continue
                new_key = key.removeprefix("language_model.")
                tensor = handle.get_tensor(key)
                tensors[new_key] = tensor
                weight_map[new_key] = output_name
                total_size += tensor_nbytes(tensor)
                tensor_count += 1

        if tensors:
            save_file(tensors, output_shard, metadata={"format": "pt"})

    if tensor_count == 0:
        raise ValueError(
            f"No text tensors found in {source_dir}; expected language_model.* tensor names"
        )

    index = {
        "metadata": {
            "total_size": total_size,
            "text_only_extraction": True,
        },
        "weight_map": dict(sorted(weight_map.items())),
    }
    (output_dir / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return tensor_count, total_size


def copy_tokenizer_files(source_dir: Path, output_dir: Path) -> list[str]:
    copied = []
    for name in TOKENIZER_FILES:
        source = source_dir / name
        if source.exists():
            shutil.copy2(source, output_dir / name)
            copied.append(name)
    return copied
