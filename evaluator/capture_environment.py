"""Print actual evaluator runtime metadata, without loading any model or data."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import hashlib


def capture() -> dict:
    record = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": dict(sorted(
            (distribution.metadata["Name"], distribution.version)
            for distribution in importlib.metadata.distributions()
        )),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "dataset": {
            "repository": os.environ.get("VEZILKA_DATASET_REPOSITORY"),
            "revision": os.environ.get("VEZILKA_DATASET_REVISION"),
        },
    }
    record["evaluator_revision"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    diff = subprocess.check_output(["git", "diff", "--no-ext-diff", "HEAD"])
    record["evaluator_diff_sha256"] = hashlib.sha256(diff).hexdigest()
    try:
        import torch

        record["torch_cuda"] = torch.version.cuda
        record["cuda_available"] = torch.cuda.is_available()
        record["cuda_devices"] = []
        if record["cuda_available"]:
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                record["cuda_devices"].append({
                    "logical_index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "compute_capability": [properties.major, properties.minor],
                })
    except Exception as error:
        record["torch_error"] = str(error)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if result.returncode:
            record["nvidia_smi_error"] = result.stderr.strip() or f"Exit code {result.returncode}"
        else:
            record["nvidia_gpus"] = [
                dict(zip(("physical_index", "uuid", "name", "driver_version"), (value.strip() for value in row)))
                for row in csv.reader(result.stdout.splitlines())
            ]
    except (OSError, subprocess.TimeoutExpired) as error:
        record["nvidia_smi_error"] = str(error)
    return record


if __name__ == "__main__":
    print(json.dumps(capture(), indent=2))
