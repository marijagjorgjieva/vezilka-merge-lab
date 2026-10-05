"""Run the pinned LVSTCK Macedonian benchmark in an isolated environment."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .config import FULL_TASKS, ROOT, load_registry


EVALUATOR_LOCK = ROOT / "evaluator" / "requirements.lock"

EVALUATOR_PATCHES = (
    ROOT / "patches" / "lvstk-chat-template.patch",
    ROOT / "patches" / "lvstk-cache-key.patch",
    ROOT / "patches" / "lvstk-dataset-revision.patch",
)


def _run_setup(command: list[str], cwd: Path | None, log: Path) -> None:
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"$ {' '.join(command)}\n")
        stream.flush()
        result = subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            raise RuntimeError(f"Evaluator setup failed; see {log}")


def ensure_evaluator(artifacts: Path) -> tuple[Path, Path]:
    spec = load_registry()["evaluator"]
    evaluator_root = artifacts / "evaluator"
    source = evaluator_root / "source"
    python = evaluator_root / "venv" / "bin" / "python"
    marker = evaluator_root / "installed.json"
    log = evaluator_root / "setup.log"
    evaluator_root.mkdir(parents=True, exist_ok=True)

    if not source.exists():
        _run_setup(["git", "clone", spec["repository"], str(source)], None, log)
        _run_setup(["git", "checkout", "--detach", spec["revision"]], source, log)
    else:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        if head != spec["revision"]:
            raise RuntimeError(f"Evaluator checkout has unexpected commit {head}; expected {spec['revision']}")

    for patch in EVALUATOR_PATCHES:
        applied = subprocess.run(["git", "apply", "--reverse", "--check", str(patch)], cwd=source, capture_output=True)
        if applied.returncode == 0:
            continue
        applicable = subprocess.run(["git", "apply", "--check", str(patch)], cwd=source, capture_output=True)
        if applicable.returncode:
            raise RuntimeError(f"Evaluator patch cannot be applied cleanly: {patch}; see {source}")
        _run_setup(["git", "apply", str(patch)], source, log)

    if not python.exists():
        uv = shutil.which("uv")
        if not uv:
            raise RuntimeError("uv is required to create the isolated evaluator environment")
        _run_setup([uv, "venv", "--python", spec["python"], str(python.parent.parent)], ROOT, log)

    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv is required to sync the locked evaluator dependencies")
    lock_hash = hashlib.sha256(EVALUATOR_LOCK.read_bytes()).hexdigest()
    # Reconcile every time: an old marker cannot hide an edited environment.
    marker.unlink(missing_ok=True)
    _run_setup(
        [uv, "pip", "sync", "--python", str(python), "--require-hashes", "--strict", str(EVALUATOR_LOCK)],
        ROOT,
        log,
    )
    # The source is pinned above; hash checking does not support editable paths.
    # Install it separately, using the locked build backend and dependencies.
    _run_setup(
        [uv, "pip", "install", "--python", str(python), "--no-deps", "--no-build-isolation", "--no-index", "-e", str(source)],
        ROOT,
        log,
    )
    _run_setup([uv, "pip", "check", "--python", str(python)], ROOT, log)
    marker.write_text(
        json.dumps({
            "repository": spec["repository"], "revision": spec["revision"],
            "dependency_lock": str(EVALUATOR_LOCK), "dependency_lock_sha256": lock_hash,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    return source, python


def _record_environment(
    python: Path, source: Path, output: Path, environment: dict[str, str],
    *, command: list[str], model_path: str, chat_template: Path | None,
    prompt: str, device: str, batch_size: int, scope: str,
) -> None:
    record = json.loads(subprocess.check_output(
        [str(python), str(ROOT / "evaluator" / "capture_environment.py")],
        cwd=source, env=environment, text=True,
    ))
    spec = load_registry()
    if record["evaluator_revision"] != spec["evaluator"]["revision"]:
        raise RuntimeError("Evaluator revision changed before evaluation")
    record["dependency_lock_sha256"] = hashlib.sha256(EVALUATOR_LOCK.read_bytes()).hexdigest()
    record["patches_sha256"] = {
        patch.name: hashlib.sha256(patch.read_bytes()).hexdigest() for patch in EVALUATOR_PATCHES
    }
    record["dataset"]["expected_files_sha256"] = spec["dataset"]["files_sha256"]
    record["evaluation"] = {
        "command": command, "device": device, "batch_size": batch_size,
        "scope": scope, "prompt_format": prompt, "dtype": "bfloat16", "num_fewshot": 0,
        "score_cache_enabled": False,
    }
    record["chat_template_sha256"] = None
    if prompt == "chat":
        template = chat_template or Path(model_path) / "chat_template.jinja"
        if template.is_file():
            record["chat_template_sha256"] = hashlib.sha256(template.read_bytes()).hexdigest()
            record["chat_template_source"] = str(template)
        else:
            tokenizer_config = Path(model_path) / "tokenizer_config.json"
            if tokenizer_config.is_file():
                template_value = json.loads(tokenizer_config.read_text()).get("chat_template")
                if template_value is not None:
                    record["chat_template_sha256"] = hashlib.sha256(
                        json.dumps(template_value, sort_keys=True).encode()
                    ).hexdigest()
                    record["chat_template_source"] = str(tokenizer_config)
    record["captured_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    (output / "environment.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def _evaluate_one(
    source: Path,
    python: Path,
    model_path: str,
    output: Path,
    *,
    scope: str,
    prompt: str,
    chat_template: Path | None,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    model_args = f"pretrained={model_path},dtype=bfloat16"
    if prompt == "chat":
        model_args += ",chat_template=true"
        if chat_template:
            model_args += f",chat_template_path={chat_template.resolve()}"
    command = [
        str(python), "main.py", "--language", "Macedonian",
        "--model", "hf-causal-experimental", "--model_args", model_args,
        "--tasks", "arc_easy" if scope == "smoke" else FULL_TASKS,
        "--batch_size", str(batch_size), "--device", device,
        "--num_fewshot", "0", "--output_path", str(output / "results.json"),
        "--no_cache",
    ]
    if scope == "smoke":
        command.extend(["--limit", "10"])
    dataset = load_registry()["dataset"]
    if dataset["repository"] != "LVSTCK/macedonian-llm-eval" or not re.fullmatch(r"[0-9a-f]{40}", dataset["revision"]):
        raise ValueError("The Macedonian benchmark must have a pinned dataset commit SHA")
    environment = os.environ.copy()
    environment["VEZILKA_DATASET_REPOSITORY"] = dataset["repository"]
    environment["VEZILKA_DATASET_REVISION"] = dataset["revision"]
    (output / "command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
    _record_environment(
        python, source, output, environment, command=command, model_path=model_path,
        chat_template=chat_template, prompt=prompt, device=device, batch_size=batch_size, scope=scope,
    )
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    with (output / "stdout_stderr.log").open("w", encoding="utf-8") as log:
        process = subprocess.run(command, cwd=source, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
    result_path = output / "results.json"
    status = {
        "status": "success" if process.returncode == 0 and result_path.is_file() else "failed",
        "exit_code": process.returncode,
        "started_utc": started,
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": model_path,
        "scope": scope,
        "prompt_format": prompt,
        "chat_template": str(chat_template) if chat_template else "model tokenizer",
        "results": str(result_path),
    }
    (output / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    if status["status"] != "success":
        raise RuntimeError(f"Benchmark failed; see {output / 'stdout_stderr.log'}")
    return status


def write_comparison(run_dir: Path, scope: str, prompts: list[str]) -> Path:
    """Compare like-for-like base and merged task metrics."""
    comparisons: dict[str, Any] = {"scope": scope, "prompt_formats": {}}
    lines = ["# Base versus merged model", "", f"Evaluation scope: {scope}", ""]
    for prompt in prompts:
        base_path = run_dir / "evaluation" / "base" / prompt / "results.json"
        merged_path = run_dir / "evaluation" / "merged" / prompt / "results.json"
        base_results = json.loads(base_path.read_text(encoding="utf-8"))["results"]
        merged_results = json.loads(merged_path.read_text(encoding="utf-8"))["results"]
        rows = []
        lines.extend([f"## {prompt}", "", "| Task | Metric | Base | Merged | Change (pp) |", "|---|---|---:|---:|---:|"])
        for task in sorted(set(FULL_TASKS.split(",")) & set(base_results) & set(merged_results)):
            metric = "acc_norm" if "acc_norm" in base_results[task] and "acc_norm" in merged_results[task] else "acc"
            base_score = base_results[task][metric]
            merged_score = merged_results[task][metric]
            change = 100 * (merged_score - base_score)
            rows.append({"task": task, "metric": metric, "base": base_score, "merged": merged_score, "delta_percentage_points": change})
            lines.append(f"| {task} | {metric} | {base_score:.4f} | {merged_score:.4f} | {change:+.2f} |")
        lines.append("")
        comparisons["prompt_formats"][prompt] = rows
    comparison_root = run_dir / "evaluation"
    (comparison_root / "comparison.json").write_text(json.dumps(comparisons, indent=2) + "\n", encoding="utf-8")
    path = comparison_root / "comparison.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def evaluate_models(
    models: dict[str, str],
    run_dir: Path,
    artifacts: Path,
    *,
    scope: str,
    prompt_format: str,
    chat_template: Path | None,
    device: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    source, python = ensure_evaluator(artifacts)
    prompts = ["plain", "chat"] if prompt_format == "both" else [prompt_format]
    statuses: list[dict[str, Any]] = []
    for label, model_path in models.items():
        for prompt in prompts:
            output = run_dir / "evaluation" / label / prompt
            print(f"Evaluating {label} ({prompt}, {scope}) → {output}", flush=True)
            statuses.append(
                _evaluate_one(
                    source, python, model_path, output,
                    scope=scope, prompt=prompt, chat_template=chat_template,
                    device=device, batch_size=batch_size,
                )
            )
    if "base" in models and "merged" in models:
        print(f"Comparison: {write_comparison(run_dir, scope, prompts)}", flush=True)
    return statuses
