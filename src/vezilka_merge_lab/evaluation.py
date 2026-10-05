"""Run the pinned LVSTCK Macedonian benchmark in an isolated environment."""

from __future__ import annotations

import datetime as dt
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .config import FULL_TASKS, ROOT, load_registry


EVALUATOR_PACKAGES = (
    "torch==2.12.1",
    "transformers==4.57.6",
    "datasets==2.14.6",
    "peft==0.19.1",
    "accelerate==1.14.0",
    "numpy==1.26.4",
    "pyarrow==14.0.2",
    "huggingface-hub==0.36.2",
    "fsspec==2023.10.0",
)

EVALUATOR_PATCHES = (
    ROOT / "patches" / "lvstk-chat-template.patch",
    ROOT / "patches" / "lvstk-cache-key.patch",
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

    if not marker.exists():
        uv = shutil.which("uv")
        if not uv:
            raise RuntimeError("uv is required to install the pinned evaluator dependencies")
        _run_setup(
            [uv, "pip", "install", "--python", str(python), "-e", str(source), *EVALUATOR_PACKAGES],
            ROOT,
            log,
        )
        marker.write_text(
            json.dumps({"repository": spec["repository"], "revision": spec["revision"], "packages": EVALUATOR_PACKAGES}, indent=2) + "\n",
            encoding="utf-8",
        )
    return source, python


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
    ]
    if scope == "smoke":
        command.extend(["--limit", "10"])
    (output / "command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    with (output / "stdout_stderr.log").open("w", encoding="utf-8") as log:
        process = subprocess.run(command, cwd=source, stdout=log, stderr=subprocess.STDOUT, check=False)
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
