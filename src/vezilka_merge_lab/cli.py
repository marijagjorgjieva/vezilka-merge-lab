"""Command-line entry point for Vezilka Merge Lab."""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from .config import (
    CONFIG_DIR,
    ROOT,
    config_digest,
    custom_slerp,
    is_bundled,
    load_yaml,
    model_refs,
    override_slerp_models,
    validate_config,
)
from .evaluation import evaluate_models
from .models import check_compatibility, prepare_config, prepare_reference


def _add_merge_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, help="MergeKit YAML to run")
    parser.add_argument("--base-model", help="Base model ID or local directory")
    parser.add_argument("--donor-model", help="Single donor model ID or local directory")
    parser.add_argument("--method", choices=["slerp"], help="Direct-command merge method")
    parser.add_argument("--t", type=float, help="Direct-command SLERP donor contribution")
    parser.add_argument("--artifacts-dir", type=Path, help="Directory for downloads and run outputs")
    parser.add_argument("--device", default="cuda:0", help="Evaluation device and merge CUDA mode")
    parser.add_argument("--dry-run", action="store_true", help="Show the plan without downloading or writing")


def _add_evaluation_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--eval-scope", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--prompt-format", choices=["plain", "chat", "both"], default="plain")
    parser.add_argument("--chat-template", type=Path, help="Jinja template to use for chat evaluation")
    parser.add_argument("--batch-size", type=int, default=1)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="merge-model", description="Reproduce and adapt Vezilka model merges.")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("list", help="List the included merge configurations")
    doctor = commands.add_parser("doctor", help="Check local tools and hardware")
    doctor.add_argument("--artifacts-dir", type=Path)

    validate = commands.add_parser("validate", help="Validate YAML without downloading models")
    validate.add_argument("config", nargs="?", type=Path)
    validate.add_argument("--all", action="store_true", help="Validate all bundled YAMLs")

    for name, help_text in (
        ("prepare", "Download models, extract known donors, and check compatibility"),
        ("merge", "Prepare models and run MergeKit"),
        ("run", "Prepare, merge, and evaluate in one command"),
    ):
        sub = commands.add_parser(name, help=help_text)
        _add_merge_inputs(sub)
        if name == "run":
            _add_evaluation_options(sub)

    evaluate = commands.add_parser("evaluate", help="Evaluate an existing model or completed merge run")
    evaluation_source = evaluate.add_mutually_exclusive_group(required=True)
    evaluation_source.add_argument("--model", help="Model directory or Hugging Face ID")
    evaluation_source.add_argument("--run", type=Path, help="Previous merge run directory; reuse its base and merged models")
    evaluate.add_argument("--base-model", help="Optional base model for comparison")
    evaluate.add_argument("--artifacts-dir", type=Path)
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.add_argument("--dry-run", action="store_true")
    _add_evaluation_options(evaluate)
    return parser.parse_args(argv)


def _artifacts_dir(args: argparse.Namespace) -> Path:
    value = args.artifacts_dir or os.environ.get("VEZILKA_ARTIFACTS_DIR") or ROOT / "artifacts"
    return Path(value).expanduser().resolve()


def _configure_cache(artifacts: Path) -> None:
    os.environ.setdefault("HF_HOME", str(artifacts / "huggingface"))


def _check_device(device: str) -> None:
    if device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; use merge-model doctor to inspect this machine")


def _load_input(args: argparse.Namespace) -> tuple[dict[str, Any], Path | None, bool]:
    if args.config:
        if args.method is not None or args.t is not None:
            raise ValueError("--method and --t are for direct commands; edit a copy of the YAML to change merge parameters")
        path = args.config.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        config = load_yaml(path)
        validate_config(config)
        overridden = bool(args.base_model or args.donor_model)
        if overridden:
            config = override_slerp_models(config, args.base_model, args.donor_model)
        return config, path, overridden

    if not args.base_model or not args.donor_model or args.t is None:
        raise ValueError("Supply --config, or supply --base-model, --donor-model, and --t")
    if args.method not in (None, "slerp"):
        raise ValueError("Direct commands currently support SLERP only")
    return custom_slerp(args.base_model, args.donor_model, args.t), None, True


def _run_label(config: dict[str, Any], source: Path | None, overridden: bool) -> str:
    if source and not overridden:
        return source.stem
    return f"custom-{config['merge_method']}-{config_digest(config)[:8]}"


def _new_run_dir(artifacts: Path, label: str, config: dict[str, Any]) -> Path:
    safe_label = re.sub(r"[^a-z0-9-]+", "-", label.lower()).strip("-")
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = artifacts / "runs" / f"{safe_label}-{timestamp}-{config_digest(config)[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_json(path: Path, data: dict[str, Any] | list[Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _environment() -> dict[str, str]:
    values = {"python": sys.version.split()[0], "uv": shutil.which("uv") or "not found"}
    for name in ("mergekit", "torch", "transformers", "huggingface-hub", "safetensors", "pyyaml"):
        try:
            values[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            values[name] = "not installed"
    return values


def _template(args: argparse.Namespace, bundled: bool) -> Path | None:
    candidate = args.chat_template or (
        ROOT / "configs" / "evaluation" / "gemma_chat_template.jinja" if bundled else None
    )
    if candidate and not candidate.is_file():
        raise FileNotFoundError(f"Chat template does not exist: {candidate}")
    return candidate.resolve() if candidate else None


def _require_model_chat_template(model_dir: Path) -> None:
    if (model_dir / "chat_template.jinja").is_file():
        return
    tokenizer_config = model_dir / "tokenizer_config.json"
    if tokenizer_config.is_file():
        data = json.loads(tokenizer_config.read_text(encoding="utf-8"))
        if data.get("chat_template"):
            return
    raise ValueError(
        f"No chat template found for {model_dir}; supply --chat-template PATH or use plain prompts"
    )


def _merge(resolved_yaml: Path, output: Path, device: str, log: Path) -> None:
    command = [
        sys.executable, "-m", "vezilka_merge_lab.mergekit_yaml_compat",
        str(resolved_yaml), str(output), "--copy-tokenizer",
    ]
    if device.startswith("cuda"):
        command.append("--cuda")
    _write_json(log.with_name("merge_command.json"), command)
    print(f"Merging → {output}", flush=True)
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"MergeKit exited with code {result.returncode}; see {log}")
    if not (output / "config.json").is_file():
        raise RuntimeError(f"MergeKit did not produce a model config in {output}; see {log}")


def _execute(args: argparse.Namespace) -> int:
    config, source_yaml, overridden = _load_input(args)
    bundled = is_bundled(source_yaml, overridden)
    label = _run_label(config, source_yaml, overridden)
    artifacts = _artifacts_dir(args)
    if args.dry_run:
        print(f"Plan: {args.command} {label}")
        print(f"Models: {', '.join(model_refs(config))}")
        print(f"Artifacts: {artifacts}")
        if args.command == "run":
            print(f"Evaluation: {args.eval_scope}, {args.prompt_format}, base and merged")
        return 0

    if args.command == "run" and args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    template = _template(args, bundled) if args.command == "run" else None
    if args.command in ("merge", "run"):
        _check_device(args.device)
    _configure_cache(artifacts)
    run_dir = _new_run_dir(artifacts, label, config)
    original = run_dir / "source_merge.yaml"
    if source_yaml:
        shutil.copy2(source_yaml, original)
    else:
        original.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    manifest: dict[str, Any] = {
        "status": "preparing",
        "command": args.command,
        "label": label,
        "source_yaml": str(source_yaml) if source_yaml else None,
        "bundled_configuration": bundled,
        "config_sha256": config_digest(config),
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "environment": _environment(),
    }
    _write_json(run_dir / "manifest.json", manifest)
    try:
        resolved, sources = prepare_config(
            config, artifacts / "prepared-models", source_yaml.parent if source_yaml else Path.cwd()
        )
        manifest["sources"] = sources
        manifest["compatibility"] = check_compatibility(sources)
        resolved_yaml = run_dir / "resolved_merge.yaml"
        resolved_yaml.write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
        manifest["status"] = "prepared"
        _write_json(run_dir / "manifest.json", manifest)
        if args.command == "run" and args.prompt_format in ("chat", "both") and not template:
            _require_model_chat_template(Path(resolved["base_model"]))
        if args.command in ("merge", "run"):
            output = run_dir / "merged-model"
            _merge(resolved_yaml, output, args.device, run_dir / "merge_stdout_stderr.log")
            manifest["merged_model"] = str(output)
            manifest["status"] = "merged"
            _write_json(run_dir / "manifest.json", manifest)
        if args.command == "run":
            manifest["evaluation"] = evaluate_models(
                {"base": resolved["base_model"], "merged": str(output)},
                run_dir, artifacts, scope=args.eval_scope, prompt_format=args.prompt_format,
                chat_template=template, device=args.device, batch_size=args.batch_size,
            )
            manifest["status"] = "complete"
        manifest["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        _write_json(run_dir / "manifest.json", manifest)
        print(f"Done: {run_dir}")
        return 0
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        manifest["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        _write_json(run_dir / "manifest.json", manifest)
        raise


def _previous_run_models(run: Path) -> tuple[dict[str, str], bool]:
    manifest_path = run / "manifest.json"
    resolved_yaml = run / "resolved_merge.yaml"
    if not manifest_path.is_file():
        raise ValueError(f"Not a merge run directory: {run}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("command") not in ("merge", "run") or not manifest.get("merged_model"):
        raise ValueError(f"Previous run has no completed merge: {run}")
    if not resolved_yaml.is_file():
        raise ValueError(f"Previous run has no resolved merge YAML: {run}")
    base_ref = load_yaml(resolved_yaml).get("base_model")
    if not isinstance(base_ref, str):
        raise ValueError(f"Previous run has no resolved base model: {run}")
    base = Path(base_ref)
    if not base.is_absolute():
        base = run / base
    models = {"base": base.resolve(), "merged": (run / "merged-model").resolve()}
    for label, path in models.items():
        if not (path / "config.json").is_file() or not (
            (path / "model.safetensors.index.json").is_file() or any(path.glob("*.safetensors"))
        ):
            raise ValueError(f"Previous run's {label} checkpoint is missing or incomplete: {path}")
    bundled = manifest.get("bundled_configuration", manifest.get("bundled_experiment", False))
    return {label: str(path) for label, path in models.items()}, bool(bundled)


def _evaluate_existing(args: argparse.Namespace) -> int:
    artifacts = _artifacts_dir(args)
    source_run = args.run.expanduser().resolve() if args.run else None
    if source_run and args.base_model:
        raise ValueError("--base-model cannot be combined with --run; the previous run supplies its base model")
    previous_models, bundled = _previous_run_models(source_run) if source_run else ({}, False)
    model_ref = previous_models.get("merged", args.model)
    base_ref = previous_models.get("base", args.base_model)
    if args.dry_run:
        print(f"Plan: evaluate {model_ref} ({args.eval_scope}, {args.prompt_format})")
        if base_ref:
            print(f"Base comparison: {base_ref}")
        if source_run:
            print(f"Source run: {source_run}")
        return 0
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    template = _template(args, bundled=bundled)
    _check_device(args.device)
    _configure_cache(artifacts)
    run_dir = _new_run_dir(artifacts, "evaluation", {"model": model_ref, "base": base_ref})
    prepared_root = artifacts / "prepared-models"
    manifest: dict[str, Any] = {
        "status": "preparing", "command": "evaluate",
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "environment": _environment(),
    }
    if source_run:
        manifest["source_run"] = str(source_run)
    _write_json(run_dir / "manifest.json", manifest)
    try:
        if source_run:
            prepared = {
                label: {
                    "requested": path, "repo": None, "revision": None,
                    "path": path, "extracted_text_only": False,
                }
                for label, path in previous_models.items()
            }
        else:
            prepared = {"merged": prepare_reference(model_ref, prepared_root)}
            if base_ref:
                prepared = {"base": prepare_reference(base_ref, prepared_root), **prepared}
        selected = {name: source["path"] for name, source in prepared.items()}
        manifest["sources"] = prepared
        if args.prompt_format in ("chat", "both") and not template:
            for path in selected.values():
                _require_model_chat_template(Path(path))
        manifest["evaluation"] = evaluate_models(
            selected, run_dir, artifacts, scope=args.eval_scope, prompt_format=args.prompt_format,
            chat_template=template, device=args.device, batch_size=args.batch_size,
        )
        manifest["status"] = "complete"
        manifest["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        _write_json(run_dir / "manifest.json", manifest)
        print(f"Done: {run_dir}")
        return 0
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        manifest["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        _write_json(run_dir / "manifest.json", manifest)
        raise


def _list() -> int:
    for path in sorted(CONFIG_DIR.glob("*/*.yaml")):
        config = load_yaml(path)
        print(f"{path.relative_to(ROOT)}  [{config['merge_method']}] ")
    return 0


def _validate(args: argparse.Namespace) -> int:
    if bool(args.config) == bool(args.all):
        raise ValueError("Pass one YAML path or --all")
    paths = sorted(CONFIG_DIR.glob("*/*.yaml")) if args.all else [args.config]
    if not paths:
        raise ValueError("No bundled YAMLs found")
    for path in paths:
        validate_config(load_yaml(path))
    print(f"Validated {len(paths)} MergeKit YAML(s).")
    return 0


def _doctor(args: argparse.Namespace) -> int:
    artifacts = _artifacts_dir(args)
    print(f"Project: {ROOT}")
    print(f"Artifacts: {artifacts}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"uv: {shutil.which('uv') or 'not found'}")
    print(f"git: {shutil.which('git') or 'not found'}")
    print(f"HF_TOKEN set: {bool(os.environ.get('HF_TOKEN'))}")
    try:
        import torch
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch: not installed")
    disk = shutil.disk_usage(artifacts if artifacts.exists() else ROOT)
    print(f"Free disk: {disk.free / 1024**3:.1f} GiB")
    print(f"Evaluator installed: {(artifacts / 'evaluator' / 'installed.json').is_file()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env", override=False)
    args = parse_args(argv)
    try:
        if args.command == "list":
            return _list()
        if args.command == "doctor":
            return _doctor(args)
        if args.command == "validate":
            return _validate(args)
        if args.command == "evaluate":
            return _evaluate_existing(args)
        return _execute(args)
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
