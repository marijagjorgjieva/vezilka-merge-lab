"""Checks for merge configurations, compatibility, and evaluation reuse."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from vezilka_merge_lab.cli import _previous_run_models, main
from vezilka_merge_lab.config import CONFIG_DIR, ROOT, custom_slerp, load_yaml, override_slerp_models, validate_config
from vezilka_merge_lab.evaluation import write_comparison
from vezilka_merge_lab.models import check_compatibility


class EnvironmentTests(unittest.TestCase):
    def test_cli_loads_project_env_without_overriding_shell(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text(
                "HF_TOKEN=file-token\nVEZILKA_ARTIFACTS_DIR=/tmp/custom-artifacts\n",
                encoding="utf-8",
            )
            with (
                patch("vezilka_merge_lab.cli.ROOT", root),
                patch.dict(os.environ, {"HF_TOKEN": "shell-token"}, clear=True),
                patch("vezilka_merge_lab.cli._list", return_value=0),
            ):
                self.assertEqual(main(["list"]), 0)
                self.assertEqual(os.environ["HF_TOKEN"], "shell-token")
                self.assertEqual(os.environ["VEZILKA_ARTIFACTS_DIR"], "/tmp/custom-artifacts")

    def test_cli_does_not_require_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch("vezilka_merge_lab.cli.ROOT", Path(tmp)),
                patch("vezilka_merge_lab.cli._list", return_value=0),
            ):
                self.assertEqual(main(["list"]), 0)


class ConfigurationTests(unittest.TestCase):
    def test_all_included_configurations_are_available(self) -> None:
        paths = sorted(CONFIG_DIR.glob("*/*.yaml"))
        self.assertEqual(len(paths), 18)
        self.assertFalse(any("fail" in path.stem for path in paths))
        for path in paths:
            with self.subTest(path=path.name):
                validate_config(load_yaml(path))
        self.assertEqual(
            load_yaml(CONFIG_DIR / "e2" / "e2-slerp-020.yaml")["parameters"]["t"][-1]["value"],
            0.20,
        )

    def test_model_override_preserves_merge_parameters(self) -> None:
        original = load_yaml(CONFIG_DIR / "e2" / "e2-slerp-020.yaml")
        changed = override_slerp_models(original, "example/new-base", "example/new-donor")
        self.assertEqual(changed["base_model"], "example/new-base")
        self.assertEqual(changed["models"][1]["model"], "example/new-donor")
        self.assertEqual(changed["parameters"], original["parameters"])
        self.assertEqual(original["base_model"], "finki-ukim/VezilkaLLM-Instruct")

    def test_direct_slerp_protects_embeddings(self) -> None:
        config = custom_slerp("example/base", "example/donor", 0.20)
        self.assertEqual(config["parameters"]["t"][0], {"filter": "embed_tokens", "value": 0.0})
        with self.assertRaises(ValueError):
            custom_slerp("example/base", "example/donor", 1.2)


class CompatibilityTests(unittest.TestCase):
    def _model(self, root: Path, name: str, width: int) -> dict:
        model = root / name
        model.mkdir()
        (model / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "gemma3_text",
                    "architectures": ["Gemma3ForCausalLM"],
                    "vocab_size": 4,
                    "hidden_size": 2,
                    "intermediate_size": 8,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 1,
                    "num_key_value_heads": 1,
                    "head_dim": 2,
                }
            ),
            encoding="utf-8",
        )
        (model / "tokenizer.json").write_text("{}", encoding="utf-8")
        save_file({"model.layers.0.weight": torch.zeros(width)}, model / "model.safetensors")
        return {"requested": name, "path": str(model)}

    def test_tensor_shape_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = self._model(root, "base", 2)
            donor = self._model(root, "donor", 3)
            with self.assertRaisesRegex(ValueError, "shape mismatches"):
                check_compatibility([base, donor])

    def test_matching_models_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = self._model(root, "base", 2)
            donor = self._model(root, "donor", 2)
            self.assertEqual(check_compatibility([base, donor])["tensor_count"], 1)

    def test_prepare_command_records_local_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = self._model(root, "base", 2)
            donor = self._model(root, "donor", 2)
            artifact_root = root / "artifacts"
            result = main(
                [
                    "prepare", "--base-model", base["path"], "--donor-model", donor["path"],
                    "--t", "0.2", "--artifacts-dir", str(artifact_root),
                ]
            )
            self.assertEqual(result, 0)
            manifests = list((artifact_root / "runs").glob("*/manifest.json"))
            self.assertEqual(len(manifests), 1)
            self.assertEqual(json.loads(manifests[0].read_text())["status"], "prepared")


class EvaluationTests(unittest.TestCase):
    def _previous_run(self, root: Path, *, bundled: bool = True) -> Path:
        run = root / "runs" / "previous-merge"
        base = root / "base-model"
        merged = run / "merged-model"
        for model in (base, merged):
            model.mkdir(parents=True)
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"checkpoint")
        (run / "resolved_merge.yaml").write_text(f"base_model: {base}\n", encoding="utf-8")
        (run / "manifest.json").write_text(json.dumps({
            "command": "run", "merged_model": str(merged), "bundled_configuration": bundled,
        }), encoding="utf-8")
        return run

    def test_evaluate_previous_run_reuses_local_models_and_fixed_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifacts = Path(tmp) / "artifacts"
            previous = self._previous_run(artifacts)
            with (
                patch("vezilka_merge_lab.cli._check_device"),
                patch("vezilka_merge_lab.cli.prepare_reference", side_effect=AssertionError("download attempted")),
                patch("vezilka_merge_lab.cli._merge", side_effect=AssertionError("merge attempted")),
                patch("vezilka_merge_lab.cli.evaluate_models", return_value=[]) as evaluate,
            ):
                result = main([
                    "evaluate", "--run", str(previous), "--artifacts-dir", str(artifacts),
                    "--prompt-format", "both", "--eval-scope", "full",
                ])
            self.assertEqual(result, 0)
            self.assertEqual(evaluate.call_args.args[0], {
                "base": str(artifacts / "base-model"),
                "merged": str(previous / "merged-model"),
            })
            self.assertEqual(
                evaluate.call_args.kwargs["chat_template"],
                ROOT / "configs" / "evaluation" / "gemma_chat_template.jinja",
            )
            manifests = list((artifacts / "runs").glob("evaluation-*/manifest.json"))
            self.assertEqual(len(manifests), 1)
            manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["source_run"], str(previous))

    def test_evaluate_previous_run_rejects_missing_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            previous = self._previous_run(Path(tmp))
            (previous / "merged-model" / "model.safetensors").unlink()
            with self.assertRaisesRegex(ValueError, "merged checkpoint is missing"):
                _previous_run_models(previous)

    def test_previous_manifest_keeps_fixed_template_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            previous = self._previous_run(Path(tmp))
            manifest_path = previous / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["bundled_experiment"] = manifest.pop("bundled_configuration")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            _, bundled = _previous_run_models(previous)
            self.assertTrue(bundled)

    def test_evaluate_previous_run_rejects_unfinished_merge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            previous = self._previous_run(Path(tmp))
            (previous / "manifest.json").write_text(json.dumps({
                "command": "run", "status": "failed",
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no completed merge"):
                _previous_run_models(previous)

    def test_evaluate_previous_run_rejects_base_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            previous = self._previous_run(Path(tmp))
            self.assertEqual(main([
                "evaluate", "--run", str(previous), "--base-model", "example/other", "--dry-run",
            ]), 2)

    def test_comparison_uses_normalized_accuracy_where_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "evaluation" / "base" / "plain"
            merged = root / "evaluation" / "merged" / "plain"
            base.mkdir(parents=True)
            merged.mkdir(parents=True)
            (base / "results.json").write_text(json.dumps({"results": {
                "arc_easy": {"acc": 0.3, "acc_norm": 0.4},
                "boolq": {"acc": 0.5},
            }}), encoding="utf-8")
            (merged / "results.json").write_text(json.dumps({"results": {
                "arc_easy": {"acc": 0.9, "acc_norm": 0.45},
                "boolq": {"acc": 0.6},
            }}), encoding="utf-8")
            write_comparison(root, "full", ["plain"])
            rows = json.loads((root / "evaluation" / "comparison.json").read_text())["prompt_formats"]["plain"]
            self.assertEqual(rows[0]["metric"], "acc_norm")
            self.assertAlmostEqual(rows[0]["delta_percentage_points"], 5.0)
            self.assertEqual(rows[1]["metric"], "acc")


if __name__ == "__main__":
    unittest.main()
