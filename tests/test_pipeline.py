"""Checks for merge configurations, compatibility, and evaluation reuse."""

from __future__ import annotations

import json
import hashlib
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from vezilka_merge_lab.cli import _merge, _previous_run_models, main
from vezilka_merge_lab.config import CONFIG_DIR, ROOT, config_digest, custom_slerp, load_yaml, override_slerp_models, validate_config
from vezilka_merge_lab.evaluation import EVALUATOR_LOCK, _evaluate_one, _record_environment, ensure_evaluator, write_comparison
from vezilka_merge_lab.extraction import extract_tensors
from vezilka_merge_lab.models import _extract_known, check_compatibility


class MergeFallbackTests(unittest.TestCase):
    def test_cuda_oom_retries_cpu_and_preserves_failed_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            output = root / "merged-model"
            log = root / "merge_stdout_stderr.log"
            def run(command, **kwargs):
                output.mkdir()
                if "--cuda" in command:
                    (output / "partial").write_text("unfinished")
                    kwargs["stdout"].write("torch.OutOfMemoryError: CUDA out of memory\n")
                    return subprocess.CompletedProcess(command, 1)
                (output / "config.json").write_text("{}")
                return subprocess.CompletedProcess(command, 0)
            with patch("vezilka_merge_lab.cli.subprocess.run", side_effect=run) as process:
                _merge(root / "resolved.yaml", output, "cuda:0", log)
            self.assertEqual(process.call_count, 2)
            self.assertNotIn("--cuda", process.call_args.args[0])
            self.assertTrue((root / "merged-model.cuda-failed" / "partial").is_file())
            attempts = json.loads((root / "merge_attempts.json").read_text())
            self.assertEqual([attempt["exit_code"] for attempt in attempts], [1, 0])
            self.assertIn("CUDA out of memory", Path(attempts[0]["log"]).read_text())

    def test_other_errors_and_cpu_oom_do_not_retry(self) -> None:
        for device, message in (("cuda:0", "Invalid config"), ("cpu", "CUDA out of memory")):
            with self.subTest(device=device), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                def run(command, **kwargs):
                    kwargs["stdout"].write(message)
                    return subprocess.CompletedProcess(command, 1)
                with patch("vezilka_merge_lab.cli.subprocess.run", side_effect=run) as process:
                    with self.assertRaisesRegex(RuntimeError, "MergeKit exited"):
                        _merge(root / "resolved.yaml", root / "merged-model", device, root / "merge.log")
                self.assertEqual(process.call_count, 1)

    def test_failed_cpu_retry_stops_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def run(command, **kwargs):
                kwargs["stdout"].write("CUDA out of memory" if "--cuda" in command else "CPU allocation failed")
                return subprocess.CompletedProcess(command, 1)
            with patch("vezilka_merge_lab.cli.subprocess.run", side_effect=run) as process:
                with self.assertRaisesRegex(RuntimeError, "MergeKit exited"):
                    _merge(root / "resolved.yaml", root / "merged-model", "cuda:0", root / "merge.log")
            self.assertEqual(process.call_count, 2)


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

    def test_self_merge_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "distinct model references"):
            custom_slerp("example/base", "example/base", 0.20)
        config = custom_slerp("example/base", "example/donor", 0.20)
        for entries in (["example/base", "example/base"], [{"model": "example/base"}] * 2):
            with self.subTest(entries=entries):
                config["models"] = entries
                with self.assertRaisesRegex(ValueError, "distinct model references"):
                    validate_config(config)

    def test_overrides_cannot_create_self_merge(self) -> None:
        config = custom_slerp("example/base", "example/donor", 0.20)
        for base, donor in (("example/donor", None), (None, "example/base"), ("example/same", "example/same")):
            with self.subTest(base=base, donor=donor):
                with self.assertRaisesRegex(ValueError, "distinct model references"):
                    override_slerp_models(config, base, donor)

    def test_earlier_matching_rule_cannot_override_embedding_protection(self) -> None:
        earlier_rules = [
            {"value": 0.20},
            {"filter": "*", "value": 0.20},
            {"filter": None, "value": 0.20},
            {"filter": "model", "value": 0.20},
            {"filter": "embed_tokens", "value": 0.20},
        ]
        for rule in earlier_rules:
            with self.subTest(rule=rule):
                config = custom_slerp("example/base", "example/donor", 0.20)
                config["parameters"]["t"].insert(0, rule)
                with self.assertRaisesRegex(ValueError, "first SLERP t rule"):
                    validate_config(config)

    def test_nonmatching_rule_can_precede_embedding_protection(self) -> None:
        config = custom_slerp("example/base", "example/donor", 0.20)
        config["parameters"]["t"].insert(0, {"filter": "self_attn", "value": 0.30})
        validate_config(config)


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
            manifest = json.loads(manifests[0].read_text())
            self.assertEqual(manifest["status"], "prepared")
            effective = load_yaml(manifests[0].parent / "effective_merge.yaml")
            self.assertEqual(effective, load_yaml(manifests[0].parent / "source_merge.yaml"))
            self.assertEqual(config_digest(effective), manifest["config_sha256"])
            self.assertEqual(manifest["model_overrides"], {})

    def test_prepare_saves_effective_configuration_after_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            base = self._model(root, "new-base", 2)
            donor = self._model(root, "new-donor", 2)
            source = root / "input.yaml"
            original_bytes = (CONFIG_DIR / "e2" / "e2-slerp-020.yaml").read_bytes()
            source.write_bytes(original_bytes)
            artifacts = root / "artifacts"
            result = main([
                "prepare", "--config", str(source),
                "--base-model", base["path"], "--donor-model", donor["path"],
                "--artifacts-dir", str(artifacts),
            ])
            self.assertEqual(result, 0)
            manifests = list((artifacts / "runs").glob("*/manifest.json"))
            self.assertEqual(len(manifests), 1)
            run = manifests[0].parent
            manifest = json.loads(manifests[0].read_text())
            effective = load_yaml(run / "effective_merge.yaml")
            expected = override_slerp_models(load_yaml(source), base["path"], donor["path"])
            self.assertEqual(effective, expected)
            self.assertEqual(config_digest(effective), manifest["config_sha256"])
            self.assertEqual(Path(manifest["effective_yaml"]), run / "effective_merge.yaml")
            self.assertEqual(manifest["model_overrides"], {
                "base_model": base["path"], "donor_model": donor["path"],
            })
            self.assertEqual((run / "source_merge.yaml").read_bytes(), original_bytes)
            self.assertEqual(source.read_bytes(), original_bytes)
            resolved = load_yaml(run / "resolved_merge.yaml")
            self.assertEqual(resolved["base_model"], str(Path(base["path"]).resolve()))
            self.assertEqual(resolved["models"][1]["model"], str(Path(donor["path"]).resolve()))


class ExtractionTests(unittest.TestCase):
    SPEC = {"source": "example/multimodal", "revision": "abcdef1234567890", "prepared_name": "text-only"}

    def _source(self, root: Path, *, text: bool = True) -> Path:
        raw = root / self.SPEC["revision"]
        raw.mkdir()
        (raw / "config.json").write_text(json.dumps({"text_config": {"hidden_size": 2}}))
        (raw / "tokenizer.json").write_text("{}")
        tensors = {"vision_tower.weight": torch.ones(2)}
        if text:
            tensors["language_model.model.embed_tokens.weight"] = torch.arange(4).reshape(2, 2).float()
        save_file(tensors, raw / "model.safetensors")
        return raw

    def test_empty_extraction_does_not_write_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = self._source(root, text=False)
            output = root / "output"
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "No text tensors"):
                extract_tensors(raw, output)
            self.assertFalse((output / "model.safetensors.index.json").exists())

    def test_failed_extraction_leaves_no_cache_and_can_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = self._source(root, text=False)
            prepared = root / "prepared"
            with patch("vezilka_merge_lab.models._download", return_value=raw):
                with self.assertRaisesRegex(ValueError, "No text tensors"):
                    _extract_known(self.SPEC, prepared)
                self.assertEqual(list(prepared.iterdir()), [])
                save_file({"language_model.model.embed_tokens.weight": torch.ones(2, 2)}, raw / "model.safetensors")
                result = _extract_known(self.SPEC, prepared)
            self.assertTrue((Path(result["path"]) / "extraction.json").is_file())

    def test_invalid_cached_extraction_is_rejected(self) -> None:
        cases = ("zero_count", "empty_index", "missing_shard", "count_mismatch", "missing_metadata")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                raw = self._source(root)
                with patch("vezilka_merge_lab.models._download", return_value=raw):
                    target = Path(_extract_known(self.SPEC, root / "prepared")["path"])
                    metadata_path = target / "extraction.json"
                    metadata = json.loads(metadata_path.read_text())
                    if case in ("zero_count", "count_mismatch"):
                        metadata["tensor_count"] = 0 if case == "zero_count" else 2
                        metadata_path.write_text(json.dumps(metadata))
                    elif case == "empty_index":
                        (target / "model.safetensors.index.json").write_text('{"weight_map": {}}')
                    elif case == "missing_shard":
                        next(target.glob("*.safetensors")).unlink()
                    else:
                        metadata_path.unlink()
                    with self.assertRaisesRegex(ValueError, "Invalid prepared model"):
                        _extract_known(self.SPEC, root / "prepared")

    def test_valid_extraction_preserves_text_tensors_and_reuses_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = self._source(root)
            with patch("vezilka_merge_lab.models._download", return_value=raw):
                result = _extract_known(self.SPEC, root / "prepared")
                target = Path(result["path"])
                metadata = json.loads((target / "extraction.json").read_text())
                self.assertEqual(metadata["tensor_count"], 1)
                self.assertEqual(metadata["tensor_bytes"], 16)
                with safe_open(next(target.glob("*.safetensors")), framework="pt", device="cpu") as handle:
                    self.assertEqual(list(handle.keys()), ["model.embed_tokens.weight"])
                    self.assertTrue(torch.equal(handle.get_tensor("model.embed_tokens.weight"), torch.arange(4).reshape(2, 2).float()))
                with patch("vezilka_merge_lab.models.extraction.extract_tensors", side_effect=AssertionError("cache not reused")):
                    self.assertEqual(_extract_known(self.SPEC, root / "prepared"), result)


class EvaluatorSetupTests(unittest.TestCase):
    def _existing_environment(self, root: Path) -> Path:
        evaluator = root / "evaluator"
        (evaluator / "source").mkdir(parents=True)
        python = evaluator / "venv" / "bin" / "python"
        python.parent.mkdir(parents=True)
        python.touch()
        marker = evaluator / "installed.json"
        marker.write_text('{"packages": ["old-unlocked-dependency"]}')
        return marker

    def test_existing_environment_is_synced_from_hashed_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker = self._existing_environment(root)
            revision = json.loads((ROOT / "configs/model_sources.json").read_text())["evaluator"]["revision"]
            with (
                patch("vezilka_merge_lab.evaluation.subprocess.check_output", return_value=revision),
                patch("vezilka_merge_lab.evaluation.subprocess.run", return_value=subprocess.CompletedProcess([], 0)),
                patch("vezilka_merge_lab.evaluation.shutil.which", return_value="uv"),
                patch("vezilka_merge_lab.evaluation._run_setup") as setup,
            ):
                source, python = ensure_evaluator(root)
            commands = [call.args[0] for call in setup.call_args_list]
            self.assertEqual(commands[0], [
                "uv", "pip", "sync", "--python", str(python),
                "--require-hashes", "--strict", str(EVALUATOR_LOCK),
            ])
            self.assertEqual(commands[1], [
                "uv", "pip", "install", "--python", str(python),
                "--no-deps", "--no-build-isolation", "--no-index", "-e", str(source),
            ])
            self.assertEqual(commands[2], ["uv", "pip", "check", "--python", str(python)])
            metadata = json.loads(marker.read_text())
            self.assertEqual(metadata["dependency_lock_sha256"], hashlib.sha256(EVALUATOR_LOCK.read_bytes()).hexdigest())

    def test_failed_sync_removes_old_success_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker = self._existing_environment(root)
            revision = json.loads((ROOT / "configs/model_sources.json").read_text())["evaluator"]["revision"]
            with (
                patch("vezilka_merge_lab.evaluation.subprocess.check_output", return_value=revision),
                patch("vezilka_merge_lab.evaluation.subprocess.run", return_value=subprocess.CompletedProcess([], 0)),
                patch("vezilka_merge_lab.evaluation.shutil.which", return_value="uv"),
                patch("vezilka_merge_lab.evaluation._run_setup", side_effect=RuntimeError("sync failed")),
            ):
                with self.assertRaisesRegex(RuntimeError, "sync failed"):
                    ensure_evaluator(root)
            self.assertFalse(marker.exists())


class EvaluationTests(unittest.TestCase):
    def test_evaluations_disable_score_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for scope in ("smoke", "full"):
                for prompt in ("plain", "chat"):
                    with self.subTest(scope=scope, prompt=prompt):
                        output = root / f"{scope}-{prompt}"

                        def fake_run(command, **kwargs):
                            result = Path(command[command.index("--output_path") + 1])
                            result.write_text('{"results": {}}', encoding="utf-8")
                            return subprocess.CompletedProcess(command, 0)

                        with (
                            patch("vezilka_merge_lab.evaluation.subprocess.run", side_effect=fake_run) as run,
                            patch("vezilka_merge_lab.evaluation._record_environment") as capture,
                        ):
                            _evaluate_one(
                                root, root / "python", "example/model", output,
                                scope=scope, prompt=prompt, chat_template=root / "template.jinja",
                                device="cpu", batch_size=1,
                            )
                        command = run.call_args.args[0]
                        self.assertIn("--no_cache", command)
                        self.assertEqual(json.loads((output / "command.json").read_text()), command)
                        environment = run.call_args.kwargs["env"]
                        dataset = json.loads((ROOT / "configs/model_sources.json").read_text())["dataset"]
                        self.assertEqual(environment["VEZILKA_DATASET_REVISION"], dataset["revision"])
                        self.assertEqual(environment["VEZILKA_DATASET_REPOSITORY"], dataset["repository"])
                        self.assertEqual(capture.call_args.args[3], environment)

    def test_unpinned_dataset_is_rejected_before_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (
                patch("vezilka_merge_lab.evaluation.load_registry", return_value={
                    "dataset": {"repository": "LVSTCK/macedonian-llm-eval", "revision": "main"},
                }),
                patch("vezilka_merge_lab.evaluation.subprocess.run") as run,
                patch("vezilka_merge_lab.evaluation._record_environment") as capture,
            ):
                with self.assertRaisesRegex(ValueError, "pinned dataset commit"):
                    _evaluate_one(root, root / "python", "example/model", root / "output",
                                  scope="full", prompt="plain", chat_template=None, device="cpu", batch_size=1)
            run.assert_not_called()
            capture.assert_not_called()

    def test_environment_records_evaluator_runtime_and_template_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = root / "template.jinja"
            template.write_text("{{ bos_token }} test", encoding="utf-8")
            registry = json.loads((ROOT / "configs/model_sources.json").read_text())
            observed = {
                "python": "3.11 evaluator", "packages": {"torch": "observed-version"},
                "evaluator_revision": registry["evaluator"]["revision"],
                "dataset": registry["dataset"].copy(),
                "cuda_available": False, "nvidia_smi_error": "No driver",
            }
            command = ["evaluator-python", "main.py", "--no_cache"]
            environment = {"VEZILKA_DATASET_REVISION": registry["dataset"]["revision"]}
            with patch("vezilka_merge_lab.evaluation.subprocess.check_output", return_value=json.dumps(observed)) as probe:
                _record_environment(
                    root / "evaluator-python", root, root, environment,
                    command=command, model_path="example/model", chat_template=template,
                    prompt="chat", device="cpu", batch_size=2, scope="full",
                )
            self.assertEqual(probe.call_args.args[0][0], str(root / "evaluator-python"))
            self.assertEqual(probe.call_args.kwargs["env"], environment)
            record = json.loads((root / "environment.json").read_text())
            self.assertEqual(record["python"], "3.11 evaluator")
            self.assertEqual(record["packages"], observed["packages"])
            self.assertEqual(record["nvidia_smi_error"], "No driver")
            self.assertEqual(record["dataset"]["revision"], registry["dataset"]["revision"])
            self.assertEqual(record["chat_template_sha256"], hashlib.sha256(template.read_bytes()).hexdigest())
            self.assertEqual(record["dependency_lock_sha256"], hashlib.sha256(EVALUATOR_LOCK.read_bytes()).hexdigest())
            self.assertIn("lvstk-dataset-revision.patch", record["patches_sha256"])
            self.assertEqual(record["evaluation"]["command"], command)

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
            artifacts = Path(tmp).resolve() / "artifacts"
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
                "winogrande": {"acc": 0.5},
                "excluded_task": {"acc": 0.1},
            }}), encoding="utf-8")
            (merged / "results.json").write_text(json.dumps({"results": {
                "arc_easy": {"acc": 0.9, "acc_norm": 0.45},
                "boolq": {"acc": 0.6},
                "winogrande": {"acc": 0.6},
                "excluded_task": {"acc": 0.9},
            }}), encoding="utf-8")
            write_comparison(root, "full", ["plain"])
            rows = json.loads((root / "evaluation" / "comparison.json").read_text())["prompt_formats"]["plain"]
            self.assertEqual(rows[0]["metric"], "acc_norm")
            self.assertAlmostEqual(rows[0]["delta_percentage_points"], 5.0)
            self.assertEqual(rows[1]["metric"], "acc")
            self.assertEqual([row["task"] for row in rows], ["arc_easy", "boolq", "winogrande"])


if __name__ == "__main__":
    unittest.main()
