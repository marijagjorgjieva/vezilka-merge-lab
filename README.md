# Vezilka Merge Lab

CLI for merging compatible Gemma 3 4B text models with MergeKit and evaluating them on Macedonian benchmarks.

The repository includes 18 merge configurations and 46 reference result sets. It supports reproducing these configurations and running custom merges. The recipient is VezilkaLLM or VezilkaLLM-Instruct, with English, Bulgarian, and Ukrainian donors in the included configurations.

## Requirements

- Linux, git, [uv](https://docs.astral.sh/uv/), and network access for initial downloads.
- Python 3.13 for the CLI. The evaluator uses a separate Python 3.11 environment, created automatically on first use.
- A CUDA GPU for the examples below. CPU execution is available with `--device cpu`.
- Tens of gigabytes of disk space for source models, merged weights, and evaluation outputs.
- Hugging Face access to the selected models, including a token and accepted model terms where required.

Minimum RAM and VRAM requirements have not been measured for this package. Runtime depends on hardware, downloads, and evaluator caching. `doctor` reports CUDA availability and free disk space.

## Setup

Run these commands from the repository root:

```bash
cp .env.example .env
uv sync --locked
uv run merge-model doctor
```

The CLI automatically loads `.env` from the repository root. Existing shell variables take precedence. Set `HF_TOKEN` in `.env` if required. Set `VEZILKA_ARTIFACTS_DIR` to store downloads and outputs on another disk. Both `.env` and the default `artifacts/` directory are ignored by Git.

## Run an included configuration

List and validate the configurations without downloading models:

```bash
uv run merge-model list
uv run merge-model validate --all
```

Preview a run:

```bash
uv run merge-model run \
  --config configs/merge/e2/e2-slerp-020.yaml --dry-run
```

Prepare the models, check compatibility, merge, and evaluate:

```bash
uv run merge-model run \
  --config configs/merge/e2/e2-slerp-020.yaml \
  --eval-scope full --prompt-format both --device cuda:0
```

`--dry-run` prints the plan without downloading models or writing run files.

Without evaluation flags, `run` evaluates 10 ARC Easy examples with plain prompts. This smoke evaluation still performs the full merge; its scores are not comparable to full benchmark results.

## Custom merges

Use a compatible donor ID or local model path:

```bash
uv run merge-model run \
  --base-model finki-ukim/VezilkaLLM-Instruct \
  --donor-model organization/my-gemma3-4b-text-model \
  --method slerp --t 0.20 --device cuda:0
```

Custom models must pass Gemma 3 text architecture, tensor-name, and tensor-shape checks. The included Bulgarian and Ukrainian donors are extracted to text-only checkpoints before merging.

- Direct SLERP runs retain the base model's token embeddings at `t=0.0` and select its tokenizer with `tokenizer_source: base`. Gemma 3's tied output head is preserved with the embeddings. MergeKit can trim unused embedding rows and change tokenizer special-token metadata; the saved tokenizer is not guaranteed to be an exact copy.
- Append `@<commit>` to a Hugging Face model ID to pin a revision.
- Pass `--config` with `--base-model` and `--donor-model` to substitute models in an included SLERP recipe. This is recorded as a custom run.
- Direct CLI merge parameters support SLERP only. Other complete MergeKit recipes can be supplied with `--config`.

Automatic text extraction supports the BgGPT and MamayLM sources in [model_sources.json](configs/model_sources.json). Other multimodal checkpoints must be converted to compatible text-only checkpoints before use.

The included TIES configuration does not apply the SLERP embedding protection rule.

## Run your own YAML

Pass the path to a complete MergeKit YAML. Validate and preview it before running:

```bash
uv run merge-model validate /path/to/my-merge.yaml
uv run merge-model run --config /path/to/my-merge.yaml --dry-run
uv run merge-model run \
  --config /path/to/my-merge.yaml \
  --eval-scope full --prompt-format both \
  --chat-template configs/evaluation/gemma_chat_template.jinja \
  --device cuda:0
```

The YAML must include `merge_method`, `base_model`, `models`, `parameters`, `tokenizer_source: base`, and `dtype: bfloat16`. SLERP requires exactly two models, including the base, and a `parameters.t` entry with `filter: embed_tokens` and `value: 0.0`. Model overrides are supported only for SLERP recipes.

The explicit chat template keeps custom-run evaluations consistent with the included configurations. Use `merge` instead of `run` to merge without evaluation.

## Commands

Use `uv run merge-model <command>`. Each command provides `--help`.

| Command | Purpose |
|---|---|
| `list` | List included configurations. |
| `validate --all` | Validate all included YAMLs without model downloads. |
| `doctor` | Report local tools, CUDA availability, and free disk space. |
| `prepare --config PATH` | Download or extract models and check compatibility. |
| `merge --config PATH` | Prepare and merge without evaluation. |
| `run --config PATH` | Prepare, merge, and evaluate the recipient and merged model. |
| `evaluate --run PATH` | Evaluate a completed merge using its local models. |
| `evaluate --model PATH` | Evaluate a model directly, with optional `--base-model ID` for comparison. |

## Evaluation

| Option | Values |
|---|---|
| `--eval-scope` | `smoke` (default): 10 ARC Easy examples; `full`: all six tasks. |
| `--prompt-format` | `plain` (default), `chat`, or `both` as separate conditions. |
| `--chat-template` | Path to a custom Jinja template. |
| `--batch-size` | Evaluation batch size, default `1`. |

Full evaluation covers ARC Challenge, ARC Easy, HellaSwag, OpenBookQA, PIQA, and WinoGrande using a pinned version of the [LVSTCK Macedonian evaluation suite](https://github.com/LVSTCK/macedonian-llm-eval). It is installed on first use with the chat-template and cache-filename patches in `patches/`.

Included configurations use the [fixed Gemma chat template](configs/evaluation/gemma_chat_template.jinja). Custom runs use the model's tokenizer template unless `--chat-template` is supplied. Results are saved separately for plain and chat prompts.

Benchmark accuracy does not measure Macedonian fluency or instruction following. Plain uses no chat template or BOS. Fixed Gemma chat adds BOS and strips leading continuation whitespace. These are separate evaluation conditions; their difference is not an isolated template effect. Scores use `acc_norm` for ARC Challenge, ARC Easy, HellaSwag, OpenBookQA and PIQA, and `acc` for WinoGrande. Any macro summary is the unweighted mean of these six metrics, computed before rounding.

To evaluate a completed merge, use the run directory printed after `Done:`:

```bash
uv run merge-model evaluate \
  --run artifacts/runs/YOUR-RUN-DIRECTORY \
  --eval-scope full --prompt-format both --device cuda:0
```

## Outputs

Runs are saved under `artifacts/runs/`. A successful run prints `Done:` followed by its directory.

| Path within the run directory | Contents |
|---|---|
| `merged-model/` | Merged checkpoint and tokenizer. |
| `manifest.json` | Run settings, resolved source revisions, and status. |
| `merge_stdout_stderr.log` | Merge log. |
| `evaluation/comparison.md` | Recipient and merged scores, with differences in percentage points. |
| `evaluation/base/` and `evaluation/merged/` | Benchmark results and logs. |

## Troubleshooting

| Issue | Check |
|---|---|
| Missing `.env.example` or CLI command | Run setup from the directory containing this README. |
| Model access denied | Check model access and `HF_TOKEN`. |
| CUDA unavailable | Run `doctor`; check the GPU environment or use `--device cpu`. |
| GPU memory exhausted | Evaluation defaults to batch size 1. Use more VRAM or try CPU execution. |
| Merge or evaluation failure | Inspect the log path in the error. Evaluator setup logs are in `artifacts/evaluator/setup.log`. |

## Reference results

- [Merge configurations](configs/merge/): 18 configurations.
- [Reference results](reference-results/README.md): 23 plain and 23 chat result sets, with model and configuration details.
- [Source revisions](configs/model_sources.json): pinned model and evaluator revisions used by this package.

## Tests

Run the automated checks without downloading models or executing merges:

```bash
uv run --locked python -m unittest discover -s tests -v
uv run --locked merge-model validate --all
```

These checks cover configuration handling, model compatibility, and evaluation reuse. A real pipeline check requires a merge followed by smoke evaluation:

```bash
uv run --locked merge-model run \
  --config configs/merge/e2/e2-slerp-020.yaml \
  --eval-scope smoke --prompt-format both --device cuda:0
```

This command downloads any missing models and performs a full merge. Smoke scores check execution and do not establish full benchmark performance.

## License

The code is licensed under [Apache-2.0](LICENSE). Model weights, benchmark data, and the external evaluator retain their own licenses and terms. Weights and datasets are not committed here.
