# Evaluator dependencies

The main `uv.lock` covers the Python 3.13 CLI. The older pinned LVSTCK evaluator
uses Python 3.11 and different Transformers, Hugging Face Hub and Accelerate
versions, so its environment has a separate lock.

`requirements.in` records the evaluator's direct requirements and build backend.
`requirements.lock` pins all 109 dependencies with distribution hashes for Linux
x86_64 and Python 3.11. The initial lock preserves versions from the working
evaluator environment rather than upgrading them.

Before evaluation, the CLI syncs this lock with `uv pip sync --require-hashes
--strict`. It then installs the pinned, patched evaluator checkout with
`--no-deps --no-build-isolation --no-index`, using the locked setuptools version,
and runs `uv pip check`. Sync runs even when an installation marker exists.
The marker records the lock's SHA-256 hash after successful setup.

To update dependencies deliberately, regenerate the lock and validate the
resulting environment before running models:

```bash
uv pip compile evaluator/requirements.in \
  --python artifacts/evaluator/venv/bin/python \
  --python-platform x86_64-unknown-linux-gnu \
  --generate-hashes --no-header --no-annotate \
  --output-file evaluator/requirements.lock
```

To preserve an existing environment's exact versions while regenerating, add
`--constraint` with a requirements file from `uv pip freeze --exclude-editable`.
The pinned evaluator's `setup.py` requirements must remain covered by
`requirements.in`. Updating the dependency lock does not rerun historical
evaluations.

The dataset commit and expected hashes of the six retained benchmark files are
recorded separately in `configs/model_sources.json`. The dataset patch requires
that commit SHA and passes it to `datasets.load_dataset`; it never defaults to
`main` for Macedonian tasks. The initial pinned revision matches the six local
cached benchmark files byte for byte.

Each new evaluation invokes `capture_environment.py` using the evaluator's
Python interpreter. Its `environment.json` records installed versions, CUDA and
GPU/driver observations, actual evaluator commit and working-tree diff hash,
the configured dataset revision, and lock/patch/template hashes. These records
describe new runs and do not fill missing historical metadata.
