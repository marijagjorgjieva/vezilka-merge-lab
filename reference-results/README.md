# Reference results

46 benchmark result sets: 23 plain and 23 fixed-Gemma-chat evaluations.

## Directory layout

| Directory | Models |
|---|---|
| `baselines/` | Unmerged Vezilka models and English, Bulgarian, and Ukrainian donors. |
| `e1/` | VezilkaLLM + English donor, SLERP. |
| `e2/` | VezilkaLLM-Instruct + English donor, SLERP with constant or layer-dependent coefficients. |
| `e3/` | VezilkaLLM-Instruct + Bulgarian donor, SLERP; one English and Bulgarian multi-donor TIES configuration. |
| `e4/` | VezilkaLLM-Instruct + Ukrainian donor, SLERP. |

Merge result directories match the filenames under [configs/merge](../configs/merge/). A `-chat` suffix indicates fixed-Gemma-chat evaluation; directories without it contain plain evaluation results.

## Files

| File | Contents |
|---|---|
| `results.json` | Per-task metrics, standard errors, task versions, and evaluator settings. |
| `model_info.json` | Model or merge configuration and prompt format. |
| `status.json` | Evaluation status, timing, device, batch size, and warnings. |
| `environment.json` | Python and library versions, CUDA information, and evaluator commit. |

Local checkpoint and template paths in evaluator arguments are replaced with placeholders. Model details and the bundled template path are provided in `model_info.json`.

For a new run, compare the matching task metrics and prompt format with these results. New run comparisons are written to `evaluation/comparison.md` inside the run directory.
