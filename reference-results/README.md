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
| `status.json` | Historical run status converted to the current CLI's status schema. |
| `original-metadata/status.json` | Original wrapper metadata, including timing, task list, device, batch size, and warnings. |
| `original-metadata/environment.json` | Original Python and library versions, CUDA information, and evaluator commit. |

Local checkpoint and template paths in evaluator arguments are replaced with placeholders. Model details and the bundled template path are provided in `model_info.json`.

For a new run, compare the matching task metrics and prompt format with these results. New run comparisons are written to `evaluation/comparison.md` inside the run directory.

## Current evaluation scope

The current protocol uses seven benchmark tasks. Original result JSON and archived run metadata are retained unchanged as historical evidence; comparisons include all seven tasks. Current report macros use their unrounded metrics. Historical plain/chat runs used batch sizes 1/2.
