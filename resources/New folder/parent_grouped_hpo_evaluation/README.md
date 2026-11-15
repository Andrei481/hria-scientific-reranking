# Evaluate the completed parent-grouped Llama refit

This is the next step **after the full refit has completed**. It evaluates the
selected `lr_5e-6` adapter on LitSearch and CSFCube. It does not repeat training.

Keep this directory beside `parent_grouped_hpo`, not inside it. The completed HPO
study hashes its Python files; leave those files and its frozen configuration
unchanged. This companion writes to a separate external-evaluation directory.

The recipe uses **one H200, GPU 0**, and two terminals. vLLM serves the pinned
base with the saved rank-128 LoRA adapter. There is no merged 16 GB model export.
Requests explicitly select a cohort-specific `hpo_grouped_<hash>` alias; they never take the first model
returned by the server, which could be the base model.

## 1. Copy this companion bundle to the server

Upload `parent_grouped_hpo_evaluation.zip` to `/home/andreij/`, using your usual
file-transfer method. For example, from local Windows PowerShell (replace the
SSH hostname if you normally connect using an IP address):

```powershell
scp "D:\Code\disertatie\New folder\parent_grouped_hpo_evaluation.zip" andreij@hria-01-all:~/
```

On the Linux server, extract the archive into your home directory:

```bash
conda activate grouped_hpo
python -m zipfile -e "$HOME/parent_grouped_hpo_evaluation.zip" "$HOME"
cd "$HOME/parent_grouped_hpo_evaluation"
```

Do this before preparing the external cohort. After preparation, the companion
Python files are also frozen by hash; do not update them in place during a run.

## 2. Check the existing resources and disk space

In this terminal, set these variables explicitly:

```bash
export HPO_ARTIFACTS="$HOME/parent_grouped_hpo/artifacts"
export HPO_EXTERNAL="$HOME/parent_grouped_hpo_external"
export LITSEARCH_INPUTS="$HOME/tie_sensitivity_artifacts/evaluation_inputs"

df -h "$HOME"
ls "$HPO_ARTIFACTS/refit/grouped/lr_5e-6/result.json"
ls "$HPO_ARTIFACTS/refit/grouped/lr_5e-6/adapter/adapter_config.json"
ls "$LITSEARCH_INPUTS"/{manifest.json,bm25.jsonl,grit.jsonl,qwen3.jsonl}
ls "$HOME/CSFCube/queries-release.csv"
python -B -m unittest discover -s . -v
```

Use the **existing frozen LitSearch inputs from the tie-sensitivity experiment**.
The helper checks their hashes and reuses them without making another large copy.
It needs the complete CSFCube checkout, including `eval_scripts`, its three gold
annotation files, `abstracts-csfcube-preds.jsonl`, `evaluation_splits.json`, and
`queries-release.csv`.

The tests use synthetic responses, not a GPU. The original-notebook parity test
may be skipped on the server because that notebook is only present in the local
repository. The real CSFCube input and scorer tests should run if `~/CSFCube` is
present and NumPy is installed.

If a resource is missing, locate the existing resource before proceeding. Do not
recreate candidate pools with a different retrieval run. Free disk space is still
needed for outputs and vLLM caches; this recipe avoids a merged model but cannot
operate on a completely full filesystem.

## 3. Prepare and check the evaluation cohort once

Still in `grouped_hpo`:

```bash
python -B prepare.py \
  --hpo-bundle "$HOME/parent_grouped_hpo" \
  --hpo-artifacts "${HPO_ARTIFACTS:?Set HPO_ARTIFACTS first}" \
  --litsearch-inputs "${LITSEARCH_INPUTS:?Set LITSEARCH_INPUTS first}" \
  --csfcube "$HOME/CSFCube" \
  --csfcube-prompt original \
  --output "${HPO_EXTERNAL:?Set HPO_EXTERNAL first}"
```

This checks the completed validation grid, selected learning rate, final refit,
adapter hashes, and evaluation resources. It tokenizes every prompt and checks
that the full prompt plus output budget fits the 131,072-token serving context.
This is CPU preparation; it performs no generation.

Success ends with:

```text
Prepared ...: 1791 LitSearch requests and 50 CSFCube requests; no generation performed.
```

`--csfcube-prompt original` is intentional: you confirmed that the original
notebook produced the comparison results. It preserves both **no facet
instruction** and the notebook's Python-list representation of sentence-list
abstracts. Facets still determine which candidate pools and gold labels are
scored. The alternative `facet` mode is a different prompt condition, not part
of this comparison.

The output directory must be new. On subsequent sessions, use the existing
manifest and continue at step 4. If preparation fails, resolve the reported
problem and use a fresh output directory. A context overflow must be investigated;
do not silently truncate abstracts, remove candidates, or shorten the output
budget to force a run through.

## 4. Start the model server in Terminal A

Use the existing serving environment; do not upgrade the training environment:

```bash
conda activate tie_serve
cd "$HOME/parent_grouped_hpo_evaluation"
export HPO_EXTERNAL="$HOME/parent_grouped_hpo_external"
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1

python -c 'from importlib.metadata import version; print("vLLM", version("vllm"))'
python -B serve.py --artifacts "${HPO_EXTERNAL:?}" --execute
```

Keep this terminal running. Wait for vLLM to report that the API server is ready.
The launcher reads the exact base revision and adapter location from the prepared
manifest. It records the command and serving-package versions in `serving.json`.
It serves `hpo_base_<hash>` and `hpo_grouped_<hash>` on
`http://127.0.0.1:8000/v1`. The hash distinguishes this cohort from previous
servers. The launcher refuses to start on an occupied port.

`HF_HUB_OFFLINE=1` prevents an unexpected model download on the nearly full root
disk. The pinned base model should already be cached from training. If vLLM reports
a missing cached model or file, stop and locate the training cache; do not simply
disable offline mode and start a large download. If port 8000 is already occupied,
use `--port 8001` on the **first** launch; clients read the port from `serving.json`.

The official [vLLM LoRA documentation](https://docs.vllm.ai/en/v0.19.0/features/lora/)
describes serving named adapters and selecting them through the request's model
field. This launcher uses that route with maximum LoRA rank 128.

## 5. Run a small smoke evaluation in Terminal B

Open another server terminal and set its environment separately:

```bash
conda activate grouped_hpo
cd "$HOME/parent_grouped_hpo_evaluation"
export HPO_EXTERNAL="$HOME/parent_grouped_hpo_external"

python -B evaluate.py --artifacts "${HPO_EXTERNAL:?}" --limit 2
```

This requests two queries from each of the six pools: three LitSearch retrievers
and three CSFCube facets, for 12 requests. It checks that the adapter alias exists
and explicitly sends requests to it. Smoke outputs go under
`results_smoke/grouped/`; they cannot be passed to the official scoring command.

Success creates `results_smoke/grouped/summary.json` with `status: smoke_only`.
Check for serving errors before starting the full run. A repaired ranking is
recorded as such, including missing IDs and output-length stops.

## 6. Run the complete external evaluation

In Terminal B, remove `--limit`:

```bash
python -B evaluate.py --artifacts "${HPO_EXTERNAL:?}"
```

This makes 1,841 requests: 597 queries for each of BM25, GRIT, and Qwen3, plus all
50 CSFCube query-facet cases. The client runs sequentially against GPU 0; it does
not occupy all eight GPUs. Each request uses temperature 0, top-p 1, and seed 0.
The output budget is 1,024 tokens for LitSearch and 4,096 for CSFCube.

Each response is saved immediately, with the input hash, raw text, repaired
ranking, parser diagnostics, usage, and finish reason. Missing ranking IDs are
appended in original candidate order, matching the retained parser. Failed HTTP
requests stop the run rather than silently dropping queries.

If interrupted, keep the same files and rerun the **same command**. It checks the
saved prefix and resumes from the next query. Smoke results are separate and are
not reused. If a crash leaves a partial final JSONL record, the helper stops;
preserve the file and repair that record before resuming. It does not silently
discard malformed output.

Full success creates `results/grouped/summary.json` with `status: complete`.
That status requires all six pools to finish. Parser failures and length stops
remain visible in the summary and should be reported alongside the metrics.

## 7. Compute the original CSFCube metrics and read the results

In Terminal B:

```bash
python -B score_csfcube.py --artifacts "${HPO_EXTERNAL:?}"

python -m json.tool "$HPO_EXTERNAL/results/grouped/summary.json"
cat "$HPO_EXTERNAL/results/grouped/csfcube_all_metrics.txt"
```

`score_csfcube.py` checks that complete ranking files match the retained raw
responses, then runs the original CSFCube evaluation script for `all`,
`background`, `method`, and `result`. It retains the original folds, grade
thresholds, and ranking-metric implementation. For NumPy 2, it provides the removed
`asfarray` operation in memory using an equivalent float conversion for numeric
relevance grades; it does not edit CSFCube or downgrade your environment.

The useful output files are:

| File under `results/grouped/` | Meaning |
|---|---|
| `summary.json` | LitSearch R@5, R@20, nDCG@5, nDCG@20 for each retriever, plus parser diagnostics for all pools |
| `csfcube_all_metrics.txt` | Official aggregate CSFCube dev/test output |
| `csfcube_background_metrics.txt`, `csfcube_method_metrics.txt`, `csfcube_result_metrics.txt` | Official facet results |
| `test-pid2pool-csfcube-hpo_grouped-*-perq.csv` | Original scorer's per-query metrics |
| `litsearch_*.jsonl`, `csfcube_*.jsonl` | Auditable raw responses and rankings |

For CSFCube, **NDCG%20 means the top 20% of each candidate pool**, whereas
NDCG@20 means the first 20 documents. Keep those columns distinct. The official
dev/test fold outputs overlap by construction; do not describe them as separate
independent held-out benchmark populations.

## 8. Optional: evaluate the base model in the same serving environment

If you need a matched current base-model row, leave the same server running:

```bash
python -B evaluate.py --artifacts "${HPO_EXTERNAL:?}" --model base
python -B score_csfcube.py --artifacts "${HPO_EXTERNAL:?}" --model base
```

This uses the cohort's `hpo_base_<hash>` alias, with the same frozen inputs and decoding settings, writing
to `results/base/`. It does not train another model or save another weight copy.
After completing whichever evaluations you need, stop Terminal A with Ctrl+C.

## What these results establish for the reviewer

Your grouped validation grid selected **5e-6 on all three splits**, rather than the
reference 1e-5. Therefore, report that you **adopted parent-grouped selection**;
do not claim the selected learning rate was unchanged. The default grid checks
learning rate only, with other hyperparameters fixed.

The external scores here belong to the selected model freshly refitted on all
4,494 training lists. They complement the internal-test NLLs already obtained
before refitting. They must not be used for further hyperparameter selection.

This is a new evaluation cohort. It preserves the requested original CSFCube
prompt and frozen candidate pools, but the serving environment and direct-LoRA
execution can differ from historical merged-model runs. Historical score changes
alone do not isolate the learning-rate effect. That would require a matched
full-data 1e-5 refit and evaluation under the same environment, prompts, pools,
and decoding settings. The 80-parent HPO trial is not that full-data control.

This bundle covers your completed **Llama** study. It does not establish grouped
HPO results for the paper's other model families. Also, describe this CSFCube
evaluation as using the original prompt **without explicit facet instructions**;
do not present it as evidence for a facet-conditioned prompt.

## Local validation and limitations

See `VALIDATION.md`. CPU checks cover parser behavior, real released candidate
pools, original notebook prompt parity, explicit adapter requests and resume,
and the original scorer's NumPy compatibility. No H200 generation or completed
server-side refit verification was possible from the local Windows workspace.
The server preparation and smoke commands are required checks before full use.
