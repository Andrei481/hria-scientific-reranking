# Llama tie-sensitivity experiment

This bundle tests whether arbitrary ordering **within a relevance grade** affects a listwise Llama reranker. It creates new matched controls using the same frozen DORIS-MAE lists, then evaluates on the same LitSearch candidate pools.

## What you will run

| Condition | Assistant target within each grade | Runs |
| --- | --- | --- |
| `deterministic` | Candidate position in the frozen input list | Seeds 42, 2026, 3407 |
| `epoch_random` | Independently shuffled each logical epoch | Seeds 42, 2026, 3407 |
| `fixed_random` (optional) | Shuffled once per example and seed; reused in both epochs | Same three seeds |

Every target still places grade 2 before grade 1 before grade 0. Query text, document text, candidate order, labels, example schedule, model revision, and hyperparameters are held fixed within each matched seed. Only target tie ordering changes. The deterministic rule is **input-list position**, not the original DORIS corpus index.

The minimum study is **six new trainings**. Start with one matched seed as a pilot, then complete the other two seeds. Do not substitute a historical checkpoint unless its exact data, schedule, masking, model, and recipe match this protocol; that identity has not been established here.

**Scope:** this implements the *tie-sensitivity* part of the proposed experiment, not a mathematically tie-aware objective. Randomized targets still use ordinary assistant-token cross-entropy and still penalize other valid permutations on each presentation. A genuine set-/grade-aware loss needs a separate implementation and matched training condition. These results cannot quantify a percentage of signal that is “genuine graded relevance.” They test sensitivity to arbitrary supervision within grades. CSFCube evaluation is not included in this bundle.

## Files and prerequisites

- `prepare_data.py`: validates and freezes the existing 4,494 training lists.
- `train.py`: tokenization preflight, two-step GPU smoke test, matched LoRA training, merged checkpoint export.
- `export_checkpoint.py`: re-export an existing adapter against its pinned base and reload it, without retraining.
- `verify_checkpoint.py`: verify an existing standalone checkpoint in a fresh native-Transformers process, without importing Unsloth.
- `prepare_evaluation.py`: freezes LitSearch prompts for BM25, GRIT, and Qwen3 (597 queries each; top 100 candidates).
- `evaluate.py`: saves raw model replies, parser diagnostics, repaired rankings, and metrics; resumes completed query prefixes.
- `analyze.py`: per-seed scores, seed mean/SD, paired query confidence intervals, and multiplicity-adjusted tests.
- `experiment.py`, `test_experiment.py`, `test_pipeline.py`: shared helpers and local tests.
- `01_prepare_and_train.ipynb`, `02_evaluate_and_analyze.ipynb`: optional wrappers around the same scripts.
- `environment-train.yml`, `environment-serve.yml`: pinned Linux environments adapted from this repository's existing environment exports, with machine-specific prefixes removed.

The commands below use **Bash on a Linux GPU server**, not Windows PowerShell. Use one BF16-capable NVIDIA GPU with substantial memory (the intended target is an H200), Conda, and access to the gated `meta-llama/Llama-3.1-8B-Instruct` repository. This is 16-bit LoRA, not QLoRA. Allow well over 100 GB of working storage for six merged 8B checkpoints, adapters, optimizer checkpoints, downloads, and data. Exact peak VRAM and runtime must be measured on your machine.

No weights, training data, or newly generated results are embedded in this directory. Keep access to these existing repository resources, or transfer them alongside the bundle:

```text
New folder/modernbert_supervised_baseline/modernbert_supervised_baseline/dorismae_listwise_4494.jsonl
LitSearch/results/retrieval/LitSearch.title_abstract.bm25.jsonl
LitSearch/results/retrieval/LitSearch.title_abstract.grit.jsonl
LitSearch/results/retrieval/LitSearch.title_abstract.qwen3.jsonl
```

The preparation command copies the training JSONL into your artifact directory. The LitSearch corpus and Llama weights/tokenizer are downloaded from Hugging Face using pinned revisions. Never commit credentials or tokens.

## 1. Set up paths and environments

Adjust the first two paths to your server. Run the experiment from a stable copy of this bundle: Python source hashes are recorded and checked. Do not edit scripts or change package versions between conditions.

```bash
cd /DATA/j_andrei/disertatie
export TIE_DIR="$PWD/New folder/tie_sensitivity"
export TIE_ARTIFACTS=/DATA/j_andrei/tie_sensitivity_artifacts
export TIE_REPO="$PWD"
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$TIE_ARTIFACTS"

conda env create -f "$TIE_DIR/environment-train.yml"
conda env create -f "$TIE_DIR/environment-serve.yml"
conda activate tie_train
hf auth login
python -B -m unittest discover -s "$TIE_DIR" -p 'test_*.py' -v
```

Accept the Llama license on Hugging Face using the account you authenticate with. If an environment cannot resolve on your server, resolve that **before** starting the cohort, export the working environment, and use it for every condition. The provided environments and GPU path have not been executed on an H200 as part of building this bundle.

Resolve the model commit **once** and retain the printed value for every training, including later terminal sessions:

```bash
export LLAMA_REVISION="$(python -c 'from huggingface_hub import HfApi; print(HfApi().model_info("meta-llama/Llama-3.1-8B-Instruct").sha)')"
echo "$LLAMA_REVISION"
```

In a new terminal, restore the same variables and use `export LLAMA_REVISION=YOUR_SAVED_40_CHARACTER_SHA`; do not resolve a newer commit halfway through the experiment. Each run records the SHA.

## 2. Freeze the training lists once

```bash
python "$TIE_DIR/prepare_data.py" \
  --source "$TIE_REPO/New folder/modernbert_supervised_baseline/modernbert_supervised_baseline/dorismae_listwise_4494.jsonl" \
  --output "$TIE_ARTIFACTS/data"
```

Expected SHA-256:

```text
7c649c816c78778777afcddc011dafd345a6847d2fd6d4e082b58bad75e5bc4f
```

The script rejects a different source. Expected audit: 4,494 examples, with 4,312 lists of length 50 and 182 of length 51. Some lists contain more than five positives. These properties are deliberately preserved across conditions; fixing the sampler at the same time would confound the comparison. Do not regenerate or reshuffle input candidate lists for each run.

Outputs: `data/lists.jsonl` and `data/manifest.json`.

## 3. Preflight the first matched pair

Preflight loads only the tokenizer, checks **all 8,988 presentations** (4,494 lists × two logical epochs), verifies complete assistant supervision, and rejects overlength examples rather than silently truncating targets.

```bash
set -e
for policy in deterministic epoch_random; do
  python "$TIE_DIR/train.py" \
    --data "$TIE_ARTIFACTS/data" --output "$TIE_ARTIFACTS/runs" \
    --policy "$policy" --seed 42 --revision "$LLAMA_REVISION" \
    --preflight-only
done
```

Inspect each run's `preflight.json`. Check:

- Both conditions have identical `schedule_sha256`, data SHA, and model revision.
- Each logical epoch has 4,494 examples and nonzero response tokens.
- `changed_targets_between_epochs` is zero for deterministic and nonzero for epoch-random.
- Maximum tokenized length is within 16,384.

The per-epoch target hashes include presentation order, so differing epoch hashes alone do **not** prove that individual targets changed. Use `changed_targets_between_epochs`.

If an example exceeds the limit, stop and choose a shared protocol change before training either condition (for example, a larger supported `--max-length` passed consistently to all runs). Do not remove examples or truncate targets for just one condition.

## 4. Run a two-step GPU smoke test

This validates loading, forward/backward execution, response masking, and checkpoint export on your actual server. It also runs the full tokenization preflight. Smoke checkpoints are marked and excluded from the final analysis.

```bash
for policy in deterministic epoch_random; do
  python "$TIE_DIR/train.py" \
    --data "$TIE_ARTIFACTS/data" --output "$TIE_ARTIFACTS/smoke_runs" \
    --policy "$policy" --seed 42 --revision "$LLAMA_REVISION" \
    --smoke-steps 2
done
```

Expect `smoke_runs/deterministic_seed42_SMOKE/` and `smoke_runs/epoch_random_seed42_SMOKE/`, each with `training_manifest.json`, `adapter/`, and `final/`. Check the manifest says `smoke_only` and two optimizer steps. A successful smoke test is not evidence of final quality or full-run numerical stability.

## 5. Train the matched pilot, then the remaining seeds

```bash
for policy in deterministic epoch_random; do
  python "$TIE_DIR/train.py" \
    --data "$TIE_ARTIFACTS/data" --output "$TIE_ARTIFACTS/runs" \
    --policy "$policy" --seed 42 --revision "$LLAMA_REVISION"
done
```

Run the evaluation below for this pair before committing to the remaining compute. Use the pilot to check feasibility and protocol correctness, not to selectively keep a favorable seed. Once the protocol is fixed:

```bash
for seed in 2026 3407; do
  for policy in deterministic epoch_random; do
    python "$TIE_DIR/train.py" \
      --data "$TIE_ARTIFACTS/data" --output "$TIE_ARTIFACTS/runs" \
      --policy "$policy" --seed "$seed" --revision "$LLAMA_REVISION"
  done
done
```

If absent, preflight runs automatically before training. Expected final directories:

```text
runs/
  deterministic_seed42/
  epoch_random_seed42/
  deterministic_seed2026/
  epoch_random_seed2026/
  deterministic_seed3407/
  epoch_random_seed3407/
```

Recipe: BF16, no quantization; LoRA rank 128/alpha 256/dropout 0.05 on attention and MLP projections; learning rate 1e-5; batch size 2; accumulation 1; cosine schedule; 20 warmup steps; weight decay 0.01; two logical epochs; maximum length 16,384. Loss supervises the assistant response only. The explicit dataset schedule contains both epochs, so `Trainer` runs **one pass over that schedule**: 8,988 presentations and 4,494 optimizer steps. Do not change `num_train_epochs` to 2.

Each run records package versions, hyperparameters, source hashes, peak training VRAM, training time (excluding preflight and export), and a merged 16-bit checkpoint for serving. No training jobs are launched by this README itself.

Loading uses `use_exact_model_name=True`. Export saves the adapter first, then uses PEFT's `merge_and_unload(safe_merge=True)` on the loaded base and standard `save_pretrained`; it does not use Unsloth's mirror-resolving merged exporter. The trainable-parameter count is captured before the merge removes adapter modules.

Optional fixed-random control: run the same command for `--policy fixed_random` for all three seeds, then evaluate those checkpoints too. This distinguishes changing tie targets across epochs from choosing a different but stable arbitrary order. Decide whether to include it before interpreting the final comparisons.

## 6. Freeze the LitSearch evaluation inputs once

In `tie_train`, resolve the corpus commit once, then prepare all three retrievers:

```bash
export LITSEARCH_REVISION="$(python -c 'from huggingface_hub import HfApi; print(HfApi().dataset_info("princeton-nlp/LitSearch").sha)')"
echo "$LITSEARCH_REVISION"
python "$TIE_DIR/prepare_evaluation.py" \
  --retrieval-root "$TIE_REPO/LitSearch/results/retrieval" \
  --corpus-revision "$LITSEARCH_REVISION" \
  --output "$TIE_ARTIFACTS/evaluation_inputs"
```

This preserves each first-stage top-100 candidate order and uses title/abstract text with the same 300-word candidate limit as training. It validates 597 unique queries per retriever and matching query identities. Reuse this exact directory for every checkpoint; do not rerun retrieval. Preparation requires an empty output directory and refuses to overwrite frozen inputs.

## 7. Serve and evaluate each checkpoint

Use two terminals, restoring the path variables in both. Never serve a model while training on the same GPU.

**Terminal A** (`tie_serve`):

```bash
conda activate tie_serve
export CUDA_VISIBLE_DEVICES=0
export RUN_NAME=deterministic_seed42
python -c 'import vllm; print(vllm.__version__)'
vllm serve "$TIE_ARTIFACTS/runs/$RUN_NAME/final" \
  --served-model-name "$RUN_NAME" \
  --dtype bfloat16 --tensor-parallel-size 1 \
  --max-model-len 65536 --gpu-memory-utilization 0.90 \
  --generation-config vllm
```

Wait for the server to become ready. Record the printed vLLM version; the supplied serving export pins `0.19.0`. Keep the launch arguments identical for every checkpoint. The larger serving context accommodates 100-candidate evaluation lists; it does not change the training context limit. An API/context error aborts evaluation rather than reducing the number of candidates.

**Terminal B** (`tie_train`):

```bash
conda activate tie_train
export RUN_NAME=deterministic_seed42
export TIE_VLLM_VERSION=0.19.0  # replace if the actual serving version differs

# Five queries per retriever, stored separately from the full evaluation:
python "$TIE_DIR/evaluate.py" \
  --run "$TIE_ARTIFACTS/runs/$RUN_NAME" \
  --inputs "$TIE_ARTIFACTS/evaluation_inputs" \
  --served-model "$RUN_NAME" --limit 5

# Full evaluation: 597 queries per retriever.
python "$TIE_DIR/evaluate.py" \
  --run "$TIE_ARTIFACTS/runs/$RUN_NAME" \
  --inputs "$TIE_ARTIFACTS/evaluation_inputs" \
  --served-model "$RUN_NAME"
```

The server alias is checked, but an alias alone cannot prove which checkpoint you loaded: verify Terminal A points at the matching `final/` directory. Repeat for each of the six run names: stop the old server with Ctrl+C, change `RUN_NAME` in **both terminals**, launch the next checkpoint, and evaluate it. Do not rename an old served checkpoint and treat it as a new one.

Generation uses temperature 0, top-p 1, seed 0, and a 1,024-token output budget. Raw replies and finish reasons are retained. The parser keeps the first valid occurrence of each ID and appends missing IDs in original candidate order. Quality is measured on the repaired full ranking; raw compliance is reported separately. `exact_permutation` measures ID coverage, not strict output-only formatting (`format_only` is a separate diagnostic). Inspect truncation and fallback rates alongside quality.

Rerunning the same evaluation command resumes an aligned saved prefix. Outputs are under each run's `evaluation/`: three JSONLs, `protocol.json`, `summary.json`, and `complete.json`. The `evaluation_SMOKE/` directory is never used for analysis.

## 8. Generate comparison tables

After evaluating just seed 42, obtain a clearly labeled pilot:

```bash
python "$TIE_DIR/analyze.py" --runs "$TIE_ARTIFACTS/runs" \
  --seeds 42 --output "$TIE_ARTIFACTS/analysis_pilot"
```

After evaluating all six full runs:

```bash
python "$TIE_DIR/analyze.py" --runs "$TIE_ARTIFACTS/runs" \
  --seeds 42 2026 3407 --output "$TIE_ARTIFACTS/analysis_full"
```

If you also completed all three fixed-random runs, use a new output directory:

```bash
python "$TIE_DIR/analyze.py" --runs "$TIE_ARTIFACTS/runs" \
  --seeds 42 2026 3407 --comparisons epoch_random fixed_random \
  --output "$TIE_ARTIFACTS/analysis_with_fixed_random"
```

Outputs:

| File | Purpose |
| --- | --- |
| `per_seed.csv` | R@5, R@20, nDCG@20 and parser diagnostics for every seed/retriever |
| `mean_sd.csv` | Mean and sample SD across seeds; one-seed SD is left blank |
| `paired_uncertainty.csv` | Candidate-minus-deterministic deltas in percentage points, 95% query-bootstrap CIs, sign-flip p-values and Holm-adjusted p-values |
| `analysis_manifest.json` | Protocol settings and statistical interpretation |

Statistics first average each query's matched difference across seeds, then resample the 597 queries (10,000 replicates). Queries are also paired across retrievers for the retriever-average summary. Do not interpret the seed/query combinations as 1,791 independent queries. These confidence intervals condition on the trained checkpoints; seed SD separately describes training variability. Holm adjustment is within each metric across all selected comparisons and the four retriever summaries, not across the three metrics. Prespecify a primary metric (for example, retriever-average R@5); regard others as secondary or broaden the adjustment if making joint confirmatory claims.

## 9. Interpret and report

Report the frozen data hash, exact tie rules, seed-level results, mean ± seed SD, paired effect and CI, raw validity/fallback rates, and training cost. State that these are **new matched controls**, not necessarily reproductions of the historical Llama checkpoint.

- Similar performance with a narrow interval supports robustness to this tie manipulation **under this protocol**. Nonsignificance alone does not establish equivalence.
- A consistent change shows tie supervision affects generalization; its direction does not by itself identify the mechanism.
- A change in output validity can contribute to a change in repaired ranking quality; inspect both.
- LitSearch evaluation uses binary relevance. This study therefore measures downstream transfer sensitivity, not a direct decomposition of the model's understanding of the three DORIS grades.
- A true tie-aware-loss condition and/or held-out graded DORIS evaluation would be additional experiments. Do not label `epoch_random` a tie-aware loss.

## Troubleshooting and reproducibility

- **Already completed the old smoke runs?** After copying the updated bundle to the server, validate the new exporter without retraining: `python "$TIE_DIR/export_checkpoint.py" --run "$TIE_ARTIFACTS/smoke_runs_exact/deterministic_seed42_SMOKE"`, then repeat for `epoch_random_seed42_SMOKE`. This loads the recorded base revision and saved adapter, runs the same export helper used by training, and reloads the resulting checkpoint for a finite-logit forward check. It writes a separate `export_check/` directory inside each smoke run, preserving the old artifacts. This is not an equivalence or ranking-quality test. An existing `export_check/` is protected; use `--output` with a new directory for a retry. Refresh both normal-run preflights after updating the scripts, before full training. No dependency upgrade is required.
- **Export completed but reload failed with `LlamaAttention ... apply_qkv`?** The old verifier reloaded native Transformers in the same process after Unsloth had patched its classes. Upload updated `export_checkpoint.py` and new `verify_checkpoint.py`; `train.py` needs no further edit. Verify the already-written artifact directly: `python "$TIE_DIR/verify_checkpoint.py" --checkpoint "$TIE_ARTIFACTS/smoke_runs_exact/deterministic_seed42_SMOKE/export_check/final"`. Do not re-export or delete that directory. Repeat for the randomized run if its export exists; otherwise export that run using the updated exporter. Future exports automatically launch verification in a clean subprocess. Success writes `final/reload_check.json`; direct recovery does not fabricate an `export_manifest.json` for an earlier interrupted command.
- **Interrupted training:** automatic optimizer/scheduler resumption is intentionally not implemented. Keep the interrupted directory for audit and restart in a fresh `--output` root. Collect completed run directories under one common root for analysis, without altering their files. Never merge two partial runs.
- **Preflight mismatch:** scripts, revision, or settings changed. Establish one fixed protocol and rerun preflight; if training already started, use a fresh cohort. Do not weaken the comparison checks.
- **Tokenizer mismatch:** Unsloth and the preflight tokenizer differ. Investigate the environment/template before continuing; do not bypass the guard.
- **GPU OOM:** stop and choose a common feasible recipe for both conditions. Changing batch size, quantization, context length, or accumulation only for one run invalidates the matched comparison.
- **Evaluation failure:** correct the server/resource issue and rerun the identical command. Any change to protocol parameters requires a separately preserved evaluation cohort. If a process kill leaves a partial final JSONL line, preserve the damaged file before repairing that final line; ordinary HTTP failures do not append a result.
- **Existing output rejected:** completion artifacts are protected from overwrite. Choose a new artifact/output directory; retain old results for audit.
- **Notebook use:** launch Jupyter from this bundle directory with the appropriate environment/kernel; set the configuration cells explicitly. Notebook wrappers launch scripts as subprocesses so Unsloth is imported in a fresh training process. They do not implement a second training recipe.

Local tests cover CPU logic, masking/backpropagation, data/order invariants, parsing, metrics, and analysis plumbing using synthetic fixtures. They do **not** certify GPU training, gated-model downloads, environment resolution, or live vLLM compatibility. Complete the server smoke test before the six-run study.
