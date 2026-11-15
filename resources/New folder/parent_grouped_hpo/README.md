# Parent-query-grouped hyperparameter selection

This folder contains the data and scripts to address the reviewer comment about
correlated DORIS-MAE examples crossing training/validation boundaries during HPO.
Copy this **entire folder** to your Linux GPU server. No other files from the
dissertation repository are needed for the HPO experiment.

The default experiment trains **nine fresh Llama 3.1 8B adapters**: three learning
rates across three parent-grouped splits. An optional nine-run list-level control
compares which learning rate the two protocols select. The scripts generate CSV
and Markdown results; they do not assume the original setting wins.

If you encountered missing campaign/Gemma functions or the NumPy warning, follow
[`RECOVERY.md`](RECOVERY.md). The optional four-model, 144-trial experiment is in
[`CAMPAIGN.md`](CAMPAIGN.md); it does not change this default nine-trial workflow.

**Scope:** the original HPO trial records have not been established. Therefore,
the supplied grid is a **new learning-rate sensitivity study**, not evidence that
every originally tuned hyperparameter is unchanged. To repeat the original HPO,
replace the grid with the actual retained candidate settings and preserve its
selection objective and training budget before preparing the experiment. Do not
claim an exact historical replay if other settings or the training implementation
differ. The default addresses grouping for new selection, with other parameters
fixed; its conclusions apply to this Llama model and this grid.

## What is included

| File | Purpose |
|---|---|
| `resources/` | All 4,494 frozen lists, deduplicated document texts, 100 parents and source checksums. About 4.2 MB total. |
| `config.json` | Three-rate search, fixed training recipe, split seeds and selection objective. |
| `prepare.py` | Freeze configuration and generate auditable parent/list split manifests. |
| `preflight.py` | Pin model revision and tokenize every complete prompt/response; reject truncation. |
| `train.py` | Fresh-base trial, selected-checkpoint internal test or fresh final refit. |
| `run_matrix.py` | Print or execute trials; optional `--gpus 0,1,2,3,4,5,6,7` runs one trial per GPU. |
| `campaign.py` | Optional four-model study preparation, stage commands and verified status. |
| `analyze.py` | Verify complete trial grids and select using validation scores only. |
| `protocol.py` | Shared data, grouping, masking, metrics and integrity checks. |
| `environment-train.yml` | Linux Conda environment adapted from the retained Unsloth training export. |
| `test_*.py` | CPU tests; synthetic fixtures are not experimental results. |
| `PROTOCOL.md`, `VALIDATION.md` | Experimental decisions and what was actually verified locally. |
| `build_resources.py` | Optional maintainer utility for rebuilding resources from the original repository. You do not need to run it. |

The data retain the historical sampler's actual behavior: 4,312 lists have 50
candidates; 182 have 51; 1,392 contain more than five positives. No sampling,
reshuffling of candidates, text rewriting or label changes are performed here.
This isolates the selection-split issue from the separate sampler issue.
The frozen artifact contains 8,477 unique candidate documents; this count refers
to the actual bundled lists, not a newly reconstructed sample.

Parent mapping was verified using aspect ID, query, category, candidate membership
and relevance labels against the original full parent pools. Some aspect IDs are
reused across parents, and frozen/source row orders differ; neither identifier
alone is used as a parent ID. All variants of each aspect have the same verified
parent. Document overlap across different parents is allowed: this experiment
tests **parent disjointness**, not document disjointness.

## 1. Decide and freeze the experiment

Default settings:

- Base: `meta-llama/Llama-3.1-8B-Instruct`, pinned to one commit for the whole study.
- Learning rates: `5e-6`, `1e-5` (the reference setting), `2e-5`.
- Split seeds: `42`, `2026`, `3407`; training seed always `3407`.
- Two epochs; BF16 LoRA; rank 128, alpha 256, dropout 0.05.
- Batch size 2, accumulation 1; AdamW; cosine schedule; 20 warmup steps; weight decay 0.01.
- Maximum sequence length: 16,384. Preflight checks this rather than silently truncating targets.
- Selection: lowest mean **parent-macro response NLL** across the three split seeds.

For each example, response NLL is the mean teacher-forced negative log likelihood
over assistant tokens, including the chat termination tokens. The prompt is
masked. Example losses are averaged within each parent, then over validation
parents. This gives equal weight to independent parents. `list_macro_nll` and
`token_micro_nll` are also recorded. These objectives are not ranking metrics;
this experiment tests hyperparameter selection under a grouped split.

Do not pick an objective after looking at which one favors the current setting.
If reproducing an old loss-based HPO, match the old weighting explicitly; do not
assume `Trainer.eval_loss` always equals this bundle's parent-macro objective.

There are five categories, each with 20 parents. Two parents per category form a
**fixed internal test set**. For each split seed, the remaining 18 become 16
training and 2 validation parents: 80/10/10 overall. Test parents never enter any
trial's training or validation. The optional `list_control` splits only the 90
non-test parents' lists, approximately 8/9 training and 1/9 validation. It allows
train/validation parent overlap deliberately, while preserving the held-out test
parents. This control is not the original historical list-level 80/10/10 split.

The different split seeds vary parent membership, not the training seed. Report
the spread as **variation across splits**, not training-seed uncertainty or an
independent-replicate standard error. Across splits, train/validation membership
can change as in cross-validation; separation is enforced within every trial.

## 2. Set up the Linux GPU environment

Use one BF16-capable NVIDIA GPU with enough memory for the original 8B, long-list
recipe. An H200 with approximately 141 GiB is the intended hardware; this bundle
does not promise it will fit a smaller GPU. Allow roughly 20 GB for base weights,
several GB of host RAM for the token cache, and additional disk space for adapters
and optional merged weights. Run one trial per GPU. With several GPUs, follow
[`PARALLEL.md`](PARALLEL.md) to schedule independent trials concurrently.

The following commands use **Bash on Linux**, not Windows PowerShell. Edit only the
path on the first line to the location where you copied this folder:

```bash
cd /DATA/j_andrei/parent_grouped_hpo
export HPO_BUNDLE="$PWD"
export HPO_ARTIFACTS="$HPO_BUNDLE/artifacts"
export CUDA_VISIBLE_DEVICES=0

conda env create -f environment-train.yml
conda activate grouped_hpo
hf auth login
```

Accept the Llama model license using the same Hugging Face account. Authentication
is for downloading the base model/tokenizer; no proprietary model API is used by
these scripts. DORIS-MAE's released relevance labels themselves originate from
its ChatGPT-based annotation pipeline, which should still be disclosed.

The environment YAML is a retained Linux package snapshot, not a newly tested
installation. If the resolver fails, settle on a working environment and export
it **before any paper trials**. Run smoke validation in that environment and use
it for the whole study. Do not mix package versions across trials; analysis
rejects differing recorded training package versions.

```bash
python -B -m unittest discover -s "$HPO_BUNDLE" -p 'test_*.py' -v
python -m pip check
python -c 'import numpy, torch; print(numpy.__version__); print(torch.from_numpy(numpy.zeros(2)))'
python -c 'import torch; print(torch.__version__); print(torch.cuda.get_device_name(0)); print(torch.cuda.is_bf16_supported())'
```

## 3. Prepare and inspect the grouped splits

Edit `config.json` now if you have the actual original trial grid. `shared` defines
the default parameters and each entry in `trials` overrides specific values.
Supported trial overrides are the keys in `shared`; optimizer, scheduler,
masking and sampling are fixed in the code. Optional top-level `load_in_4bit`
is a boolean, defaults to false and fixes the loading policy for the whole study.
The four-model campaign enables it only for Gemma 27B. `reference_trial` must name the
configuration used in your paper. `original_grid_replay` is a provenance label,
not automatic proof that a configuration reproduces the historical HPO.

```bash
python -B prepare.py --config config.json --output "$HPO_ARTIFACTS"
python -B run_matrix.py --artifacts "$HPO_ARTIFACTS"
```

Preparation verifies the bundled data checksums and writes:

```text
artifacts/config.json
artifacts/study.json
artifacts/splits/grouped_42.json
artifacts/splits/grouped_2026.json
artifacts/splits/grouped_3407.json
artifacts/splits/list_control_42.json
artifacts/splits/list_control_2026.json
artifacts/splits/list_control_3407.json
```

Each manifest includes exact list indices, parent IDs, counts and parent-overlap
diagnostics. Grouped train/validation overlap must be empty, and every grouped
partition has 80/10/10 parents. List counts need not be 80/10/10 because parents
have different aspect counts. The dry run prints nine commands and launches no
training. Inspect these outputs before continuing.

Preparation refuses an existing output directory. To change code or configuration,
use a fresh study directory, e.g. set `HPO_ARTIFACTS` to `artifacts_v2` and repeat
preparation. Never modify frozen artifacts to bypass the integrity checks.

## 4. Pin the base model and preflight all training sequences

Resolve the model SHA once and retain the resulting file:

```bash
python -c 'from huggingface_hub import HfApi; import json; from pathlib import Path; c=json.loads(Path("config.json").read_text()); Path("model_revision.txt").write_text(HfApi().model_info(c["model"]).sha + "\n")'
export HPO_MODEL_REVISION="$(cat model_revision.txt)"

python -B preflight.py \
  --artifacts "$HPO_ARTIFACTS" \
  --revision "$HPO_MODEL_REVISION"
```

This downloads the tokenizer and caches all 4,494 encoded examples. It verifies
the exact prompt/response token boundary, nonempty supervised responses and
complete sequences within the context limit. The full 100-parent set is checked
so all trials and any later refit use the same tokenization policy. Inspect
`artifacts/tokens/audit.json` and the success manifest at `tokens/manifest.json`.

**If any example exceeds 16,384 tokens:** do not truncate it silently or filter it
after splitting. The failed preflight leaves an audit but no success manifest,
and training cannot start. To preserve complete examples, choose an adequately
large context length supported by the model and GPU, set it in `config.json`,
prepare a fresh artifact directory and repeat preflight. This is then a new
longer-context sensitivity cohort: report the departure from the original recipe
and do not claim an exact original-context replay. If you instead need to study
the original truncation policy, it must be implemented and audited explicitly
before training; this bundle intentionally does not guess that policy.

On a resumed shell, restore `HPO_BUNDLE`, `HPO_ARTIFACTS`, `CUDA_VISIBLE_DEVICES`
and the Conda environment. Do not resolve a different model SHA midway through
the study. The successful preflight manifest becomes the authoritative SHA.

## 5. Run the two-step GPU smoke check

```bash
python -B train.py \
  --artifacts "$HPO_ARTIFACTS" \
  --protocol grouped --split-seed 42 --trial lr_1e-5 \
  --smoke-steps 2
```

This loads the pinned base, initializes a new LoRA adapter, trains two optimizer
steps, saves the adapter and evaluates two validation examples. Inspect:

```text
artifacts/smoke/grouped/split_42/lr_1e-5/result.json
```

Its status must be `smoke_only`, with finite validation losses. It is not a paper
result and is stored separately from the real matrix. If you changed trial IDs,
replace `lr_1e-5` in the command with your reference trial ID.

Fix environment/runtime problems before the full matrix. Never change precision,
batching or dependencies halfway through the paper trials to get one run to fit;
freeze the revised setup in a new cohort if changes are necessary.

## 6. Run the grouped hyperparameter search

For the eight-H200 server, use the parallel commands in
[`PARALLEL.md`](PARALLEL.md). The command below keeps the sequential default.

```bash
python -B run_matrix.py --artifacts "$HPO_ARTIFACTS" --execute
```

The runner executes nine trials sequentially. Each starts in a fresh Python
process from the original pinned base; no trial initializes from another trial's
adapter or an existing full-data checkpoint. It trains for the configured epoch
budget and evaluates validation once at the end. There is no validation-driven
early stopping or hidden checkpoint selection.

For each trial you get:

```text
artifacts/runs/grouped/split_42/lr_1e-5/
  started.json
  adapter/
  validation_scores.json
  result.json
```

The result contains exact training settings, package versions, model revision,
training time, peak training allocation/reservation, optimizer steps, per-parent
and per-list validation aggregates, and adapter checksums. It does not contain
LitSearch or CSFCube scores.

Rerunning the matrix skips compatible completed results. Interrupted runs are not
silently resumed from partially trained adapters. Preserve the failed directory
by moving it outside its `runs/<protocol>/split_<seed>/<trial>` location, record
the failure, then rerun that trial from the base. Never mix partial and complete
results. If resolving the failure changes code/configuration/environment, start
and rerun a fresh study instead. Check disk availability before the full sweep.

## 7. Generate the grouped-selection results

```bash
python -B analyze.py --artifacts "$HPO_ARTIFACTS" --protocols grouped
```

Read these outputs:

```text
artifacts/analysis/validation_summary.csv
artifacts/analysis/report.md
artifacts/analysis/selection.json
```

The table reports each configuration's mean validation NLL and sample standard
deviation across split seeds. `selection.json` records the aggregate winner,
winner for each split, whether the reference was retained, and source result
hashes. Exact score ties are broken deterministically by trial ID, not by giving
the reference special treatment. Inspect near ties and split-specific winners;
the output is not a statistical equivalence claim.

Analysis fails if any required run is missing, unfinished, smoke-only or from a
different frozen study. It validates the per-example validation membership and
recalculates summaries before selecting. It never reads test scores.

## 8. Optional: compare grouped selection with a list-level control

This adds nine trainings using the same grid, pinned base, frozen lists, fixed
test parents, training seed and objective:

```bash
python -B run_matrix.py \
  --artifacts "$HPO_ARTIFACTS" --protocols list_control --execute

python -B analyze.py \
  --artifacts "$HPO_ARTIFACTS" --protocols grouped list_control
```

The report now states whether the winners agree. Compare the **selected
configuration**, not the absolute validation losses as if the validation sets
were identical. Validation membership differs between protocols; lower loss in
the leaky list control is not evidence of better generalization. Training-list
counts can also differ slightly because grouped splits partition whole parents.

This comparison is stronger evidence about split sensitivity than comparing
against an undocumented historical search, but remains limited to the supplied
grid and model. It does not prove all hyperparameters are invariant to grouping.

## 9. Optional: evaluate the selected configuration on the internal test

Do this only after finishing selection and committing to the winner:

```bash
for split_seed in 42 2026 3407; do
  python -B train.py \
    --artifacts "$HPO_ARTIFACTS" --mode test \
    --protocol grouped --split-seed "$split_seed"
done
```

Adjust the seed loop if you edited `config.json`. The script automatically loads
the **aggregate selected configuration's existing 80-parent checkpoint** for that
split. Results are in `artifacts/tests/grouped/split_<seed>/<winner>/`.
The same ten test parents are used for all three checkpoints; these are repeated
evaluations on the same test set, not three independent test samples. Do not use
the test scores to change the winner, grid, epochs or preprocessing.

## 10. Optional: refit on all 100 parents

If grouped selection changes the configuration, or you want a new final checkpoint
with fully documented selection provenance:

```bash
python -B train.py \
  --artifacts "$HPO_ARTIFACTS" --mode refit \
  --protocol grouped --export-merged
```

This initializes another fresh base and adapter, trains on all 4,494 lists, then
saves both the adapter and merged model under:

```text
artifacts/refit/grouped/<winner>/adapter/
artifacts/refit/grouped/<winner>/final/
artifacts/refit/grouped/<winner>/result.json
```

The original internal test parents now participate in training. Do not report
their performance as held out for this refitted model. Evaluate the new model
with your existing frozen LitSearch/CSFCube harness if updating the paper's
external benchmark tables; keep this new cohort separate from historical runs.
External evaluation is not required to produce the grouped HPO selection table,
and those benchmark resources are intentionally not dependencies of this bundle.

If the reference setting wins and the recipe is otherwise unchanged, the study
supports a qualified selection-sensitivity statement. It does not retroactively
turn historical list-level selection into grouped selection. State what was
performed originally and what was checked during revision.

## 11. What to put in the paper and reviewer response

Include the split manifest, search space, selection objective, exact model
revision, training budget and a validation table with every trial. Report all
split-specific winners as well as the aggregate winner. Do not call learning-rate
sensitivity “all hyperparameters unchanged.”

After results exist, adapt this wording using the actual outcomes:

> We conducted an additional parent-grouped model-selection study for Llama 3.1
> 8B. Ten category-balanced DORIS-MAE parents were reserved as a fixed internal
> test set. Across three predefined splits of the remaining parents, training and
> validation contained 80 and 10 parents, respectively; all aspects and augmented
> lists from each parent remained within one partition. Each trial initialized
> from the same base-model revision. We selected the learning rate using mean
> parent-macro validation response NLL over the three splits. Other training
> settings were fixed. The selected learning rate was [actual winner].

If `1e-5` wins, you can add that the paper's learning rate was also selected under
this grouped protocol. If it loses, report the new selection and any new final
model results. If the optional control was run, report whether its winner agrees.
With only the default grid, explicitly limit the conclusion to learning rate.

For a claim about the complete original HPO, use the actual original candidate
configurations and objective. A new local grid cannot establish that untested
historical choices would remain unchanged.

## References and execution limits

- [Grouped validation guidance](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data)
- [Transformers Trainer documentation](https://huggingface.co/docs/transformers/main_classes/trainer)
- [DORIS-MAE annotation provenance, Section 4](https://proceedings.neurips.cc/paper_files/paper/2023/file/78f9c04bdcb06f1ada3902912d8b64ba-Paper-Datasets_and_Benchmarks.pdf)

See `VALIDATION.md` for exact local checks. No H200 training, completed GPU HPO
results or successful fresh Conda installation are claimed by delivering this
folder. The GPU smoke and real experiment commands above must be run on your
target machine before reporting experimental outcomes.
