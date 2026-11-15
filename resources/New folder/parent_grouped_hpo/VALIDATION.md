# Local verification record

## Missing campaign/Gemma implementation fixed: 2026-10-06

This entry supersedes the five unresolved failures in the earlier parallel-runner
entry below. Implemented the missing four-model campaign and Gemma model-loading
helpers; added a regression test excluding vision projection targets. No tests
were skipped or removed. The complete current suite passed **25 of 25 tests**,
also independently rerun by the code reviewer.

Fresh campaign CLI preparation and status checks passed: four studies, 144
planned grouped trials, no invented selections/refits and no training launched.
The search dry run generated four model-stage commands with eight-GPU scheduling.
All Python files parsed successfully. The default nine-trial configuration is
unchanged; the larger campaign must be selected explicitly.

The independent review found no verified major defect. Campaign completion/refit
branches still lack direct end-to-end CPU fixture coverage; the existing tests
cover pending status and corrupted campaign scope/configuration. Actual GPU
loading, Gemma adapter insertion, 4-bit training and merged export have not been
run here. Server preflight and smoke checks remain required.

The reported missing NumPy is a separate server-environment problem. The retained
YAML already includes NumPy; `RECOVERY.md` gives installation, dependency-check
and Torch/NumPy interoperability commands. They have not been executed on the
remote server by this assistant. Record the repaired environment before trials.

## Parallel-runner update: 2026-10-06

Added `run_matrix.py --gpus` and instructions in `PARALLEL.md`. Three new
CPU tests passed using real subprocesses, covering device isolation and reuse,
concurrent execution, separate logs, duplicate-index rejection and failure
handling. A fresh prepared-study dry run produced nine commands and an
eight-worker queue without starting training.

The full suite at this update ran 24 tests: **19 passed, 5 failed**. All 16
original tests and the three new parallel tests passed. The failures come from
tests added on 2026-10-05, before this update, for implementation absent from the
current workspace:

- `test_campaign.CampaignTests.test_grid_covers_all_models_and_includes_reference`
- `test_campaign.CampaignTests.test_prepare_is_immutable_and_status_does_not_invent_results`
- `test_campaign.CampaignTests.test_scope_cannot_be_reduced_by_editing_manifest`
- `test_training.TrainingTests.test_gemma_loading_is_text_only`
- `test_training.TrainingTests.test_trainable_vision_and_unmaterialized_weights_are_rejected`

The campaign tests require the missing `campaign.py`; the two additional model
tests require missing `model_interface` and `validate_trainable_model` functions.
Those tests were preserved and their requested features are outside the parallel
scheduler change. The entire current suite is therefore **not green**. No
multi-GPU model training was run locally.

## Original bundle verification: 2026-10-03

Verified on 2026-10-03, on Windows with Python 3.12. These are implementation and
data checks, not experimental model results. No hyperparameter winner has been
established by these checks.

## Checks completed

The following command passed all **16 tests**:

```bash
python -B -m unittest discover -s . -p 'test_*.py' -v
```

Tests cover parent identity despite reordered rows and reused aspect IDs;
grouped separation and a fixed test set; independent category draws; deliberately
overlapping list controls; parent-macro loss weighting; response-only masking;
invalid/non-finite inputs; complete-grid selection and smoke exclusion; actual
bundled data; analysis/selection round trips; and rejection of changed settings,
mixed package versions and incorrect validation membership. Training scores used
in integration tests are synthetic, written only inside temporary directories.

Additional checks passed:

- All Python files parsed successfully, and all six executable scripts accepted
  `--help` without downloading models or initializing GPU training.
- Fresh preparation verified the packaged resource checksums and generated all
  six split manifests.
- The combined grouped/control dry run printed 18 training commands.
- An independent count directly from the original frozen JSONL confirmed 4,494
  lists and 8,477 unique candidate document IDs.
- Independent code review found no critical or important defects. Its minor
  observation about identical positional draws across categories was addressed
  with category-specific deterministic seeds and a regression test.

The machine-readable data and split audit is in
[`local_validation/data_and_splits.json`](local_validation/data_and_splits.json).
It records parent membership and counts, not trained model scores.

## Actual grouped split sizes

Each grouped split has 80 training, 10 validation and 10 test parents. All three
have zero train/validation parent overlap. The same test parents are held out
globally, comprising 444 lists.

| Split seed | Training lists | Validation lists | Test lists |
|---|---:|---:|---:|
| 42 | 3,591 | 459 | 444 |
| 2026 | 3,642 | 408 | 444 |
| 3407 | 3,594 | 456 | 444 |

The optional controls each have 3,600 training and 450 validation lists. They
have 88, 89 and 90 overlapping train/validation parents, respectively, as expected
for a list-level control. Their ten test parents remain disjoint from both.

## Checks still required on the target server

The retained Conda snapshot was not freshly installed locally. The GPU runner,
Unsloth/PEFT integration, adapter loading, merged export and real model losses
have not been executed on an H200 here. Run the README's tokenizer preflight and
two-step GPU smoke check before launching paper trials.

An attempted offline tokenizer audit with the older local Transformers/tokenizers
installation could not load the retained tokenizer: its exported
`TokenizersBackend` class and serialized tokenizer format were unsupported by
that installation. No successful token-length audit is claimed. In particular,
it is not yet established that every full sequence fits 16,384 tokens. The server
preflight must pass, or the documented context-length procedure must be followed
before training.

No full training, HPO selection, internal-test evaluation or external benchmark
evaluation has been run for this new cohort. The original HPO candidate records
and exact historical checkpoint input identity are not established. The default
three-rate grid therefore supports new learning-rate sensitivity evidence only;
it cannot show that every originally tuned hyperparameter is unchanged.
