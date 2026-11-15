# Exact-base export change: local validation

The GPU/Unsloth/PEFT execution path must still be checked on the server. This
workstation does not have the target GPU environment. No remote training or
export was launched as part of this edit.

## Bundle tests

`python -B -m unittest discover -s "New folder/tie_sensitivity" -p "test_*.py" -v`

Result: 23 tests passed. The new regression checks cover preserving the adapter
before merge, exporting loaded weights rather than invoking the mirror-resolving
exporter, retaining the pre-merge trainable-parameter count, refusing to overwrite
an export directory, and rejecting an unpinned revision before model loading.
These CPU checks use a small stateful model double at the GPU/PEFT boundary;
they do not claim to validate PEFT or Unsloth itself. The export-only CLI help and
Python syntax checks also passed. An additional regression saves a tiny real
Llama checkpoint and tokenizer, deliberately patches the parent process's
attention implementation, and successfully reloads/executes the checkpoint in a
clean subprocess using native Transformers on CPU. A separate guard rejects a
verifier process that already imported Unsloth. This tests process isolation, not
the target H200 environment. The local HF cache is read-only under the sandbox,
so Transformers emits a cache-location warning; the temporary local checkpoint
check still passes without downloading a model.

## Broader repository tests

`python -B -m unittest discover -s tests -v`

Result: 73 tests run, six failures (including subtest failures), two errors.
These involve files outside `New folder/tie_sensitivity`; they were not changed
for this task:

- `test_reranker_inference_regressions.RerankerInferenceRegressionTests.test_gemma_notebooks_keep_adapter_backup_outside_merged_export_dir`: four failing subtests for `train_gemma3_12b.ipynb`, `train_gemma3_27b.ipynb`, `DorisMAE/train_gemma3_12b.ipynb`, and `DorisMAE/train_gemma3_27b.ipynb`. The notebooks contain the adapter-save path forbidden by the test.
- `test_train_llama_notebook.TrainLlamaNotebookTests.test_llama_notebooks_check_for_meta_parameters_before_trainer_init`: failure for `train_llama.ipynb`; expected meta-parameter guard absent.
- `test_train_llama_notebook.TrainLlamaNotebookTests.test_llama_notebooks_use_materialized_unsloth_4bit_checkpoint`: failure for `train_llama.ipynb`; expected quantized model name absent.
- `test_litsearch_build_index_revision.LitSearchBuildIndexRevisionTests.test_main_loads_pinned_revision_and_builds_index`: `SystemExit: 2` from required `--index_type`/`--key` arguments during import.
- `test_litsearch_build_index_revision.LitSearchBuildIndexRevisionTests.test_parser_defaults_to_pinned_dataset_revision`: same argument-parsing error.

## Server validation without retraining

Upload `train.py`, `export_checkpoint.py`, and `verify_checkpoint.py` beside the
existing `experiment.py`. For each saved smoke run, run:

```bash
python -u "$TIE_DIR/export_checkpoint.py" \
  --run "$TIE_ARTIFACTS/smoke_runs_exact/deterministic_seed42_SMOKE"
```

Repeat for `epoch_random_seed42_SMOKE`. Success prints
`Exact-base export and reload check passed` and writes
`export_check/export_manifest.json`. This loads the recorded base SHA, attaches
the saved adapter, invokes the same export helper as training, then reloads the
standalone checkpoint for a finite-logit forward check. It does not establish
numerical equivalence or reranking quality. Old manifests and exports are not
overwritten or relabeled as new results.

If the old export completed but its in-process reload hit the `apply_qkv` error,
do not repeat export: run `verify_checkpoint.py --checkpoint PATH/export_check/final`
in a new Python process. This verifies the existing files and records
`final/reload_check.json`; it does not reconstruct a missing old export manifest.

Refresh preflight for both full-training conditions after copying the scripts,
since their recorded source hashes and export configuration changed.
