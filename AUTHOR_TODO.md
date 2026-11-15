# Links and files still needed from the author

This list concerns resources used in the article, not requests for new experiments.
Everything already recovered is indexed in [resources.md](resources.md).

## 1. Checkpoint links and their experiment mapping

- [ ] Primary fine-tuned Llama 3.1 8B, Gemma 3 12B and Qwen 2.5 14B: provide
  the Hugging Face URL (and revision if retained) for each published primary run.
- [ ] Gemma 3 27B: confirm which primary LitSearch checkpoint corresponds to each
  result set and precision condition. The FT V2 CSFCube link
  `Andrei48155/20260218-181123` is already listed; add its evaluated commit if known.
- [ ] Parent-grouped HPO refit: publish/link the adapter from
  `/home/andreij/parent_grouped_hpo/artifacts/refit/grouped/lr_5e-6/adapter`
  and its tokenizer/base-model information. Do not substitute a tie-study adapter.
- [ ] Adapted GTE-ModernBERT: upload/link the local `final/` checkpoint identified
  in [models and data](docs/models_and_data.md). The model files are locally
  available; only their publication URL is missing. The same applies to the
  earlier plain ModernBERT control if you want to distribute that checkpoint.
- [ ] Where available, link reconstructed, seed, matched-ablation, parent-budget,
  FIRST and precision-control checkpoints and identify their saved result folder.
  If weights were not retained, state that rather than linking a different run.

The six tie-sensitivity checkpoint links and exact commits have already been
recovered from publication receipts. You do not need to supply those again.

## 2. Parent-grouped HPO files still on the server

The local collection includes code, frozen data, configuration, and a clearly
labelled note containing the results reported in the conversation. It does not
contain the raw run outputs below. Copy or link the small records, not caches or
all intermediate weights:

- [ ] `parent_grouped_hpo/artifacts/study.json`, realized splits, trial manifests,
  validation scores, and saved selection/analysis outputs for the nine trials.
- [ ] `artifacts/tests/grouped/` and the refit completion/training manifest.
- [ ] LitSearch `results/Llama31_8B_parent_grouped_lr5e6/` (both evaluated depths,
  including rankings and metric JSON files).
- [ ] CSFCube `eval_scripts/runs_hpo_grouped_lr5e6_vllm_full_list_4096/`
  (ranked JSON files, per-query CSVs, metric logs).
- [ ] The actual environment export and serving command/configuration for that
  run, if retained. The bundled environment recipe is not a historical snapshot.

## 3. Exact source data and historical input identity

- [ ] Provide the download/version link for the exact complete DORIS-MAE release
  used. The official upstream link is supplied; hashes of the large local corpus,
  annotation and full dataset files are recorded in `docs/omitted_large_resources.json`.
- [ ] If retained, link the compact-study server inputs named in its preparation
  manifest: aspect supervision hash
  `010261f259e715a362e9f52bb478603a674e020fdc07156ec66de376889dc42b`
  and corpus hash
  `25d3fb7cd9153ba099f9b13225cf7d39e680788b203130e3eabb8500eabdf7f2`.
  The final frozen listwise file is already included. The local HRIA aspect file
  and full local corpus must not be labelled as those server files without a match.
- [ ] For historical primary checkpoints, add the exact formatted training files
  or hashes if you still have them. The later compact/HPO frozen lists alone do
  not prove historical input identity.
- [ ] Add any retained run manifests that resolve the Gemma 12B batch/accumulation
  discrepancy and exact model/tokenizer repositories used.

## 4. Optional missing detail behind retained summaries

- [ ] Supply the FIRST result record supporting 70.8 / 35.8 / 35.0. No matching
  metric file or evaluation summary was located in the packaged resources.
- [ ] Add raw per-query prompt-repetition results if retained. All nine published
  rows match packaged terminal summaries, but their prediction files are missing.
- [ ] Identify the exact producing code revision or server copy if a local notebook
  differs from what ran. Existing variants are retained and labelled, not merged.
- [x] The three `tab:top-rank-failures` examples were checked against packaged
  primary Qwen3-based rankings on 2026-10-28; their metrics match. No additional
  result file is needed to support those numerical examples.

## 5. Publication metadata

- [ ] Confirm the final GitHub URL. `CITATION.cff` currently names
  `andrei-joldea/scientific-reranking-reproducibility`; the article names
  `Andrei481/hria-scientific-reranking`. Neither was changed automatically.
- [ ] Add the final article URL/DOI to README and citation metadata when available.
- [ ] Choose the license for your original code. Existing LitSearch and CSFCube
  notices have been preserved; upstream data/model terms remain separate.

Missing historical material can be marked unavailable. No new training is needed
to complete this resource description.
