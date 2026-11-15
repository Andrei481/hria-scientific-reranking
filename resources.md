# Resources used in the article

The following links point to retained resources, not newly run experiments.
Table names use stable LaTeX labels because printed table numbers can change.
All paths under `resources/` retain the original workspace names.

| Study / article label | Code and configuration | Data and results |
|---|---|---|
| Primary model training; `tab:model-training-settings` | [Numbered notebooks](notebooks/), [original DORIS notebooks](resources/DorisMAE/), [retained Gemma notebooks](resources/HRIA/), [environment exports](configs/environments/) | [Prepared aspect rows](resources/HRIA/reranking_dataset_doris_mae.jsonl); checkpoint links/status in [models and data](docs/models_and_data.md). Original checkpoint input identity is not established by the later frozen list artifact. |
| LitSearch primary results and depth; `tab:litsearch-main`, `tab:inference-depth`, `tab:inference-depth-detailed` | [Reranking/retrieval code](resources/LitSearch/eval/), [evaluation](resources/LitSearch/evaluate_results.py), [local model client](resources/LitSearch/utils/local_llm_utils.py) | [Primary rankings and metrics](resources/LitSearch/results/reranking/), [retrieval pools](resources/LitSearch/results/retrieval/); additional depth outputs retained under [New folder](resources/New%20folder/). |
| CSFCube main/facet results; `tab:csfcube`, `tab:csfcube-facet-gains-appendix` | [Original no-facet notebook](resources/New%20folder/csfcube_eval_original.ipynb), [official evaluator](resources/CSFCube/eval_scripts/ranking_eval.py), [metric definitions](resources/CSFCube/eval_scripts/rank_metrics.py) | [Gold/query files](resources/CSFCube/), [selected run directories](resources/CSFCube/eval_scripts/), [retained aggregate logs](resources/CSFCube/results/), [Llama/Qwen FT case records](resources/New%20folder/results_review/). Corrected Gemma paths are specified below. |
| Paired uncertainty; `tab:litsearch-bootstrap`, `tab:csfcube-uncertainty`, `tab:litsearch-r5-uncertainty` | [Analysis script](resources/tools/paired_statistical_uncertainty.py), [helper](resources/tools/litsearch_document_disjoint.py), [full comparison configuration](resources/configs/paired_statistical_uncertainty.json) | [Earlier complete analysis](resources/results/statistical_uncertainty_corrected/), [corrected CSFCube V2 families](resources/results/statistical_uncertainty_csfcube_v2_20261020/). Use the latter for affected CSFCube comparisons; retain full Holm families. |
| Gemma precision control; `tab:gemma27b-precision` | [Configuration](resources/configs/gemma27b_precision_uncertainty.json), [paired analysis script](resources/tools/paired_statistical_uncertainty.py) | [Precision analysis](resources/results/statistical_uncertainty_gemma27b_precision/); exact compared input paths are in its configuration. |
| Parent-grouped learning-rate selection; `tab:grouped-hpo-selection`, `tab:grouped-hpo-litsearch` | [Study bundle](resources/New%20folder/parent_grouped_hpo/), [CSFCube notebook](resources/New%20folder/csfcube_eval_hpo), [evaluation support bundle](resources/New%20folder/parent_grouped_hpo_evaluation/) | [Frozen lists/documents/parents](resources/New%20folder/parent_grouped_hpo/resources/), [reported-results evidence note](resources/HRIA/grouped_hpo_reported_evidence.md). Raw remote trial/refit and benchmark outputs are still missing; see AUTHOR_TODO. |
| Parent supervision budget; `tab:parent-budget-main`, `tab:parent-budget-detailed`, `fig:parent-query-learning-curve` | [Training notebook](resources/New%20folder/train_llama_parent_query_learning_curve.ipynb), [figure/aggregation script](resources/HRIA/generate_parent_query_learning_curve.py) | Named `parent_budget_*` directories in [results_review](resources/New%20folder/results_review/), [retained figure](resources/HRIA/fig_parent_query_learning_curve.pdf). |
| Ablations and seeds; `tab:litsearch-ablation-main`, `tab:litsearch-ablation-detailed`, `tab:pool-augmentation-uncertainty`, `tab:seed-stability` | [Numbered ablation/seed notebooks](notebooks/), [retained seeded notebook](resources/New%20folder/train_llama_seed.ipynb), [statistical configuration](resources/configs/paired_statistical_uncertainty.json) | [VM139 outputs](resources/New%20folder/results_vm_139/), [VM250 outputs](resources/New%20folder/results_vm_250/), [statistical results](resources/results/statistical_uncertainty_corrected/). Historical timestamps are retained; unresolved checkpoint mappings are not guessed. |
| Matched ablations; `tab:matched-main-ablation` | [Matched training notebook](resources/New%20folder/train_llama_matched_ablations.ipynb) | Named `matched_ablation_*` directories in [results_review](resources/New%20folder/results_review/). |
| Tie sensitivity; `tab:tie-sensitivity`, `tab:tie-sensitivity-uncertainty` | [Study scripts/notebooks](resources/New%20folder/tie_sensitivity/), [report and budget audit](resources/tie_sensitivity_report_20261006/) | [Six final run records](resources/tie_sensitivity_artifacts/runs_disk_safe/), [final analysis](resources/tie_sensitivity_artifacts/analysis_final_three_seeds/), [frozen evaluation inputs](resources/tie_sensitivity_artifacts/evaluation_inputs/), [checkpoint links](docs/models_and_data.md#tie-sensitivity-checkpoints). |
| Leakage/disjoint evaluation; `tab:leakage-overlap`, `tab:litsearch-document-disjoint` | [Leakage notebook](resources/DorisMAE/leakage_analysys.ipynb), [preparation](resources/tools/prepare_litsearch_document_disjoint.py), [comparison](resources/tools/compare_litsearch_document_disjoint.py) | [Overlap outputs](resources/leakage_analysis_outputs/), [analysis outputs](resources/results/litsearch_document_disjoint/), [strict-rerun results and filtered pools](resources/New%20folder/results_vm_139/litsearch_document_disjoint/strict_rerun/). |
| Open reranker baselines; `tab:open-reranker-baselines`, `tab:open-reranker-uncertainty` | [Retained reranker implementations](resources/LitSearch/eval/reranking/), [paired analysis configuration](resources/configs/paired_statistical_uncertainty.json) | [Local baseline outputs](resources/LitSearch/results/others/), [analysis](resources/results/statistical_uncertainty_corrected/). |
| Compact supervised GTE reranker; `tab:compact-litsearch`, `tab:compact-csfcube`, `tab:compact-compute`, `tab:compact-uncertainty` | [Training/evaluation/compute notebooks](resources/New%20folder/ranking_pretrained_compact_baseline/), [shared helper](resources/tools/modernbert_supervised_baseline.py) | [Rankings, metrics, training/evaluation manifests and compute measurements](resources/New%20folder/ranking_pretrained_compact_baseline/ranking_pretrained_compact_baseline/); the adapted weights exist locally but still need a public link. |
| Earlier plain ModernBERT control (distinct from GTE) | [Original bundle](resources/New%20folder/modernbert_supervised_baseline/) | [Frozen listwise supervision and outputs](resources/New%20folder/modernbert_supervised_baseline/modernbert_supervised_baseline/). Included because the shared frozen data and retained analysis refer to it; do not relabel it as GTE. |
| Reconstructed models; `tab:reconstructed-main-replication` | [Retained training notebooks](resources/DorisMAE/), [environments](configs/environments/) | [VM139](resources/New%20folder/results_vm_139/), [VM250](resources/New%20folder/results_vm_250/), [other named model result folders](resources/New%20folder/); distinct from primary results. |
| Exploratory FIRST, prompt repetition and query rewriting; `tab:first-ablation`, `tab:prompt-repetition` | [FIRST notebooks](resources/HRIA/train_unsloth_12b_FIRST.ipynb), [second FIRST notebook](resources/HRIA/train_unsloth_12b_first_v2.ipynb), [rewriting notebook](resources/HRIA/rewrite_queries.ipynb), [reranking scripts](resources/LitSearch/eval/reranking/) | [Rewritten query data](resources/HRIA/rewritten_queries.jsonl), retained `results*.txt` under [HRIA](resources/HRIA/). These text summaries are not complete per-query records. |
| Figures and literature comparisons; `fig:litsearch-gains`, `fig:gpt4o-comparison`, `tab:gpt4o`, `tab:csfcube-external-baselines` | [Retained figure scripts and figures](resources/HRIA/) | Literature-derived rows retain the sources cited in the article; they are not local model runs. |

## Numerical evidence and GTE uncertainty

See [the numerical evidence audit](docs/numerical_evidence_audit_2026-10-28.md)
for remaining gaps and checks actually performed. The saved GTE comparison
results are now included in [statistical_uncertainty_gte](resources/results/statistical_uncertainty_gte/).
These 22 rows were recovered from an existing analysis, not recomputed.

## Corrected Gemma CSFCube files

- Base: [runs_gemma_3_27b_base_19102026_vllm_full_list_4096](resources/CSFCube/eval_scripts/runs_gemma_3_27b_base_19102026_vllm_full_list_4096/).
- FT V2: [runs_gemma_3_27b_20260218-181123_vllm_full_list_4096](resources/CSFCube/eval_scripts/runs_gemma_3_27b_20260218-181123_vllm_full_list_4096/).
- Paired results: [statistical_uncertainty_csfcube_v2_20261020](resources/results/statistical_uncertainty_csfcube_v2_20261020/).

The main CSFCube aggregate uses the released fold averaging; the paired table
uses pooled cases with clustering by query paper. Do not substitute one mean for
the other or mix the earlier Gemma checkpoint's uncertainty with FT V2.

## Reading the collection

The table identifies resource groups. It does not certify a one-to-one historical
checkpoint mapping for every timestamped run. Missing mappings and files are
listed in [AUTHOR_TODO.md](AUTHOR_TODO.md). The source path and SHA-256 of every
collected file are in [the inventory](docs/resource_inventory.csv).

The root `benchmarks/` and `results/` directories from the initial scaffold are
navigation points; the collected files live under `resources/` to preserve the
paths used by existing scripts. No experiments were rerun during packaging.
