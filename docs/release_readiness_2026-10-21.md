# Resources accompanying the article

The collection has now been assembled. Use [the current resource index](../resources.md)
and [AUTHOR_TODO](../AUTHOR_TODO.md) for included files and remaining items.
The notes below record the earlier scope discussion.

Scope clarified by the author on 2026-10-21: provide the resources actually used
for the article. This replaces the earlier, broader reproduction-framework
checklist. No CPU reproduction workflow, CI, new command-line framework, or
retraining campaign is required for this resource collection.

## Suggested structure

Keep the existing notebooks, scripts, configurations and environment exports.
Organize additional resources by experiment, without rewriting the original code
merely to impose a common interface.

```text
README.md
CITATION.cff
LICENSE
notebooks/                  # Existing preparation, training and evaluation notebooks
scripts/                    # Scripts actually used
configs/                    # Actual experiment settings and environment exports
experiments/
  parent_grouped_hpo/
  tie_sensitivity/
  compact_reranker/
  parent_budget/
  matched_ablations/
  document_disjoint/
benchmarks/
  litsearch/                # Evaluation/reranking scripts used
  csfcube/                  # Evaluation scripts used
results/                    # Retained rankings, metrics, statistical outputs
resources.md                # Data/checkpoint links and experiment-to-file guide
LICENSES/                   # Notices for included third-party resources
```

This is a suggested arrangement, not a list of new implementations to build.
Keep an experiment's existing folder layout where that is clearer. Do not
create empty folders for unavailable material.

## Resources to add from the existing workspace

Paths below are relative to the surrounding workspace, not this repository.

| Resource | Existing location |
|---|---|
| LitSearch reranking and evaluation | `LitSearch/eval/`, `LitSearch/utils/`, `LitSearch/evaluate_results.py` |
| CSFCube evaluation | `CSFCube/eval_scripts/ranking_eval.py`, `CSFCube/eval_scripts/rank_metrics.py` |
| Corrected Gemma base and FT V2 rankings and metrics | `CSFCube/eval_scripts/runs_gemma_3_27b_base_19102026_vllm_full_list_4096/`, `CSFCube/eval_scripts/runs_gemma_3_27b_20260218-181123_vllm_full_list_4096/` |
| Paired statistical analysis | `tools/paired_statistical_uncertainty.py`, its imported helpers, relevant configurations, and corrected saved outputs |
| Parent-grouped HPO | `New folder/parent_grouped_hpo/`, including frozen resources; add actual trial/selection/refit records retained on the server |
| Tie sensitivity | `New folder/tie_sensitivity/`, `tie_sensitivity_report_20261006/`, retained run outputs |
| Compact supervised reranker | `New folder/ranking_pretrained_compact_baseline/` and the helpers it actually imports |
| Parent-budget study | `New folder/train_llama_parent_query_learning_curve.ipynb`, `HRIA/generate_parent_query_learning_curve.py`, corresponding results |
| Matched ablations | `New folder/train_llama_matched_ablations.ipynb`, corresponding results under `New folder/results_review/` |
| Document-disjoint study | Existing preparation/comparison scripts under `tools/`, configurations, and retained strict-rerun results |
| Other primary and supplementary experiments | Select the producing notebooks and outputs from `LitSearch/results/` and relevant folders under `New folder/`; identify them in `resources.md` |

Select resources that actually underlie the article. Do not copy every historical
or abandoned run. Keep earlier plain ModernBERT and current GTE-ModernBERT studies
separate. The corrected Gemma V2 statistical outputs supersede the affected
comparisons, not every other statistical result.

## What to write in the README and resource guide

- State that this repository collects resources used in the article.
- Describe each folder briefly.
- For each experiment, link its notebook/script, input data, checkpoint and saved
  results; identify the corresponding table or figure by its LaTeX label/name.
- Link source datasets and prepared data actually used. Preserve parent mappings
  and split files for parent-grouped studies where retained.
- Link Hugging Face checkpoints; include exact revisions when known. Do not infer
  missing historical revisions from the current state of a repository.
- Include the commands/settings actually used where available, especially serving
  model IDs, prompts, candidate depths and output budgets.
- Identify unavailable resources or unresolved provenance plainly.

A simple Markdown table is sufficient; a new machine-readable registry is not
required. Retain the environment exports you have and identify their experiments.

## Accuracy issues to describe or reconcile

These are about accurately identifying the supplied resources, not implementing
new infrastructure:

- Preparation currently defaults to 100 candidates while the primary YAML says
  50. Identify the command/configuration actually used for the relevant data.
- The extracted preparation code drops `original_query_id` and its sampler differs
  from the historical notebook. Do not label this refactor as the exact producing
  implementation without evidence. Supply retained parent mappings/frozen inputs.
- The grouped-HPO resource manifest describes 4,494 lists, including 182 of length
  51 and 1,392 with more than five positives. Preserve this artifact's description;
  it does not establish the training inputs of historical checkpoints.
- The Gemma 12B notebook uses batch 1/accumulation 2, whereas the YAML/article list
  2/1. Identify the actual producing settings if recoverable; otherwise flag the
  discrepancy rather than silently rewriting the historical notebook.
- Distinguish actual base repositories from repackaged/alias identifiers, and the
  executed Llama grouped-HPO study from any unexecuted four-model campaign code.
- CSFCube aggregate and paired-uncertainty tables use different aggregation rules.
  Keep the corresponding analysis outputs clearly identified, with complete Holm
  comparison families where applicable.

## Before uploading

- Include saved outputs behind the reported results, or stable download links.
- Keep weights on Hugging Face and link them rather than copying them into Git.
- Check that `.gitignore` does not silently exclude intended JSONL evidence.
- Exclude credentials, caches, local environments and unrelated temporary files.
- Choose a license for your own code and retain upstream notices for copied code.
- Use the same actual GitHub URL in the article and `CITATION.cff`.

The current revision only updates this guidance. It does not migrate resources,
change experiment code, rerun experiments, or publish anything.
