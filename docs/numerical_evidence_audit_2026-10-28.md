# Numerical evidence audit (2026-10-28)

Scope: compare the current `HRIA/articol_latex.tex` with resources in this
collection. Distinguish preserved results from reported summaries and code-only
resources. This is not an exhaustive cell-by-cell certification of every number
in the manuscript, and no training/inference or statistical resampling was rerun.

## Remaining gaps

| Claim / table | What is available | What remains missing |
|---|---|---|
| `tab:grouped-hpo-selection`: validation means 0.070711, 0.072112, 0.074852 and SDs 0.009043, 0.009905, 0.008772 | Code, configuration, frozen data, split audit, and the [author-reported evidence note](../resources/HRIA/grouped_hpo_reported_evidence.md) | Actual trial result records and saved selection output from the server. The split audit explicitly contains no trained-model results. |
| Grouped-HPO internal test: 0.076731, 0.076040, 0.076456 | Same reported evidence note | Saved internal-test outputs. |
| `tab:grouped-hpo-litsearch`: for example BM25 66.31 / 55.11 / 50.03 / 46.66 | Same note reproducing author-supplied evaluation output | Rankings and metric JSONs from `results/Llama31_8B_parent_grouped_lr5e6/`. |
| Grouped-HPO CSFCube: test R@20 0.4083, nDCG@20 0.4351, nDCG%20 0.4320 | Same note with author-reported evaluator output | `runs_hpo_grouped_lr5e6_vllm_full_list_4096` ranked JSONs, per-query records and metric logs. |
| `tab:first-ablation`, FIRST row: 70.8 / 35.8 / 35.0 | FIRST training/scoring code | No matching evaluation summary or metric file was located in the collection. The presence of code does not substantiate those scores. Obtain the original FIRST run's summary or predictions. |
| `tab:prompt-repetition`, nine rows | All nine displayed metric tuples match retained `HRIA/results_repeat.txt` / `results_2.txt` terminal summaries | The corresponding full prediction files were not found. The values have summary-level evidence, not complete retained per-query evidence. |

Reported-output evidence is not the same as an unsupported invented value, but
it is weaker than shipping the original saved experiment files. Missing model
weights alone do not invalidate numerical evidence in preserved predictions.

## Checks that passed

- All 72 metric cells in `tab:litsearch-main` match the packaged primary metric
  JSON files at one decimal place.
- All 120 displayed numerical fields (delta, both interval bounds, raw and adjusted
  p-values) in `tab:litsearch-bootstrap` match the retained saved analysis, at the
  printed precision or `<.001` threshold.
- The six compact LitSearch rows match their six metric JSONs at two decimals.
- All three `tab:top-rank-failures` examples are supported by the primary Qwen3
  retrieval-based base/FT rankings. Recomputed R@20, R@5 and nDCG@5 match the table:
  Qwen 0.6131 -> 0.2641, Gemma 12B 0.3869 -> 0.2372, Gemma 27B 0.2961 -> 0.1815.
  Relevant positions shown in the table refer to the top 20; other relevant
  documents can appear later in the saved lists.
- All nine prompt-repetition rows match retained summary lines.
- The compact compute manifest records latency, throughput and GPU-memory values
  backing the rounded values in `tab:compact-compute`.

Source groups:
[primary metrics](../resources/LitSearch/results/reranking/),
[primary statistical analysis](../resources/results/statistical_uncertainty_corrected/),
[compact outputs](../resources/New%20folder/ranking_pretrained_compact_baseline/ranking_pretrained_compact_baseline/),
[prompt-repetition summaries](../resources/HRIA/results_repeat.txt).

## Packaging omission corrected during this audit

The initial package contained GTE prediction files and analysis configuration but
omitted the saved GTE uncertainty output. The existing workspace file
`results/statistical_uncertainty/paired_uncertainty.csv` contains the required
four GTE comparison families. Their 22 rows have now been included in
[statistical_uncertainty_gte](../resources/results/statistical_uncertainty_gte/).
All 110 displayed numerical fields in `tab:compact-uncertainty` match those rows.

The extraction preserves values and complete correction families. It does not
rerun statistics or reintroduce older Gemma CSFCube base/FT results from the same
larger source file. Source hashes are recorded in the new folder's provenance.

## Other preserved evidence and limits

Corrected Gemma CSFCube ranked JSONs, metric logs, per-query CSVs and paired
uncertainty are present. Tie sensitivity includes six sets of final predictions,
run manifests and final three-seed summaries. Parent-budget, matched-ablation,
seed/depth and disjoint result collections are also present. Their presence is
not a claim that every number in these studies was recalculated in this audit.

Parent-budget aspect/step counts are reconstructed recipe counts recorded in the
figure script; they are not all separately recovered Trainer logs. The local
Gemma 12B notebook and article differ in batch/accumulation settings, and exact
historical model/input mappings remain incomplete. These are provenance issues,
separate from whether stored rankings support a metric value.

GPT-4o and external CSFCube baseline rows are explicitly literature-derived.
Their evidence belongs to the cited publications; absence of local runs is
expected. Those external source values were not independently re-audited here.

See [AUTHOR_TODO](../AUTHOR_TODO.md) for server paths and remaining author inputs.
The article itself was not modified.
