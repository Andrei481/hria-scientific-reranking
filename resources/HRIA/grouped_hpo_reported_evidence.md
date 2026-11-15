# Parent-grouped HPO manuscript revision ? 2026-10-12

Only `articol_latex.tex` was edited. The pre-edit version is retained as
`articol_latex.before_grouped_hpo_2026-10-12.tex`. The complete patch below shows
all additions and replacements; existing primary/reconstructed result tables
were not changed.

## Evidence for numerical additions

The remote experimental results below were supplied by the user in this
conversation. They were not regenerated locally or independently read from the
remote artifacts. Protocol facts were checked against the local experiment code
and resources. No additional training, scoring, significance tests, or inferred
improvements are claimed by this edit.

| Added information | Supporting evidence |
|---|---|
| Llama 3.1 8B; rates 5e-6, 1e-5, 2e-5; split seeds 42/2026/3407; training seed 3407; two epochs | `New folder/parent_grouped_hpo/config.json`; user reports for completed grid and refit |
| Five categories; per-category 16 training / 2 validation / 2 test parents; total 80/10/10 parents | `protocol.make_split`; locally regenerated all three grouped splits from `resources/parents.json` and `resources/lists.jsonl.gz` |
| Same ten test parents across splits, selected with seed 42; no train/validation parent overlap | `protocol.make_split`, `config.json`; verified parent-ID sets locally |
| Selection averages response-token NLL per list, then lists within parents, then parents, then splits | `train.evaluate_losses`, `protocol.summarize_losses`, `protocol.select_results` |
| 0.070711 / 0.009043, 0.072112 / 0.009905, 0.074852 / 0.008772 (mean / SD) | User-pasted `analyze.py --protocols grouped` output, in ascending learning-rate order |
| 5e-6 wins all three splits | Same output: winners 42, 2026, 3407 all `lr_5e-6` |
| Internal-test NLLs 0.076731, 0.076040, 0.076456 | User-pasted completed split 42/2026/3407 checks; packages match=True |
| Refit on 4,494 lists, two epochs | Config/resources plus user training output ending epoch=2.0 and complete refit path `artifacts/refit/grouped/lr_5e-6`; train.py refit uses all rows |
| PyTorch 2.14.1+cu130; Transformers 4.57.6; Unsloth 2026.10.1 | User-pasted run environment header; internal-test package comparisons matched training packages |
| Hugging Face Trainer / Unsloth FastLanguageModel | `train.py`, including `model_interface` for Llama |
| LitSearch top-100: BM25 66.31/55.11/50.03/46.66; GRIT 80.88/69.13/64.23/60.71; Qwen3 81.09/69.21/64.40/60.84 | User-pasted `evaluate_results.py` overall_avg rows, ordered R@20/R@5/NDCG@20/NDCG@5; count=597 in each; output paths under k_100 |
| CSFCube aggregate test R@20=40.83%, NDCG@20=43.51%, NDCG%20=43.20% | User-pasted official all-facet test output: 0.4083, 0.4351, 0.4320; conversion to percentages only |
| CSFCube 4,096 output tokens; full pools; original no-facet prompt | User-confirmed original notebook; inspected `New folder/csfcube_eval_hpo` configuration and prompt builder |
| CSFCube 131,072-token server context | Serving restart procedure supplied after observed 65,536-token overflow; user subsequently reported completion of all facets. This is the run procedure, not an independently retrieved server configuration snapshot |
| Direct adapter serving | User `/v1/models` output: `hpo_grouped_lr5e6` root points to full-data refit adapter; parent `llama31_base` |
| Historical list counts 3,594/450/450 and split seed 42 | Already present in original article; retained and explicitly described as historical, not new grouped counts |

The new section intentionally covers only the completed Llama study. It does
not claim unchanged hyperparameters, isolate a causal effect of grouping, or
extend grouped-selection validation to other model families. Historical base?FT
comparisons and learning curves remain separate.

## Verification

- Regenerated all three grouped splits and checked parent counts, disjointness,
  and fixed test-parent identity.
- Checked numerical table entries against the user-reported results above.
- Checked new labels/references and LaTeX environment balance.
- Independent manuscript review checked numerical fidelity and cohort scope.
- No LaTeX compiler was found in the available PATH; PDF compilation/layout was
  not verified.

