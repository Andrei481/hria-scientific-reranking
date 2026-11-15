# CSFCube Gemma 3 27B source correction — 2026-10-20

Table 25 previously used non-V2 Gemma predictions, while Tables 24 and 26 used
V2 aggregate logs. The newly supplied base and FT reruns reproduce those V2
logs and now supply the matching per-case observations for statistical analysis.
No models were trained or served during this local correction.

## Exact inputs

Base per-case scores:

`CSFCube/eval_scripts/runs_gemma_3_27b_base_19102026_vllm_full_list_4096/test-pid2pool-csfcube-gemma_3_27b_base_19102026_full_list-4096-all-perq.csv`

Fine-tuned per-case scores:

`CSFCube/eval_scripts/runs_gemma_3_27b_20260218-181123_vllm_full_list_4096/test-pid2pool-csfcube-gemma_3_27b_20260218-181123_full_list-4096-all-perq.csv`

Each directory also contains three facet ranking JSONs, facet CSVs, and metric
logs. Historical comparison logs are
`reproducibility_package/results/csfcube/gemma3_27b/base/results.txt` and
`reproducibility_package/results/csfcube/gemma3_27b/FT/results.txt`.

`verify_sources.py` checks query identities, unique complete candidate
permutations, and agreement of all saved per-case metrics with a fresh
calculation from rankings and released gold labels. It compares all six reported
metrics for every facet and the overall aggregate, on both dev and test, against
the retained V2 logs. Both systems cover 50 query-facet cases, 34 parent papers,
and 6,244 candidate occurrences. All checks passed; the largest per-case numeric
difference was 1.1102230246251565e-16. All aggregate comparisons matched at four
decimal places. Matching rounded aggregates does not prove identity with the
unavailable historical per-case predictions or establish a checkpoint revision.

Full-precision aggregates and SHA-256 hashes of the checked inputs and evaluator
are recorded in `source_verification.json`.

## Reproduce from the repository root

```text
python results/statistical_uncertainty_csfcube_v2_20261020/verify_sources.py
python -m unittest tests.test_paired_statistical_uncertainty
python -m tools.paired_statistical_uncertainty --config configs/paired_statistical_uncertainty_csfcube_v2_20261020.json --workspace-root . --output-dir results/statistical_uncertainty_csfcube_v2_20261020
```

The analysis retains the existing settings: 10,000 paired percentile-bootstrap
replicates, 95% intervals, 10,000 two-sided paired sign-flip permutations, master
seed 20260815, comparison-specific derived seeds, and parent-paper clustering.
Holm adjustment includes all four base–FT comparisons separately for each
metric. All families are complete; `missing_inputs.json` is empty. The 10
existing statistical implementation tests passed.

The focused configuration also reruns the existing Gemma-versus-ModernBERT-base
comparison because it used the same outdated FT input. This is distinct from
the paper's GTE-ModernBERT comparison, which uses Llama as its generative
comparator. No GTE results were substituted or changed.

## Corrected Table 25 entries

Every number below comes from `paired_uncertainty.csv` (also JSON):

| Model | Metric | Base | FT | Delta | 95% CI | p | Holm p |
|---|---|---:|---:|---:|---|---:|---:|
| Gemma 3 27B | R@20 | 0.387 | 0.388 | 0.001 | [-0.059, 0.061] | .9840 | 1.0000 |
| Gemma 3 27B | nDCG%20 | 0.448 | 0.425 | -0.023 | [-0.069, 0.023] | .3315 | .6629 |

The full-precision means are R@20 base 0.38686794469147406, FT
0.38759608691961633; nDCG%20 base 0.44786999721758297, FT
0.4246071546657621. These are direct pooled CSV means, not estimates inferred
from rounded dev/test logs. Deltas are calculated before rounding.

Qwen 2.5 14B's nDCG%20 Holm p-value changes from .6149 to .6629 because the
Gemma p-value changed within the same family. Its raw p-value and other
statistics are unchanged. All other Table 25 model rows are unchanged. Neither
Gemma metric is statistically significant, so the surrounding interpretation
remains valid. Tables 24 and 26 require no numerical changes.

## File changes and provenance

- `HRIA/articol_latex.tex`: corrected the two Gemma rows and the Qwen nDCG Holm
  value in `tab:csfcube-uncertainty`; added LaTeX comments identifying sources.
- `configs/paired_statistical_uncertainty.json`: replaced the old base path once
  and old FT path twice. Future full analyses will use the matching reruns.
- `configs/paired_statistical_uncertainty_csfcube_v2_20261020.json`: complete
  affected families only; copied verbatim to `analysis_manifest.json` by the tool.
- `config_before_correction.json` and `articol_latex_before.tex` preserve the
  pre-edit files. Historical statistical output directories remain unchanged;
  use this dated directory for the corrected affected comparisons.
- `HRIA/csfcube_tables_24_26_audit_2026-10-19.md`: added a dated resolution note.

The original weighting explanation in Table 25 is now consistent for Gemma:
pooled case means differ from the released equal-fold test averages. The old
non-V2 point estimates and intervals must not be reused for this comparison.
