# Notes on the retained resources

## Scope and paths

The collection preserves original code, configurations and saved results. It is
not a newly implemented reproduction framework. `resources/` corresponds to the
original workspace root for scripts with paths such as `LitSearch/...`,
`CSFCube/...`, `New folder/...` and `tools/...`.

Original notebook outputs and execution counts were cleared. Standalone result
files were retained. Larger JSONL files were gzip-compressed without changing the
uncompressed bytes. `resource_inventory.csv` records both source and included
hashes. Original source files were not edited.

Server paths and old execution instructions inside retained notebooks/READMEs
are historical context. Configure those paths if running the files elsewhere.
For example, some figure scripts default to an old `HRIA/LitSearch/results`
location; the actual collected main results are at `resources/LitSearch/results`.
Use their path arguments or update a local configuration as needed.

## Avoid mixing experiments

- Primary, reconstructed, grouped-HPO and tie-study checkpoints are different
  experiments. Saved outputs do not imply their adapters are interchangeable.
- The completed grouped-HPO results concern Llama. The bundle also contains a
  four-model campaign implementation; its presence does not establish execution.
- The HPO results note is sourced from author-reported terminal output. It is not
  a replacement for the remote trial/refit result files listed in AUTHOR_TODO.
- `resources/HRIA/LitSearch_original` is an upstream-style historical snapshot.
  Its prompt differs from the customized reranker in `resources/LitSearch`.
  Do not assume that directory name identifies the executed server version.
- The supplemental HPO evaluation bundle contains proposed evaluation tooling;
  the author's reported run used the original LitSearch pipeline and no-facet
  CSFCube notebook. Inclusion does not imply every bundled entry point was run.
- Keep the adapted GTE-ModernBERT study separate from the earlier plain ModernBERT
  control. Both are retained because data preparation and statistical configurations
  refer to them.
- The original no-facet CSFCube notebook and other retained notebook variants
  must not be silently substituted for one another.

## Statistics

`results/statistical_uncertainty_corrected` is the earlier complete analysis.
For affected CSFCube comparisons use `results/statistical_uncertainty_csfcube_v2_20261020`.
The latter contains the full four-model base/FT family and an earlier plain
ModernBERT comparison, not the entire paper's statistical analysis.
The full current configuration is `configs/paired_statistical_uncertainty.json`.
Gemma precision-control statistics have their own configuration/output folder.
The four saved GTE comparison families are in `results/statistical_uncertainty_gte`,
added on 2026-10-28 after finding their omission from the initial collection.

CSFCube's equal-fold aggregates and pooled paired case means differ by design.
The paired analysis clusters cases by query paper. Holm corrections require the
complete comparison family; replacing a single row's unadjusted p-value is not
sufficient. The retained corrected files preserve that family calculation.

## Data and configuration discrepancies

The standalone extracted preparation script at the repository root defaults to
100 candidates, whereas the primary YAML specifies 50, and it omits the parent
ID from its final rows. Its sampling behavior is not identical to the historical
notebook. It was retained as an existing utility, not used to regenerate the
packaged historical data.

The frozen compact/HPO cohort has 4,494 lists: 4,312 of length 50 and 182 of
length 51, with 1,392 lists above five positives (HPO resource manifest). Preserve
these historical properties rather than describing a strict cap that the artifact
does not satisfy. This cohort does not prove historical checkpoint input identity.

The retained Gemma 12B notebook has batch 1/accumulation 2, while the article/YAML
specify 2/1. Some training notebooks use Unsloth repositories whereas the article
names Google/Qwen models. These require producing-run metadata; packaging did
not silently reconcile them or change the paper.

## Tie backup extraction

Final evaluations and analysis were recovered from
`backups/tie_final_primary1024_backup_20261006/report_and_resources.tar.gz`.
Only final run records, evaluation inputs and final three-seed analysis were
collected; smoke/pilot outputs and intermediate weights were excluded. The
included budget-audit script expects the original backup archives, which are not
part of Git. Its retained report is supplied; packaging did not rerun the audit.
The frozen training list is provided once with the compact-study resources;
its hash matches the retained tie data manifest.
