# Tie-sensitivity experiment: primary results and token-budget audit

Prepared on 2026-10-06 from the verified local backup
`backups/tie_final_primary1024_backup_20261006`.

## Scope and completed experiment

Six training runs compare deterministic within-grade tie ordering with
epoch-randomized within-grade ordering, using seeds 42, 2026, and 3407.
The deterministic condition uses presented input-list position to break ties.
This is a tie-ordering sensitivity experiment. It does not implement a distinct
tie-aware loss, and it does not identify a percentage of learned signal that is
attributable to graded relevance rather than ordering.

All six models were evaluated on the same 597 LitSearch queries for BM25,
GRIT, and Qwen3 candidate lists: 100 candidates per query and 10,746 saved
responses in total. Training used 50/51-candidate DORIS-MAE lists, so list-length
transfer is an explicit limitation of this evaluation.

The recorded recipe uses `meta-llama/Llama-3.1-8B-Instruct`, revision
`0e9e39f249a16976918f6564b8830bc894c89659`, unquantized BF16 LoRA with rank 128,
alpha 256, dropout 0.05, learning rate 1e-5, and batch size 2. Two logical epochs
are represented as one sequential Trainer pass of 8,988 presentations and 4,494
optimizer steps. Trainer's reported epoch value of 1 is therefore expected.

Inference used temperature 0, top-p 1, generation seed 0, and a 1,024-token
output budget. The frozen parser keeps the first occurrence of each valid ID,
discards repeated or invalid IDs, and appends missing candidates in their input
order. Quality metrics refer to these repaired rankings. Exact-permutation and
other compliance measures describe the raw responses separately.

## Primary quality results

Values below are metric scores multiplied by 100, shown as mean +/- sample
standard deviation across the three training seeds. The source is
`analysis_final_three_seeds/mean_sd.csv` inside the backup.

| Retriever | Tie policy | R@5 | R@20 | nDCG@20 |
|---|---|---:|---:|---:|
| BM25 | Deterministic | 54.50 +/- 0.50 | 67.21 +/- 0.70 | 49.89 +/- 0.30 |
| BM25 | Epoch-randomized | 56.17 +/- 0.25 | 67.21 +/- 0.22 | 47.74 +/- 1.26 |
| GRIT | Deterministic | 69.78 +/- 0.05 | 82.57 +/- 0.80 | 64.79 +/- 0.16 |
| GRIT | Epoch-randomized | 60.98 +/- 1.86 | 80.52 +/- 0.52 | 53.33 +/- 2.10 |
| Qwen3 | Deterministic | 69.92 +/- 0.24 | 80.94 +/- 0.73 | 64.84 +/- 0.24 |
| Qwen3 | Epoch-randomized | 62.63 +/- 2.06 | 79.37 +/- 0.52 | 53.26 +/- 1.93 |

Paired differences below are epoch-randomized minus deterministic, averaged
across retrievers and seeds. Units are percentage points.

| Metric | Difference | 95% query-bootstrap CI | Holm-adjusted p |
|---|---:|---:|---:|
| R@5 | -4.806 | [-6.669, -3.029] | 0.0004 |
| R@20 | -1.207 | [-2.411, -0.038] | 0.1404 |
| nDCG@20 | -8.396 | [-9.911, -6.904] | 0.0004 |

The analysis averages each query's paired difference across seeds before
bootstrapping queries or applying sign-flip tests (10,000 replicates). It does
not treat the three evaluations of a query as independent queries. Holm
correction is applied within each metric across the three retriever-specific
comparisons and the retriever average. The confidence intervals are unadjusted;
therefore the R@20 interval excluding zero does not contradict its adjusted
p-value exceeding 0.05. This is not evidence of equivalence. The query-based
uncertainty conditions on these trained checkpoints; seed standard deviations
are reported separately.

## Output compliance

Across equally sized retriever/seed groups, exact-permutation rates average
52.67% for deterministic targets and 0.242% for randomized targets. Mean missing
candidate counts are 1.42 and 32.25, respectively. All full runs have zero
complete-fallback rate, but this does not imply valid raw permutations: missing
IDs can still be appended by the parser.

Randomized-target models also differ considerably by seed. Seed 2026 often
stops after around 50/51 distinct IDs. Seed 3407 often repeats IDs until the
output limit. Both behaviors should be reported; low training loss or valid
surface formatting alone does not establish ranking quality or completeness.

## Does the 1,024-token limit explain the quality gap?

The saved raw outputs allow a direct audit without another inference run.
`audit_budget.py` independently verifies the two downloaded archive hashes,
training-manifest and evaluation-protocol hashes, all 18 result-file hashes,
query counts, and the reconstruction of rankings from first-occurrence IDs.

Observed counts:

| Tie policy | Seed | Responses | Finished at token limit |
|---|---:|---:|---:|
| Deterministic | 42 | 1,791 | 37 |
| Deterministic | 2026 | 1,791 | 15 |
| Deterministic | 3407 | 1,791 | 24 |
| Epoch-randomized | 42 | 1,791 | 277 |
| Epoch-randomized | 2026 | 1,791 | 0 |
| Epoch-randomized | 3407 | 1,791 | 996 |
| Total | | 10,746 | 1,349 |

Of the 1,349 truncated responses, 1,348 already contain at least 20 distinct
valid IDs. Appending more text to those same responses cannot change the first
20 distinct valid IDs under the frozen parser, so R@5, R@20, and nDCG@20 cannot
change through that mechanism.

The sole shorter truncated response is GRIT, epoch_random_seed42, query
`ba01d24256902be6c6b0c142358f7d87b3dca7919f715c1bf78c20741309aa75`.
It emits IDs `[1], [17], [39], [94]` before repetition. Its sole relevant
document is already first; its saved R@5, R@20, and nDCG@20 are all 1.0.
Appending further IDs cannot alter that relevant document's first position.

Consequently, extending the recorded truncated responses while preserving
their prefixes would leave all three reported quality metrics unchanged.
Responses that stopped before the cap are not directly extended merely by
raising the maximum token allowance.

This is an audit of recorded prefixes and parser behavior, not a completed
2,048-token experiment. A fresh serving run can produce different prefixes
because of numerical or execution differences, and a larger budget can change
coverage, duplicate counts, stop behavior, latency, and token consumption.
Those remain valid subjects for an optional additional generation-behavior
experiment. The audit supports finishing the primary report without requiring
six more full evaluations solely to test whether the existing outputs were
cut off before enough IDs were available for the measured top-k metrics.

## Reproducibility and preservation

Training used Transformers 5.5.0; the serving environment used vLLM 0.19.0 with
Transformers 4.57.6. A validated `tokenizer_v4` sidecar was shared across runs.
Serving commands used `--hf-overrides '{"rope_theta": 500000.0}'` to preserve
the intended Llama 3.1 rotary-position configuration across the version boundary.
These deployment details come from the recorded workflow and server logs;
the evaluation protocol does not itself capture every serving flag.

The report/resources archive contains scripts, frozen inputs, six-run results,
manifests, logs, and final analysis tables, excluding large model weights.
The final adapter archive in this backup contains only epoch_random_seed3407;
the earlier verified backup directories preserve the other five final adapters.
Keep all of those local backups. Deleted merged checkpoints can be regenerated
from the pinned base and retained adapters. Deleted optimizer states were not
included in the report backup and are not recoverable from it.

To reproduce the budget audit from the repository root in PowerShell:

```powershell
python .\tie_sensitivity_report_20261006\audit_budget.py `
  .\backups\tie_final_primary1024_backup_20261006
```

The script reads archives without extracting or modifying their contents and
prints detailed per-run/retriever counts as JSON.

## Suggested report conclusion

Under the matched three-seed protocol, epoch-wise randomization of within-grade
target order reduced average R@5 and nDCG@20 relative to deterministic
input-position tie breaking, while the R@20 difference was inconclusive after
multiplicity correction. Randomization also substantially reduced exact-list
compliance. These results demonstrate sensitivity to target serialization under
the tested DORIS-MAE-to-LitSearch transfer and repair protocol; they do not
separate semantic relevance learning from list-length generalization and
generation compliance. An audit of truncated response prefixes shows that
merely extending those recorded responses cannot improve the measured top-20
quality scores. A distinct tie-aware objective remains a separate experiment.
