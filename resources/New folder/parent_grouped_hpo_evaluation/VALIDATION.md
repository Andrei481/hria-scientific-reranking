# Validation record

Validated locally on 2026-10-10 in the Windows repository. These checks use
fixtures and released evaluation resources; they are not new model results.

Command:

```text
python -B -m unittest discover -s "New folder/parent_grouped_hpo_evaluation" -v
Ran 11 tests
OK
```

The checks cover:

- Ranking repair: first valid occurrence, invalid/duplicate IDs, original-order
  completion, and the no-valid-ID fallback.
- LitSearch metric denominators using all gold documents, including documents
  absent from the candidate pool.
- Original no-facet prompt formatting, including list-form abstracts, versus
  the distinct optional facet prompt.
- Exact prompt equality with the retained original notebook's extracted builder.
- All 50 released CSFCube cases and their complete candidate pools.
- Rejection of modified adapter files, mismatched saved queries, and rankings
  inconsistent with raw responses.
- Pinned serving revision, saved tokenizer, and rank-128 adapter command.
- Real local HTTP requests with explicit model selection, output retention,
  prefix resume, rejection of an unrelated server's generic aliases, and a clear
  error for an incomplete final JSONL record.
- Full evaluator orchestration with synthetic responses: smoke and full outputs
  remain separate, smoke produces no official ranking files, full completion
  requires all six pools, and rerunning performs no duplicate requests.
- Original CSFCube scorer execution over all released pools with fixture rankings.
  With local NumPy 1.26.4, the report is identical when `asfarray` is available
  and when its absence is simulated and the compatibility conversion is used.
  The compatibility operation is removed again after scoring.

All four CLI entry points (`prepare.py`, `serve.py`, `evaluate.py`, and
`score_csfcube.py`) also passed their `--help` invocation.

An independent code review checked prompt parity against the original notebook
for all 50 real CSFCube cases (candidate pool sizes 88–253), official scorer
ordering/folds, refit validation, and the serving/resume path. Its important
finding about shared server aliases was addressed with cohort-specific aliases
and an occupied-port check; the follow-up review found no remaining blocking or
important issue.

No original `parent_grouped_hpo` Python files were changed for this companion.
No GPU model was loaded locally. Actual server-side refit verification, tokenizer
preflight, cached-model loading, H200 memory use, and model generation remain to
be exercised by README steps 3–6. The NumPy compatibility test exercises the
removed API path; it is not a complete test of every NumPy 2 release.

The bundle intentionally contains no generated benchmark scores. The existing
frozen LitSearch inputs, CSFCube checkout, original HPO bundle and artifacts, and
cached base model stay on the server and are verified/reused in place.
