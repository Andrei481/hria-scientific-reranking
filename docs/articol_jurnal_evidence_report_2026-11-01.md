# Evidence report for `articol_jurnal.tex`

Audit date: 2026-11-01. Line numbers refer to the current [manuscript](../articol_jurnal.tex). Its content hash and numerical check results are recorded in [the check output](manuscript_evidence_checks_2026-11-01.json).

## Answer

**Not every claim can be independently substantiated from the repository.** The main experimental evidence gap is the parent-grouped hyperparameter study: 24 reported outcome values and its selection/completion claims have an author-reported note, but no retained original experimental outputs. External comparison numbers also lack a locally checkable primary numerical source. Historical training configuration and input identity remain incompletely established.

This does **not** mean these numbers are fabricated or have literally no documentary support. The grouped-HPO note is evidence of what the author reported. It is not independent evidence that the experiments produced those results. Accordingly, this report separates absent experimental evidence, external-source gaps, provenance conflicts, and small numerical discrepancies.

Most major result families have retained rankings, metric files, statistical outputs, or measurement summaries. The checks below found no mismatches in the tested primary result tables. Two minor learning-curve differences and one overlap-count interpretation need correction or clarification.

## 1. Results lacking original experimental evidence

All four groups below rely on [grouped_hpo_reported_evidence.md](../resources/HRIA/grouped_hpo_reported_evidence.md), whose opening explicitly says results were supplied in a conversation and were not independently read from remote artifacts. The [local split audit](../resources/New%20folder/parent_grouped_hpo/local_validation/data_and_splits.json) identifies itself as **“CPU data and split audit; no trained model results.”** The retained study directory contains code, configuration, frozen data and that split audit, but no completed trial/refit/evaluation outputs. [AUTHOR_TODO.md](../AUTHOR_TODO.md) independently documents the missing server files.

| ID | Location and exact claim | Missing evidence and action |
|---|---|---|
| H1 | Lines 709–734, `tab:grouped-hpo-selection`: validation mean/SD **0.070711 / 0.009043**, **0.072112 / 0.009905**, **0.074852 / 0.008772** for learning rates 5e-6, 1e-5 and 2e-5. Also: 5e-6 wins **each of three splits**. | No nine-trial validation records or saved selection result. Supply trial manifests, validation losses and the output of selection/analysis. The configured search grid establishes the intended experiment, not its outcome. |
| H2 | Lines 737–741: internal-test NLLs **0.076731, 0.076040, 0.076456** for seeds 42, 2026 and 3407; final refit completed on **4,494 lists for two epochs**. | No saved internal-test results or refit completion/training manifest. List count and planned epochs have data/configuration support; actual completion is only reported in the note. Supply `artifacts/tests/grouped/` and refit records. |
| H3 | Lines 743–759, `tab:grouped-hpo-litsearch`: BM25 **66.31 / 55.11 / 50.03 / 46.66**; GRIT **80.88 / 69.13 / 64.23 / 60.71**; Qwen3 **81.09 / 69.21 / 64.40 / 60.84**, ordered R@20 / R@5 / nDCG@20 / nDCG@5. | No original rankings or metric JSONs for this refit. Supply `results/Llama31_8B_parent_grouped_lr5e6/`, especially the top-100 evaluations over 597 queries. Outputs from primary Llama or tie-policy checkpoints cannot substitute. |
| H4 | Lines 762–764: grouped-refit CSFCube test **R@20 40.83%, nDCG@20 43.51%, nDCG%@20 43.20%**. | No original ranked pools, per-query records or evaluator logs. Supply `runs_hpo_grouped_lr5e6_vllm_full_list_4096/`. |

**Recommended treatment:** recover these small result files, or explicitly label the results as author-reported with original artifacts unavailable. If the paper requires independently auditable evidence for every experiment, withhold these outcome claims until the records are supplied. Missing model weights alone would not require removing scores if their evaluated rankings were retained.

## 2. External numerical comparisons without a locally checkable primary source

| ID | Location and values | Repository support and remaining gap |
|---|---|---|
| E1 | Lines 1163–1166, `tab:csfcube-external-baselines`: fabs_tfidfcbow200 **0.231 / 0.314**; UnSimCSE-BERT **0.344 / 0.426**; SPECTER-ID **0.521 / 0.572**, ordered R@20 / nDCG%@20. | They are described as published baselines, not local experiments. No matching original numerical table, evaluation output or exact table/page attribution was located in the inspected resources. The retained CSFCube README describes evaluation but does not substantiate these three numerical pairs. Add precise source/table/version references and the facet/fold aggregation calculation. This also affects the comparative prose at lines 1146–1151. |
| E2 | Lines 1304–1311: GPT-4o + GritLM Specific R@20 **85.9%**, and Gemma's differences **−2.3 pp Broad R@20** and **−3.4 pp Specific R@5**. | [generate_graphs.py](../resources/HRIA/generate_graphs.py), lines 90–92, hard-codes the external tuple **75.3, 79.2, 85.9** and attributes it to the LitSearch paper. This is a transcription, not a retained primary-source table or evaluation record. Add an exact published table/page reference. The Gemma side has local metric evidence; the external reference was not rerun, as the manuscript correctly states. |

These are **local verification gaps**, not findings that the external results are wrong. External literature was not independently audited for this repository-only report. A properly identified published source can support a literature comparison without local model predictions.

## 3. Numerical and provenance discrepancies

### P1. Gemma 12B batch/accumulation settings are unresolved

Lines 459 and 464 claim batch **2**, accumulation **1**, effective batch **2** for primary Gemma 3 12B. The retained [train_gemma3_12b.ipynb](../resources/DorisMAE/train_gemma3_12b.ipynb), source lines 554–555, instead specifies batch **1**, accumulation **2**. [models.yaml](../configs/training/models.yaml) agrees with the manuscript, so this is conflicting evidence rather than a number absent everywhere. Both settings give effective batch 2, but the exact execution claim cannot be resolved without the producing run's manifest or Trainer configuration.

The same notebook loads `unsloth/gemma-3-12b-it`, whereas the manuscript names `google/gemma-3-12b-it`. The misleadingly named `resources/HRIA/train_unsloth_12b_20251229_original.ipynb` actually loads a 27B model; its batch 2/accumulation 1 settings cannot establish the primary 12B configuration. Repository notes already acknowledge incomplete model/tokenizer mappings.

### P2. Historical training-set identity is not established

Lines 151–155, 209–212, 353–356 and 439–465 associate the 4,494-list recipe with historical primary checkpoints. The repository supports the recipe and later frozen datasets, but does not retain an exact formatted input/hash-to-checkpoint mapping for every original run. The compact/HPO artifact exposes **8,477** document IDs; the reconstructed LLM preparation exposes **8,480**. The paper correctly acknowledges that distinction at lines 483–487 and 860–862.

Do not infer historical input identity from later data artifacts. Supply original training-file hashes and run-to-checkpoint mappings, or retain an explicit provenance limitation. This gap does not invalidate metric values backed by saved predictions.

### P3. LitSearch overlap prose confuses matched pairs with unique DORIS documents

Lines 400–403 say that, of 8,480 DORIS documents, **227** match LitSearch, with **202** matches among positive DORIS documents. The [prompt-level overlap summary](../resources/leakage_analysis_outputs/prompt_level_overlap_summary.csv) distinguishes:

| Population | Matched pairs | Unique DORIS documents (`text_hash_left_docs`) | Unique LitSearch documents (`text_hash_right_docs`) |
|---|---:|---:|---:|
| All prompt candidates | 227 | **226** | 227 |
| Positive prompt candidates | 202 | **201** | 202 |

Thus 227/202 are supported as pair counts or matched LitSearch-document counts, **not as counts of distinct DORIS documents**. Rewrite the prose accordingly, or use 226/201 for the DORIS side. Table `tab:leakage-overlap` should define “Matches.” The strict-filter blacklist of 227 LitSearch documents remains consistent with this evidence.

### P4. Two learning-curve deltas use rounded operands

Recomputed all **45** top-100 ranking files: five budgets × three seeds × three retrievers, each containing 597 queries. Following the manuscript's averaging order:

| Claim | Paper, lines 692–695 | Difference from full-precision results | Rounded result |
|---|---:|---:|---:|
| R@20: 100 parents minus 25 | 0.45 pp | 0.4404739748 pp | **0.44 pp** |
| R@5: 25 parents minus 100 | 1.02 pp | 1.0267386314 pp | **1.03 pp** |

The printed numbers can be obtained by subtracting already-rounded means. They are minor precision issues, not unsupported experiments. The 1.03-point R@20 difference against ten parents and 0.70-point nDCG difference are consistent. The highest R@20 at 100 parents and highest R@5/nDCG@20 at 25 parents are confirmed.

## 4. Missing supplement and limits on verification

The current manuscript repeatedly directs the reader to **Online Resource 1** (including Tables S2, S4, S8, S10 and S12). No standalone supplementary manuscript was located in this checkout; `articol_jurnal.tex` is the only `.tex` file found. Supporting code, figures and experiment outputs are often present, but they are not the referenced supplementary document. Include it so the additional numerical tables, protocols and literature attributions are accessible.

The older [2026-10-28 audit](numerical_evidence_audit_2026-10-28.md) concerns `HRIA/articol_latex.tex`. Its unsupported FIRST scores **70.8 / 35.8 / 35.0** are **not present in the current `articol_jurnal.tex`** and are therefore not counted as an unsupported current-paper result. Prompt-repetition rows and top-rank examples are likewise no longer printed here. Audit them separately if they appear in the missing supplement.

Large original DORIS corpus/annotation files are omitted, with hashes recorded in [omitted_large_resources.json](omitted_large_resources.json). Consequently, the full **363,133-document corpus / 2011–2021** description and original-annotation audits cannot be fully reconstructed offline from the packaged subset alone. They have dataset attribution or retained audit summaries; missing raw files are not equivalent to no evidence. The missing-relevance audit explicitly supports **1,501 / 1,498 rows**, **163,969 judged pairs**, and **88,285 explicit-zero pairs**.

## 5. Checks performed and evidence that should not be flagged as absent

The reproducible [audit script](../scripts/audit_manuscript_evidence.py) runs with Python's standard library:

```text
python scripts/audit_manuscript_evidence.py
```

The final run performed **130 grouped checks**, with one failing group containing the two rounding discrepancies above. It inspected 227 metric JSON files while locating disjoint-result evidence. Specifically:

- All **72 primary LitSearch metric cells** match saved metric JSONs at printed precision.
- All **120 primary LitSearch uncertainty fields** match retained deltas, interval bounds and raw/Holm p-values.
- All **56 CSFCube uncertainty fields** match the corrected V2 statistical output.
- All **36 tie-policy mean/SD fields** match the final three-seed summary. Its raw validity summary also supports 52.67% versus 0.24% exact permutations and 1.42 versus 32.25 missing IDs.
- All **18 compact LitSearch metric cells** match saved metric JSONs.
- All **27 strict document-disjoint table cells** match recomputation from six ranking files over 595 queries, including the printed deltas. The actual results are under `resources/New folder/results_vm_250/new/disjoint/`; the VM139 strict-rerun directory holds filtered inputs.
- The compact compute JSON contains the reported latency, throughput, GPU-memory, parameter-count and validity measurements. The measured time ratio supports approximately **41×**.
- The learning-curve conclusion was independently recomputed from rankings, as described above.

Other families have identifiable retained evidence, including ablations, matched controls, seed/depth experiments, precision controls, compact CSFCube evaluation, local open baselines, overlap analyses and tie-budget diagnostics; see [resources.md](../resources.md). Their presence was inspected, but this report does not claim a fresh cell-by-cell recomputation of every number in every family. Statistical resampling, model training/inference and external publications were not rerun or re-audited. Bibliographic years, page ranges, equation constants and illustrative index sequences were not treated as experimental outcomes.

The manuscript was not edited. The report and verification script/output are new audit artifacts.
