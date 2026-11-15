# Models and data

These links are taken from retained local resources or the author's explicit
checkpoint identification. No new model upload was performed. Historical model
revisions are recorded only where available; remote availability was not retested
during this packaging step.

## Datasets and prepared supervision

| Resource | Included files or source |
|---|---|
| DORIS-MAE upstream dataset | [Official project](https://github.com/kwang927/Doris-Mae-Dataset); use its dataset download link. The old `Real-Doris-Mae` URL redirects there. |
| Query and aspect metadata | [DorisMAE](../resources/DorisMAE/) |
| Prepared aspect-level rows | [HRIA/reranking_dataset_doris_mae.jsonl](../resources/HRIA/reranking_dataset_doris_mae.jsonl) |
| Frozen listwise supervision used by compact studies | [dorismae_listwise_4494.jsonl.gz](../resources/New%20folder/modernbert_supervised_baseline/modernbert_supervised_baseline/dorismae_listwise_4494.jsonl.gz), [preparation manifest](../resources/New%20folder/modernbert_supervised_baseline/modernbert_supervised_baseline/preparation_manifest.json) |
| Grouped-HPO documents, lists and parent mappings | [HPO resources](../resources/New%20folder/parent_grouped_hpo/resources/) |
| LitSearch queries/corpus | [princeton-nlp/LitSearch](https://huggingface.co/datasets/princeton-nlp/LitSearch); [saved retrieval lists](../resources/LitSearch/results/retrieval/) are included |
| CSFCube gold, queries and abstracts | [Included collection](../resources/CSFCube/); [upstream project](https://github.com/iesl/CSFCube) |
| Tie-study frozen inputs | [Evaluation inputs](../resources/tie_sensitivity_artifacts/evaluation_inputs/); large JSONL inputs are gzip-compressed |

The uncompressed frozen listwise artifact has SHA-256
`7c649c816c78778777afcddc011dafd345a6847d2fd6d4e082b58bad75e5bc4f`.
It is included as a gzip copy, not regenerated. The separate HPO list/document
representation is also included. Neither establishes the exact input file used
by every historical primary model.

The full DORIS corpus, annotation JSON and complete dataset JSON are available in
the author's workspace but omitted from Git because of size. Their local hashes
are recorded in [omitted_large_resources.json](omitted_large_resources.json).
The compact preparation manifest refers to server input hashes that differ from
some local files; those exact server inputs must not be silently replaced.
See [AUTHOR_TODO](../AUTHOR_TODO.md).

## Checkpoints with known references

| Model / role | Reference | Evidence and limits |
|---|---|---|
| Gemma 3 27B FT V2 evaluated on CSFCube | [Andrei48155/20260218-181123](https://huggingface.co/Andrei48155/20260218-181123) | Identified by the author and the reported serving model ID. Exact evaluated repository commit was not recovered locally. This does not identify all historical Gemma runs. |
| Base Llama for grouped HPO and tie sensitivity | [meta-llama/Llama-3.1-8B-Instruct, pinned revision](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/tree/0e9e39f249a16976918f6564b8830bc894c89659) | Recorded in HPO reported evidence and tie run manifests. |
| Base GTE compact reranker | [Alibaba-NLP/gte-reranker-modernbert-base, pinned revision](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base/tree/204bdd4d405882a9e8ca9452c223fc49f02e8451) | Recorded in compact training configuration/manifests. |
| Other base models | [google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it), [google/gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it), [Qwen/Qwen2.5-14B-Instruct](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) | Model-family links, not proof of the exact historical revision/repack. Some retained training notebooks use Unsloth repositories. |

## Tie-sensitivity checkpoints

The following links and commits come from the retained
[publication receipts](../resources/backups/tie_huggingface_publication_20261007_233631/receipts/).
The receipts record public download/hash checks on 2026-10-07; this collection
preserves that evidence rather than claiming a new remote check.

| Run | Checkpoint | Commit |
|---|---|---|
| deterministic_seed2026 | [Andrei48155/Llama-3.1-8B-DORIS-tie-deterministic-seed2026](https://huggingface.co/Andrei48155/Llama-3.1-8B-DORIS-tie-deterministic-seed2026/tree/b5508f87a5bcad546c9e147be8639898024a6ff1) | `b5508f87a5bcad546c9e147be8639898024a6ff1` |
| deterministic_seed3407 | [Andrei48155/Llama-3.1-8B-DORIS-tie-deterministic-seed3407](https://huggingface.co/Andrei48155/Llama-3.1-8B-DORIS-tie-deterministic-seed3407/tree/455642ae41b89a271dd91f427fd70b894420d3f8) | `455642ae41b89a271dd91f427fd70b894420d3f8` |
| deterministic_seed42 | [Andrei48155/Llama-3.1-8B-DORIS-tie-deterministic-seed42](https://huggingface.co/Andrei48155/Llama-3.1-8B-DORIS-tie-deterministic-seed42/tree/c84172fcf31b61ef8d942ece91ed5c4ce094730d) | `c84172fcf31b61ef8d942ece91ed5c4ce094730d` |
| epoch_random_seed2026 | [Andrei48155/Llama-3.1-8B-DORIS-tie-epoch-random-seed2026](https://huggingface.co/Andrei48155/Llama-3.1-8B-DORIS-tie-epoch-random-seed2026/tree/da03f5e2ae0032fbb367c2e478bb61ae5ccd5fa8) | `da03f5e2ae0032fbb367c2e478bb61ae5ccd5fa8` |
| epoch_random_seed3407 | [Andrei48155/Llama-3.1-8B-DORIS-tie-epoch-random-seed3407](https://huggingface.co/Andrei48155/Llama-3.1-8B-DORIS-tie-epoch-random-seed3407/tree/5c13c80ee5971654afe34f335f966987209b2b9b) | `5c13c80ee5971654afe34f335f966987209b2b9b` |
| epoch_random_seed42 | [Andrei48155/Llama-3.1-8B-DORIS-tie-epoch-random-seed42](https://huggingface.co/Andrei48155/Llama-3.1-8B-DORIS-tie-epoch-random-seed42/tree/67e37fd987e5b2661f898ee24b80e5bf7112fe52) | `67e37fd987e5b2661f898ee24b80e5bf7112fe52` |

## Weights that still need public links

The adapted GTE checkpoint exists locally at
`New folder/ranking_pretrained_compact_baseline/ranking_pretrained_compact_baseline/models/gte_modernbert_dorismae_lambda_seed3407/final/`.
Its manifest/configuration are included, but weights are not copied into Git.
The earlier plain ModernBERT checkpoint is also a separate local model.

Public checkpoint mappings for the primary/reconstructed models, grouped-HPO
refit, parent-budget and ablation runs are not established by the current local
resource collection. See the explicit list in [AUTHOR_TODO](../AUTHOR_TODO.md).
