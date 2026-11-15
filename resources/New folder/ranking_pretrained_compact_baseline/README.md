# Ranking-pretrained compact reranker: exact runbook

This bundle fine-tunes `Alibaba-NLP/gte-reranker-modernbert-base`, a
149M-parameter ranking-pretrained cross-encoder, on the exact frozen 4,494
DORIS-MAE-derived lists used by the existing generic ModernBERT control. The
checkpoint is pinned to revision
`204bdd4d405882a9e8ca9452c223fc49f02e8451`.

The only controlled training difference from the generic ModernBERT baseline
is initialization. LambdaLoss, list order and labels, two epochs, learning rate,
batching, seed, maximum length, precision, and quantization configuration are
unchanged. The model has prior ranking supervision, so the paper must say
"the same DORIS-MAE adaptation supervision," not "the same total lifetime
supervision."

## Step 1 — Create or reuse the GPU environment

From the repository root on the H200 server:

```bash
conda create -n gte_modernbert_baseline python=3.11 -y
conda activate gte_modernbert_baseline
python -m pip install --upgrade pip
python -m pip install -r "New folder/ranking_pretrained_compact_baseline/requirements-gte-modernbert.txt"
python -m ipykernel install --user --name gte_modernbert_baseline --display-name "Python 3 (modernbert)"
jupyter lab
```

If the already-created `modernbert_baseline` environment still has the exact
versions in the requirement file, it can be reused. Do not upgrade packages
between the generic and ranking-pretrained runs.

## Step 2 — Point to and verify the frozen supervision

Do not run the preparation notebook again. Reuse the exact artifact from the
generic ModernBERT experiment:

```bash
export DORISMAE_LISTWISE_PATH=/DATA/j_andrei/modernbert_supervised_baseline/dorismae_listwise_4494.jsonl
export RANKING_PRETRAINED_COMPACT_ARTIFACT_DIR=/DATA/j_andrei/ranking_pretrained_compact_baseline
sha256sum "$DORISMAE_LISTWISE_PATH"
```

The required SHA-256 is:

```text
7c649c816c78778777afcddc011dafd345a6847d2fd6d4e082b58bad75e5bc4f
```

Notebook 01 checks the same hash, 4,494 examples, the 4,312/182 list-size
histogram, and the complete 0/1/2 label set. Stop if any assertion fails.

## Step 3 — Train the ranking-pretrained compact model

Open `01_train_gte_modernbert.ipynb`, select the `Python 3 (modernbert)`
kernel, and run all cells once.

The fixed run is:

- model: `Alibaba-NLP/gte-reranker-modernbert-base`;
- revision: `204bdd4d405882a9e8ca9452c223fc49f02e8451`;
- seed: 3407;
- 4,494 frozen listwise examples;
- LambdaLoss with default NDCGLoss2++ weighting and `k=20`;
- two epochs, learning rate `2e-5`, warmup ratio `0.06`;
- list batch size 1, gradient accumulation 8, document mini-batch size 8;
- BF16, SDPA attention, maximum pair length 1,024;
- full-parameter training and no quantization;
- no validation or benchmark-based checkpoint selection.

Retain the complete directory:

```text
/DATA/j_andrei/ranking_pretrained_compact_baseline/models/gte_modernbert_dorismae_lambda_seed3407/
```

It must contain `final/` and `training_manifest.json`. The final cell must
reload the model and print two finite smoke-test scores.

## Step 4 — Evaluate zero-shot and adapted models on all LitSearch pools

Restart the kernel to release trainer memory. Open
`02_evaluate_litsearch.ipynb` and run all cells.

The notebook first evaluates the pinned, off-the-shelf GTE reranker and then
the DORIS-MAE-adapted checkpoint. Both variants rerank exactly the existing
top 100 candidates for each of the 597 queries. It must create twelve result
files under
`/DATA/j_andrei/ranking_pretrained_compact_baseline/litsearch/`:

```text
LitSearch.title_abstract.bm25.gte_modernbert_dorismae.reranked.jsonl
LitSearch.title_abstract.bm25.gte_modernbert_dorismae.reranked.metrics.json
LitSearch.title_abstract.grit.gte_modernbert_dorismae.reranked.jsonl
LitSearch.title_abstract.grit.gte_modernbert_dorismae.reranked.metrics.json
LitSearch.title_abstract.qwen3.gte_modernbert_dorismae.reranked.jsonl
LitSearch.title_abstract.qwen3.gte_modernbert_dorismae.reranked.metrics.json
LitSearch.title_abstract.bm25.gte_modernbert_zeroshot.reranked.jsonl
LitSearch.title_abstract.bm25.gte_modernbert_zeroshot.reranked.metrics.json
LitSearch.title_abstract.grit.gte_modernbert_zeroshot.reranked.jsonl
LitSearch.title_abstract.grit.gte_modernbert_zeroshot.reranked.metrics.json
LitSearch.title_abstract.qwen3.gte_modernbert_zeroshot.reranked.jsonl
LitSearch.title_abstract.qwen3.gte_modernbert_zeroshot.reranked.metrics.json
```

Also retain `litsearch_evaluation_manifest.json`. The assertions verify that
each output contains the exact same candidate set as its input.

## Step 5 — Evaluate CSFCube

Restart the kernel. Open `03_evaluate_csfcube.ipynb` and run all cells.

The notebook evaluates both the pinned zero-shot model and the adapted model
on background, method, result, and pooled facets. Retain the complete
directory:

```text
/DATA/j_andrei/ranking_pretrained_compact_baseline/csfcube/
```

In particular, this file is required for clustered paired uncertainty:

```text
test-pid2pool-csfcube-gte-modernbert-dorismae-lambda-seed3407-all-perq.csv
test-pid2pool-csfcube-gte-modernbert-zeroshot-all-perq.csv
```

The notebook stops if the evaluator does not create either file. It also
retains six `*-ranked.json` files, four adapted metric text files, four
zero-shot metric text files, and the evaluation manifest.

## Step 6 — Measure matched inference cost

Restart the kernel. Open `04_compute_benchmark.ipynb` and run the compact-model
cells once. It benchmarks 20 fixed LitSearch queries against the same top-100
Qwen3 candidates, with one warmup and three measured repetitions.

For the generative comparison, serve the same fine-tuned Llama 3.1 8B
checkpoint used in the quality comparison, then set:

```bash
export LLM_BASE_URL=http://127.0.0.1:8000/v1
export LLM_MODEL_ID=/absolute/path/to/the/llama-3.1-8b-ft-checkpoint
export LLM_API_KEY=EMPTY
export LLM_PEAK_VRAM_GIB=REPLACE_WITH_SERVER_PEAK
```

First run all cells with `RUN_LLM_BENCHMARK = False`, before starting the
Llama server. The compact benchmark cell releases the GTE model after saving
its measurements. Then start the Llama server, set
`RUN_LLM_BENCHMARK = True` in the live kernel, and rerun only the Llama cell
and the final manifest-writing cell. Do not rerun the compact-model cells
while the Llama server is loaded. This keeps the measurements sequential on
the same otherwise-idle GPU. Retain:

```text
/DATA/j_andrei/ranking_pretrained_compact_baseline/compute/matched_compute_benchmark.json
```

The manifest records total and trainable parameters, median query latency,
queries/s, documents/s, peak inference VRAM, the input protocol, and the
software/hardware environment. Record Llama server peak VRAM using `nvidia-smi`
or the server process; the client cannot inspect a different process's CUDA
allocator.

## Step 7 — Copy results back and regenerate paired statistics

Copy the entire server artifact directory into this exact local location:

```text
D:\Code\disertatie\New folder\ranking_pretrained_compact_baseline\ranking_pretrained_compact_baseline\
```

The resulting layout must therefore include:

```text
New folder/ranking_pretrained_compact_baseline/ranking_pretrained_compact_baseline/models/
New folder/ranking_pretrained_compact_baseline/ranking_pretrained_compact_baseline/litsearch/
New folder/ranking_pretrained_compact_baseline/ranking_pretrained_compact_baseline/csfcube/
New folder/ranking_pretrained_compact_baseline/ranking_pretrained_compact_baseline/compute/
```

The main statistical configuration contains four matched Llama-versus-GTE
quality contrasts and four adapted-versus-zero-shot contrasts.
From `D:\Code\disertatie`, run:

```bat
python -m tools.paired_statistical_uncertainty ^
    --config configs\paired_statistical_uncertainty.json ^
    --workspace-root . ^
    --output-dir results\statistical_uncertainty
```

Verify that all inputs were found:

```bat
type results\statistical_uncertainty\missing_inputs.json
```

It must print:

```text
[]
```

Finally run:

```bat
python -c "import csv; r=list(csv.DictReader(open(r'results\statistical_uncertainty\paired_uncertainty.csv',encoding='utf-8-sig'))); g=[x for x in r if 'GTE-ModernBERT' in x['comparison']]; print('total rows=',len(r)); print('GTE rows=',len(g)); print('all corrected=',all(x['p_holm'].strip() for x in g))"
```

Expected output after the eight GTE-related contrasts are complete:

```text
total rows= 101
GTE rows= 22
all corrected= True
```

## Step 8 — Report the complete compact-reranker comparison

Use only the retained metric files and manifests. Report:

- LitSearch R@5, R@20, and nDCG@20 for zero-shot and adapted GTE on BM25,
  GRIT, and Qwen3 pools;
- CSFCube pooled and per-facet R@20 and nDCG%20;
- the paired adapted-minus-zero-shot and Llama-minus-adapted estimates with
  95% intervals and Holm-corrected tests;
- total and trainable parameters, training wall time, peak training VRAM,
  median query latency, queries/s, documents/s, and peak inference VRAM;
- the exact GPU, precision, batch size, candidate depth, truncation, warmup,
  and repetition count.

The zero-shot comparison measures the additional value of DORIS-MAE
adaptation. The Llama comparison measures the quality/compute trade-off. Say
that the models receive the same DORIS-MAE adaptation supervision; do not say
that they received the same total lifetime supervision.

Do not update the paper from notebook screen output. Use the saved metric
files, manifests, and `paired_uncertainty.csv` after these checks pass.
