# Controlled compact supervised reranker: exact runbook

This bundle trains `answerdotai/ModernBERT-base` (149M parameters) as a
non-generative cross-encoder using exactly the same DORIS-MAE-derived source
rows, graded labels, pool-sampling algorithm, three variants, target list size
50, five-positive cap, 100-document hard-negative prefix, and 300-word paper
truncation as the LLM pipeline. Training uses listwise LambdaLoss, so relevance
ties remain ties rather than becoming an arbitrary target permutation.

The original sampler has a small fallback-size quirk that this control preserves:
4,312 lists have 50 papers and 182 have 51. In those 182 cases it first fills 50
negative slots, then inserts a required positive. Correcting that only for
ModernBERT would no longer be the same supervision; the preparation manifest
therefore records and asserts the exact size histogram.

The model is a generic masked-language-model checkpoint, not an off-the-shelf
MS MARCO reranker. This is what makes it a controlled test of whether the large
generative architecture is needed. The model card is
<https://huggingface.co/answerdotai/ModernBERT-base>; LambdaLoss is documented at
<https://sbert.net/docs/package_reference/cross_encoder/losses.html>.

## Step 1 — Create the environment

On the GPU server, from the repository root:

```bash
conda create -n modernbert_baseline python=3.11 -y
conda activate modernbert_baseline
python -m pip install --upgrade pip
python -m pip install -r "New folder/modernbert_supervised_baseline/requirements-modernbert.txt"
python -m ipykernel install --user --name modernbert_baseline --display-name "Python 3 (modernbert)"
jupyter lab
```

The requirement file is an exact environment target. If the server's CUDA
driver cannot load the pinned PyTorch wheel, install the PyTorch 2.13.0 wheel
for that CUDA runtime, then run the remaining requirements. Do not silently
change versions: notebook 02 writes the actually loaded versions, GPU name,
driver, and memory to `training_manifest.json`.

## Step 2 — Freeze the same 4,494 supervision examples

Open `01_prepare_supervision.ipynb`, select the `modernbert_baseline` kernel,
and run all cells.

The repository contains the 1,501-row source JSONL. Notebook 01 automatically
checks both `/DorisMAE/splits/Corpus.json` and
`/DATA/j_andrei/DorisMAE/splits/Corpus.json`. Your mounted layout shown in
Jupyter is the first one. You only need an override if the file is elsewhere:

```bash
export DORIS_CORPUS_PATH=/DorisMAE/splits/Corpus.json
export MODERNBERT_ARTIFACT_DIR=/DATA/j_andrei/modernbert_supervised_baseline
```

Required terminal output:

- input rows: 1,501;
- eligible rows: 1,498;
- removed zero-positive rows: 3;
- output examples: 4,494;
- list-size histogram: 4,312 of length 50 and 182 of length 51;
- minimum and maximum generated variants: both 3;
- a SHA-256 for `dorismae_listwise_4494.jsonl`.

If any assertion fails, stop. Do not edit the expected counts to make it run.
Retain both `dorismae_listwise_4494.jsonl` and `preparation_manifest.json`.

## Step 3 — Train the compact reranker once

Restart the kernel to release the large pandas corpus, then open
`02_train_modernbert.ipynb` and run all cells once. This is one seed, 3407—not
three seeds.

The fixed run is:

- ModernBERT-base, one regression logit per query-document pair;
- listwise LambdaLoss with default NDCGLoss2++ weighting and `k=20`;
- 4,494 labeled lists (4,312 length 50; 182 length 51), two epochs;
- list batch size 1, gradient accumulation 8;
- document mini-batch size 8 inside LambdaLoss;
- learning rate `2e-5`, warmup ratio `0.06`, weight decay `0.01`;
- BF16, max pair length 1,024, SDPA attention, no quantization;
- no validation/test tuning: the configuration is fixed a priori.

The last cell reloads the saved checkpoint and runs a smoke test. Retain the
whole model directory and `training_manifest.json`; it contains wall-clock
training time and peak allocated/reserved VRAM.

## Step 4 — Run both quality evaluations

Restart the kernel. Run `03_evaluate_litsearch.ipynb` from top to bottom. It
downloads the pinned LitSearch `corpus_clean` revision `9573fb2`, truncates each
existing candidate file to the same top 100 used in the paper, reranks without
changing candidate membership, and calls the existing metric script.

Retain for BM25, GRIT, and Qwen3:

- each `*.modernbert.reranked.jsonl` file;
- each adjacent `*.metrics.json` file;
- `litsearch_evaluation_manifest.json`.

Restart the kernel again and run `04_evaluate_csfcube.ipynb`. Retain:

- three `test-pid2pool-csfcube-...-ranked.json` files;
- `metrics-background.txt`, `metrics-method.txt`, `metrics-result.txt`, and
  `metrics-all.txt`;
- the generated `*-all-perq.csv` file if the CSFCube evaluator writes it;
- `csfcube_evaluation_manifest.json`.

The original CSFCube `rank_metrics.py` uses `np.asfarray`, so this bundle pins
NumPy 1.26.4. NumPy 2.x removed that function. If this environment was created
from the earlier requirements file, repair it with:

```bash
python -m pip install --force-reinstall "numpy==1.26.4"
```

Restart the Jupyter kernel after the installation before rerunning notebook 04.

Do not claim that ModernBERT beats SPECTER-ID unless the result actually does.
The necessary controlled claim is narrower: how a compact supervised reranker
compares with the generative models under identical DORIS-MAE-derived labels.

## Step 5 — Compare compute on matched inputs

Run `05_compute_benchmark.ipynb` on an otherwise idle GPU. Its ModernBERT
section runs immediately: 20 fixed LitSearch queries × the same top-100 Qwen3
candidates, one warmup and three measured repeats.

For the generative comparison, serve the representative Llama 3.1 8B FT
checkpoint with the same vLLM settings used in the paper. Set:

```bash
export LLM_BASE_URL=http://127.0.0.1:8000/v1
export LLM_MODEL_ID=/absolute/path/to/the/llama-3.1-8b-ft-checkpoint
export LLM_API_KEY=EMPTY
export LLM_PEAK_VRAM_GIB=REPLACE_WITH_SERVER_PEAK
```

Then set `RUN_LLM_BENCHMARK = True` in notebook 05 and rerun it. The notebook
uses the same listwise prompt and 300-word paper truncation as the existing LLM
pipeline. Record peak server VRAM from the vLLM process or `nvidia-smi`; client
PyTorch cannot observe another process's allocator.

Report median query latency, queries/s, documents/s, and peak VRAM. Also report
training wall time and peak VRAM from the training manifests. If valid LLM timing
already exists for exactly this protocol and hardware, you may reuse it; otherwise
rerun inference timing only. You do not need to retrain the LLM.

## Step 6 — Add the result and paired uncertainty to the paper

Use `paper_reporting_template.md` for the method and tables. Fill every placeholder
from the saved artifacts; do not copy numbers from notebook screen output without
retaining the corresponding file.

Add ModernBERT as a local-baseline comparison in the paired statistical analysis.
`paired_stats_config_additions.json` gives the exact path pattern. Copy its
contrasts into `configs/paired_statistical_uncertainty.json`, update that family's
expected contrast count, then run:

```bash
python tools/paired_statistical_uncertainty.py \
  --config configs/paired_statistical_uncertainty.json \
  --output-dir artifacts/paired_statistical_uncertainty
```

For LitSearch, compare per-query outputs under each frozen retriever. For
CSFCube, compare per-case output with a clustered bootstrap over the 34 unique
query papers, not over the 50 facet cases as if they were independent.

The experiment is complete only when the paper reports:

- compact-model quality on all three LitSearch pools and CSFCube;
- paired 95% intervals/delta tests against the representative FT LLM;
- one seed and the exact fixed training configuration;
- parameter count, training time, peak VRAM, and matched inference throughput;
- the factual conclusion supported by the observed quality/compute trade-off.
