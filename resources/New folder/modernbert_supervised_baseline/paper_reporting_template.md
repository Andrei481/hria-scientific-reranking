# Paper reporting template

Replace every bracketed placeholder with a value from a retained manifest,
metrics file, or statistical-analysis output.

## Methods paragraph

To test whether a large generative reranker is necessary under our supervision
regime, we trained a compact non-generative cross-encoder initialized from
ModernBERT-base (149M parameters). The checkpoint has generic masked-language-
model pretraining and no prior ranking supervision. We used exactly the same
1,498 eligible DORIS-MAE aspect rows and deterministic pool-construction process
as the generative models, yielding the same 4,494 lists (4,312 with the intended
50 candidates and 182 with 51 candidates due to the original positive-fallback
behavior). The
cross-encoder received the original graded relevance labels (0/1/2) and was
optimized for two epochs with LambdaLoss (default NDCGLoss2++ weighting,
`k=20`), learning rate `2e-5`, warmup ratio `0.06`, weight decay `0.01`, list
batch size 1, gradient accumulation 8, and seed 3407. We used BF16, a maximum
query-document length of 1,024 tokens, and no quantization. Hyperparameters were
fixed a priori; neither LitSearch nor CSFCube was used for model selection.

## LitSearch quality table

| Reranker | Parameters | Retriever | R@20 | R@5 | nDCG@20 |
|---|---:|---|---:|---:|---:|
| ModernBERT supervised | 149M | BM25 | [ ] | [ ] | [ ] |
| ModernBERT supervised | 149M | GRIT | [ ] | [ ] | [ ] |
| ModernBERT supervised | 149M | Qwen3 | [ ] | [ ] | [ ] |
| Representative FT LLM | [ ] | BM25 | [ ] | [ ] | [ ] |
| Representative FT LLM | [ ] | GRIT | [ ] | [ ] | [ ] |
| Representative FT LLM | [ ] | Qwen3 | [ ] | [ ] | [ ] |

Immediately below the table, report paired deltas as “FT LLM − ModernBERT” with
paired 95% intervals and Holm-adjusted p-values for every reported metric and
retriever. A negative delta favors ModernBERT.

## CSFCube quality table

| Reranker | Background nDCG%20 | Method nDCG%20 | Result nDCG%20 | All |
|---|---:|---:|---:|---:|
| ModernBERT supervised | [ ] | [ ] | [ ] | [ ] |
| Representative FT LLM | [ ] | [ ] | [ ] | [ ] |
| SPECTER-ID (published/local reference, label provenance explicitly stated) | [ ] | [ ] | [ ] | 0.572 |

For the FT-LLM versus ModernBERT delta, report the clustered paired 95% interval
and corrected p-value using the unique query-paper ID as the resampling cluster.

## Compute table

| Model | Parameters | Train time | Train peak VRAM | Median query latency | Queries/s | Documents/s | Inference peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| ModernBERT supervised | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Representative FT LLM | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |

State that inference timing used the same 20 fixed LitSearch queries, the same
top-100 Qwen3 pools, title/abstract inputs truncated to 300 words, one warmup,
three measured repetitions, sequential execution, and an otherwise idle GPU.

## Interpretation paragraph: choose only after seeing results

- If ModernBERT is statistically indistinguishable from the FT LLM while being
  much cheaper, conclude that the DORIS-MAE data recipe—not generative decoding
  or multi-billion-parameter scale—explains most of the observed gain.
- If the FT LLM has a clear paired advantage, conclude that larger generative
  capacity provides measurable quality beyond what a strong compact model can
  extract from identical supervision, and quantify the compute premium.
- If performance varies by benchmark or first-stage retriever, report the mixed
  result and avoid a universal architectural claim.

In every case, preserve the separate limitation that the proposed models remain
below SPECTER-ID on CSFCube if that remains true after the controlled rerun.
