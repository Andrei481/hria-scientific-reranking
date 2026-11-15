Parent-grouped hyperparameter selection

This bundle implements the experiment authorized in the conversation: keep all
aspects and augmented lists from a parent together during model selection. It
creates a new Llama 3.1 8B cohort and never silently reuses historical FT weights.

The bundled data preserve the existing 4,494 sampled lists, including 50/51
candidate lengths and positive-cap exceptions. Parent identity is recovered from
aspect IDs and the original complete parent pools, not numeric source-row order.
Text is deduplicated for transfer without changing prompt content.

Ten category-balanced parents are fixed as the internal test set. For each split
seed, the remaining 90 parents become 80 training and 10 validation parents.
An optional list-control condition divides the same non-test lists into training
and validation, deliberately permitting parent overlap there. Its test parents
remain held out. This is a newly controlled comparison, not an exact replay of
the original list-level 80/10/10 partition.

Default search: learning rates 5e-6, 1e-5 and 2e-5; split seeds 42, 2026 and 3407;
training seed 3407; two epochs; rank 128; alpha 256; dropout 0.05; BF16 LoRA;
batch size 2; gradient accumulation 1. Selection minimizes mean parent-macro
assistant-token negative log likelihood across the prescribed split seeds.
Within a parent, each list contributes its mean response-token loss equally.
The default is new learning-rate sensitivity evidence, not historical HPO replay.

Freeze configuration, model revision and code before preflight/training. Every
trial starts from the same pinned base. Token preflight rejects truncation,
prefix-mask mismatch and empty targets. Smoke runs cannot enter selection.
Incomplete grids, mixed revisions/configurations and non-finite scores are errors.
Selection never reads internal-test scores or LitSearch/CSFCube results.

After selecting, optional internal-test evaluation uses only that trial's
80-parent checkpoints; these are diagnostic scores, not the final 100-parent
model. An optional fresh refit uses all 100 parents, followed by the existing
external benchmark harness. Once used for the refit, the internal test is no
longer a held-out set for that model.

Implementation sequence:

- [x] Test and implement provenance mapping, grouped/control splits and metrics.
- [x] Package audited text/lists and generate immutable split manifests.
- [x] Implement token preflight, trial runner, selection, selected-test and refit.
- [x] Document Linux GPU commands, expected outputs and reviewer reporting.
- [x] Run CPU tests and actual-data audits; independently review the bundle.

Scope: GPU training is to be run by the user on the target machine. Local CPU
verification must not be represented as H200 execution or experimental results.
