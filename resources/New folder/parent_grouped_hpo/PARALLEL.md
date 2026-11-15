# Parallel trials on the eight-H200 server

Use one independent training process per GPU. Each process runs an entire trial
with the existing batch size, accumulation, seed, split and training budget.
The queue gives each child exactly one visible GPU using `CUDA_VISIBLE_DEVICES`;
inside that child it is CUDA device 0. See
[NVIDIA's device visibility documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html).

This is parallel hyperparameter search. Do not launch these commands with
`torchrun` or `accelerate launch`: the trainer expects one process and one GPU
per trial. More GPUs increase the number of concurrent trials, without changing
the effective batch size of any trial.

## Before starting

Copy the updated bundle to the server **before preparing a new study**. The
study fingerprints all production Python files, including `run_matrix.py`.
Replacing this script in a previously prepared study will trigger an integrity
error. Keep any existing study and its original code intact; use a fresh artifact
directory for this version and follow the README's preparation and preflight
steps. Do not edit a study manifest to bypass the check.

Complete one tokenizer preflight and the single-GPU smoke run first. The smoke
run downloads the pinned base weights into the shared model cache before eight
processes load them. Each worker also loads its own token data and model copy in
host memory. Check available system RAM with `free -h` and disk space with
`df -h`; reduce the GPU list if host memory or shared storage becomes a bottleneck.
The CPU/data-loading work is replicated for every active trial.

Use only GPUs allocated to you. The examples below assume the displayed server
is available to this job. `--gpus` explicitly overrides an inherited
`CUDA_VISIBLE_DEVICES=0` from the README's smoke setup. In a cluster/container,
use indices valid within that allocation rather than assuming host indices.
Launch only one matrix runner against a given study at a time; this is a local
queue, not a cross-process GPU reservation service.

## Launch the default nine-trial grouped search

In the activated training environment, from the bundle directory:

```bash
export HPO_BUNDLE="$PWD"
export HPO_ARTIFACTS="$HPO_BUNDLE/artifacts"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Inspect the matrix first. This command does not train.
python -B run_matrix.py --artifacts "$HPO_ARTIFACTS" \
  --gpus 0,1,2,3,4,5,6,7

# Up to eight concurrent trials, one per selected GPU.
python -B run_matrix.py --artifacts "$HPO_ARTIFACTS" \
  --gpus 0,1,2,3,4,5,6,7 --execute

python -B analyze.py --artifacts "$HPO_ARTIFACTS" --protocols grouped
```

Use your actual artifact path if you prepared a differently named directory.
Run inside an existing `tmux`/`screen` session if you want it to survive an SSH
disconnect. Monitor GPU activity from another terminal with `watch -n 2 nvidia-smi`.
The first eight trials start together; the ninth starts as soon as a GPU is free.
If each trial takes approximately T hours, nine serial trials take about 9T and
this schedule about 2T: roughly 4.5x for this nine-trial matrix, before shared
CPU/storage overhead. Measure actual runtime; eight GPUs do not guarantee an
eightfold speedup for nine jobs.

## Optional list-level control

After the grouped matrix finishes, run the additional nine control trials:

```bash
python -B run_matrix.py --artifacts "$HPO_ARTIFACTS" \
  --protocols list_control --gpus 0,1,2,3,4,5,6,7 --execute

python -B analyze.py --artifacts "$HPO_ARTIFACTS" \
  --protocols grouped list_control
```

Alternatively, to put all 18 trials into one queue from the outset:

```bash
python -B run_matrix.py --artifacts "$HPO_ARTIFACTS" \
  --protocols grouped list_control --gpus 0,1,2,3,4,5,6,7 --execute
```

Use just `--gpus 0,1,2,3` to limit concurrency to four. Without `--gpus`, the
runner keeps the original sequential behavior and inherited GPU visibility.

## Logs, failures and reproducibility

The scheduler prints each trial's assigned GPU and log path. Each invocation
creates a fresh directory under `artifacts/logs/matrix_*/`, containing one log per
launched trial. Trial results still use the original `artifacts/runs/...` paths,
so the analysis commands and validation-only selection are unchanged.

If a trial exits unsuccessfully, the queue stops starting new trials, waits for
already active trials to finish, and exits with an error. Read the failed log
and preserve any partial run directory as described in the main README before
retrying. Completed compatible results are skipped on the next invocation.
Ctrl+C stops active children; interrupted trials require the same recovery.

Parallelism does not add training seeds or independent split replicates. Keep
the same pinned environment and base revision across workers. GPU scheduling
does not promise bitwise deterministic training or comparable per-trial timing
under shared I/O contention; the scientific selection protocol is unchanged.

## Local verification

The scheduler's CPU tests launch actual lightweight Python subprocesses. They
verify simultaneous execution on distinct device assignments, no overlap on the
same assigned GPU, reuse after completion, separate logs, rejection of duplicate
GPU indices, and stopping the queue after a failure while preserving active work.
They do not run CUDA or establish that a real model fits GPU/host memory.

```bash
python -B -m unittest discover -s . -p 'test_parallel.py' -v
```
