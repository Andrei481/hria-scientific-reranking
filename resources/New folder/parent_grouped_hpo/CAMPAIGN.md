# Optional four-model grouped-selection campaign

The default README remains a nine-trial Llama learning-rate study. This separate
campaign supplies the broader experiment expected by `test_campaign.py`:

| Model | Hugging Face base | Loading |
|---|---|---|
| `llama8b` | `meta-llama/Llama-3.1-8B-Instruct` | BF16 LoRA |
| `gemma12b` | `google/gemma-3-12b-it` | BF16 LoRA |
| `qwen14b` | `Qwen/Qwen2.5-14B-Instruct` | BF16 LoRA |
| `gemma27b` | `google/gemma-3-27b-it` | NF4 QLoRA, double quantization, BF16 compute |

The loading choices follow the primary-cohort table in `HRIA/articol_latex.tex`.
Gemma uses Unsloth `FastModel` and language-only LoRA targets; vision weights
remain frozen. Llama and Qwen use `FastLanguageModel`. The runner rejects
unmaterialized weights, trainable vision parameters and unexpected trainable
non-LoRA weights. See the [official Unsloth Gemma example](https://github.com/unslothai/notebooks/blob/main/nb/Gemma3_%284B%29.ipynb).

Each model uses a new factorial sensitivity grid: learning rate in
`{5e-6, 1e-5, 2e-5}`, LoRA rank in `{64, 128}`, dropout in `{0, 0.05}`, and
alpha always twice the rank. Twelve configurations across three grouped splits
give **36 trials per model, 144 total**, plus optional smoke/refit runs. The
reference is `lr_1e-5_r128_d05`. Epochs, batching and other settings come from the
base `config.json`; defaults match the README. Alpha varies with rank, so this
grid does not estimate an independent alpha effect.

This is a newly specified sensitivity grid, not a verified reconstruction of the
original HPO. Report its actual dimensions and outcomes. It cannot establish that
untested choices, such as other schedules or epoch budgets, would be unchanged.
No trained campaign results are included in the bundle.

## Prepare, pin and preflight

Use the same activated Linux environment and bundled resources as the README.
Accept each required model license. From the bundle directory:

```bash
export HPO_CAMPAIGN="$PWD/artifacts_campaign"
export CUDA_VISIBLE_DEVICES=0

python -B campaign.py prepare --root "$HPO_CAMPAIGN"
python -B campaign.py status --root "$HPO_CAMPAIGN"
```

Preparation creates four independent studies with identical parent splits. It
refuses an existing root. Status initially reports pending selection for all
four models; it does not infer results from configurations or directory names.

Resolve one model revision per study, once. Do not overwrite this revision file
midway through the campaign:

```bash
python - <<'PY'
import json, os
from pathlib import Path
from huggingface_hub import HfApi
root = Path(os.environ['HPO_CAMPAIGN'])
manifest = json.loads((root / 'campaign.json').read_text())
revisions = {name: HfApi().model_info(item['model']).sha
             for name, item in manifest['models'].items()}
with (root / 'model_revisions.json').open('x') as f:
    json.dump(revisions, f, indent=2)
PY

python - <<'PY'
import json, os, subprocess, sys
from pathlib import Path
root = Path(os.environ['HPO_CAMPAIGN'])
revisions = json.loads((root / 'model_revisions.json').read_text())
for name, revision in revisions.items():
    subprocess.run([sys.executable, '-B', 'preflight.py', '--artifacts',
                    str(root / name), '--revision', revision], check=True)
PY
```

Each model needs its own tokenizer cache. If a preflight exceeds 16,384 tokens,
follow the README's no-truncation procedure. To prepare the entire campaign at
a larger, predeclared context length, use a fresh root and
`campaign.py prepare --root <new-path> --max-length <chosen-limit>`. Report that
recipe departure. The larger models still need individual GPU smoke checks;
an eight-H200 server does not prove every long-context trial fits one H200.

## Smoke, search and analyze

Commands print their planned stages by default. Add `--execute` to run:

```bash
python -B campaign.py smoke --root "$HPO_CAMPAIGN" --execute
python -B campaign.py search --root "$HPO_CAMPAIGN" \
  --gpus 0,1,2,3,4,5,6,7
python -B campaign.py search --root "$HPO_CAMPAIGN" \
  --gpus 0,1,2,3,4,5,6,7 --execute
python -B campaign.py analyze --root "$HPO_CAMPAIGN" --execute
python -B campaign.py status --root "$HPO_CAMPAIGN"
```

Models are processed sequentially, with up to eight simultaneous trials within
each model. Smoke runs are sequential on the one exposed GPU. This avoids
multiple model queues contending for the same devices. Each model's reports are
under `<campaign-root>/<model>/analysis/`; its parallel logs are under `logs/` in
that model's study. A failure stops progression to later models. Preserve failed
directories and follow the main README recovery procedure before restarting.

Analyze and report each model's winner independently; do not compare absolute
token losses across model vocabularies. The same fixed internal test parents
remain unavailable to HPO for all models.

## Optional refit

After selection is complete for all models:

```bash
export CUDA_VISIBLE_DEVICES=0
python -B campaign.py refit --root "$HPO_CAMPAIGN" --execute
python -B campaign.py status --root "$HPO_CAMPAIGN"
```

This sequentially trains four fresh full-data adapters at the selected settings.
It saves adapters without automatically exporting merged weights. If you need
merged export, use the main `train.py --mode refit --export-merged` command for
each study **instead of** its campaign refit command, before that refit output
directory exists. The 4-bit model uses Unsloth's `merged_16bit` exporter rather
than merging into quantized weights. This exporter and GPU training still need
validation in your pinned server environment.

Campaign status validates selection against complete trial grids and refit
provenance/adapter hashes. It always states that external evaluation is separate.
Even four completed refits do not establish LitSearch/CSFCube results. Evaluate
each fresh model with the frozen benchmark harness before changing paper tables.
