# Fix the five missing-implementation failures and NumPy warning

The three `test_campaign` failures were caused by a missing `campaign.py`. The
two additional `test_training` failures were caused by missing Gemma loading and
model-validation helpers. The corrected bundle implements these features; no
tests have been disabled or deleted. It also contains the parallel-runner update.

## 1. Copy the complete corrected folder to the server

Update `/home/andreij/parent_grouped_hpo` with the complete current bundle,
including `campaign.py`, `train.py`, `protocol.py`, `run_matrix.py` and all tests.
Copying only `campaign.py` is insufficient. Keep any existing experiment outputs.

Your output ran 21 tests; the corrected bundle currently has 25, including three
parallel-runner tests and an additional Gemma language-target regression test.
The local full suite passed all 25. These tests run on CPU and do not need model
downloads or CUDA initialization.

## 2. Repair and check the activated Python environment

```bash
cd /home/andreij/parent_grouped_hpo
conda activate grouped_hpo

python -m pip install numpy
python -m pip check
python -c 'import sys, numpy, torch; print(sys.executable); print("NumPy:", numpy.__version__); print("Torch:", torch.__version__); print(torch.from_numpy(numpy.zeros(2))); print("GPU:", torch.cuda.get_device_name(0)); print("BF16:", torch.cuda.is_bf16_supported())'

python -B -m unittest discover -s . -p 'test_*.py' -v
```

Use `python -m pip` so installation targets the same interpreter running the
tests. NumPy was already present in the retained environment YAML; its absence
on the server means that environment's dependencies were not fully installed.
The command above repairs the reported missing package. `pip check` may reveal
other missing dependencies or version conflicts; resolve those before smoke
training. Successful GPU identification alone does not validate the full stack.
See [NumPy installation guidance](https://numpy.org/install/).

Because this repair may differ from the old package snapshot, record the actual
working environment before paper trials:

```bash
python -m pip freeze > environment-server-pip.txt
conda env export > environment-server.yml
```

Keep the same environment across all trials. Do not replace the existing Torch
or CUDA installation merely because of this NumPy warning: the supplied output
already identifies an H200 and reports BF16 support.

## 3. Prepare fresh artifacts after updating the code

If you have only run the setup tests, continue with README step 3. If you already
prepared a study, preserve it and use a new path, for example:

```bash
export HPO_BUNDLE="$PWD"
export HPO_ARTIFACTS="$HPO_BUNDLE/artifacts_fixed"
export CUDA_VISIBLE_DEVICES=0
python -B prepare.py --config config.json --output "$HPO_ARTIFACTS"
```

Then repeat README steps 4 and 5 (pinned tokenizer preflight and the two-step GPU
smoke check). Updated production files invalidate the previous study's source
hashes by design. Do not edit manifests or reuse old results to bypass that check.

After smoke validation, launch the default nine-trial Llama search:

```bash
python -B run_matrix.py --artifacts "$HPO_ARTIFACTS" \
  --gpus 0,1,2,3,4,5,6,7 --execute
python -B analyze.py --artifacts "$HPO_ARTIFACTS" --protocols grouped
```

The optional four-model campaign is documented in `CAMPAIGN.md`. Its presence
does not change the default experiment or automatically start 144 trials.
