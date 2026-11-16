# Scientific-paper reranking: article and research resources

Materials for **Are 100 Queries Enough? Small-Data Adaptation of Open LLMs for Scientific Reranking**.

Repository: https://github.com/Andrei481/hria-scientific-reranking

## Availability and limitations

This repository contains the available code, notebooks, experiment configurations,
analysis scripts, and experimental results accompanying the article. Trained model
checkpoints and some raw experimental files are **not included**. The released
resources should not be interpreted as a complete archive of every training and
inference run. In particular, notebooks may refer to external model weights,
corpora, local paths, or intermediate files that are absent from this release.

The source benchmark datasets are maintained by their original authors:

- **DORIS-MAE:** [archived dataset](https://doi.org/10.5281/zenodo.8299749) and
  [upstream project](https://github.com/Real-Doris-Mae/Doris-Mae-Dataset).
  The local `resources/DorisMAE/` directory includes queries, judgments, aspect
  mappings, and notebooks; the original `Corpus.json` is not included.
- **LitSearch:** [dataset](https://huggingface.co/datasets/princeton-nlp/LitSearch)
  and [upstream project](https://github.com/princeton-nlp/LitSearch).
  The upstream dataset has `query`, `corpus_clean`, and `corpus_s2orc`
  configurations. Local evaluation code and retained results are under
  `resources/LitSearch/` and the experiment directories below.
- **CSFCube:** [upstream project and releases](https://github.com/iesl/CSFCube).
  The local `resources/CSFCube/` directory contains the benchmark's documentation,
  annotations, and evaluation resources.

Consult the original dataset documentation and retained third-party license files
for reuse terms. References in the manuscripts identify the benchmark papers.
