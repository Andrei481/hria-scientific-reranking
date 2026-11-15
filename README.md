# Scientific Reranking Reproducibility

This repository collects the resources used for **Are 100 Queries Enough?
Small-Data Adaptation of Open LLMs for Scientific Reranking**: notebooks,
scripts, configurations, prepared data, saved rankings, and analysis results.

Start with [resources.md](resources.md) for the experiment-to-file guide and
[models and data](docs/models_and_data.md) for download links. Resources that
still need an author-supplied link or file are listed in
[AUTHOR_TODO.md](AUTHOR_TODO.md).

## Repository layout

```text
resources.md                 experiment-to-file guide
docs/models_and_data.md       available model and data links
AUTHOR_TODO.md                remaining links/files to supply
notebooks/                   existing numbered notebook collection
configs/environments/        retained environment exports
resources/                   collected experimental resources
  DorisMAE/                  query/aspect metadata and original notebooks
  HRIA/                      prepared aspect data and figure scripts
  LitSearch/                 reranking, evaluation, retrieval pools and results
  CSFCube/                   gold data, evaluation and selected result files
  New folder/                additional experiment bundles and saved outputs
  tools/  configs/  results/  statistical/disjoint analysis resources
  tie_sensitivity_artifacts/ final tie-study evaluations and manifests
docs/resource_inventory.csv  source paths, hashes and packaging changes
LICENSES/                    upstream notices
```

The paths below `resources/` preserve the original workspace layout because
existing scripts and configurations refer to those paths. For these scripts,
`resources/` corresponds to the original workspace root. The numbered notebooks
at the repository root remain available as a convenient overview.

This is a collection of retained experimental resources. Original scripts may
contain server paths, environment-specific settings, or model IDs that must be
set for a new machine. The individual experiment READMEs document their commands;
there is no new common execution framework. Files ending in `.jsonl.gz` are
compressed copies: decompress them to the same filename without `.gz` when an
original script expects JSONL. Model weights are linked rather than bundled.

Historical, reconstructed, and later sensitivity experiments are kept separate.
See [resource notes](docs/resource_notes.md) before treating an earlier run or
refactored preparation script as the source of a published result. The existing
`src/hria_reranking` package and preparation script are retained utilities; they
are not asserted to be the exact producers of all historical checkpoints.

Notebook outputs and execution counts were cleared in collected copies. Saved
result files are retained separately; source files in the author's workspace
were not changed. [The inventory](docs/resource_inventory.csv) records the
original and included hashes, including compressed files and notebook changes.

Third-party notices are described in [LICENSES/README.md](LICENSES/README.md).
The author still needs to select a license for original contributions and
confirm the final repository URL in `CITATION.cff` and the article.
