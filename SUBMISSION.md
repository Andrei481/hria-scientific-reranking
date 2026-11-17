# Journal of Intelligent Information Systems submission

Guidelines checked on 17 November 2026:
https://link.springer.com/journal/10844/submission-guidelines

## Manuscript preparation

- Both manuscripts use the Springer Nature `sn-jnl` template with
  `sn-basic,Numbered`, as documented in the supplied template manual.
- The guideline page recommends this modern template and `sn-basic.bst`, but
  also retains an older `smallcondensed` instruction. `smallcondensed` is not an
  option of `sn-jnl`; it has not been added or emulated.
- Both reference lists are generated from `references.bib`, with numbered
  citations and DOI links. The template sample bibliography is not used.
- The main article has six keywords and an abstract within the 150–250 word
  range. The abstract citation has been removed.
- Romanian names use explicit LaTeX accents, including comma-below accents.
- The article includes funding, competing interests, author contributions,
  and data/code availability under Statements and Declarations.
- The author supplied the no-competing-interests declaration and confirmation
  that all authors reviewed and approved the manuscript. On 17 November 2026,
  the equal-contribution statement was replaced with the supplied individual
  roles: Andrei-Răzvan Joldea led conception, code and experiments, analysis, and
  original writing; the other four authors contributed to conceptualization,
  methodology, supervision, and critical revision.
- The availability statement explicitly excludes trained model checkpoints and
  some raw experimental files. It does not promise access on request or complete
  end-to-end reproducibility from the repository alone.
- Online Resource 1 includes the article title, journal name, author names,
  affiliation, and corresponding-author email.

## Upload files

The `submission/` directory contains the main article PDF, `ESM_1.pdf`, and
separate source archives. Each source archive has a flat layout with its own
LaTeX entry point, bibliography, class/style files, referenced PDF figures,
and editable SVG counterparts for the external figures.

Upload the main article source archive and compiled PDF as manuscript files.
Upload `ESM_1.pdf` as **Online Resource 1**. If supplementary source is requested,
use its separate archive; do not merge both LaTeX entry points into the main
manuscript archive.

Suggested supplementary-file description:

> Detailed methods, robustness checks, complete supporting results, and
> exploratory variants for the scientific-paper reranking study.

Author contributions and competing interests must also be entered in the journal's
submission form. Enter the confirmed manuscript statements there. Check the final
PDF produced by the submission portal before completing submission.

## Validation

- Both documents compiled with pdfLaTeX and BibTeX (TeX Live 2026), including
  repeated LaTeX passes to resolve citations and cross-references.
- Final lengths: main article **24 pages** including references (below the
  journal's 25-page limit); supplementary PDF **23 pages**.
- The final logs contain no undefined references/citations, missing characters,
  or overfull boxes. Underfull spacing, math-font substitutions, and nonfatal PDF-bookmark warnings remain.
- All 17 main-article table bodies and 22 supplement tabular bodies were compared
  with the original sources and are unchanged. All 27 unique reference keys
  are preserved. BEIR now cites the published NeurIPS Datasets and Benchmarks
  2021 proceedings, with the official proceedings URL replacing its preprint DOI.
- Rendered pages were visually checked for clipping, overlap, and table placement.
- Table captions retain their original below-table placement. The supplied class
  wraps tables in `threeparttable`; moving captions above scaled tables triggered
  a reproducible grouping error. No class internals were patched.
- The standard `combelow.sty` package is included unchanged, with its license
  header, to preserve Romanian comma-below accents in a flat source package.
- The source packages use the supplied `sn-basic.bst` with both `SORT` commands
  commented out, as instructed in its header. Each document's references are
  numbered consecutively in order of first citation.

Recheck pagination if the manuscript or declarations change. The final portal
build can differ because it may use a different TeX Live release.

## Final figure and reference corrections (17 November 2026)

- Figures 1, 2 and S2 were regenerated as vector PDFs at the manuscript text
  width, with 8.5--9 pt lettering and embedded TrueType fonts. Their editable
  SVG counterparts are included in the corresponding source archives.
- Figure 1 now states a target list size of 50 and realized sizes of 50 or 51.
- Figure 2 uses distinct line styles and markers. Figure S2 uses patterned bars
  so the configurations remain distinguishable without color.
- Figure S1 uses black numbers; the darkest red cell has approximately 5.73:1
  text contrast. All heatmap values and all manuscript table bodies are unchanged.
- Run `python scripts/render_submission_figures.py` from the repository root to
  reproduce the external figures using the retained experiment data. This uses
  Python with Matplotlib and NumPy, plus the existing research aggregation code.
- Both source archives were independently extracted and compiled after packaging.
  Their page text and rendered pixels match the corresponding submission PDFs.

The submission portal still requires the authors' declarations and approval of
its generated PDF. The journal's conflicting template instructions remain noted
above. Confirm any disclosure for AI use beyond copyediting, if applicable;
no unverified disclosure or promise of access to missing data has been added.
