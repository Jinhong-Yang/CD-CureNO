# CD-CureNO SoftwareX Overleaf package v0.0.2

This package follows the SoftwareX Original Software Publication structure and
uses Elsevier's `elsarticle` class. The revision uses only frozen P0--P5
evidence. It does not convert the unfinished P6 held-out campaign, P2 one-seed
results, or the P5 validation pilot into confirmatory performance claims.

Version `v0.0.2` is a documentation and release-metadata patch. It aligns the
manuscript title, CRediT statement, citation metadata, traceability table,
public README, and executable command descriptions. It does not change a
scientific algorithm, numerical result, checkpoint, split, or gate decision.

## Build in Overleaf

1. Upload this directory or its ZIP archive to a new Overleaf project.
2. Set `main.tex` as the Main document.
3. Select pdfLaTeX.
4. Recompile until bibliography and cross-references are resolved.

## Local build

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

A TeX installation with `latexmk` can instead use `latexmk -pdf main.tex`.
The supplied `main.bbl` also allows the manuscript to compile when BibTeX is
not available in an initial source check.

## Submission gate

```bash
python submission_check.py
```

The gate enforces the 3,000-word limit used for this package, required
SoftwareX sections and C1--C8 metadata, the version-pinned README, the
generative-AI declaration heading, and the absence of unresolved markers.

## Supporting QA files

- `VALIDATION_REPORT.md`: clean-install, build, rendering, and evidence QA.
- `CLEAN_INSTALL_QA.md`: environment, tests, command checks, and raw-log hashes.
- `SOURCE_TRACEABILITY.csv`: manuscript claim-to-artifact mapping.
- `SUBMISSION_CHECKLIST.md`: completed and author-confirmation checks.
- `PACKAGE_MANIFEST.sha256`: checksums for this exact source package.
- `CITATION.cff`: versioned software citation metadata.
- `LICENSE.txt`: Apache License 2.0 text.

Before journal upload, the corresponding author should reconfirm the frozen
CRediT roles, funding, competing-interest statement, and exact generative-AI
tool/version. The GitHub `v0.0.2` tag and release URL must be live before the
manuscript is submitted.
