# CD-CureNO SoftwareX Overleaf package

This package follows the official **SoftwareX original article template,
Version 6 (March 2026)** while using Elsevier's `elsarticle` LaTeX class for
Overleaf. The manuscript preserves the required metadata table and the five
mandatory sections:

1. Motivation and significance
2. Software description
3. Illustrative examples
4. Impact
5. Conclusions

The main body is intentionally software-centered. It uses only passed P0-P5
artifacts and does not treat the unfinished P6 held-out campaign as evidence.

## Overleaf

1. Upload the ZIP or all files in this directory to a new Overleaf project.
2. Set `main.tex` as the Main document.
3. Use pdfLaTeX.
4. Replace every `SOFTWAREX-REPLACE` item in `author_metadata.tex` and confirm
   the competing-interest statement in `main.tex`.
5. Recompile until references resolve.

## Local build

```powershell
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Clean auxiliary files with:

```powershell
latexmk -C
```

## Submission gate

The package compiles with visible red placeholders so that missing human
metadata cannot be mistaken for final content. Structural QA can be run with:

```powershell
python submission_check.py --allow-placeholders
```

The strict pre-submission check is:

```powershell
python submission_check.py
```

The strict check must fail until the public GitHub URL, documentation URL,
CRediT statement, acknowledgements, and conflict-of-interest confirmation have
been completed. Author, affiliation, corresponding/support email, funding, and
Apache-2.0 license fields have been populated from author-supplied decisions.

## Important release constraint

SoftwareX requires a public GitHub repository with a documented `README.md`
and `License.txt`. CD-CureNO uses the OSI-approved Apache License 2.0, but the
current working repository still has no public remote. The upstream ResFNO
repository/data have no explicit license recorded in this project; do not
redistribute or relicense them by assumption.

## Package contents

- `main.tex`: SoftwareX manuscript.
- `author_metadata.tex`: single human-input surface.
- `references.bib`: numerical references.
- `figures/`: frozen, non-result-selected manuscript figure.
- `highlights.txt`: Elsevier highlights draft.
- `LICENSE.txt`: Apache License 2.0 distribution terms.
- `CITATION.cff`: GitHub/software citation metadata.
- `SOURCE_TRACEABILITY.csv`: claim-to-artifact map.
- `VALIDATION_REPORT.md`: evidence QA and remaining limitations.
- `SUBMISSION_CHECKLIST.md`: final human/editorial gate.
- `submission_check.py`: automated structural and placeholder gate.
