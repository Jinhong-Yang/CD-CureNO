# CD-CureNO v0.0.2 SoftwareX package validation report

## Overall assessment

The `v0.0.2` release manuscript is structurally and
editorially suitable for SoftwareX submission without additional scientific
experiments. It uses only existing P0--P5 artifacts and maintains the
distinction between software validation and confirmatory predictive evidence.
Publication of the immutable GitHub tag and release was authorized by the
corresponding author; the release workflow includes post-publication URL
verification before handoff.

## Automated manuscript checks

- Mandatory SoftwareX sections: present
- Metadata C1--C8: present
- Estimated manuscript words: 2,677
- Figure environments: 2
- Unresolved `SOFTWAREX-REPLACE` markers: 0
- Forbidden unsupported claim phrases: absent
- Strict `submission_check.py`: PASS

## Clean-install and command QA

- Environment: Windows, Python 3.11.9, PyTorch 2.13.0+cpu
- Editable install from the local `v0.0.2` candidate: PASS
- Full test suite: 210 passed, 30 explicitly skipped
- Focused P4 contract suite: 17 passed
- Noncanonical P4 plan review: PASS; 512 cases, no labels generated
- P5 inflation and verification command help: PASS
- Full P4 label generation, checkpoint inflation, model training, and other
  scientific experiments: not run

The first installation attempt under a deeply nested Windows path failed with
`WinError 206` while installing PyTorch's license tree. Repeating the same
installation from a short virtual-environment path succeeded. The public
README now documents this Windows path-length prerequisite. Detailed command
results and raw-log hashes are recorded in `CLEAN_INSTALL_QA.md`.

## Compilation and visual QA

- Engine: pdfLaTeX
- Bibliography: `bibtexu` with `elsarticle-num.bst`
- Final passes: two pdfLaTeX runs after bibliography generation
- LaTeX log scan: no fatal errors, unresolved citations/references, rerun
  warnings, or overfull boxes
- Final page count: 16
- Embedded fonts: all PDF fonts embedded
- Visual QA: all 16 pages rendered at 140 dpi and inspected; commands, figures,
  declarations, and references are visible without clipping

## Evidence and claim boundaries

- Evidence cutoff: the frozen P0--P5 artifacts underlying public release
  `v0.0.1`; `v0.0.2` changes documentation and release metadata only.
- P2 joint-model values remain one-seed descriptive results.
- P4 512-case and 27-check values remain solver/release validation.
- P5 restriction and future-invariance values remain label-free contract tests.
- The unfinished P6 campaign is not used for superiority, OOD, or experimental
  validation claims.
- No new training, data generation, statistical analysis, or experiment was
  performed for this revision.
- The existing Table 3 layout was retained without modification, as requested.

## Data and license review

- CD-CureNO code: Apache-2.0.
- Upstream ResFNO arrays: not redistributed; upstream terms and manifest
  checksums govern acquisition.
- Generated P4 core arrays: approximately 918 MB and not bundled in the tagged
  code release; frozen plan/config/code/hash and compact summaries are present.
- The manuscript states these boundaries explicitly.

## Release status and remaining author checks

The package, manuscript, citation metadata, and README all identify `v0.0.2`.
The corresponding author authorized publication of this exact immutable
commit, tag, and release. Every versioned GitHub URL is checked as part of the
release handoff. Funding, competing-interest, generative-AI, and co-author
CRediT confirmations remain author responsibilities.
