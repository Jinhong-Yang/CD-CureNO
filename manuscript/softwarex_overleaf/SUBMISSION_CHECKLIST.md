# SoftwareX submission checklist

## Hard blockers before submission

- [x] Confirm the final author list, order, affiliations, corresponding author,
  and email from the supplied author record.
- [x] Apply the OSI-approved Apache License 2.0 and add root-level
  `LICENSE.txt` to the public release.
- [x] Publish the exact manuscript version in public GitHub repository
  `Jinhong-Yang/CD-CureNO`.
- [x] Create stable tag/release `v0.0.1`.
- [x] Replace C2 with the permanent GitHub link to that exact version.
- [x] Provide public developer documentation through the repository README and
  support email `jinhong@inje.ac.kr`.
- [x] Verify that the release README contains installation, data acquisition,
  minimal examples, expected outputs, tests, and citation instructions.
- [x] Exclude upstream ResFNO code/data from redistribution unless explicit
  permission or a compatible license is documented. A submodule pointer or
  user-run acquisition/checksum workflow is safer than bundling unlicensed
  bytes.
- [x] Fill the supplied funding statement.
- [x] Record acknowledgements as none.
- [ ] Fill CRediT and confirm the competing-interest statement.

## Manuscript compliance

- [x] Keep the five mandatory SoftwareX sections and C1-C8 metadata table.
- [x] Keep the main text below 4,000 words; prioritize this limit over page
  count as the official template instructs.
- [x] Keep figures at six or fewer.
- [ ] Do not upgrade P2/P5 one-seed or validation-only evidence to confirmatory
  superiority.
- [ ] Do not claim geometry OOD, direct experimental validation, broad
  source-seed invariance, state of the art, or a first-ever method.
- [x] Confirm every number against `SOURCE_TRACEABILITY.csv` and the current
  frozen artifacts.
- [x] Re-run the current public-release test suite in a clean environment:
  211 passed and 29 upstream/artifact-dependent tests skipped with reasons.
- [ ] Refresh literature immediately before submission.
- [ ] Run `python submission_check.py` and require exit code 0.
- [ ] Compile from a clean Overleaf project and visually inspect every page.

## Submission-system extras

- [ ] Upload `highlights.txt` if requested by Editorial Manager.
- [ ] Supply the source ZIP and compiled manuscript PDF.
- [ ] Add a software citation/reference for the tagged release; add a DOI if
  the GitHub release is archived, while retaining the mandatory GitHub C2 URL.
- [ ] Confirm data-availability wording against what is actually public.
