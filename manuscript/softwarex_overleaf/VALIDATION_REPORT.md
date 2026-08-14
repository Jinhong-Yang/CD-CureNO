# Validation report

## Overall assessment: Share with caveats

The manuscript is structurally aligned with the official SoftwareX original
article template (Version 6, March 2026), compiles as an Elsevier `elsarticle`
manuscript, and confines scientific statements to stored P0-P5 evidence. It is
not yet administratively submission-ready because the CRediT and conflict
statements remain unresolved. Authorship, affiliations, corresponding/support
email, funding, acknowledgements, Apache-2.0 open-source license, public GitHub
URL, and documentation URL have been supplied.

## Methodology review

- Audience: research-software reviewers and computational materials users.
- Question answered: what the software does, why it is needed, how it is used,
  and which stored evidence validates its current functionality.
- Main claim class: software architecture, safeguards, reproducibility, and
  staged validation; not final P6 superiority.
- Evidence cutoff: the P0--P5 artifacts packaged in public release `v0.0.1`.
  Existing P6 outputs are excluded from the public release and are not used
  for empirical manuscript claims.
- Display selection: case 100, fixed by identifier and documented in the P2
  figure contract; no performance-ranked case selection.

## Calculation spot checks

- P2 model counts and metrics: verified against
  `outputs/tables/p2_extended_seed0_runs.csv`.
- P3 public solver metrics: verified against
  `outputs/tables/p3_gate_summary.json` and `analysis/p3_validation.md`.
- P4 case count, shape, energy residual, and replay: verified against
  `outputs/tables/p4_gate_summary.json` and `analysis/p4_validation.md`.
- P5 causal tensor mapping, restriction, and future invariance: verified
  against `outputs/tables/p5_causal_restriction_gate.json`.
- P5 pilot results are described only as a one-seed validation gate, consistent
  with `analysis/p5_pilot_results.md`.
- No value from a P6 ID-test/OOD statistics bundle is used; that bundle does
  not exist in the inspected workspace.

## Visualization review

- The architecture diagram uses a left-to-right process flow and a separate
  guardrail band; it does not encode quantitative magnitudes.
- The included P2 temperature heatmap has common temperature/error scales,
  units, case/seed labels, and a sign explanation. Its frozen QA contract and
  SHA-256 are stored in `analysis/p2_figure_contract.md`.
- The caption and adjacent text identify the display as 1+1-D, one-seed, and
  descriptive.

## Material caveats

1. **Medium:** the public GitHub release, permanent tag URL, README
   documentation, and Apache-2.0 license satisfy the software-sharing package
   structure, subject to publisher review.
2. **Medium:** author identity, affiliations, corresponding/support email,
   funding, acknowledgements, and Apache-2.0 license are supplied, but CRediT
   roles and competing-interest confirmation remain author-level decisions.
3. **Medium:** the upstream ResFNO license remains `NOASSERTION`; public release
   packaging must not silently redistribute upstream code/data.
4. **Medium:** P6 confirmatory held-out evidence and final external validation
   are unavailable; the article therefore avoids final superiority claims.
5. **Medium:** the canonical two-dimensional labels are generated numerical
   fields; the BDF cross-check shares physical inputs and is not a fully
   independent external solver.
6. **Low:** the official SoftwareX source template is DOCX, so the package maps
   its required structure into Elsevier's accepted `elsarticle` class for
   Overleaf rather than reproducing Word styling.

## Required final action

The package compiled locally to a 13-page PDF with no unresolved citations,
references, LaTeX errors, or overfull boxes. All 13 final rendered pages were
visually inspected for clipping, overlap, table fit, figure legibility, and
placeholder visibility. The placeholder-tolerant structural checker passed
with 2,060 estimated words, two figure environments, and two intentionally
unresolved human-input markers.

Resolve every item in `SUBMISSION_CHECKLIST.md`, run the strict checker, and
perform one final clean Overleaf build after the human metadata is supplied.
