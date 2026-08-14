# P1 temperature-field figure contract

## Analytical question

For one fixed, non-performance-selected Case1 test sample, how do the exact and
corrected position-wise ResFNO stacks reproduce the through-thickness
temperature field, and where are their errors concentrated?

## Takeaway boundary

The figure may show spatial/temporal error structure but may not imply that the
legacy stack is a spatially coupled 2-D operator. It also may not claim
statistical superiority from one displayed case.

## Selection and grain

- Fixed case ID: 100, chosen by identifier rather than error rank.
- Exact field: seed 1.
- Corrected field: seed 1, enabling a same-seed descriptive comparison.
- Grain: 51 through-thickness positions × 223 time points.
- Source: frozen `temperature_field.npz` artifacts referenced by tracked field
  summaries.

## Chart family and surface

- Family: matrix/heatmap small multiples.
- Static Matplotlib export for manuscript use.
- Panels: truth, exact prediction, exact error, corrected prediction, corrected
  error, and exact-vs-corrected absolute-error difference.

## Palette and scale policy

- Temperature panels: one perceptually uniform root (`viridis`) with identical
  Kelvin limits.
- Error panels: diverging `coolwarm`, centered at zero with one shared symmetric
  limit.
- Error-difference panel: diverging, centered at zero; negative means corrected
  has lower absolute error.
- Color is supplemented by panel titles and labeled color bars.

## Export and QA

- PNG at 200 dpi and vector PDF under `outputs/figures/`.
- Inspect the exported PNG for clipping, legibility, shared scales, units, and
  correct case/seed labels before freezing.

## Frozen QA record

- Inspection date: 2026-07-23.
- Result: passed visual inspection after layout iteration.
- Verified: no clipped labels, readable panel and color-bar text, shared
  temperature and prediction-error scales, correct Kelvin/mm/min units, and
  explicit case/seed/uncoupled-model context.
- Selection remains fixed at case 100; no result-dependent case selection was
  introduced during iteration.

## Versioned common-case correction

The displayed case 100 is in both saved seed-1 populations, so the frozen
figure and its visual QA remain valid. The aggregate 2.04% improvement statement
does not: the exact summary averaged cases 50--199, whereas the corrected
summary averaged cases 75--199. The versioned comparison therefore intersects
and aligns the 125 common IDs before computing any paired result.

On cases 75--199, exact versus corrected mean relative L2 is
`0.0020070584` versus `0.0020275023` for the full field,
`0.0018733012` versus `0.0019000263` for the composite, and
`0.0021625017` versus `0.0021782592` for the tool. Corrected is better on
46/125, 44/125, and 52/125 cases, respectively. These are descriptive seed-1
results and support no corrected-accuracy gain. The immutable source hashes,
case membership, and full-precision values are recorded in
`outputs/tables/p1_common125_correction_v2.json`; all v1 artifacts remain
preserved.
