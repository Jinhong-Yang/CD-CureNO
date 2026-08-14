# P2 joint-field figure contract

## Analytical question

For fixed Case1 test case 100, how do the frozen seed-0 factorized and causal
joint operators reproduce temperature and degree-of-cure fields, and where do
their absolute errors differ?

## Selection and grain

- Case 100 is selected by identifier, not error rank.
- Both models use their validation-selected best checkpoints.
- Grain: one complete case with 223 time points and 51 positions.
- Sources: frozen physical predictions under the two P2 extended run IDs.

## Figure design

- Separate temperature and α figures to keep units and scales unambiguous.
- Top row: truth, factorized prediction, causal prediction on one shared field
  scale.
- Bottom row: factorized absolute error, causal absolute error on one shared
  error scale, then causal-minus-factorized absolute-error difference.
- Negative difference means the causal model has lower absolute error.
- Temperature uses Kelvin; α is dimensionless.

## Export and QA

- Static PNG at 200 dpi and vector PDF under `outputs/figures/`.
- Inspect both PNGs for clipping, readable color bars, correct case/seed labels,
  shared scales, units, and the sign explanation before freezing.
- The figures are descriptive one-case views and cannot support statistical
  superiority or true-2-D claims.

## Frozen QA record

- Inspection date: 2026-07-23.
- Result: both PNGs passed visual inspection after moving adjacent error
  color-bar labels to short top titles.
- Verified: no clipping or overlapping labels, readable units and sign
  explanation, shared model-comparison scales, correct case/seed labels, and
  explicit tool/composite structure in α.
- Temperature PNG SHA-256:
  `351C85E8632978991CCE6DDA08513B75CE88B5E73A2DEC97372BE4338C8D313A`.
- Alpha PNG SHA-256:
  `A6AA9DAD297D64A3F232B25AADBCE0D0370D91455E4E65AE87DD7FB8190D9D02`.
