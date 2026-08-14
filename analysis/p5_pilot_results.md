# P5 RP-FFNO target-pilot result

Status: **passed the pre-registered one-seed validation-only stage gate**.
This is not a confirmatory superiority result and does not use target ID-test
or OOD label values.

## Frozen comparison

- methods: scratch target RP-FFNO and restriction-preserving source inflation;
- target label budgets: nested prefixes 8 and 16;
- seed: 0;
- target validation cases: frozen IDs 256 through 287;
- parameter count: 172,610 for every run;
- trainability: full T2 for every run;
- maximum epochs: 120, validation-only selection, patience 20 after epoch 40;
- full-resolution micro-batch 2, accumulation 2, effective batch 4;
- implementation SHA-256:
  `65ed1e956118e0cdf6ab2dff892c7bd44224316f297ecc59a91538cf015df6a0`.

All four runs record Git commit
`7bbceb2dc78782f7ad00d174e0e11227cff442a2`. Shared target files were
checksum-read and memory-mapped, but ID-test/OOD label values were not indexed,
materialized, evaluated, or used for checkpoint selection.

## Validation results

| Budget | Initialization | Selected / executed epochs | T relative L2 mean | Alpha relative L2 mean | Peak error mean (K) |
|---:|---|---:|---:|---:|---:|
| 8 | scratch | 117 / 120 | 0.00378278 | 0.288232 | 1.30668 |
| 8 | restriction transfer | 65 / 85 | 0.00279367 | 0.0171695 | 0.854004 |
| 16 | scratch | 119 / 120 | 0.00308202 | 0.147662 | 1.45280 |
| 16 | restriction transfer | 108 / 120 | 0.00260857 | 0.0167332 | 0.744281 |

Transfer reduced mean temperature relative L2 by `26.15%` at budget 8 and
`15.36%` at budget 16. At budget 16, the paired case-wise median relative
improvement was `25.24%`; transfer was better on 28 of 32 validation cases.
The alpha transfer/scratch ratio was `0.1133`, and the peak-error ratio was
`0.5123`, so both pre-registered 10% guardrails passed by improvement rather
than tolerated degradation.

## Integrity and restriction checks

The independent aggregator:

- reproduced every saved per-case metric from the physical-unit prediction
  arrays, material mask, and `x/z` coordinates;
- checked history continuity and that the selected epoch was the validation
  argmin;
- verified best/last checkpoint, history, prediction, per-case, and
  restriction-report hashes;
- required exact equality of the frozen YAML, data/checkpoint hashes, resource
  profile, implementation hash, model, optimizer, losses, and label access
  within each matched pair;
- bound the pilot to the independently recreated 61-tensor inflation and
  actual P3/P4 F0 restriction gate;
- required the selected-model homogeneous lateral-invariance score to be at
  most `1e-6` and normalized lateral range to be at most `1e-5`.

All 13 aggregate checks passed. Machine-readable evidence is
`outputs/tables/p5_rp_ffno_pilot_gate.json`, SHA-256
`366107f63b74a70e5a5dacc422cab7c00c7eb135e0d3c521789dcc651db0eea3`.

## Decision

P5 passes and unblocks P6 pre-registration. The demonstrated claim is limited
to a one-seed validation-stage benefit for the noncausal RP-FFNO
initialization. The final causal CD-CureNO comparison still requires at least
five seeds, the frozen multi-budget protocol, and the pre-registered ID/OOD
statistical analysis.

