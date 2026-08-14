# P5 checkpoint-inflation validation

Status: **architecture/initialization gate passed; the complete P5 gate is not
yet passed**. The pilot must still show a transfer advantage over random
initialization, and the source/target/OOD split audit must remain clean.

## Frozen inputs

- Source checkpoint:
  `outputs/runs/p3-source-factorized-v4-seed0/best.pt`
- Source SHA-256:
  `82d7a497f99a884f2b08eeac799fd7e7ab6391309d3e3037864ecddd8fa3f5f6`
- Target contract: `configs/model/cdcureno_2d.yaml`
- Target-contract SHA-256:
  `6fb813d5d90e6868f132de1d4c6041b3e2d434bdc954ede4408ce30183dc8c7f`
- Machine-readable evidence:
  `outputs/tables/p5_checkpoint_inflation.json`
- Generated checkpoint (ignored by Git):
  `outputs/checkpoints/p5_inflated_source.pt`
- Generated checkpoint SHA-256:
  `b66918640ddd80bbfccfd0d16c3557155ef26cb415bf280ba965c94d9f2264f0`

The verification input is deterministic and synthetic. It does not open any
P4 temperature or cure label.

The target contract pins the exact source checkpoint SHA-256 and inflation
seed. Inflation fails before tensor mapping if either differs, even when an
alternate checkpoint has a compatible architecture.

## Mapping

All 52 source tensors are copied bitwise:

- the first source lift is mapped to a 14-channel shared projection;
- a separate six-channel target-only projection is exactly zero;
- each copied `z` spectral, temporal spectral, local mixing, normalization,
  and head tensor retains its source value;
- each block gains one high-pass low-rank `x` adapter;
- the adapter input factor is deterministic and nonzero, while its output
  factor is exactly zero;
- the `k_x=0` coefficient is never consumed by the adapter.

The split lift is equivalent to a conceptual 20-channel input matrix whose
first 14 columns are copied and whose final six columns are zero. Unlike a
single expanded `Linear`, this representation lets Stage T0 freeze copied
columns completely, including against optimizer weight decay.

## Restriction argument

Let the shared target input be the lateral extrusion of a source input. At
inflation, the target-only projection contributes zero, so the target lift at
every `x` equals the source lift.

Assume block input `h_l(t,z,x)` is independent of `x` and equals the source
hidden state. The copied `z` spectral and spatial-local branches therefore
equal the source branches at every `x`. The lateral branch is zero because its
output factor is zero; independently, its immutable high-pass construction
omits `k_x=0`. The copied GELU, residual addition, and layer normalization thus
produce the source spatial update. The copied temporal spectral and local
branches then produce the source temporal update independently at every `x`.
This proves the induction step. The copied output heads use the same baseline,
material mask, and initial-cure prefix channels, so temperature, cure rate,
and integrated cure equal the source outputs.

`Nx=1` is bitwise equal in the implementation. Batched FFT kernels introduce
small float32 ordering differences for larger `Nx`; these are tested against
field-specific numerical tolerances.

## Measured verification

| `Nx` | max temperature mismatch | max alpha mismatch | max cure-rate mismatch | temperature RVS | temperature LIS |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 0 | 0 | 0 |
| 2 | 1.19e-7 | 1.79e-7 | 6.56e-6 | 3.95e-8 | 3.64e-8 |
| 7 | 1.19e-7 | 1.79e-7 | 6.79e-6 | 7.23e-8 | 5.05e-8 |
| 40 | 1.19e-7 | 2.38e-7 | 8.34e-6 | 7.06e-8 | 5.46e-8 |

The prespecified absolute tolerances are `1e-6` for temperature, temperature
residual, and alpha, and `1e-5` for cure rate.

## Actual frozen-input gate

`scripts/verify_p5_restriction.py` independently loaded actual P3 validation
case `80` and tested extrusions at `Nx=1,2,7,40`. It also constructed the full
`[112,50,40,20]` input for P4 validation F0 case `267` exclusively from the
pre-label plan, exogenous coordinate/mask arrays, and the label-free coarse
1-D solver. It did not open a P4 temperature or cure array.

Across the P3 checks, maximum temperature and alpha mismatches were
`8.94e-8` and `1.19e-7`; maximum temperature RVS and LIS were
`3.43e-8` and `5.97e-8`. On the real P4 F0 input, temperature mismatch,
RVS, and LIS were `5.96e-8`, `3.54e-8`, and `1.86e-8`. `Nx=1` remained
bitwise equal for all four outputs. Every pre-specified restriction check
passed. The complete evidence is
`outputs/tables/p5_restriction_gate.json`.

The gate also recreates the complete untrained target state from the pinned
source checkpoint, target contract, and seed. All 61 target tensors must match
bitwise, the embedded copied/initialized tensor map and hashes must match the
recreation, the six-channel lift must remain zero, every lateral input factor
must retain its deterministic nonzero value, and every lateral output factor
must remain exactly zero. This prevents a trained or manually modified
checkpoint from passing merely because homogeneous inputs hide its high-pass
lateral branch. Dry-run performs the same parse, binding, tensor, and
frozen-input checks; it only suppresses report writing.

## Training stages and runtime smoke

- T0: target-only lift plus lateral adapters;
- T1: T0 modules plus output heads and copied channel mixers;
- T2: all parameters.

On an NVIDIA GeForce RTX 5080 with PyTorch `2.13.0+cu130`, a label-free
`[1,9,7,5,20]` T0 forward/backward smoke produced finite loss, a nonzero
lateral output-factor gradient, 12,480 trainable parameters out of 172,610,
and 70,435,328 bytes peak allocated CUDA memory.

## Separate causal production path

The selected P3 source uses a noncausal temporal FFT. Exact inflation therefore
produces a target marked `causal: false`. The CLI rejects `causal: true` or a
different temporal family rather than silently claiming an impossible exact
mapping. This RP-FFNO path remains limited to the one-seed P5 stage pilot.

The production causal path is now implemented separately. Causal source
checkpoint
`81bde9a8eb9c23404ea4011cead994be9162abe62ed2cf789fae362be708a0c4`
was inflated under causal target configuration
`bb82c56a7cb2bfc21c392bdada4abe1dd7b4795a32115cdbab6aa569ba12fdab`
to checkpoint
`e8626ff782a2e1f079f013300656d88022ab8cf7bc627618c972af522443084f`.
All 100 source tensors were copied bitwise, 17 tensors were initialized, and
the complete 117-tensor state was independently recreated from
source/config/seed. The target contains eight causal temporal convolutions and
zero temporal FFT modules. `Nx=1` was bitwise equal for all four outputs,
`Nx=2,7,40` passed the numerical bounds, and all-channel future perturbations
changed every tested prefix by exactly zero. The actual P4 F0 case 267 also
passed without reading a P4 temperature or cure label. Evidence:
`outputs/tables/p5_causal_checkpoint_inflation.json` and
`outputs/tables/p5_causal_restriction_gate.json`.

## Reproduction

```powershell
$env:PYTHONPATH='src'
.venv\Scripts\python.exe scripts\inflate_1d_to_2d_checkpoint.py --overwrite
.venv\Scripts\python.exe scripts\verify_p5_restriction.py
.venv\Scripts\python.exe -m pytest -q `
  tests/unit/test_checkpoint_inflation.py `
  tests/scientific/test_restriction_preserving_transfer.py
```
