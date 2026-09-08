# Public CPU example: solver → source → transfer → true 2-D training → independent metric check

This small example demonstrates functionality and reproducibility. It is not the
canonical P3/P4/P5 workflow, does not reuse or alter their frozen populations, and
does not provide evidence of predictive superiority or a P6 result. It needs no
upstream ResFNO files, private checkpoint, GPU, or downloaded dataset.

After installing the pinned release dependencies, run from the repository root:

```bash
python scripts/run_public_demo.py --config configs/demo/public_demo_v1.json --output outputs/public_demo/run_v1
python scripts/verify_public_demo.py --input outputs/public_demo/run_v1
```

Use a new empty output directory for each run. Existing evidence is never silently
replaced. The example uses CPU only with one PyTorch thread. The tested environment,
run time and numerical outputs accompany the archived example. Larger systems and
the full 512-case benchmark have different resource requirements.

The frozen configuration specifies a 6 × 10 cell-centred x-z grid and 17 output
times over 1200 seconds. Eight target cases have different imposed air-temperature
ramps and lateral heat-transfer profiles. Cases 0–3 are training, 4–5 validation,
and 6–7 held out. All labels come from the production `simulate_cure_2d` solver
with the public material/kinetics definitions and project-generated conditions.
The upper Robin coefficient varies in x; the cases are not 1-D fields relabeled
as two-dimensional data. The lateral faces are insulated.

Six homogeneous source cases use the same production solver and z grid. Their
x-invariant fields are reduced to (t,z) without interpolation. The source model
is the production `CausalFactorizedOperator`; the target is the production
`CausalAxisFactorized2DOperator`, both width 8 and depth 2. The example calls the
production tensor-copy/zero-initialization helper `_copy_source_state` and records
every copied and initialized tensor. Its demo checkpoint format and receipt are
explicitly separate from the canonical P5 checkpoint/provenance protocol.

The seven shared input channels are imposed air temperature, a causal lumped
temperature baseline, z coordinate, time, composite mask, mean upper Robin
coefficient, and initial degree of cure. The two added channels are x coordinate
and the local upper Robin coefficient. The baseline is a first-order response
with a fixed 600-second time constant, updated from previous air inputs only;
it is a simple illustrative input, not a temperature label or a calibrated cure
model. The source-training temperature mean and standard deviation normalize the
two temperature input channels and the temperature training target for both models.
Validation and held-out cases do not fit any normalization statistic.

Training uses full-batch Adam for 32 source and 48 target epochs with a fixed seed
and learning rate 0.003. The objective is normalized temperature MSE plus
composite-region degree-of-cure MSE. The minimum validation objective selects each
checkpoint; all epochs and the selected epoch are recorded. The selected target
checkpoint is then reloaded in a separate process, which first opens the held-out
demo population and saves its predictions, per-case metrics and an x-z figure.

`verify_public_demo.py` imports NumPy but neither PyTorch nor the trainer/model
modules. It checks the artifact hashes, split identities and normalization, then
recalculates composite-region metrics directly from saved fields using physical
cell volumes and trapezoidal time weights. These metrics are temperature relative
L2, temperature RMSE in K, and degree-of-cure RMSE. Its receipt distinguishes
direct calculations from training/inflation/causality checks reported by their
respective execution receipts. This is separate postflight implementation within
the same project, not reproduction by an unaffiliated team or physical validation.

The fixed acceptance conditions concern finite outputs and changed trained
parameters, disjoint complete cases, training-only normalization, nonzero lateral
variation, cure bounds/monotonicity, inflation restriction, future-input prefix
invariance and agreement of independently recomputed metrics. **No prediction
accuracy threshold is used to decide whether the example passed.** The small
training budget is intended to exercise the workflow, not achieve a publishable
accuracy target. Float tolerances are frozen in the configuration; bitwise model
or NPZ-file identity across operating systems is not promised.

## Expected artifacts

| Artifact | Purpose |
|---|---|
| `config.json`, `provenance.json` | Frozen conditions, code/config hashes, actual CPU environment |
| `source_{train,validation}.npz`, `target_{train,validation,heldout}.npz` | Public generated inputs and labels, explicit case IDs |
| `grid.npz`, `solver_diagnostics.json` | Coordinates, volume weights, masks, solver convergence records |
| `normalization.json` | Recomputable source-training-only statistics |
| `source_best.pt`, `target_inflated.pt`, `target_best.pt` | Learned source, inflated initialization and learned target |
| `source_history.csv`, `target_history.csv`, `inflation.json` | All losses/selection and tensor-transfer evidence |
| `heldout_predictions.npz`, `per_case_metrics.csv`, `heldout_xz_fields.png` | Held-out outputs, metrics and true x-z temperature/cure views |
| `evaluation.json`, `run_summary.json` | Checkpoint reload/process record, checks and step timings |
| `artifact_manifest.json`, `postflight_receipt.json` | Bound artifact hashes and independent metric-check result |

The run succeeds only if the final postflight receipt reports `"status": "passed"`.
An integration test also changes a reported metric and updates its checksum to
show that independent numerical recomputation detects the false result.
