# Measured resource requirements and canonical-array replay

Measurements taken on 2026-09-08; independent artifact audit completed on the
same date. Both executions started with fresh output directories. Setup,
dependency installation, and downloads are excluded. These are observed
single-run costs on one workstation, not minimum hardware requirements or
guarantees for other systems.

## Complete-command resource measurements

| Workflow | Full command wall time | Sampled peak process-tree RSS | Saved output size |
|---|---:|---:|---:|
| Full canonical P4, 512 cases | 214.5049962 s (about 3 min 35 s) | 1,410,904,064 bytes (1.31401 GiB) | Eight NPY files: 917,819,936 bytes; all generation artifacts: 926,166,622 bytes |
| Public end-to-end true-2-D demonstration, final `clean_linux_v2` | 9.785031148 s | 751,235,072 bytes (716.434 MiB) | 403,458 bytes across 23 output files |

Decimal MB/GB and binary MiB/GiB are distinct: the eight P4 NPY files total
917.820 MB, or 875.301 MiB. The memory figures above use binary GiB/MiB with
exact bytes retained for audit.

The workstation was an Intel Core Ultra 7 265K, with 20 logical CPUs reported
and nominally 64 GiB host RAM. P4 ran on Windows build 26200 with a clean
Python 3.11.9 environment, NumPy 2.2.6, SciPy 1.15.3, and PyYAML 6.0.2.
It used four generator workers, maximum eight in-flight cases, and one
BLAS/OpenMP thread per process. The profiler observed up to eight live
processes, including command-launch and multiprocessing support processes.
Visible host RAM was 68,022,812,672 bytes; available RAM at launch was
9,979,523,072 bytes.

The final demonstration ran in a clean Ubuntu WSL2 Python 3.12.3 environment on the
same host, with PyTorch 2.13.0+cpu, NumPy 2.2.6, and one computation thread.
The WSL environment exposed 33,308,184,576 bytes of memory. Both measurements
were CPU-only; unrelated GPU work was active on the host. The times are
therefore observations under the recorded concurrent workload, without a
repeat-run timing distribution.

## Memory measurement method and limitations

The profiler used psutil 7.2.2 to sample the sum of RSS for the launched command
and its live descendants at a nominal interval of 0.05 s. Raw sample counts
were 3,504 for P4 and 193 for the final demonstration. An independent audit of the
CSV samples reproduced each profile's peak RSS, sample count, and maximum
live-process count.

Summed RSS can count shared resident pages more than once; short peaks between
samples can be missed. Polling and process inspection introduce scheduling
overhead, so 0.05 s is the requested interval rather than a guarantee of exact
sample spacing. These values are sampled aggregate working-set measurements,
not exclusive physical memory, peak virtual address space, GPU memory, or
minimum RAM requirements. The profiler's parent process and unrelated host
processes are outside the measured command tree.

## Full command time versus internal timers

The P4 result `generation_summary.json` contains an internal timer of
**212.5802220 s**. The generator captures this value before `_finalize`, which
builds and hashes final payloads and publishes the output directory, metadata,
and manifest. The external profiler's **214.5049962 s** covers the complete
CLI, including startup, finalization, and process exit. Use the latter for a
reader's complete regeneration budget; do not substitute the internal timer.

The final demonstration's **9.785031148 s** likewise covers the complete command:
data generation, source and target training, checkpoint reload, evaluation,
plot generation, and postflight verification. Its internal
`pre_postflight_wall_seconds=7.311793575` does not include the complete workflow
and should not replace the profiled command duration. Individual internal
stage times are preserved in the copied `demo_v2_run_summary.json`.

## What was regenerated and verified

P4 used the original deterministic plan
`splits/p4_2d_core_v1_plan.json`, with frozen plan hash
`a97511e192450a11d3af3c966bae3fd26523b3eb12b8df5728ca49b25d55b0bb`.
All **512 cases passed**, no cases failed, and all **14 generation acceptance
checks** passed. This 14-item generation summary is distinct from the original
manuscript's 27 solver checks and from the new mathematical verification suite.

An independent read-only audit streamed every byte of all eight generated
`.npy` files through SHA-256 without importing the generator's hash function.
Each computed file hash matches both the newly generated manifest and the
original frozen manifest, `splits/p4_2d_core_v1.json`:

| Array file | File bytes | SHA-256 |
|---|---:|---|
| `air_temperature_K.npy` | 229,504 | `9b63fdaedbe3c598d47645567cbfc61c7ab38c67b332a836c71463c844f4d270` |
| `alpha.npy` | 458,752,128 | `229162bce49c3e5b108b7e9f5b279c90be64605ca9f385f83435a280061f6f06` |
| `composite_mask.npy` | 2,128 | `023bf5c66343c9212f9f04203ee09e4e9d1a3bdbc412364eedc41f49fd976af3` |
| `temperature_K.npy` | 458,752,128 | `2a6aa1f75ee91041e296522649c57f94d5225ffb60a9a07c50899e1b0619e18b` |
| `time_s.npy` | 1,024 | `b193db76ab3e4833ca0ba0c250ab9a34f857a57b0847c7b80d9c4f8e6b620762` |
| `top_h_W_m2_K.npy` | 82,048 | `19dd34d37de9faef1348d6f47346e5d8a172bec32195e9bcb6caa596223a0ac8` |
| `x_m.npy` | 448 | `3889c82360e38e5ae507d634b5258f8e4bc4d29bc276b215953eeae3cdba0854` |
| `z_m.npy` | 528 | `0e70b3ed41ee2c76017f6182c3bb557a4aca5169d52051acf7d07cba68ac702d` |

The two principal float32 fields have shape `[512,112,50,40]` each. Their raw
payload alone is 917,504,000 bytes. That payload count excludes NPY headers and
the six ancillary arrays; it is not the total storage figure.

The complete `core_arrays` directory occupies **924,206,207 logical file
bytes**, including the eight arrays, plan, metadata, manifest, summary, and
512 case-completion receipts. The full generation directory occupies
**926,166,622 bytes**, including the additional published copies of metadata,
manifest, and summary. The external profiling directory adds **121,407 bytes**.
These are final file sizes, not peak temporary disk use or filesystem allocated
block sizes. Environments, code checkout, dependency caches, and profiler files
are excluded from the 926,166,622-byte generated-artifact total.

For the final demonstration, all output files total **403,458 bytes**, including
data, source/inflated/target checkpoints, histories, predictions, metrics,
figure, provenance, and postflight artifacts. External profiling files add
**12,887 bytes**. Its small output and runtime do not describe the cost of
reproducing historical P3/P5 results or the full P4 benchmark.

## Manifest identity and provenance boundaries

**The regenerated full manifest is not byte-identical to the original.** Four
top-level fields differ: `array_artifact_root`, `metadata_path`,
`generation_summary_path`, and `generation_summary_sha256`. Output paths were
deliberately redirected to a separate profiling directory, and the new
generation summary records this run's time and resources. These differences
must not be hidden under a claim that the entire manifest replayed identically.

All other manifest fields compare equal, including the eight array file
hashes, all 512 case-definition and case-artifact hash records, split map,
plan hash, and metadata-content hash. The new metadata and generation-summary
bytes independently match their respective hashes in the new manifest.

The actual profiled source was a clean checkout of submitted release commit
`d9ee97f2d231d7aa2088139548a7352dfcb51efe`; tracked source was unchanged during
execution. The frozen plan retains its historical generation-origin commit
`6d0b31f63428e00524a89445e9f4d7148f46d8f7`. These are different provenance
roles. Hashes of the generator, production 2-D solver, and material-law module
match across the generation record, baseline checkout, and current revision
checkout. No generator, solver, configuration, frozen plan, or acceptance guard
was changed for this measurement.

The successful byte replay applies to the recorded **Windows/Python 3.11.9**
environment. The earlier Linux/Python 3.12.3 attempt failed before generation
because of exact-plan/provenance requirements. See
`docs/revision_canonical_replay.md`; these measurements do not establish
cross-platform byte-identical regeneration.

## Final demonstration and preserved earlier execution

The final `clean_linux_v2` run follows the shared-channel contiguity fix in
the target model. Its predeclared configuration hash remains
`4ab10a2bf3a5094c411a7e96228b572d53a353d195073139cf1139c683a6b696`.
The recorded source commit is
`5ccaa3cb86cfc6fdb810b2301ede9651ccb9a9d9`, with the source-file hashes and
working-tree context preserved in `demo_v2_provenance.json` and the profile.
The per-case temperature-relative-L2, temperature-RMSE, and alpha-RMSE CSV
values match the earlier execution exactly when parsed as floats. This is
not a claim of identical checkpoint, metadata, or full-output bytes.

The earlier `clean_linux_v1` measurement is retained as pre-fix evidence:
10.284674577 s, 752,074,752 bytes sampled peak tree RSS, and 403,456 output
bytes. Its original `demo_profile.json`, `demo_memory_samples.csv`,
`demo_stdout.log`, `demo_run_summary.json`, and `demo_provenance.json` copies
remain unchanged. The prior audit is saved in `summary_demo_v1.json`, and
the earlier audit helper is retained as `audit_measurements_v1.py`.
Neither timing was chosen on performance grounds: final v2 is used because
it corresponds to the final target-model implementation.

## Reproducible evidence package

`outputs/revision_resource_measurements/summary.json` records the independent
audit, storage inventories, source hashes, exact manifest differences, and
original evidence paths. Its `demo` entry selects final v2; `demo_previous`
preserves the v1 measurement. The folder contains the P4 profile and both
demonstration versions, their raw memory CSVs and stdout, the original/new
P4 manifests, P4 generation summary, and demonstration provenance/stage
summaries. Final demo copies use the explicit `demo_v2_` prefix. `checksums.json` binds
these copies, and `audit_measurements.py` documents the read-only procedure.
The approximately 918 MB arrays themselves are not duplicated into this compact
measurement-evidence directory.

No additional full generation or demonstration run was performed for this
audit update. The main resource table uses the completed P4 execution and
final demo v2 execution, with the earlier demo v1 evidence retained separately.
