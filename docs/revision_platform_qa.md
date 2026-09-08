# Revision platform QA

## Scope and baseline

The submitted release is `v0.0.2`, commit
`d9ee97f2d231d7aa2088139548a7352dfcb51efe`. Its preserved manual QA
record reports Windows `10.0.26200`, Python 3.11.9, CPU PyTorch 2.13.0,
and **210 passed, 30 skipped in 42.40 s**. This is a historical Windows
record, not a Linux result or a count for the revised source.

The historical 30 skips were 28 tests requiring separately acquired upstream
ResFNO code or Case1 data, one P4/P3 integration smoke test requiring
unbundled generated arrays or a source checkpoint, and one CUDA-only device
regression. A skipped test is not verified functionality. Counts from new
environments must be reported with their actual reasons rather than forced
to match this historical count.

## Canonical path correction

`scripts/freeze_p4_splits.py` now checks `PureWindowsPath(value).drive`
in addition to POSIX absolute-path and normalization checks. This rejects
Windows absolute paths, drive-relative paths such as `C:relative/file.json`,
and bare drive prefixes on Linux as well as Windows. The check runs before
file loading, preserving the intended `ValueError` for a noncanonical path.

The existing integration regression was extended with lowercase drive,
drive-relative, bare-drive, POSIX absolute, UNC, dot-prefixed, and doubled
separator inputs. These cases exercise full manifest validation rather than
reimplementing the resolver in the test. The existing successful immutable
manifest-freeze integration test continues to cover valid repository-relative
paths.

## Automated checks

`.github/workflows/tests.yml` defines a Linux CPU job on Ubuntu 24.04 and
Python 3.11. It reads the PyTorch version from `pyproject.toml`, installs that
version from the CPU wheel index, installs the remaining pinned package
dependencies, and rejects a PyTorch build compiled with CUDA. The job runs
the full suite with explicit skip reasons and JUnit output. It retains
installation logs, source identity, environment details, test results and
skip reasons as a GitHub Actions artifact for 30 days, including when a
preceding step fails.

The workflow is configured for pushes, pull requests and manual dispatch.
Its presence does not establish a successful remote run. Record the actual
run URL and tested commit below after execution. Python versions or operating
systems absent from this job are not represented as CI-tested.

## Revision execution evidence

The Windows checks below were executed on 2026-09-08 using the **reused**
`C:/qa_cdcureno_v002_20260816` environment, Python 3.11.9 and
PyTorch `2.13.0+cpu` (compiled CUDA: `None`). This is not a fresh Windows
installation. `PYTHONPATH` selected each checkout's own `src` directory;
the original and revised source were tested separately. The revised source
had the eight new path regression cases and no new numerical evidence from
the other revision work packages at the time of this run.

The local Linux check used a newly prepared CPU environment with Python
3.12.3 and PyTorch `2.13.0+cpu` on Ubuntu/WSL2,
kernel `6.6.87.2-microsoft-standard-WSL2`, glibc 2.39. The package versions
were recorded in `linux_freeze.txt`; this environment's Python 3.12 differs
from the historical Windows 3.11 and the configured remote CI 3.11.
`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and `OPENBLAS_NUM_THREADS` were all 2.
The Linux revised suite included the eight additional path cases and three
new solver-verification tests, accounting for its 221 passing tests. The
source-file SHA-256 inventory was unchanged between the before/after
snapshots of this execution. Later revisions require their own test record.

A subsequent **fresh Windows environment** at
`C:/cdcureno-r1-20260908/env_windows` was used for the full revised suite:
Windows `10.0.26200`, Python 3.11.9 and PyTorch `2.13.0+cpu`.
The same three CPU thread settings were 2. The environment freeze and
before/after source manifests were retained; the source-file hashes did not
change during this run. This suite included all eight additional path cases,
four solver-verification tests and the public-demo/tampering integration
test: 210 + 8 + 4 + 1 = 223 passing tests. The 30 skips retained their
historical prerequisite categories.

| Check | Source/environment | Result | Evidence |
|---|---|---|---|
| Original reported POSIX regression | Unmodified submitted commit, Linux | 1 failed, 2 passed, 26 deselected; 1.66 s | `linux_baseline_path.log`, `linux_baseline_path.xml` |
| Full original suite | Unmodified submitted commit, Linux | 1 failed, 209 passed, 30 skipped; 37.75 s | `linux_baseline_repaired_pytest.log`, `linux_baseline_repaired_pytest.xml` |
| Full original suite | Submitted commit, reused Windows CPU environment | 210 passed, 30 skipped; 45.08 s | `windows_baseline_pytest.log`, `windows_baseline_pytest.xml` |
| Corrected path integration tests | Revised source, Linux | All 11 invalid-path cases passed in full suite | `linux_revised_pytest.xml` |
| Full revised suite | Revised source, Linux CPU | 221 passed, 30 skipped; 33.37 s | `linux_revised_pytest.log`, `linux_revised_pytest.xml` |
| Full revised suite | Revised source, reused Windows CPU environment | 218 passed, 30 skipped; 36.66 s | `windows_revised_pytest.log`, `windows_revised_pytest.xml` |
| Full revised suite | Revised source, fresh Windows CPU environment | 223 passed, 30 skipped; 57.07 s | `windows_clean_pytest.log`, `windows_clean_pytest.xml` |
| Remote Linux CI | Exact revised commit | Pending | Pending |

Local revision logs are retained outside the source checkout under
`../logs/w1/`. Environment differences from the historical release pins,
including unavailable package versions, must be recorded with the results.
Successful tests in a replacement environment do not retrospectively verify
an unavailable historical environment.

The Windows skip breakdown remained 28 upstream-dependent, one P4/P3
artifact-dependent, and one CUDA-device regression in both runs. The full
test command was `python -m pytest -q -ra --junitxml=<log-path>`.
The YAML workflow parsed successfully with PyYAML; this structural check is
not a substitute for the pending remote Actions execution.

The original Linux failure was exactly the reported drive-prefixed path
reaching `FileNotFoundError` instead of `ValueError`. An earlier full baseline
attempt additionally encountered 14 Git lookup failures because the worktree
metadata contained a Windows-absolute `gitdir` pointer. That pointer was
changed to a relative path, and both Windows and Linux then resolved the same
submitted commit. No baseline source was changed. The initial 15-failure log
is preserved as `linux_baseline_pytest.log`; the corrected-environment run
above isolated the one software regression. This worktree setup issue is not
a claimed defect in the released source.
