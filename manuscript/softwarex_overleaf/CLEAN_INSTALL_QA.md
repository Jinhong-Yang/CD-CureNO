# CD-CureNO v0.0.2 clean-install QA

Date: 2026-08-16
Scope: release engineering only; no training, label generation, checkpoint
inflation, or scientific experiment was executed.

## Environment

- Platform: Windows 10.0.26200
- Python: 3.11.9
- CD-CureNO: 0.0.2
- NumPy: 2.2.6
- SciPy: 1.15.3
- PyYAML: 6.0.2
- PyTorch: 2.13.0+cpu
- pytest: 9.1.1
- CUDA available: false

The first virtual environment was created under the deeply nested project
path. Pip built `cdcureno-0.0.2` successfully but Windows rejected a deeply
nested PyTorch license path with `WinError 206`. The source was unchanged and
installation was repeated at the short temporary path
`C:\qa_cdcureno_v002_20260816`, where it completed successfully. The public
README now advises Windows users to enable long paths or use a short virtual-
environment path.

## Commands and results

```text
python -m pip install -e ".[dev]"
RESULT: PASS; installed cdcureno-0.0.2 and all pinned dependencies

python -m pytest -q
RESULT: PASS; 210 passed, 30 skipped in 42.40 s
```

Skip reasons were explicit:

- 28 upstream-dependent skips: separately acquired ResFNO code or Case1 was
  unavailable.
- 1 P4/P3 integration smoke skip: generated canonical arrays or source
  checkpoint unavailable.
- 1 CUDA-device regression skip on the CPU-only environment.

```text
python -m pytest -q tests/unit/test_p4_dataset_pipeline.py
RESULT: PASS; 17 passed in 3.17 s

python scripts/generate_p4_2d.py --tier core --plan-only \
  --plan tmp/p4_2d_core_v1_plan_review_v002.json
RESULT: PASS; 512-case plan verified; labels_generated=false
review plan SHA-256: 2352560e2f13c41fee9c790bac05c9c8a1b0d8297d6f72360c9385026c91f4ab

python scripts/inflate_causal_1d_to_2d_checkpoint.py --help
RESULT: PASS

python scripts/verify_p5_causal_restriction.py --help
RESULT: PASS
```

The default post-freeze `--plan-only` path correctly refused to overwrite the
non-identical canonical plan because release provenance advanced from the
pre-label commit. A separate review-plan path was therefore used. Relative to
the immutable canonical plan, the review copy had zero case-definition
differences and zero code-hash differences; only the expected Git provenance
and derived plan hash differed. Production regeneration loads the included
canonical plan instead of recreating it.

## Preserved local log hashes

The raw local QA logs were retained outside the submission ZIP. Their hashes
bind this summary to the executed console records:

- environment: `737f58a735c32156951d91a1d5ea7c1295775b4a21cd97ad4bfac6ea499e412f`
- pip install: `32f78a7823d351857e3c235a7e1637f01c9f1d4254631d7e74b28c4531af7576`
- full pytest: `25be2c954fdbc184a297c448c089a6ffebc5f1ea84df703c637ee2f4468bb87e`
- focused P4 pytest: `c46247c18efc5f11f6f8a9b06eebe489811519ec122b4da88b296511375933d0`
- P4 review plan: `e4ce583da2012c6f4b700a579a54e97a1ca0a0bf54c976ca54bb3dd393524f25`
- P5 inflation help: `a885530f1e9382f5ddeca8c771d45c06953625d517b835c20c58b8db5bd8dc90`
- P5 verification help: `436b2f185462033d5ac7f254909463891ba23208b2f2711c2bcc6475a102f345`
