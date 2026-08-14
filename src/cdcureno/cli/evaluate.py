"""Verify a completed run by recomputing metrics from saved predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cdcureno.evaluation.run_verification import verify_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independently verify saved complete-case run metrics."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[3]
    run_dir = (project_root / args.output_root / args.run_id).resolve()
    if args.dry_run:
        print(json.dumps({"run_dir": str(run_dir), "exists": run_dir.is_dir()}, indent=2))
        return 0
    verification = verify_run(run_dir)
    (run_dir / "verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0 if verification["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
