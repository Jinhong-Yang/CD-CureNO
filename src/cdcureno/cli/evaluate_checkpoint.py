"""Evaluate a completed run checkpoint on a frozen manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cdcureno.evaluation.checkpoint import evaluate_run_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a selected run checkpoint on a frozen case manifest."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[3]
    run_dir = (project_root / args.output_root / args.run_id).resolve()
    manifest = (project_root / args.manifest).resolve()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "run_exists": run_dir.is_dir(),
                    "manifest": str(manifest),
                    "manifest_exists": manifest.is_file(),
                },
                indent=2,
            )
        )
        return 0
    result = evaluate_run_checkpoint(run_dir, manifest, device=args.device)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
