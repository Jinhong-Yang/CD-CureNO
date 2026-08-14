"""Evaluate the pinned upstream x=35 checkpoint on a frozen manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cdcureno.evaluation.checkpoint import evaluate_upstream_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a pinned upstream ResFNO checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("external/ResFNO/logs/ResFNO_T_X35_net_params.pkl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("splits/p1_common_holdout_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/audit/upstream_checkpoint_x35.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    checkpoint = (project_root / args.checkpoint).resolve()
    manifest = (project_root / args.manifest).resolve()
    output = (project_root / args.output).resolve()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "checkpoint": str(checkpoint),
                    "checkpoint_exists": checkpoint.is_file(),
                    "manifest": str(manifest),
                    "manifest_exists": manifest.is_file(),
                    "output": str(output),
                },
                indent=2,
            )
        )
        return 0
    result = evaluate_upstream_checkpoint(
        project_root, checkpoint, manifest, location=35
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
