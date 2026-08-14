"""Reconstruct and score a full legacy field from a completed P1 sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cdcureno.evaluation.field_reconstruction import reconstruct_field


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stack 51 verified location runs into a temperature field."
    )
    parser.add_argument("--sweep-state", type=Path, required=True)
    parser.add_argument(
        "--field-root", type=Path, default=Path("outputs/runs/_fields")
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="Optional small tracked JSON summary path.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    sweep_state = (project_root / args.sweep_state).resolve()
    field_root = (project_root / args.field_root).resolve()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "sweep_state": str(sweep_state),
                    "sweep_state_exists": sweep_state.is_file(),
                    "field_root": str(field_root),
                },
                indent=2,
            )
        )
        return 0
    summary = reconstruct_field(project_root, sweep_state, field_root)
    if args.summary_output:
        summary_path = (project_root / args.summary_output).resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
