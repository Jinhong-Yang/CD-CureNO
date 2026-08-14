"""Inflate the completed causal P3 source into the causal P5 target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cdcureno.models.causal_checkpoint_inflation import (
    inflate_causal_checkpoint_file,
)


ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Copy the structurally causal P3 source into the distinct causal "
            "true-2-D target, audit every tensor hash, and verify restriction "
            "and counterfactual future invariance without target labels."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "runs"
            / "p3-source-causal-v1-seed0-fix1"
            / "checkpoints"
            / "best.pt"
        ),
    )
    parser.add_argument(
        "--target-config",
        type=Path,
        default=ROOT / "configs" / "model" / "cdcureno_causal_2d.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "checkpoints"
            / "p5_causal_inflated_source.pt"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "tables"
            / "p5_causal_checkpoint_inflation.json"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inflate and run all gates in memory without writing artifacts.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace existing causal inflation artifacts.",
    )
    parser.add_argument("--print-report", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = inflate_causal_checkpoint_file(
        args.source,
        args.target_config,
        args.output,
        args.report,
        seed=args.seed,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
    if args.print_report:
        output = report
    else:
        restriction = report["verification"]["restriction"]
        output = {
            "passed": report["passed"],
            "dry_run": args.dry_run,
            "source_checkpoint_sha256": report["source"][
                "checkpoint_sha256"
            ],
            "target_config_sha256": report["target"]["config_sha256"],
            "source_family": report["source"]["family"],
            "target_family": report["target"]["family"],
            "temporal_family": report["target"]["temporal_family"],
            "copied_tensor_count": report["tensor_mapping"]["copied_count"],
            "initialized_tensor_count": report["tensor_mapping"][
                "initialized_count"
            ],
            "tested_nx": restriction["tested_nx"],
            "maximum_future_prefix_difference": report["verification"][
                "future_invariance"
            ]["maximum_prefix_abs_difference"],
            "target_checkpoint": report["artifacts"]["target_checkpoint"],
            "report": report["artifacts"]["report"],
        }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
