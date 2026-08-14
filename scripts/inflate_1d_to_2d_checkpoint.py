"""Inflate and verify the frozen P3 source checkpoint for the P5 target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cdcureno.models.checkpoint_inflation import inflate_checkpoint_file


ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministically copy a spectral P3 FactorizedFNO into the "
            "restriction-preserving true-2-D target and verify a label-free "
            "synthetic extrusion batch."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "runs"
            / "p3-source-factorized-v4-seed0"
            / "best.pt"
        ),
        help="Trusted local P3 checkpoint containing model and channel metadata.",
    )
    parser.add_argument(
        "--target-config",
        type=Path,
        default=ROOT / "configs" / "model" / "cdcureno_2d.yaml",
        help="Frozen target architecture/channel YAML contract.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "checkpoints" / "p5_inflated_source.pt",
        help="Output target checkpoint (large artifact; ignored by Git).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "outputs" / "tables" / "p5_checkpoint_inflation.json",
        help="Detailed JSON tensor-mapping and restriction-verification report.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260726,
        help="Deterministic lateral-factor and verification seed.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu",),
        default="cpu",
        help=(
            "Inflation is intentionally CPU-only so tensor mapping and "
            "verification do not depend on GPU kernels."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate, inflate in memory, and verify without writing artifacts.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace existing output/report artifacts.",
    )
    parser.add_argument(
        "--print-report",
        action="store_true",
        help="Print the complete report instead of a compact gate summary.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = inflate_checkpoint_file(
        args.source,
        args.target_config,
        args.output,
        args.report,
        seed=args.seed,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
    if args.print_report:
        payload = report
    else:
        per_nx = report["verification"]["per_nx"]
        payload = {
            "passed": report["passed"],
            "dry_run": args.dry_run,
            "source_checkpoint_sha256": report["source"]["checkpoint_sha256"],
            "target_config_sha256": report["target"]["config_sha256"],
            "copied_tensor_count": report["tensor_mapping"]["copied_count"],
            "initialized_tensor_count": report["tensor_mapping"][
                "initialized_count"
            ],
            "tested_nx": report["verification"]["tested_nx"],
            "maximum_absolute_mismatch": {
                nx: {
                    field: metrics["maximum_absolute_mismatch"]
                    for field, metrics in values["fields"].items()
                }
                for nx, values in per_nx.items()
            },
            "target_checkpoint": report["artifacts"]["target_checkpoint"],
            "report": report["artifacts"]["report"],
        }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
