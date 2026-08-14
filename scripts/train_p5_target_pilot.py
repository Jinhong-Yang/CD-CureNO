"""Preflight, train, or resume the frozen P5 RP-FFNO pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cdcureno.training.target_2d import (
    PILOT_BUDGETS,
    PILOT_METHODS,
    PILOT_SEEDS,
    load_target_pilot_config,
    run_target_resource_preflight,
    target_pilot_dry_run,
    train_target_pilot,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "configs" / "experiment" / "p5_rp_ffno_pilot_v1.yaml"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the leakage-safe P5 RP-FFNO pilot. The training path can "
            "index only the frozen target training budget and validation "
            "label values; ID-test and OOD loaders are unavailable."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--method",
        choices=PILOT_METHODS,
        help="Initialization method; all later training conditions are shared.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        choices=PILOT_BUDGETS,
        help="Frozen nested target label budget.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        choices=PILOT_SEEDS,
        help="Frozen one-seed pilot seed.",
    )
    parser.add_argument(
        "--device",
        help="Device override: cuda, cuda:N, cpu, or auto.",
    )
    parser.add_argument(
        "--run-id",
        help="Explicit run ID; required when resuming a timestamped run.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Repository-relative or absolute output root override.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Epoch override for non-training dry-run diagnostics only.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Atomically resume model/optimizer/scheduler/RNG/samplers.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate hashes, architecture, split restrictions, and the "
            "frozen resource profile without indexing target label values."
        ),
    )
    parser.add_argument(
        "--resource-preflight",
        action="store_true",
        help=(
            "Run the canonical full-resolution CUDA optimizer-step preflight "
            "and freeze micro-batch/gradient accumulation for both methods."
        ),
    )
    parser.add_argument(
        "--overwrite-resource-profile",
        action="store_true",
        help="Explicitly replace an existing resource-preflight profile.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.resume and not args.run_id:
        raise SystemExit("--resume requires the exact existing --run-id.")
    if args.dry_run and args.resource_preflight:
        raise SystemExit(
            "Choose either --dry-run or --resource-preflight, not both."
        )
    if args.epochs is not None and not args.dry_run:
        raise SystemExit(
            "--epochs is not permitted for resource preflight or pilot "
            "training; the canonical 120-epoch ceiling is frozen."
        )
    config_path = (
        args.config.resolve()
        if args.config.is_absolute()
        else (ROOT / args.config).resolve()
    )
    config = load_target_pilot_config(
        config_path,
        project_root=ROOT,
        method=args.method,
        label_budget=args.budget,
        seed=args.seed,
        device=args.device,
        run_id=args.run_id,
        output_root=args.output_root,
        epochs=args.epochs,
        resume=args.resume,
    )
    if args.resource_preflight:
        result = run_target_resource_preflight(
            config, overwrite=args.overwrite_resource_profile
        )
    elif args.dry_run:
        result = target_pilot_dry_run(config)
    else:
        result = train_target_pilot(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("status") == "paused":
        return 0
    return 0 if result.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
