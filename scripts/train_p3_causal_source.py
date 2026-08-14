"""Train or audit the frozen P3 structurally causal source experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cdcureno.training.causal_source_1d import (
    causal_source_dry_run,
    load_causal_source_config,
    train_causal_source,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "configs" / "experiment" / "p3_causal_source_pretrain_v1.yaml"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the auditable causal P3 source model on the frozen v3 "
            "source data and splits."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--device",
        help="Device override: cuda, cuda:N, cpu, or auto.",
    )
    parser.add_argument("--run-id", help="Override the configured run ID.")
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Repository-relative or absolute output root override.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Epoch override for controlled smoke or follow-up runs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume atomically from the selected run's last.pt.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify config, hashes, architecture, and device without writing.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_path = (
        args.config.resolve()
        if args.config.is_absolute()
        else (ROOT / args.config).resolve()
    )
    config = load_causal_source_config(
        config_path,
        project_root=ROOT,
        device=args.device,
        run_id=args.run_id,
        output_root=args.output_root,
        epochs=args.epochs,
        resume=args.resume,
    )
    if args.dry_run:
        summary = causal_source_dry_run(config)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["passed"] else 1
    metrics = train_causal_source(config)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if metrics["status"] != "completed":
        return 0
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
