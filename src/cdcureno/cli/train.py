"""Train auditable legacy or joint Case1 operator experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cdcureno.legacy.training import LegacyTrainConfig, train_legacy
from cdcureno.training.joint import (
    EXPERIMENT_MODELS,
    JointTrainConfig,
    train_joint,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an auditable CD-CureNO training experiment."
    )
    parser.add_argument(
        "--experiment",
        required=True,
        choices=[
            "legacy_resfno_exact",
            "legacy_resfno_corrected",
            *EXPERIMENT_MODELS,
        ],
    )
    parser.add_argument("--data", type=Path, default=Path("external/ResFNO/data/Case1.mat"))
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--location", type=int, default=35)
    parser.add_argument("--task", choices=["T", "A"], default="T")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--smoothness-weight", type=float, default=0.5)
    parser.add_argument("--temperature-weight", type=float, default=1.0)
    parser.add_argument("--alpha-weight", type=float, default=0.5)
    parser.add_argument("--gradient-weight", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--width", type=int)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--modes-time", type=int, default=16)
    parser.add_argument("--modes-space", type=int, default=12)
    parser.add_argument("--early-stopping-patience", type=int, default=60)
    parser.add_argument("--minimum-epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:N, or auto")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--no-register-result",
        action="store_true",
        help="Defer RESULTS_INDEX.csv registration for a managed sweep.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[3]
    split = args.split_manifest
    if split is None:
        name = "legacy_case1_corrected_matched_v1.json"
        if args.experiment == "legacy_resfno_exact":
            name = "legacy_case1_exact_v1.json"
        split = Path("splits") / name
    common = {
        "data_path": (project_root / args.data).resolve(),
        "split_manifest": (project_root / split).resolve(),
        "output_root": (project_root / args.output_root).resolve(),
        "project_root": project_root,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "device": args.device,
        "run_id": args.run_id,
        "resume": args.resume,
        "register_result": not args.no_register_result,
        "num_threads": args.num_threads,
    }
    if args.experiment in EXPERIMENT_MODELS:
        config = JointTrainConfig(
            experiment=args.experiment,
            temperature_weight=args.temperature_weight,
            alpha_weight=args.alpha_weight,
            gradient_weight=args.gradient_weight,
            gradient_clip=args.gradient_clip,
            width=args.width,
            depth=args.depth,
            modes_time=args.modes_time,
            modes_space=args.modes_space,
            early_stopping_patience=args.early_stopping_patience,
            minimum_epochs=args.minimum_epochs,
            **common,
        ).validated()
    else:
        config = LegacyTrainConfig(
            experiment=args.experiment,
            location=args.location,
            task=args.task,
            smoothness_weight=args.smoothness_weight,
            **common,
        ).validated()
    if args.dry_run:
        payload = {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in config.__dict__.items()
            },
            "data_exists": config.data_path.is_file(),
            "split_exists": config.split_manifest.is_file(),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    metrics = (
        train_joint(config)
        if isinstance(config, JointTrainConfig)
        else train_legacy(config)
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
