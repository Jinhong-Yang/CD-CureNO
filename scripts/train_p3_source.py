"""Train the frozen P3 source-pretraining experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cdcureno.training.source_1d import (
    SourcePretrainConfig,
    run_source_pretraining,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--modes-time", type=int, default=24)
    parser.add_argument("--modes-space", type=int, default=12)
    args = parser.parse_args()
    config = SourcePretrainConfig(
        data_path=ROOT / "data" / "processed" / "p3_source_1d_v3.npz",
        split_manifest=ROOT / "splits" / "p3_source_1d_v3.json",
        output_root=ROOT / "outputs" / "runs" / "p3-source-factorized-v4-seed0",
        summary_path=(
            ROOT
            / "outputs"
            / "tables"
            / "p3_source_pretraining_v4_seed0.json"
        ),
        cases_path=(
            ROOT
            / "outputs"
            / "tables"
            / "p3_source_pretraining_v4_cases.csv"
        ),
        epochs=args.epochs,
        num_threads=args.num_threads,
        width=args.width,
        depth=args.depth,
        modes_time=args.modes_time,
        modes_space=args.modes_space,
        minimum_epochs=min(60, args.epochs),
        early_stopping_patience=25,
    )
    summary = run_source_pretraining(config)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
