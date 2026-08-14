"""Run future-perturbation checks on a frozen P2 best checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cdcureno.data.joint_case1 import prepare_joint_case1
from cdcureno.models.joint_operators import build_joint_operator
from cdcureno.training.joint import EXPERIMENT_MODELS


def check_checkpoint(
    project_root: Path,
    run_id: str,
    case_ids: list[int],
    cutoffs: list[int],
    tolerance: float,
) -> dict:
    run_dir = project_root / "outputs" / "runs" / run_id
    config = json.loads(
        (run_dir / "config_resolved.yaml").read_text(encoding="utf-8")
    )
    prepared = prepare_joint_case1(
        Path(config["data_path"]),
        Path(config["split_manifest"]),
    )
    test_ids = set(prepared.splits["test"])
    if not case_ids or any(case_id not in test_ids for case_id in case_ids):
        raise ValueError("Every causality case ID must belong to the frozen test split.")
    if not cutoffs or any(cutoff < 0 or cutoff >= prepared.inputs.shape[1] - 1 for cutoff in cutoffs):
        raise ValueError("Cutoffs must leave at least one future time sample.")

    model_family = EXPERIMENT_MODELS[config["experiment"]]
    model = build_joint_operator(
        model_family,
        input_channels=prepared.inputs.shape[-1],
        width=config["width"],
        depth=config["depth"],
        modes_time=config["modes_time"],
        modes_space=config["modes_space"],
    ).eval()
    checkpoint = torch.load(
        run_dir / "checkpoints" / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model"])
    inputs = prepared.inputs[torch.tensor(case_ids, dtype=torch.long)]
    rows = []
    with torch.no_grad():
        original = model(inputs)
        for cutoff in cutoffs:
            perturbed = inputs.clone()
            perturbed[:, cutoff + 1 :, :, 0] += 0.25
            perturbed[:, cutoff + 1 :, :, 1] -= 0.15
            changed = model(perturbed)
            differences = {
                field: float(
                    torch.max(
                        torch.abs(
                            original[field][:, : cutoff + 1]
                            - changed[field][:, : cutoff + 1]
                        )
                    )
                )
                for field in (
                    "temperature",
                    "temperature_residual",
                    "alpha",
                    "cure_rate",
                )
            }
            rows.append(
                {
                    "cutoff_index": cutoff,
                    "future_start_index": cutoff + 1,
                    "maximum_past_difference": differences,
                    "passed": max(differences.values()) <= tolerance,
                }
            )
    architecture_is_causal = model_family == "causal_factorized"
    summary = {
        "run_id": run_id,
        "model_family": model_family,
        "architecture_is_causal": architecture_is_causal,
        "case_ids": case_ids,
        "cutoffs": cutoffs,
        "perturbed_channels": [
            "air_temperature_normalized",
            "causal_temperature_baseline_normalized",
        ],
        "tolerance": tolerance,
        "checks": rows,
        "passed": all(row["passed"] for row in rows),
        "interpretation": (
            "A causal checkpoint must pass. A noncausal checkpoint result is a "
            "descriptive sensitivity control and is not required to pass."
        ),
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check a P2 checkpoint for future-perturbation invariance."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--case-ids", default="75,100,150")
    parser.add_argument("--cutoffs", default="55,111,167")
    parser.add_argument("--tolerance", type=float, default=2e-6)
    parser.add_argument(
        "--output",
        type=Path,
        help="Default: outputs/tables/<run-id>_causality.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    case_ids = [int(value) for value in args.case_ids.split(",") if value]
    cutoffs = [int(value) for value in args.cutoffs.split(",") if value]
    run_dir = project_root / "outputs" / "runs" / args.run_id
    output = args.output or (
        Path("outputs") / "tables" / f"{args.run_id}_causality.json"
    )
    output = (project_root / output).resolve()
    if args.dry_run:
        print(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "run_exists": run_dir.is_dir(),
                    "case_ids": case_ids,
                    "cutoffs": cutoffs,
                    "output": str(output),
                },
                indent=2,
            )
        )
        return 0
    summary = check_checkpoint(
        project_root,
        args.run_id,
        case_ids,
        cutoffs,
        args.tolerance,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["architecture_is_causal"] and not summary["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
