"""Recompute demo metrics with NumPy only, without loading model or trainer code.

This separate implementation verifies saved numerical outputs and artifact
integrity. It is not an independent physical solver or a third-party replication.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify(folder: Path) -> dict:
    manifest = load_json(folder / "artifact_manifest.json")
    needed = {"config.json", "provenance.json", "normalization.json", "grid.npz",
              "source_train.npz", "source_validation.npz", "target_train.npz",
              "target_validation.npz", "target_heldout.npz", "heldout_predictions.npz",
              "source_best.pt", "target_inflated.pt", "target_best.pt", "inflation.json",
              "source_history.csv", "target_history.csv", "solver_diagnostics.json",
              "evaluation.json", "run_summary.json", "per_case_metrics.csv", "heldout_xz_fields.png"}
    require(needed.issubset(manifest), "Manifest omitted required artifacts")
    for name, expected in manifest.items():
        require(Path(name).name == name and ":" not in name and "\\" not in name,
                "Manifest paths must be single filenames")
        require((folder / name).is_file() and sha(folder / name) == expected,
                f"Artifact hash mismatch: {name}")
    config = load_json(folder / "config.json")
    acceptance = config["acceptance"]
    require(acceptance["predictive_accuracy_threshold"] is None, "Demo must not use a predictive-accuracy pass threshold")
    all_ids = sum(config["split"].values(), [])
    require(len(all_ids) == len(set(all_ids)), "Case leakage across splits")
    for kind in ("source", "target"):
        for split, ids in config["split"].items():
            if kind == "source" and split == "heldout":
                continue
            with np.load(folder / f"{kind}_{split}.npz", allow_pickle=False) as data:
                require(data["case_ids"].tolist() == ids, f"Incorrect {kind}/{split} population")
                for key in ("inputs", "temperature_K", "alpha"):
                    require(np.isfinite(data[key]).all(), f"Nonfinite {kind}/{split}/{key}")
    norm = load_json(folder / "normalization.json")
    require(norm["fit_population"] == "source_train" and norm["case_ids"] == config["split"]["train"],
            "Normalizer population is not training-only")
    with np.load(folder / "source_train.npz", allow_pickle=False) as source:
        reference_mean = float(np.mean(source["temperature_K"], dtype=np.float64))
        reference_std = max(float(np.std(source["temperature_K"], dtype=np.float64)), 1.0)
    require(abs(norm["temperature_mean_K"]-reference_mean) < 1e-12 and
            abs(norm["temperature_std_K"]-reference_std) < 1e-12,
            "Saved normalizer differs from independently recomputed training statistics")
    summary = load_json(folder / "run_summary.json")
    evaluation = load_json(folder / "evaluation.json")
    provenance = load_json(folder / "provenance.json")
    require(evaluation["pid"] != summary["training_pid"], "Evaluator did not use a separate process")
    require(evaluation["checkpoint_sha256"] == sha(folder / "target_best.pt"), "Evaluator checkpoint binding mismatch")
    require(evaluation["config_sha256"] == provenance["config_sha256"] == sha(folder / "config.json"), "Configuration binding mismatch")
    require(evaluation["case_ids"] == config["split"]["heldout"], "Wrong evaluation population")
    require(evaluation["future_prefix_maximum_absolute_difference"] <= acceptance["causality_absolute_tolerance"],
            "Future-input perturbation changed an output prefix")
    require(evaluation["reference_lateral_range_K"] > acceptance["minimum_reference_lateral_range_K"],
            "Reference demonstration is degenerate in the lateral direction")
    inflation = load_json(folder / "inflation.json")
    require(inflation["restriction_maximum_absolute_difference"] <= acceptance["restriction_absolute_tolerance"],
            "Inflated model violated the restriction tolerance")
    require(all(row["source_sha256"] == row["target_sha256"] for row in inflation["copied"]),
            "Inflation tensor hash mismatch")
    for kind in ("source", "target"):
        training = summary[f"{kind}_training"]
        require(training["parameter_absolute_change_sum"] > 0 and np.isfinite(training["parameter_absolute_change_sum"]),
                f"No finite parameter change in {kind} training")
        require(training["epochs_executed"] == config["training"][f"{kind}_epochs"], "Incomplete training epochs")
        with (folder / f"{kind}_history.csv").open(newline="", encoding="utf-8") as stream:
            history = list(csv.DictReader(stream))
        require(len(history) == training["epochs_executed"], "Incomplete history")
        selected = min(history, key=lambda row: float(row["validation_objective"]))
        require(int(selected["epoch"]) == training["selected_epoch"], "Checkpoint selection did not use validation objective")
    diagnostics = load_json(folder / "solver_diagnostics.json")
    require(all(d["all_coupling_steps_converged"] for d in diagnostics), "Unconverged solver labels")
    grid = dict(np.load(folder / "grid.npz", allow_pickle=False))
    t = grid["times_s"].astype(np.float64)
    dt = np.diff(t)
    require(np.all(dt > 0), "Invalid time coordinates")
    # Independently derive physical trapezoidal weights from actual coordinates.
    temporal_weights = np.concatenate(([dt[0]/2], (dt[:-1]+dt[1:])/2, [dt[-1]/2]))
    cell_weights = grid["cell_volume_m2"].astype(np.float64) * grid["composite_mask"]
    weights = temporal_weights[:, None, None] * cell_weights[None]
    predictions = dict(np.load(folder / "heldout_predictions.npz", allow_pickle=False))
    with np.load(folder / "target_heldout.npz", allow_pickle=False) as reference:
        require(np.array_equal(predictions["case_ids"], reference["case_ids"]), "Saved predictions use wrong cases")
        require(np.array_equal(predictions["reference_temperature_K"], reference["temperature_K"]), "Saved temperature labels changed")
        require(np.array_equal(predictions["reference_alpha"], reference["alpha"]), "Saved alpha labels changed")
    for key in ("temperature_K", "alpha", "reference_temperature_K", "reference_alpha"):
        require(np.isfinite(predictions[key]).all(), f"Nonfinite saved field: {key}")
    a = predictions["alpha"]
    tol = acceptance["alpha_tolerance"]
    require(a.min() >= -tol and a.max() <= 1+tol and np.diff(a, axis=1).min() >= -tol,
            "Degree of cure violates bounds/monotonicity")
    require(float(np.max(np.ptp(predictions["temperature_K"], axis=3))) > 0, "Target predictions have no lateral variation")
    with (folder / "per_case_metrics.csv").open(newline="", encoding="utf-8") as stream:
        reported = list(csv.DictReader(stream))
    require([int(row["case_id"]) for row in reported] == config["split"]["heldout"], "Metric rows have wrong population/order")
    recomputed = []
    max_difference = 0.0
    for i, case in enumerate(predictions["case_ids"]):
        delta = predictions["temperature_K"][i] - predictions["reference_temperature_K"][i]
        energy_error = np.einsum("tzx,tzx->", weights, delta**2)
        energy_reference = np.einsum("tzx,tzx->", weights, predictions["reference_temperature_K"][i]**2)
        cure_error = np.einsum("tzx,tzx->", weights, (a[i]-predictions["reference_alpha"][i])**2)
        values = {"case_id": int(case), "temperature_relative_l2": float(np.sqrt(energy_error/energy_reference)),
                  "temperature_rmse_K": float(np.sqrt(energy_error/weights.sum())),
                  "alpha_rmse": float(np.sqrt(cure_error/weights.sum()))}
        for key in ("temperature_relative_l2", "temperature_rmse_K", "alpha_rmse"):
            difference = abs(values[key]-float(reported[i][key]))
            max_difference = max(max_difference, difference)
            require(difference <= acceptance["metric_absolute_tolerance"], f"Metric mismatch: case {case}, {key}")
        recomputed.append(values)
    return {"status": "passed", "scope": "NumPy-only saved-output metric/integrity verification; not external physical validation",
            "verified_manifest_sha256": sha(folder / "artifact_manifest.json"),
            "verifier_sha256": sha(Path(__file__).resolve()),
            "recomputed_metrics": recomputed, "maximum_metric_absolute_difference": max_difference,
            "checks": ["artifact hashes", "complete-case split", "training-only normalizer", "separate evaluator process",
                       "checkpoint/config binding", "actual parameter change receipt", "validation checkpoint selection",
                       "nondegenerate true-2-D labels/predictions", "cure bounds/monotonicity", "causality/restriction receipts",
                       "independent saved-output metric calculation"],
            "predictive_superiority_claim": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    folder = args.input.resolve()
    try:
        receipt = verify(folder)
    except Exception as error:
        # An earlier successful receipt must not survive a failed recheck as
        # if it described the modified artifacts.
        failed = {"status": "failed", "error": f"{type(error).__name__}: {error}",
                  "verifier_sha256": sha(Path(__file__).resolve())}
        (folder / "postflight_receipt.json").write_text(json.dumps(failed, indent=2)+"\n", encoding="utf-8")
        raise
    (folder / "postflight_receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
