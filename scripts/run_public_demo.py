"""Public, standalone CPU demonstration using production CD-CureNO components.

This deliberately does not impersonate a canonical P3/P4/P5 experiment. Inputs,
splits and checkpoint schema are named demo artifacts. The held-out evaluator is
launched as a separate process after checkpoint selection is complete.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from cdcureno.models.joint_operators import CausalFactorizedOperator
from cdcureno.models.target_operators import CausalAxisFactorized2DOperator
from cdcureno.models.causal_checkpoint_inflation import _copy_source_state
from cdcureno.solvers.conservative_2d import (
    rectangular_tool_composite_grid, RobinBoundaries2D, simulate_cure_2d,
)

SOURCE_CHANNELS = (
    "air_temperature_normalized", "causal_physics_temperature_baseline_normalized",
    "z_fraction", "time_fraction", "composite_mask", "mean_upper_h_scaled",
    "initial_degree_of_cure",
)
NEW_CHANNELS = ("x_fraction", "upper_h_local_scaled")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def configure(config: dict) -> None:
    torch.set_num_threads(config["threads"])
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.use_deterministic_algorithms(True)


def target_model(config: dict) -> CausalAxisFactorized2DOperator:
    return CausalAxisFactorized2DOperator(
        source_channel_names=SOURCE_CHANNELS, new_channel_names=NEW_CHANNELS,
        adapter_seed=config["seed"], **config["model"],
    ).cpu()


def source_model(config: dict) -> CausalFactorizedOperator:
    m = config["model"]
    return CausalFactorizedOperator(input_channels=7, width=m["width"],
                                    depth=m["depth"], modes_space=m["modes_z"]).cpu()


def generate(config: dict, out: Path) -> None:
    grid = rectangular_tool_composite_grid(**config["grid"])
    s = config["solver"]
    times = np.linspace(0.0, s["duration_s"], s["output_times"])
    # Known driving conditions only; the causal baseline below uses no labels.
    xfrac = grid.x_m / config["grid"]["width_m"]
    zfrac = grid.z_m / float(np.sum(grid.control_volume_width_z_m))
    diag = []
    generated = {}
    for kind in ("source", "target"):
        ids = config["source_case_ids"] if kind == "source" else sum(config["split"].values(), [])
        for case in ids:
            amplitude = 0.0 if kind == "source" else config["cases"]["lateral_amplitude"][case]
            hmean = config["cases"]["upper_h_W_m2_K"][case]
            h = hmean * (1.0 + amplitude * (2.0 * xfrac - 1.0))
            air = s["initial_temperature_K"] + config["cases"]["air_rise_K"][case] * np.minimum(times / 600.0, 1.0)
            result = simulate_cure_2d(
                times, air, grid=grid,
                boundaries=RobinBoundaries2D(lower_h_W_m2_K=30.0, upper_h_W_m2_K=h),
                initial_temperature_K=s["initial_temperature_K"], initial_alpha=s["initial_alpha"],
                maximum_step_s=s["maximum_step_s"],
            )
            if not result.diagnostics.all_coupling_steps_converged:
                raise RuntimeError(f"Unconverged demo solver case {kind}-{case}")
            baseline = np.empty_like(times)
            baseline[0] = s["initial_temperature_K"]
            # A documented causal lumped-temperature input, not a fitted label.
            for j in range(1, len(times)):
                baseline[j] = baseline[j-1] + (1.0 - np.exp(-(times[j]-times[j-1])/600.0)) * (air[j-1]-baseline[j-1])
            shape = result.temperature_K.shape
            features = np.empty((*shape, 9), dtype=np.float64)
            features[..., 0] = air[:, None, None]
            features[..., 1] = baseline[:, None, None]
            features[..., 2] = zfrac[None, :, None]
            features[..., 3] = (times / s["duration_s"])[:, None, None]
            features[..., 4] = grid.composite_mask
            features[..., 5] = hmean / 100.0
            features[..., 6] = s["initial_alpha"] * grid.composite_mask
            features[..., 7] = xfrac[None, None, :]
            features[..., 8] = h[None, None, :] / 100.0
            temperature, alpha = result.temperature_K, result.alpha
            if kind == "source":
                # A homogeneous production 2-D solve provides the 1-D limit on
                # exactly the same cell-centred z grid, avoiding interpolation.
                if np.max(np.ptp(temperature, axis=2)) > 1e-8:
                    raise AssertionError("Homogeneous source is not laterally invariant")
                features, temperature, alpha = features[:, :, 0, :7], temperature[:, :, 0], alpha[:, :, 0]
            generated[kind, case] = (features, temperature, alpha)
            diag.append({"kind": kind, "case_id": case, **asdict(result.diagnostics)})
    for kind in ("source", "target"):
        for split, ids in config["split"].items():
            if kind == "source" and split == "heldout":
                continue
            values = [generated[kind, i] for i in ids]
            np.savez_compressed(out / f"{kind}_{split}.npz", case_ids=np.array(ids),
                                inputs=np.stack([v[0] for v in values]),
                                temperature_K=np.stack([v[1] for v in values]),
                                alpha=np.stack([v[2] for v in values]))
    np.savez_compressed(out / "grid.npz", times_s=times, x_m=grid.x_m, z_m=grid.z_m,
                        composite_mask=grid.composite_mask, cell_volume_m2=grid.control_volume_m2)
    write_json(out / "solver_diagnostics.json", diag)


def normalize(raw: np.ndarray, norm: dict) -> torch.Tensor:
    values = raw.copy()
    values[..., :2] = (values[..., :2] - norm["temperature_mean_K"]) / norm["temperature_std_K"]
    return torch.tensor(values, dtype=torch.float32)


def loss(model: torch.nn.Module, data: dict, norm: dict) -> torch.Tensor:
    pred = model(data["inputs"])
    temp = (data["temperature_K"] - norm["temperature_mean_K"]) / norm["temperature_std_K"]
    mask = data["inputs"][..., 4]
    return torch.mean((pred["temperature"] - temp)**2) + torch.sum((pred["alpha"]-data["alpha"])**2 * mask)/mask.sum()


def read_training(path: Path, norm: dict) -> dict:
    with np.load(path, allow_pickle=False) as d:
        return {"inputs": normalize(d["inputs"], norm),
                "temperature_K": torch.tensor(d["temperature_K"], dtype=torch.float32),
                "alpha": torch.tensor(d["alpha"], dtype=torch.float32)}


def train(model: torch.nn.Module, kind: str, config: dict, norm: dict, out: Path) -> dict:
    training = read_training(out / f"{kind}_train.npz", norm)
    validation = read_training(out / f"{kind}_validation.npz", norm)
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    best, selected, history = float("inf"), None, []
    started = time.perf_counter()
    for epoch in range(1, config["training"][f"{kind}_epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        objective = loss(model, training, norm)
        if not torch.isfinite(objective):
            raise AssertionError("Nonfinite training objective")
        objective.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            v = float(loss(model, validation, norm))
        if not np.isfinite(v):
            raise AssertionError("Nonfinite validation objective")
        history.append({"epoch": epoch, "training_objective": float(objective.detach()), "validation_objective": v})
        if v < best:
            best, selected = v, epoch
            torch.save({"schema": "cdcureno-public-demo-checkpoint-v1", "model_kind": kind,
                        "model_state": model.state_dict(), "config": config, "normalization": norm,
                        "config_sha256": digest(out / "config.json"), "selected_epoch": epoch}, out / f"{kind}_best.pt")
    payload = torch.load(out / f"{kind}_best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model_state"], strict=True)
    change = sum(float(torch.sum(torch.abs(model.state_dict()[key]-value))) for key, value in before.items())
    if not np.isfinite(change) or change <= 0.0:
        raise AssertionError("Selected trained model did not change parameters")
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise AssertionError("Nonfinite selected parameters")
    with (out / f"{kind}_history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader(); writer.writerows(history)
    return {"epochs_executed": len(history), "selected_epoch": selected,
            "parameter_absolute_change_sum": change, "parameter_count": sum(p.numel() for p in model.parameters()),
            "wall_seconds": time.perf_counter()-started, "training_case_ids": config["split"]["train"],
            "validation_case_ids": config["split"]["validation"], "heldout_read_during_training": False}


def evaluate(out: Path) -> None:
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    configure(config)
    payload = torch.load(out / "target_best.pt", map_location="cpu", weights_only=True)
    if payload["config_sha256"] != digest(out / "config.json"):
        raise AssertionError("Checkpoint/config binding mismatch")
    model = target_model(config)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    norm = payload["normalization"]
    with np.load(out / "target_heldout.npz", allow_pickle=False) as d:
        inputs, reference_t, reference_a, ids = d["inputs"], d["temperature_K"], d["alpha"], d["case_ids"]
    inp = normalize(inputs, norm)
    with torch.no_grad():
        result = model(inp)
        temperature = result["temperature"].numpy().astype(np.float64) * norm["temperature_std_K"] + norm["temperature_mean_K"]
        alpha = result["alpha"].numpy().astype(np.float64)
        prefix = inp.shape[1] // 2
        changed = inp.clone()
        changed[:, prefix+1:] += 0.37
        perturbed = model(changed)
        causal = max(float(torch.max(torch.abs(result[key][:, :prefix+1]-perturbed[key][:, :prefix+1]))) for key in result)
    grid = dict(np.load(out / "grid.npz", allow_pickle=False))
    tw = np.ones(len(grid["times_s"]))
    tw[[0, -1]] = 0.5
    weights = tw[:, None, None] * grid["cell_volume_m2"][None] * grid["composite_mask"][None]
    rows = []
    for i, case in enumerate(ids):
        e = temperature[i]-reference_t[i]
        rows.append({"case_id": int(case),
                     "temperature_relative_l2": float(np.sqrt(np.sum(weights*e*e)/np.sum(weights*reference_t[i]**2))),
                     "temperature_rmse_K": float(np.sqrt(np.sum(weights*e*e)/np.sum(weights))),
                     "alpha_rmse": float(np.sqrt(np.sum(weights*(alpha[i]-reference_a[i])**2)/np.sum(weights)))})
    with (out / "per_case_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(out / "heldout_predictions.npz", case_ids=ids, temperature_K=temperature, alpha=alpha,
                        reference_temperature_K=reference_t, reference_alpha=reference_a)
    write_json(out / "evaluation.json", {"pid": os.getpid(), "separate_process": True,
               "checkpoint_sha256": digest(out / "target_best.pt"), "config_sha256": digest(out / "config.json"),
               "case_ids": ids.tolist(), "future_perturbation_cutoff_index": prefix,
               "future_prefix_maximum_absolute_difference": causal,
               "reference_lateral_range_K": float(np.max(np.ptp(reference_t, axis=3))),
               "prediction_lateral_range_K": float(np.max(np.ptp(temperature, axis=3))),
               "metrics_region": "composite cells; trapezoidal time weights and cell volumes", "predictive_superiority_claim": False})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), constrained_layout=True)
    extent = [0, config["grid"]["width_m"]*1000, 0, 50]
    for r, (truth, prediction, label) in enumerate(((reference_t[0, -1], temperature[0, -1], "Temperature (K)"), (reference_a[0, -1], alpha[0, -1], "Degree of cure"))):
        lo, hi = min(truth.min(), prediction.min()), max(truth.max(), prediction.max())
        for c, data in enumerate((truth, prediction, prediction-truth)):
            error_limit = max(float(np.max(np.abs(prediction-truth))), 1e-15)
            kwargs = {"vmin": lo, "vmax": hi, "cmap": "viridis"} if c < 2 else {"vmin": -error_limit, "vmax": error_limit, "cmap": "coolwarm"}
            im = axes[r,c].imshow(data, origin="lower", extent=extent, aspect="auto", **kwargs)
            axes[r,c].set(title=("Reference", "Trained target", "Prediction − reference")[c], xlabel="x (mm)", ylabel="z (mm)")
            fig.colorbar(im, ax=axes[r,c], label=label)
    fig.suptitle(f"Public functionality demo · held-out case {ids[0]} · t={grid['times_s'][-1]:g} s\nOne seed, small training budget; no predictive-superiority claim")
    fig.savefig(out / "heldout_xz_fields.png", dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/demo/public_demo_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    if args.evaluate_only:
        evaluate(out)
        return 0
    config = json.loads(args.config.read_text(encoding="utf-8"))
    populations = list(config["split"].values())
    if len(set(sum(populations, []))) != len(sum(populations, [])):
        raise ValueError("Demo case splits must be disjoint")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new empty output directory; existing evidence is never overwritten")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    configure(config)
    write_json(out / "config.json", config)
    provenance = {"config_sha256": digest(out / "config.json"), "python": sys.version,
                  "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__,
                  "device": "cpu", "threads": torch.get_num_threads(), "training_pid": os.getpid(),
                  "scope": config["purpose"], "source_hashes": {}}
    for name in ("scripts/run_public_demo.py", "scripts/verify_public_demo.py", "src/cdcureno/solvers/conservative_2d.py", "src/cdcureno/models/joint_operators.py", "src/cdcureno/models/target_operators.py", "src/cdcureno/models/causal_checkpoint_inflation.py"):
        provenance["source_hashes"][name] = digest(ROOT / name)
    try:
        provenance["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        provenance["git_dirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        provenance["git_commit"] = None
    write_json(out / "provenance.json", provenance)
    tick = time.perf_counter(); generate(config, out); generation_seconds = time.perf_counter()-tick
    with np.load(out / "source_train.npz", allow_pickle=False) as d:
        norm = {"temperature_mean_K": float(d["temperature_K"].mean()),
                "temperature_std_K": max(float(d["temperature_K"].std()), 1.0),
                "fit_population": "source_train", "case_ids": d["case_ids"].tolist()}
    write_json(out / "normalization.json", norm)
    source = source_model(config)
    source_training = train(source, "source", config, norm, out)
    target = target_model(config)
    # Same zero-residual initialization sequence as the production inflator;
    # only the surrounding artifact contract is this standalone demo schema.
    with torch.no_grad():
        target.lift.geometry.weight.zero_()
        for index, block in enumerate(target.blocks):
            block.lateral.reset_zero_residual(config["seed"] + index)
    copied, initialized = _copy_source_state(target, source.state_dict())
    src_input = read_training(out / "source_validation.npz", norm)["inputs"]
    nx = round(config["grid"]["width_m"]/config["grid"]["spacing_x_m"])
    lifted = torch.cat((src_input.unsqueeze(3).expand(-1, -1, -1, nx, -1), torch.zeros(*src_input.shape[:-1], nx, 2)), dim=-1)
    source.eval(); target.eval()
    with torch.no_grad():
        src_out, trg_out = source(src_input), target(lifted)
        restriction = max(float(torch.max(torch.abs(trg_out[key]-src_out[key].unsqueeze(3)))) for key in src_out)
    if restriction > config["acceptance"]["restriction_absolute_tolerance"]:
        raise AssertionError("Inflated target failed the prespecified restriction tolerance")
    torch.save({"model_state": target.state_dict(), "config": config, "normalization": norm}, out / "target_inflated.pt")
    write_json(out / "inflation.json", {"mechanism": "production _copy_source_state; standalone demo, not canonical P5 receipt", "copied": copied, "initialized": initialized, "restriction_maximum_absolute_difference": restriction})
    target_training = train(target, "target", config, norm, out)
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--output", str(out), "--evaluate-only"], check=True)
    write_json(out / "run_summary.json", {"generation_wall_seconds": generation_seconds, "source_training": source_training,
               "target_training": target_training, "training_pid": os.getpid(), "pre_postflight_wall_seconds": time.perf_counter()-started,
               "normalization_fit_population": "source_train", "predictive_accuracy_threshold": None})
    write_json(out / "artifact_manifest.json", {p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file()})
    subprocess.run([sys.executable, str(ROOT / "scripts/verify_public_demo.py"), "--input", str(out)], check=True)
    print(json.dumps({"output": str(out), "status": "passed", "wall_seconds": time.perf_counter()-started}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
