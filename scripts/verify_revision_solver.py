"""Independent continuous-solution checks through the unmodified 2-D solver.

See docs/revision_solver_verification.md for the predeclared protocol.
"""
from __future__ import annotations

import os

for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                         "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_thread_variable] = "2"

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
import traceback

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cdcureno.physics import CureKinetics
from cdcureno.solvers.conservative_2d import (
    LayeredGrid2D, RobinBoundaries2D, simulate_cure_2d,
)

LX, LZ, T0, AMP, CAPACITY, KX = 0.2, 0.1, 300.0, 10.0, 1e6, 2.0
LAMBDA = (KX * (np.pi / LX) ** 2 + 0.5 * (np.pi / LZ) ** 2) / CAPACITY
SPATIAL_GRIDS = [(12, 8), (24, 16), (48, 32)]
TEMPORAL_STEPS = [20.0, 10.0, 5.0]
PROTOCOL_FILES = ["scripts/verify_revision_solver.py",
                  "docs/revision_solver_verification.md",
                  "src/cdcureno/solvers/conservative_2d.py",
                  "src/cdcureno/physics/as4_8552.py"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def grid_for(nx: int, nz: int, case: str = "N") -> LayeredGrid2D:
    x = (np.arange(nx) + 0.5) * LX / nx
    z = (np.arange(nz) + 0.5) * LZ / nz
    shape = (nz, nx)
    kz = np.full(shape, 0.5)
    mask = np.zeros(shape, dtype=bool)
    source = np.zeros(shape)
    if case == "I":
        mask = np.broadcast_to(z[:, None] >= LZ / 2, shape).copy()
        kz = np.where(mask, 0.25, 1.0)
    elif case == "C":
        mask[:] = True
        source[:] = 2e7
    return LayeredGrid2D(
        x_m=x, z_m=z,
        control_volume_width_x_m=np.full(nx, LX / nx),
        control_volume_width_z_m=np.full(nz, LZ / nz),
        composite_mask=mask,
        density_kg_m3=np.full(shape, 1000.0),
        specific_heat_J_kg_K=np.full(shape, 1000.0),
        conductivity_x_W_m_K=np.full(shape, KX),
        conductivity_z_W_m_K=kz,
        cure_source_J_m3_per_alpha=source,
    )


def exact_transient(x: np.ndarray, z: np.ndarray, times: np.ndarray,
                    coupled: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Continuous PDE solution; no discrete production routine is reused."""
    mode = np.cos(np.pi * z[:, None] / LZ) * np.cos(np.pi * x[None, :] / LX)
    temperature = T0 + AMP * np.exp(-LAMBDA * times[:, None, None]) * mode
    alpha = np.zeros_like(temperature)
    if coupled:
        alpha[:] = (1.0 - 0.9 * np.exp(-0.01 * times))[:, None, None]
        temperature = temperature + 20.0 * (alpha - 0.1)
    return temperature, alpha


def exact_steady(x: np.ndarray, z: np.ndarray, interface: bool = False
                 ) -> tuple[np.ndarray, np.ndarray, RobinBoundaries2D]:
    """Continuous piecewise solution and analytically differentiated source."""
    bx = 50.0
    if interface:
        kz = np.where(z < LZ / 2, 1.0, 0.25)
        u = np.minimum(z, LZ / 2) / 1.0 + np.maximum(z - LZ / 2, 0.0) / 0.25
        total_u = (LZ / 2) / 1.0 + (LZ / 2) / 0.25
    else:
        kz = np.full(z.shape, 0.5)
        u, total_u = z / 0.5, LZ / 0.5
    bz = 2.0 / total_u**2
    f = 1.0 + bx * x * (LX - x)
    g = 1.0 + bz * u * (total_u - u)
    temperature = T0 + AMP * g[:, None] * f[None, :]
    source = 2 * AMP * (KX * bx * g[:, None] + (bz / kz)[:, None] * f[None, :])
    return temperature, source, RobinBoundaries2D(
        bz * total_u, bz * total_u, KX * bx * LX, KX * bx * LX)


def error_norms(numerical: np.ndarray, exact: np.ndarray, volume: np.ndarray
                ) -> dict[str, float]:
    error = numerical - exact
    l2 = float(np.sqrt(np.sum(volume * error**2) / np.sum(volume)))
    denominator = float(np.sqrt(np.sum(volume * (exact - T0)**2) / np.sum(volume)))
    return {"temperature_l2_K": l2, "temperature_linf_K": float(np.max(np.abs(error))),
            "temperature_relative_l2_perturbation": l2 / denominator}


def run_case(case: str, nx: int, nz: int, step: float, times: np.ndarray,
             run_id: str, study: str, output: Path | None = None) -> dict:
    grid = grid_for(nx, nz, case)
    alpha0 = 0.0
    kinetics = None
    source: float | np.ndarray = 0.0
    if case in ("R", "I"):
        exact0, source, boundaries = exact_steady(grid.x_m, grid.z_m, case == "I")
        exact = np.broadcast_to(exact0, (len(times), *grid.shape)).copy()
        exact_alpha = np.zeros_like(exact)
        if case == "I":
            alpha0 = 1.0
            exact_alpha[:, grid.composite_mask] = 1.0
    else:
        boundaries = RobinBoundaries2D(0.0, 0.0)
        exact, exact_alpha = exact_transient(grid.x_m, grid.z_m, times, case == "C")
        if case == "C":
            alpha0 = 0.1
            kinetics = CureKinetics(A_per_s=0.02, delta_E_J_per_mol=0.0,
                                    M=0.0, N=1.0, C=0.0, denominator_offset=1.0)
    started = perf_counter()
    result = simulate_cure_2d(
        times, np.full_like(times, T0), grid=grid, boundaries=boundaries,
        kinetics=kinetics, initial_temperature_K=exact[0], initial_alpha=alpha0,
        external_heat_source_W_m3=source, maximum_step_s=step,
    )
    wall = perf_counter() - started
    per_time = []
    for i in range(1, len(times)):
        row = {"time_s": float(times[i]),
               **error_norms(result.temperature_K[i], exact[i], grid.control_volume_m2),
               "alpha_linf": float(np.max(np.abs(result.alpha[i] - exact_alpha[i])))}
        per_time.append(row)
    record = {
        "run_id": run_id, "case": case, "study": study, "nx": nx, "nz": nz,
        "maximum_step_s": step, "times_s": times.tolist(), "wall_seconds": wall,
        "diagnostics": asdict(result.diagnostics), "errors": per_time,
        "x_variation_K": float(np.max(np.ptp(exact[0], axis=1))),
        "z_variation_K": float(np.max(np.ptp(exact[0], axis=0))),
        "finite": bool(np.all(np.isfinite(result.temperature_K)) and np.all(np.isfinite(result.alpha))),
    }
    if case in ("R", "I"):
        record["steady_change_K"] = float(np.max(np.abs(result.temperature_K[-1] - result.temperature_K[-2])))
    if case == "C":
        energy_identity = np.mean(result.temperature_K, axis=(1, 2)) - T0 - 20.0 * (
            np.mean(result.alpha, axis=(1, 2)) - 0.1)
        record["mean_reaction_heat_identity_error_K"] = float(np.max(np.abs(energy_identity)))
    if output is not None:
        np.savez_compressed(output / f"{run_id}.npz", x_m=grid.x_m, z_m=grid.z_m,
                            times_s=times, temperature_K=result.temperature_K,
                            exact_temperature_K=exact, alpha=result.alpha,
                            exact_alpha=exact_alpha, volume_m2=grid.control_volume_m2)
        write_json(output / f"{run_id}.json", record)
    return record


def protocol() -> dict:
    return {
        "protocol_version": "1", "spatial_grids_nx_nz": SPATIAL_GRIDS,
        "temporal_steps_s": TEMPORAL_STEPS,
        "case_N_spatial_step_s": 0.02, "case_N_spatial_step_control_s": 0.01,
        "case_N_temporal_grid": [96, 64], "case_N_temporal_grid_control": [192, 128],
        "spatial_order_interval": [1.8, 2.2], "heat_time_order_interval": [0.85, 1.15],
        "cure_time_order_interval": [3.8, 4.4],
        "N_spatial_finest_relative_limit": 2e-4, "N_spatial_finest_linf_K_limit": 0.002,
        "N_temporal_finest_relative_limit": 4e-4, "error_contamination_fraction_limit": 0.1,
        "steady_finest_relative_limit": 0.002, "steady_finest_linf_K_limit": 0.05,
        "steady_change_K_limit": 1e-7, "cure_finest_alpha_linf_limit": 1e-6,
        "cure_finest_temperature_linf_K_limit": 0.03,
        "reaction_heat_identity_K_limit": 1e-9,
        "reference_field_axis_variation_K_min": 0.01,
    }


def assess(records: list[dict], output: Path) -> dict:
    p = protocol()
    checks: list[dict] = []
    orders = {}

    def check(name: str, passed: bool, **details: object) -> None:
        checks.append({"name": name, "passed": bool(passed), **details})

    def selected(study: str) -> list[dict]:
        return [r for r in records if r["study"] == study and "exception" not in r]

    def order_check(study: str, key: str, bounds: list[float]) -> None:
        values = [r["errors"][-1][key] for r in selected(study)]
        rates = [float(np.log2(a / b)) for a, b in zip(values[:-1], values[1:])]
        orders[study] = rates
        check(study + "_order", len(values) == 3 and all(bounds[0] <= x <= bounds[1] for x in rates),
              observed_orders=rates, acceptance_interval=bounds)

    for r in records:
        check(r["run_id"] + "_completed", "exception" not in r)
        if "exception" in r:
            continue
        check(r["run_id"] + "_finite_and_coupled", r["finite"] and r["diagnostics"]["all_coupling_steps_converged"])
        check(r["run_id"] + "_nonconstant_xz", r["x_variation_K"] > 0.01 and r["z_variation_K"] > 0.01)
        if "steady_change_K" in r:
            check(r["run_id"] + "_steady", r["steady_change_K"] < p["steady_change_K_limit"], value_K=r["steady_change_K"])
        if "mean_reaction_heat_identity_error_K" in r:
            check(r["run_id"] + "_heat_identity", r["mean_reaction_heat_identity_error_K"] < p["reaction_heat_identity_K_limit"],
                  value_K=r["mean_reaction_heat_identity_error_K"])
    for study in ("N_spatial", "R_spatial", "I_spatial"):
        order_check(study, "temperature_l2_K", p["spatial_order_interval"])
    order_check("N_temporal", "temperature_l2_K", p["heat_time_order_interval"])
    order_check("C_temporal", "alpha_linf", p["cure_time_order_interval"])
    for study, rel, linf in [("N_spatial", 2e-4, 0.002), ("N_temporal", 4e-4, None),
                             ("R_spatial", 0.002, 0.05), ("I_spatial", 0.002, 0.05)]:
        runs = selected(study)
        if len(runs) == 3:
            error = runs[-1]["errors"][-1]
            check(study + "_finest_error", error["temperature_relative_l2_perturbation"] < rel and
                  (linf is None or error["temperature_linf_K"] < linf), errors=error)
    if len(selected("C_temporal")) == 3:
        error = selected("C_temporal")[-1]["errors"][-1]
        check("C_temporal_finest_error", error["alpha_linf"] < 1e-6 and error["temperature_linf_K"] < 0.03, errors=error)
    if selected("N_spatial") and selected("N_spatial_control"):
        a = selected("N_spatial")[-1]
        b = selected("N_spatial_control")[0]
        with np.load(output / f"{a['run_id']}.npz") as aa, np.load(output / f"{b['run_id']}.npz") as bb:
            difference = aa["temperature_K"][-1] - bb["temperature_K"][-1]
            difference_l2 = float(np.sqrt(np.mean(difference**2)))
        fraction = difference_l2 / a["errors"][-1]["temperature_l2_K"]
        check("N_spatial_time_contamination", fraction < 0.1, fraction_of_error=fraction)
    if selected("N_temporal") and selected("N_temporal_control"):
        a = selected("N_temporal")[-1]["errors"][-1]["temperature_l2_K"]
        b = selected("N_temporal_control")[0]["errors"][-1]["temperature_l2_K"]
        fraction = abs(a - b) / a
        check("N_temporal_space_contamination", fraction < 0.1, fraction_of_error=fraction)
    return {"passed": all(c["passed"] for c in checks), "checks": checks, "observed_orders": orders}


def make_plot(records: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.7), constrained_layout=True)
    specs = [(axes[0, 0], ["N_spatial"], "nx", "temperature_l2_K", "Transient heat: space", "Cells along x", "Temperature RMS error (K)", 2, -1),
             (axes[0, 1], ["N_temporal"], "maximum_step_s", "temperature_l2_K", "Transient heat: time", "Time step (s)", "Temperature RMS error (K)", 1, 1),
             (axes[1, 0], ["R_spatial", "I_spatial"], "nx", "temperature_l2_K", "Steady Robin / interface: space", "Cells along x", "Temperature RMS error (K)", 2, -1),
             (axes[1, 1], ["C_temporal"], "maximum_step_s", "alpha_linf", "Special-case cure: time", "Time step (s)", "Cure maximum absolute error", 4, 1)]
    for ax, studies, xkey, ykey, title, xlabel, ylabel, expected, sign in specs:
        for j, study in enumerate(studies):
            rows = [r for r in records if r["study"] == study and "exception" not in r]
            x = np.array([r[xkey] for r in rows], dtype=float)
            y = np.array([r["errors"][-1][ykey] for r in rows])
            if len(x):
                label = {"R_spatial": "Uniform kz", "I_spatial": "Discontinuous kz"}.get(study, "Computed error")
                ax.loglog(x, y, marker=["o", "s"][j], linewidth=1.7, label=label,
                          color=["#245A8D", "#B35C23"][j])
                if j == 0:
                    ax.loglog(x, y[0] * (x / x[0]) ** (sign * expected), "--", color="#737373",
                              linewidth=1.2, label=f"Order {expected} guide")
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.grid(True, which="both", alpha=0.16)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Independent continuous-solution verification of the production 2-D solver", fontsize=12)
    fig.savefig(output / "convergence.png", dpi=180)
    fig.savefig(output / "convergence.pdf")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--environment-label", required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True)
    state = subprocess.run(["git", "status", "--short"], cwd=REPO, capture_output=True, text=True)
    environment = {"label": args.environment_label, "platform": platform.platform(),
                   "python": sys.version, "executable": sys.executable, "processor": platform.processor(),
                   "cpu_count": os.cpu_count(), "thread_limit": 2, "gpu_used": False,
                   "packages": {name: importlib.metadata.version(name) for name in ["numpy", "scipy", "matplotlib"]},
                   "git_head": git.stdout.strip(), "git_status_short": state.stdout,
                   "source_sha256": {name: sha256(REPO / name) for name in PROTOCOL_FILES}}
    plan = {"started_utc": datetime.now(timezone.utc).isoformat(), "environment": environment,
            "protocol": protocol()}
    write_json(output / "plan_before_execution.json", plan)
    configurations = []
    for nx, nz in SPATIAL_GRIDS:
        configurations.append(("N", nx, nz, 0.02, [0, 50, 100], f"N_space_{nx}x{nz}", "N_spatial"))
    configurations.append(("N", 48, 32, 0.01, [0, 50, 100], "N_space_time_control", "N_spatial_control"))
    for step in TEMPORAL_STEPS:
        configurations.append(("N", 96, 64, step, [0, 100], f"N_time_{step:g}", "N_temporal"))
    configurations.append(("N", 192, 128, 5.0, [0, 100], "N_time_space_control", "N_temporal_control"))
    for case in ("R", "I"):
        for nx, nz in SPATIAL_GRIDS:
            configurations.append((case, nx, nz, 1000.0, [0, 100000, 200000],
                                   f"{case}_space_{nx}x{nz}", f"{case}_spatial"))
    for step in TEMPORAL_STEPS:
        configurations.append(("C", 12, 8, step, [0, 100], f"C_time_{step:g}", "C_temporal"))
    write_json(output / "case_schedule.json", configurations)
    records = []
    for case, nx, nz, step, times, run_id, study in configurations:
        print(f"Running {run_id}", flush=True)
        try:
            record = run_case(case, nx, nz, step, np.array(times, dtype=float), run_id, study, output)
        except Exception:
            record = {"run_id": run_id, "case": case, "study": study, "exception": traceback.format_exc()}
            write_json(output / f"{run_id}.json", record)
        records.append(record)
        print(json.dumps(record.get("errors", record.get("exception"))), flush=True)
    summary = {**plan, "assessment": assess(records, output), "runs": records,
               "total_wall_seconds_before_plot": perf_counter() - started}
    write_json(output / "summary.json", summary)
    rows = [{"run_id": r["run_id"], "case": r["case"], "study": r["study"],
             "nx": r["nx"], "nz": r["nz"], "maximum_step_s": r["maximum_step_s"], **e}
            for r in records if "exception" not in r for e in r["errors"]]
    if rows:
        with (output / "errors.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        make_plot(records, output)
    write_json(output / "checksums.json", {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()})
    print(json.dumps(summary["assessment"], indent=2), flush=True)
    return 0 if summary["assessment"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
