"""Generate deterministic parameterized 1-D source cases for P3 pretraining."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from cdcureno.physics import PublicCase1Material
from cdcureno.solvers import (
    LayeredGrid1D,
    RobinBoundaries,
    layered_tool_composite_grid,
    simulate_cure_1d,
)


ROOT = Path(__file__).resolve().parents[1]
SEED = 20260723
FAMILIES = ("single_hold", "two_hold", "smart_cure", "three_hold")
FAMILY_TO_ID = {name: index for index, name in enumerate(FAMILIES)}
CASES_PER_FAMILY = 100
TIME_COUNT = 223
SPACE_COUNT = 51
MAXIMUM_STEP_S = 20.0
DEFAULT_DATASET_ID = "p3_source_1d_v3"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / f"{DEFAULT_DATASET_ID}.npz"
DEFAULT_MANIFEST = ROOT / "splits" / f"{DEFAULT_DATASET_ID}.json"
DEFAULT_SUMMARY = (
    ROOT / "outputs" / "tables" / f"{DEFAULT_DATASET_ID}_generation.json"
)


def _piecewise_cycle(
    control_times_min: list[float], control_temperatures_K: list[float]
) -> np.ndarray:
    output_times = np.arange(TIME_COUNT, dtype=np.float64)
    return np.interp(output_times, control_times_min, control_temperatures_K)


def _make_cycle(family: str, rng: np.random.Generator) -> np.ndarray:
    if family == "single_hold":
        ramp_end = rng.uniform(40.0, 80.0)
        hold_end = rng.uniform(145.0, 195.0)
        high = rng.uniform(430.0, 475.0)
        return _piecewise_cycle(
            [0.0, ramp_end, hold_end, 222.0],
            [293.0, high, high, 293.0],
        )
    if family == "two_hold":
        first_ramp_end = rng.uniform(25.0, 50.0)
        first_hold_end = first_ramp_end + rng.uniform(20.0, 45.0)
        second_ramp_end = first_hold_end + rng.uniform(20.0, 45.0)
        second_hold_end = rng.uniform(max(second_ramp_end + 45.0, 160.0), 200.0)
        low = rng.uniform(350.0, 405.0)
        high = rng.uniform(max(low + 35.0, 430.0), 475.0)
        return _piecewise_cycle(
            [
                0.0,
                first_ramp_end,
                first_hold_end,
                second_ramp_end,
                second_hold_end,
                222.0,
            ],
            [293.0, low, low, high, high, 293.0],
        )
    if family == "smart_cure":
        first_ramp_end = rng.uniform(30.0, 55.0)
        first_hold_end = first_ramp_end + rng.uniform(10.0, 25.0)
        cool_end = first_hold_end + rng.uniform(20.0, 35.0)
        cool_hold_end = cool_end + rng.uniform(10.0, 25.0)
        second_ramp_end = cool_hold_end + rng.uniform(20.0, 35.0)
        second_hold_end = rng.uniform(max(second_ramp_end + 30.0, 175.0), 205.0)
        first_high = rng.uniform(425.0, 460.0)
        dip = rng.uniform(350.0, 400.0)
        second_high = rng.uniform(max(first_high, 445.0), 480.0)
        return _piecewise_cycle(
            [
                0.0,
                first_ramp_end,
                first_hold_end,
                cool_end,
                cool_hold_end,
                second_ramp_end,
                second_hold_end,
                222.0,
            ],
            [
                293.0,
                first_high,
                first_high,
                dip,
                dip,
                second_high,
                second_high,
                293.0,
            ],
        )
    if family == "three_hold":
        t1 = rng.uniform(20.0, 35.0)
        t2 = t1 + rng.uniform(15.0, 25.0)
        t3 = t2 + rng.uniform(15.0, 25.0)
        t4 = t3 + rng.uniform(15.0, 25.0)
        t5 = t4 + rng.uniform(15.0, 25.0)
        t6 = rng.uniform(max(t5 + 40.0, 175.0), 205.0)
        low = rng.uniform(350.0, 390.0)
        middle = rng.uniform(max(low + 20.0, 400.0), 435.0)
        high = rng.uniform(max(middle + 20.0, 445.0), 480.0)
        return _piecewise_cycle(
            [0.0, t1, t2, t3, t4, t5, t6, 222.0],
            [293.0, low, low, middle, middle, high, high, 293.0],
        )
    raise ValueError(f"Unknown family: {family}")


def _parameter_record(
    case_id: int, family: str, rng: np.random.Generator
) -> dict[str, float | int | str | np.ndarray]:
    tool_mm = int(rng.integers(10, 41))
    composite_mm = int(rng.integers(10, 41))
    return {
        "case_id": case_id,
        "family": family,
        "family_id": FAMILY_TO_ID[family],
        "air_temperature_K": _make_cycle(family, rng),
        "tool_thickness_m": tool_mm / 1000.0,
        "composite_thickness_m": composite_mm / 1000.0,
        "lower_h_W_m2_K": float(rng.uniform(40.0, 100.0)),
        "upper_h_W_m2_K": float(rng.uniform(80.0, 160.0)),
        "composite_conductivity_scale": float(rng.uniform(0.8, 1.2)),
        "heat_of_reaction_scale": float(rng.uniform(0.9, 1.1)),
    }


def _run_case(
    record: dict[str, float | int | str | np.ndarray],
) -> dict[str, object]:
    base = PublicCase1Material()
    material = replace(
        base,
        tool_thickness_m=float(record["tool_thickness_m"]),
        composite_thickness_m=float(record["composite_thickness_m"]),
    )
    grid = layered_tool_composite_grid(0.001, material)
    conductivity = grid.conductivity_W_m_K.copy()
    conductivity[grid.composite_mask] *= float(
        record["composite_conductivity_scale"]
    )
    source = grid.cure_source_J_m3_per_alpha.copy()
    source[grid.composite_mask] *= float(record["heat_of_reaction_scale"])
    conditioned_grid = LayeredGrid1D(
        z_m=grid.z_m,
        control_volume_width_m=grid.control_volume_width_m,
        composite_mask=grid.composite_mask,
        density_kg_m3=grid.density_kg_m3,
        specific_heat_J_kg_K=grid.specific_heat_J_kg_K,
        conductivity_W_m_K=conductivity,
        cure_source_J_m3_per_alpha=source,
    )
    air = np.asarray(record["air_temperature_K"], dtype=np.float64)
    times = np.arange(TIME_COUNT, dtype=np.float64) * 60.0
    started = time.perf_counter()
    result = simulate_cure_1d(
        times,
        air,
        grid=conditioned_grid,
        boundaries=RobinBoundaries(
            float(record["lower_h_W_m2_K"]),
            float(record["upper_h_W_m2_K"]),
        ),
        maximum_step_s=MAXIMUM_STEP_S,
        maximum_coupling_iterations=16,
    )
    inert_grid = LayeredGrid1D(
        z_m=conditioned_grid.z_m,
        control_volume_width_m=conditioned_grid.control_volume_width_m,
        composite_mask=np.zeros_like(conditioned_grid.composite_mask),
        density_kg_m3=conditioned_grid.density_kg_m3,
        specific_heat_J_kg_K=conditioned_grid.specific_heat_J_kg_K,
        conductivity_W_m_K=conditioned_grid.conductivity_W_m_K,
        cure_source_J_m3_per_alpha=np.zeros_like(
            conditioned_grid.cure_source_J_m3_per_alpha
        ),
    )
    inert_result = simulate_cure_1d(
        times,
        air,
        grid=inert_grid,
        boundaries=RobinBoundaries(
            float(record["lower_h_W_m2_K"]),
            float(record["upper_h_W_m2_K"]),
        ),
        maximum_step_s=MAXIMUM_STEP_S,
        maximum_coupling_iterations=2,
    )
    coarse_material = material
    coarse_grid = layered_tool_composite_grid(
        (material.tool_thickness_m + material.composite_thickness_m) / 16.0,
        coarse_material,
    )
    coarse_conductivity = coarse_grid.conductivity_W_m_K.copy()
    coarse_conductivity[coarse_grid.composite_mask] *= float(
        record["composite_conductivity_scale"]
    )
    coarse_source = coarse_grid.cure_source_J_m3_per_alpha.copy()
    coarse_source[coarse_grid.composite_mask] *= float(
        record["heat_of_reaction_scale"]
    )
    conditioned_coarse_grid = LayeredGrid1D(
        z_m=coarse_grid.z_m,
        control_volume_width_m=coarse_grid.control_volume_width_m,
        composite_mask=coarse_grid.composite_mask,
        density_kg_m3=coarse_grid.density_kg_m3,
        specific_heat_J_kg_K=coarse_grid.specific_heat_J_kg_K,
        conductivity_W_m_K=coarse_conductivity,
        cure_source_J_m3_per_alpha=coarse_source,
    )
    coarse_result = simulate_cure_1d(
        times,
        air,
        grid=conditioned_coarse_grid,
        boundaries=RobinBoundaries(
            float(record["lower_h_W_m2_K"]),
            float(record["upper_h_W_m2_K"]),
        ),
        maximum_step_s=60.0,
        maximum_coupling_iterations=16,
    )
    normalized_z = np.linspace(0.0, 1.0, SPACE_COUNT, dtype=np.float64)
    target_z = normalized_z * result.z_m[-1]
    temperature = np.stack(
        [np.interp(target_z, result.z_m, row) for row in result.temperature_K]
    )
    alpha = np.stack(
        [np.interp(target_z, result.z_m, row) for row in result.alpha]
    )
    inert_temperature = np.stack(
        [
            np.interp(target_z, inert_result.z_m, row)
            for row in inert_result.temperature_K
        ]
    )
    coarse_temperature = np.stack(
        [
            np.interp(target_z, coarse_result.z_m, row)
            for row in coarse_result.temperature_K
        ]
    )
    coarse_alpha = np.stack(
        [
            np.interp(target_z, coarse_result.z_m, row)
            for row in coarse_result.alpha
        ]
    )
    interface_fraction = (
        float(record["tool_thickness_m"]) / result.z_m[-1]
    )
    composite_mask = target_z > float(record["tool_thickness_m"])
    alpha[:, ~composite_mask] = 0.0
    return {
        "case_id": int(record["case_id"]),
        "family_id": int(record["family_id"]),
        "air_temperature_K": air.astype(np.float32),
        "temperature_K": temperature.astype(np.float32),
        "inert_temperature_K": inert_temperature.astype(np.float32),
        "coarse_temperature_K": coarse_temperature.astype(np.float32),
        "coarse_alpha": coarse_alpha.astype(np.float32),
        "alpha": alpha.astype(np.float32),
        "composite_mask": composite_mask.astype(np.float32),
        "signed_distance_fraction": (
            normalized_z - interface_fraction
        ).astype(np.float32),
        "parameters": np.array(
            [
                record["tool_thickness_m"],
                record["composite_thickness_m"],
                record["lower_h_W_m2_K"],
                record["upper_h_W_m2_K"],
                record["composite_conductivity_scale"],
                record["heat_of_reaction_scale"],
            ],
            dtype=np.float32,
        ),
        "maximum_relative_global_energy_residual": (
            max(
                result.diagnostics.maximum_relative_global_energy_residual,
                inert_result.diagnostics.maximum_relative_global_energy_residual,
                coarse_result.diagnostics.maximum_relative_global_energy_residual,
            )
        ),
        "maximum_abs_energy_residual_W_m3": max(
            result.diagnostics.maximum_abs_energy_residual_W_m3,
            inert_result.diagnostics.maximum_abs_energy_residual_W_m3,
            coarse_result.diagnostics.maximum_abs_energy_residual_W_m3,
        ),
        "wall_seconds": time.perf_counter() - started,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cases-per-family", type=int, default=CASES_PER_FAMILY)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    records = []
    case_id = 0
    for family in FAMILIES:
        for _ in range(args.cases_per_family):
            records.append(_parameter_record(case_id, family, rng))
            case_id += 1
    wall_started = time.perf_counter()
    if args.workers == 1:
        rows = [_run_case(record) for record in records]
    else:
        context = mp.get_context("spawn")
        with context.Pool(args.workers) as pool:
            rows = list(pool.imap_unordered(_run_case, records, chunksize=1))
        rows.sort(key=lambda row: int(row["case_id"]))
    generation_wall_seconds = time.perf_counter() - wall_started
    arrays = {
        key: np.stack([np.asarray(row[key]) for row in rows])
        for key in (
            "air_temperature_K",
            "temperature_K",
            "inert_temperature_K",
            "coarse_temperature_K",
            "coarse_alpha",
            "alpha",
            "composite_mask",
            "signed_distance_fraction",
            "parameters",
        )
    }
    arrays["case_ids"] = np.array(
        [row["case_id"] for row in rows], dtype=np.int64
    )
    arrays["family_ids"] = np.array(
        [row["family_id"] for row in rows], dtype=np.int64
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    train: list[int] = []
    validation: list[int] = []
    in_family_test: list[int] = []
    held_out_family_test: dict[str, list[int]] = {}
    for family in FAMILIES:
        ids = [
            int(row["case_id"])
            for row in rows
            if int(row["family_id"]) == FAMILY_TO_ID[family]
        ]
        if family in ("single_hold", "two_hold"):
            train.extend(ids[: int(0.8 * len(ids))])
            validation.extend(ids[int(0.8 * len(ids)) : int(0.9 * len(ids))])
            in_family_test.extend(ids[int(0.9 * len(ids)) :])
        else:
            held_out_family_test[family] = ids
    manifest = {
        "schema_version": 1,
        "dataset_id": args.dataset_id,
        "seed": SEED,
        "family_to_id": FAMILY_TO_ID,
        "splits": {
            "train": train,
            "validation": validation,
            "in_family_test": in_family_test,
            "held_out_family_test": held_out_family_test,
        },
        "rules": {
            "training_families": ["single_hold", "two_hold"],
            "held_out_families": ["smart_cure", "three_hold"],
            "labels_from_held_out_families_used_in_training": False,
        },
        "array_artifact": str(args.output.relative_to(ROOT)),
        "array_sha256": _sha256(args.output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    maximum_local = max(
        float(row["maximum_abs_energy_residual_W_m3"]) for row in rows
    )
    maximum_global = max(
        float(row["maximum_relative_global_energy_residual"]) for row in rows
    )
    summary = {
        "schema_version": 1,
        "phase": "P3",
        "dataset_id": args.dataset_id,
        "case_count": len(rows),
        "case_count_by_family": {
            family: args.cases_per_family for family in FAMILIES
        },
        "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
        "array_sha256": _sha256(args.output),
        "generation_wall_seconds": generation_wall_seconds,
        "summed_case_wall_seconds": float(
            sum(float(row["wall_seconds"]) for row in rows)
        ),
        "maximum_abs_energy_residual_W_m3": maximum_local,
        "maximum_relative_global_energy_residual": maximum_global,
        "energy_acceptance": {
            "maximum_abs_energy_residual_W_m3_max": 1.0e-4,
            "maximum_relative_global_energy_residual_max": 1.0e-8,
        },
        "passed": bool(maximum_local <= 1.0e-4 and maximum_global <= 1.0e-8),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
