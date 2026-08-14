"""Plan, generate, resume, and finalize the frozen P4 true-2-D benchmark."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
import ctypes
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np
import scipy
from scipy.stats import qmc
import yaml

from cdcureno.physics import PublicCase1Material
from cdcureno.solvers import (
    LayeredGrid2D,
    RobinBoundaries2D,
    rectangular_tool_composite_grid,
    simulate_cure_2d,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "data" / "p4_2d_benchmark_v1.yaml"
REQUIRED_DESIGN_DIMENSIONS = (
    "cycle_family_selector",
    "cycle_p0",
    "cycle_p1",
    "cycle_p2",
    "cycle_p3",
    "cycle_p4",
    "cycle_p5",
    "cycle_p6",
    "cycle_p7",
    "cycle_p8",
    "difficulty_selector",
    "difficulty_variant",
    "bottom_htc",
    "top_htc_center",
    "top_htc_amplitude",
    "top_htc_frequency",
    "top_htc_phase",
    "edge_htc",
    "edge_asymmetry",
    "conductivity_scale",
    "reaction_enthalpy_scale",
)
REQUIRED_CORE_SPLITS = (
    "train",
    "validation",
    "id_test",
    "cycle_ood",
    "htc_ood",
    "pattern_ood",
    "combined_ood",
)
ACCEPTANCE_DIAGNOSTIC_THRESHOLDS = (
    (
        "maximum_abs_energy_residual_W_m3",
        "maximum_abs_energy_residual_W_m3_max",
    ),
    (
        "maximum_relative_global_energy_residual",
        "maximum_relative_global_energy_residual_max",
    ),
    (
        "maximum_interface_flux_imbalance_W",
        "maximum_interface_flux_imbalance_W_max",
    ),
    (
        "maximum_temperature_interface_jump_K",
        "maximum_temperature_interface_jump_K_max",
    ),
    (
        "maximum_robin_flux_imbalance_W",
        "maximum_robin_flux_imbalance_W_max",
    ),
    (
        "maximum_converged_coupling_update_K",
        "maximum_converged_coupling_update_K_max",
    ),
)
REQUIRED_ACCEPTANCE_KEYS = {
    *(threshold for _, threshold in ACCEPTANCE_DIAGNOSTIC_THRESHOLDS),
    "alpha_bound_violation_count_max",
    "alpha_monotonicity_violation_count_max",
    "tool_alpha_nonzero_count_max",
    "finite_field_required",
    "all_coupling_steps_converged_required",
    "silent_failure_count_max",
}


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _pretty_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(_canonical_json_bytes(list(contiguous.shape)))
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return repr(value)
    return value


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_immutable_bytes(path: Path, payload: bytes) -> str:
    """Write once, permit byte-identical replay, and reject replacement."""

    if path.exists():
        existing = path.read_bytes()
        if existing != payload:
            raise FileExistsError(
                f"Refusing to overwrite non-identical frozen artifact: {path}"
            )
        return "verified_existing"
    _atomic_write_bytes(path, payload)
    return "created"


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    try:
        with temporary.open("wb") as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _portable_path(
    path: Path,
    root: Path = ROOT,
    *,
    require_repo_relative: bool = False,
) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        if require_repo_relative:
            raise ValueError(
                f"Canonical P4 path must be inside the repository: {resolved}"
            ) from None
        return resolved.as_posix()


def _validate_repo_relative_posix_path(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty repository-relative path.")
    if "\\" in value:
        raise ValueError(f"{label} must use POSIX separators.")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or Path(value).is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{label} must be a normalized repository-relative path.")
    return value


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _code_hashes(root: Path = ROOT) -> dict[str, str]:
    relative_paths = (
        "scripts/generate_p4_2d.py",
        "src/cdcureno/physics/as4_8552.py",
        "src/cdcureno/solvers/conservative_2d.py",
    )
    missing = [name for name in relative_paths if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing code files required for P4 provenance: "
            + ", ".join(missing)
        )
    return {name: _sha256_file(root / name) for name in relative_paths}


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load and validate the authoritative P4 YAML configuration."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("P4 configuration must be a YAML mapping.")
    if payload.get("schema_version") != 2:
        raise ValueError("P4 configuration schema_version must be 2.")
    if payload.get("phase") != "P4":
        raise ValueError("P4 configuration phase must be P4.")
    dimensions = tuple(payload["sampling"]["dimensions"])
    if dimensions != REQUIRED_DESIGN_DIMENSIONS:
        raise ValueError(
            "sampling.dimensions must be the ordered, non-overlapping P4 "
            f"dimension list: {list(REQUIRED_DESIGN_DIMENSIONS)}"
        )
    if len(set(dimensions)) != len(dimensions):
        raise ValueError("sampling.dimensions contains duplicate names.")
    geometry = payload["geometry"]
    material = PublicCase1Material()
    if not np.isclose(geometry["tool_thickness_m"], material.tool_thickness_m):
        raise ValueError("Configured tool thickness differs from sourced material.")
    if not np.isclose(
        geometry["composite_thickness_m"], material.composite_thickness_m
    ):
        raise ValueError(
            "Configured composite thickness differs from sourced material."
        )
    material_config = payload["material"]
    if material_config["layup"] != "unidirectional_fibres_along_x":
        raise ValueError("P4 canonical layup must keep fibres aligned with x.")
    if (
        float(material_config["composite_base_conductivity_x_W_m_K"]) <= 0.0
        or float(material_config["composite_base_conductivity_z_W_m_K"]) <= 0.0
    ):
        raise ValueError("Configured anisotropic conductivities must be positive.")
    if not np.isclose(
        float(material_config["composite_base_conductivity_x_W_m_K"]),
        material.composite_longitudinal_k_W_m_K,
    ):
        raise ValueError("Configured x conductivity differs from sourced material.")
    if not np.isclose(
        float(material_config["composite_base_conductivity_z_W_m_K"]),
        material.composite_k_W_m_K,
    ):
        raise ValueError("Configured z conductivity differs from sourced material.")
    duration = float(payload["time"]["duration_s"])
    interval = float(payload["time"]["output_interval_s"])
    if duration <= 0 or interval <= 0 or not np.isclose(duration % interval, 0.0):
        raise ValueError("Time duration must be exactly divisible by output interval.")
    for tier, settings in payload["tiers"].items():
        raw_count = settings["case_count"]
        if isinstance(raw_count, bool) or not isinstance(raw_count, int):
            raise ValueError(f"Tier {tier} case_count must be an integer.")
        count = raw_count
        if count <= 0 or count & (count - 1):
            raise ValueError(f"Tier {tier} case_count must be a positive power of 2.")
        raw_split_counts = settings["split_counts"]
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in raw_split_counts.values()
        ):
            raise ValueError(f"Tier {tier} split counts must be positive integers.")
        split_total = sum(raw_split_counts.values())
        if split_total != count:
            raise ValueError(
                f"Tier {tier} split counts sum to {split_total}, expected {count}."
            )
        raw_budgets = settings["nested_training_budgets"]
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in raw_budgets
        ):
            raise ValueError(f"Tier {tier} budgets must contain only integers.")
        budgets = list(raw_budgets)
        if budgets != sorted(set(budgets)):
            raise ValueError(f"Tier {tier} budgets must be unique and increasing.")
        if any(value <= 0 for value in budgets):
            raise ValueError(f"Tier {tier} budgets must be strictly positive.")
        if budgets:
            train_count = int(settings["split_counts"].get("train", 0))
            if budgets[-1] > train_count:
                raise ValueError(f"Tier {tier} budget exceeds its training pool.")
        width = float(geometry["width_m"])
        total_z = float(geometry["tool_thickness_m"]) + float(
            geometry["composite_thickness_m"]
        )
        dx = float(settings["spacing_x_m"])
        dz = float(settings["spacing_z_m"])
        if (
            dx <= 0
            or dz <= 0
            or not np.isclose(round(width / dx) * dx, width)
            or not np.isclose(round(total_z / dz) * dz, total_z)
        ):
            raise ValueError(f"Tier {tier} spacings must divide the domain.")
        if float(settings["maximum_step_s"]) <= 0:
            raise ValueError(f"Tier {tier} maximum_step_s must be positive.")
    if tuple(payload["tiers"]["core"]["split_counts"]) != REQUIRED_CORE_SPLITS:
        raise ValueError(
            f"Core split order must be {list(REQUIRED_CORE_SPLITS)}."
        )
    if float(payload["tiers"]["core"]["maximum_step_s"]) != 10.0:
        raise ValueError("Canonical core maximum_step_s must remain frozen at 10 s.")
    core_budgets = [
        int(value)
        for value in payload["tiers"]["core"]["nested_training_budgets"]
    ]
    if not core_budgets or core_budgets[-1] != int(
        payload["tiers"]["core"]["split_counts"]["train"]
    ):
        raise ValueError(
            "Canonical core budgets must be nonempty and end at the train count."
        )
    acceptance = payload.get("acceptance")
    if not isinstance(acceptance, dict) or set(acceptance) != REQUIRED_ACCEPTANCE_KEYS:
        raise ValueError(
            "P4 acceptance must contain exactly the prespecified invariant and "
            "diagnostic thresholds."
        )
    for _, threshold in ACCEPTANCE_DIAGNOSTIC_THRESHOLDS:
        raw_value = acceptance[threshold]
        if isinstance(raw_value, bool) or not isinstance(
            raw_value, (int, float)
        ):
            raise ValueError(f"Acceptance threshold {threshold} must be numeric.")
        value = float(raw_value)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"Acceptance threshold {threshold} must be nonnegative.")
    for threshold in (
        "alpha_bound_violation_count_max",
        "alpha_monotonicity_violation_count_max",
        "tool_alpha_nonzero_count_max",
        "silent_failure_count_max",
    ):
        value = acceptance[threshold]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise ValueError(f"Acceptance threshold {threshold} must be an integer.")
    for requirement in (
        "finite_field_required",
        "all_coupling_steps_converged_required",
    ):
        if not isinstance(acceptance[requirement], bool):
            raise ValueError(f"Acceptance requirement {requirement} must be boolean.")
    if (
        not acceptance["finite_field_required"]
        or not acceptance["all_coupling_steps_converged_required"]
        or int(acceptance["alpha_bound_violation_count_max"]) != 0
        or int(acceptance["alpha_monotonicity_violation_count_max"]) != 0
        or int(acceptance["tool_alpha_nonzero_count_max"]) != 0
        or int(acceptance["silent_failure_count_max"]) != 0
    ):
        raise ValueError(
            "Canonical P4 invariant acceptance must require finite/converged "
            "solutions and zero alpha or silent-failure violations."
        )
    return payload


def _time_axis(config: dict[str, Any]) -> np.ndarray:
    duration = float(config["time"]["duration_s"])
    interval = float(config["time"]["output_interval_s"])
    count = int(round(duration / interval)) + 1
    return np.arange(count, dtype=np.float64) * interval


def _linear_cycle(
    time_s: np.ndarray,
    control_minutes: list[float],
    control_temperature_K: list[float],
) -> np.ndarray:
    return np.interp(
        time_s / 60.0, control_minutes, control_temperature_K
    ).astype(np.float64)


def _cycle(family: str, u: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    if family == "single_hold":
        ramp = 40.0 + 35.0 * u[0]
        hold_end = 150.0 + 45.0 * u[1]
        high = 430.0 + 45.0 * u[2]
        return _linear_cycle(
            time_s,
            [0.0, ramp, hold_end, 222.0],
            [293.0, high, high, 293.0],
        )
    if family == "two_hold":
        t1 = 25.0 + 20.0 * u[0]
        t2 = t1 + 20.0 + 25.0 * u[1]
        t3 = t2 + 20.0 + 25.0 * u[2]
        t4 = 165.0 + 35.0 * u[3]
        low = 355.0 + 45.0 * u[4]
        high = max(low + 35.0, 430.0 + 45.0 * u[5])
        return _linear_cycle(
            time_s,
            [0.0, t1, t2, t3, t4, 222.0],
            [293.0, low, low, high, high, 293.0],
        )
    if family == "smart_cure":
        t1 = 30.0 + 20.0 * u[0]
        t2 = t1 + 12.0 + 15.0 * u[1]
        t3 = t2 + 20.0 + 15.0 * u[2]
        t4 = t3 + 12.0 + 15.0 * u[3]
        t5 = t4 + 20.0 + 15.0 * u[4]
        t6 = 180.0 + 25.0 * u[5]
        first = 425.0 + 30.0 * u[6]
        dip = 355.0 + 40.0 * u[7]
        high = max(first, 445.0 + 30.0 * u[8])
        return _linear_cycle(
            time_s,
            [0.0, t1, t2, t3, t4, t5, t6, 222.0],
            [293.0, first, first, dip, dip, high, high, 293.0],
        )
    if family == "three_hold":
        t1 = 20.0 + 12.0 * u[0]
        t2 = t1 + 15.0 + 10.0 * u[1]
        t3 = t2 + 15.0 + 10.0 * u[2]
        t4 = t3 + 15.0 + 10.0 * u[3]
        t5 = t4 + 15.0 + 10.0 * u[4]
        t6 = 180.0 + 25.0 * u[5]
        low = 350.0 + 35.0 * u[6]
        middle = max(low + 20.0, 400.0 + 30.0 * u[7])
        high = max(middle + 20.0, 445.0 + 30.0 * u[8])
        return _linear_cycle(
            time_s,
            [0.0, t1, t2, t3, t4, t5, t6, 222.0],
            [293.0, low, low, middle, middle, high, high, 293.0],
        )
    raise ValueError(f"Unknown cycle family {family!r}.")


def _split_names(config: dict[str, Any], tier: str) -> list[str]:
    names: list[str] = []
    for split, count in config["tiers"][tier]["split_counts"].items():
        names.extend([str(split)] * int(count))
    return names


def _category_assignments(
    tier: str,
    case_id: int,
    split: str,
    raw: dict[str, float],
    config: dict[str, Any],
) -> tuple[str, str]:
    """Assign categories from dedicated Sobol coordinates, never reused values."""

    thresholds = config["sampling"]["category_thresholds"]
    cycle_selector = raw["cycle_family_selector"]
    if split in {"cycle_ood", "combined_ood"}:
        cycle_family = (
            "smart_cure"
            if cycle_selector < float(thresholds["ood_smart_cure_fraction"])
            else "three_hold"
        )
    else:
        cycle_family = (
            "single_hold"
            if cycle_selector < float(thresholds["id_single_hold_fraction"])
            else "two_hold"
        )
    if tier == "debug":
        debug_families = (
            "F0",
            "F0",
            "F1_smooth",
            "F1_smooth",
            "F1_piecewise",
            "F1_piecewise",
            "F2_left",
            "F2_both",
        )
        difficulty = debug_families[case_id]
    elif split == "pattern_ood":
        difficulty = (
            "F1_smooth"
            if raw["difficulty_variant"] < 0.5
            else "F1_piecewise"
        )
    elif split == "htc_ood":
        difficulty = (
            "F2_left" if raw["difficulty_variant"] < 0.5 else "F2_both"
        )
    elif split == "combined_ood":
        difficulty = "F1_F2_combined"
    else:
        selector = raw["difficulty_selector"]
        f0 = float(thresholds["f0_fraction"])
        f1_upper = float(thresholds["f1_upper_fraction"])
        if selector < f0:
            difficulty = "F0"
        elif selector < f1_upper:
            difficulty = (
                "F1_smooth"
                if raw["difficulty_variant"] < 0.5
                else "F1_piecewise"
            )
        else:
            difficulty = (
                "F2_left" if raw["difficulty_variant"] < 0.5 else "F2_both"
            )
    return cycle_family, difficulty


def _lerp(bounds: list[float], u: float) -> float:
    return float(bounds[0]) + (float(bounds[1]) - float(bounds[0])) * float(u)


def _select_frequency(values: list[int], u: float) -> int:
    index = min(int(float(u) * len(values)), len(values) - 1)
    return int(values[index])


def _build_case_definition(
    *,
    tier: str,
    case_id: int,
    split: str,
    raw: dict[str, float],
    config: dict[str, Any],
    time_s: np.ndarray,
) -> dict[str, Any]:
    settings = config["tiers"][tier]
    geometry = config["geometry"]
    width_m = float(geometry["width_m"])
    spacing_x_m = float(settings["spacing_x_m"])
    nx = int(round(width_m / spacing_x_m))
    x_m = (np.arange(nx, dtype=np.float64) + 0.5) * spacing_x_m
    cycle_family, difficulty = _category_assignments(
        tier, case_id, split, raw, config
    )
    cycle_u = np.asarray(
        [raw[f"cycle_p{index}"] for index in range(9)], dtype=np.float64
    )
    air = _cycle(cycle_family, cycle_u, time_s)
    is_htc_ood = split in {"htc_ood", "combined_ood"}
    is_pattern_ood = split in {"pattern_ood", "combined_ood"}
    ranges = config["sampling"]["ranges"]
    htc_range = ranges["ood"] if is_htc_ood else ranges["id"]
    pattern_range = ranges["ood"] if is_pattern_ood else ranges["id"]
    bottom_h = _lerp(htc_range["bottom_h_W_m2_K"], raw["bottom_htc"])
    requested_center = _lerp(
        htc_range["top_h_center_W_m2_K"], raw["top_htc_center"]
    )
    requested_amplitude = _lerp(
        (
            ranges["ood"]["top_h_amplitude_W_m2_K"]
            if is_htc_ood
            else ranges["id"]["top_h_amplitude_W_m2_K"]
        ),
        raw["top_htc_amplitude"],
    )
    requested_frequency = _select_frequency(
        pattern_range["top_h_frequencies_per_width"],
        raw["top_htc_frequency"],
    )
    phase = 2.0 * np.pi * raw["top_htc_phase"]
    top_h = np.full(nx, requested_center, dtype=np.float64)
    realized_frequency = 0
    if difficulty in {"F1_smooth", "F1_F2_combined"}:
        profile = np.cos(
            2.0 * np.pi * requested_frequency * x_m / width_m + phase
        )
        profile -= np.mean(profile)
        top_h = requested_center + requested_amplitude * profile
        realized_frequency = requested_frequency
    elif difficulty == "F1_piecewise":
        # A square wave has two constant-sign bands per full cycle. Centering
        # its sampled profile keeps the requested center equal to the realized
        # spatial mean without changing the half peak-to-peak amplitude.
        bands = (
            np.floor(
                2.0 * requested_frequency * x_m / width_m + phase / np.pi
            ).astype(int)
            % 2
        )
        profile = np.where(bands == 0, -1.0, 1.0)
        profile -= np.mean(profile)
        top_h = requested_center + requested_amplitude * profile
        realized_frequency = requested_frequency
    edge_h = _lerp(htc_range["edge_h_W_m2_K"], raw["edge_htc"])
    left_h = 0.0
    right_h = 0.0
    if difficulty in {"F2_left", "F1_F2_combined"}:
        left_h = edge_h
    elif difficulty == "F2_both":
        left_h = edge_h
        right_h = edge_h * (0.35 + 0.45 * raw["edge_asymmetry"])
    if difficulty == "F1_F2_combined":
        right_h = edge_h * (0.25 + 0.50 * raw["edge_asymmetry"])
    if difficulty == "F0":
        left_h = 0.0
        right_h = 0.0
    common = ranges["common"]
    conductivity_scale = _lerp(
        common["composite_conductivity_scale"], raw["conductivity_scale"]
    )
    reaction_scale = _lerp(
        common["reaction_enthalpy_scale"], raw["reaction_enthalpy_scale"]
    )
    realized_amplitude = float(0.5 * np.ptp(top_h))
    if difficulty in {"F0", "F2_left", "F2_both"}:
        # These case families have a uniform realized top boundary.
        if not np.allclose(top_h, requested_center):
            raise AssertionError("Uniform-top family produced a varying top HTC.")
        realized_amplitude = 0.0
        realized_frequency = 0
    maximum_edge_h = max(left_h, right_h)
    material_config = config["material"]
    base_conductivity_x = float(
        material_config["composite_base_conductivity_x_W_m_K"]
    )
    base_conductivity_z = float(
        material_config["composite_base_conductivity_z_W_m_K"]
    )
    actual_edge_conductivity = base_conductivity_x * conductivity_scale
    return {
        "tier": tier,
        "case_id": int(case_id),
        "case_key": f"p4-{tier}-{case_id:04d}",
        "split": split,
        "difficulty_family": difficulty,
        "cycle_family": cycle_family,
        "raw_design_coordinates": {
            key: float(raw[key]) for key in REQUIRED_DESIGN_DIMENSIONS
        },
        "air_temperature_K": air.tolist(),
        "bottom_h_W_m2_K": bottom_h,
        "top_h_W_m2_K": top_h.tolist(),
        "requested_top_h_center_W_m2_K": requested_center,
        "requested_top_h_amplitude_W_m2_K": requested_amplitude,
        "requested_top_h_frequency_per_width": requested_frequency,
        "top_h_phase_rad": float(phase),
        "realized_top_h_mean_W_m2_K": float(np.mean(top_h)),
        "realized_top_h_min_W_m2_K": float(np.min(top_h)),
        "realized_top_h_max_W_m2_K": float(np.max(top_h)),
        "realized_top_h_amplitude_W_m2_K": realized_amplitude,
        "realized_top_h_frequency_per_width": int(realized_frequency),
        "requested_edge_h_W_m2_K": edge_h,
        "left_h_W_m2_K": float(left_h),
        "right_h_W_m2_K": float(right_h),
        "edge_Biot_number": float(
            maximum_edge_h * width_m / actual_edge_conductivity
        ),
        "composite_layup": material_config["layup"],
        "base_composite_conductivity_x_W_m_K": base_conductivity_x,
        "base_composite_conductivity_z_W_m_K": base_conductivity_z,
        "composite_conductivity_x_W_m_K": (
            base_conductivity_x * conductivity_scale
        ),
        "composite_conductivity_z_W_m_K": (
            base_conductivity_z * conductivity_scale
        ),
        "composite_conductivity_scale": conductivity_scale,
        "reaction_enthalpy_scale": reaction_scale,
    }


def build_case_plan(
    tier: str,
    *,
    config_path: Path = DEFAULT_CONFIG,
    config: dict[str, Any] | None = None,
    project_root: Path = ROOT,
) -> dict[str, Any]:
    """Build the deterministic, pre-label case plan for one tier."""

    active_config = config or load_config(config_path)
    if tier not in active_config["tiers"]:
        raise ValueError(f"Unknown tier {tier!r}.")
    settings = active_config["tiers"][tier]
    count = int(settings["case_count"])
    dimensions = tuple(active_config["sampling"]["dimensions"])
    tier_index = list(active_config["tiers"]).index(tier)
    design_seed = int(active_config["seed"]) + tier_index
    design = qmc.Sobol(d=len(dimensions), scramble=True, seed=design_seed)
    points = design.random_base2(m=int(np.log2(count)))
    splits = _split_names(active_config, tier)
    time_s = _time_axis(active_config)
    cases = []
    for case_id, point in enumerate(points):
        raw = {name: float(value) for name, value in zip(dimensions, point)}
        definition = _build_case_definition(
            tier=tier,
            case_id=case_id,
            split=splits[case_id],
            raw=raw,
            config=active_config,
            time_s=time_s,
        )
        cases.append(
            {
                "case_definition_sha256": _sha256_bytes(
                    _canonical_json_bytes(definition)
                ),
                "definition": definition,
            }
        )
    code_hashes = _code_hashes(project_root)
    plan: dict[str, Any] = {
        "schema_version": 2,
        "phase": "P4",
        "plan_role": "pre_label_case_plan",
        "dataset_id": (
            f"p4_2d_{tier}_{active_config.get('dataset_version', 'v1')}"
        ),
        "tier": tier,
        "tier_role": settings["role"],
        "case_count": count,
        "seed": int(active_config["seed"]),
        "design_seed": design_seed,
        "design": active_config["sampling"]["design"],
        "design_dimensions": list(dimensions),
        "config_path": _portable_path(
            config_path,
            project_root,
            require_repo_relative=settings["role"] == "canonical",
        ),
        "config_sha256": _sha256_file(config_path),
        "resolved_config": active_config,
        "code_sha256": code_hashes,
        "provenance": {
            "git_commit_sha": _git_commit(project_root),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "pyyaml_version": yaml.__version__,
        },
        "time_s": time_s.tolist(),
        "splits": {
            name: [index for index, value in enumerate(splits) if value == name]
            for name in active_config["tiers"][tier]["split_counts"]
        },
        "nested_training_budgets": [
            int(value) for value in settings["nested_training_budgets"]
        ],
        "cases": cases,
    }
    plan["plan_sha256"] = _sha256_bytes(_canonical_json_bytes(plan))
    return plan


def write_or_verify_plan(plan: dict[str, Any], path: Path) -> str:
    """Persist a pre-label plan without ever replacing a different plan."""

    return _write_immutable_bytes(path, _pretty_json_bytes(plan))


def _validate_plan(
    plan: dict[str, Any],
    *,
    tier: str,
    config: dict[str, Any],
    config_path: Path,
    project_root: Path = ROOT,
) -> None:
    stored_hash = plan.get("plan_sha256")
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if stored_hash != _sha256_bytes(_canonical_json_bytes(unsigned)):
        raise ValueError("Plan SHA-256 does not match its canonical content.")
    if plan.get("tier") != tier:
        raise ValueError(f"Plan tier {plan.get('tier')!r} does not match {tier!r}.")
    if config["tiers"][tier]["role"] == "canonical":
        _validate_repo_relative_posix_path(
            plan.get("config_path"), label="Plan config_path"
        )

    # A self-consistent stored plan is not sufficient: rebuild the complete
    # deterministic Sobol design from the current authoritative YAML and
    # relevant code, then compare every scientific/design field exactly.
    expected = build_case_plan(
        tier,
        config_path=config_path,
        config=config,
        project_root=project_root,
    )
    if set(plan) != set(expected):
        raise ValueError("Plan field inventory differs from the rebuilt plan.")
    scientific_plan = {
        key: value
        for key, value in plan.items()
        if key not in {"plan_sha256", "provenance"}
    }
    expected_scientific_plan = {
        key: value
        for key, value in expected.items()
        if key not in {"plan_sha256", "provenance"}
    }
    if _canonical_json_bytes(scientific_plan) != _canonical_json_bytes(
        expected_scientific_plan
    ):
        raise ValueError(
            "Plan scientific design differs from the deterministic rebuilt plan."
        )

    provenance = plan.get("provenance")
    expected_provenance = expected["provenance"]
    if not isinstance(provenance, dict) or set(provenance) != set(
        expected_provenance
    ):
        raise ValueError("Plan provenance inventory is invalid.")
    stored_git = provenance.get("git_commit_sha")
    if not isinstance(stored_git, str) or re.fullmatch(
        r"(?:[0-9a-f]{40}|[0-9a-f]{64})", stored_git
    ) is None:
        raise ValueError("Canonical plan must preserve a non-null Git commit SHA.")
    for key, current_value in expected_provenance.items():
        if key == "git_commit_sha":
            # The frozen plan can be committed after it is generated; later
            # byte-identical replay must not require HEAD to remain unchanged.
            continue
        if provenance.get(key) != current_value:
            raise ValueError(
                f"Plan provenance {key} differs from the current environment."
            )


def load_and_validate_plan(
    path: Path,
    *,
    tier: str,
    config: dict[str, Any],
    config_path: Path,
    project_root: Path = ROOT,
) -> dict[str, Any]:
    if config["tiers"][tier]["role"] == "canonical":
        _portable_path(path, project_root, require_repo_relative=True)
    plan = json.loads(path.read_text(encoding="utf-8"))
    _validate_plan(
        plan,
        tier=tier,
        config=config,
        config_path=config_path,
        project_root=project_root,
    )
    return plan


def _condition_grid(
    definition: dict[str, Any],
    config: dict[str, Any],
) -> LayeredGrid2D:
    settings = config["tiers"][definition["tier"]]
    base = rectangular_tool_composite_grid(
        width_m=float(config["geometry"]["width_m"]),
        spacing_x_m=float(settings["spacing_x_m"]),
        spacing_z_m=float(settings["spacing_z_m"]),
        composite_conductivity_x_scale=float(
            definition["composite_conductivity_scale"]
        ),
        composite_conductivity_z_scale=float(
            definition["composite_conductivity_scale"]
        ),
    )
    source = base.cure_source_J_m3_per_alpha.copy()
    source[base.composite_mask] *= float(definition["reaction_enthalpy_scale"])
    return replace(base, cure_source_J_m3_per_alpha=source)


def _acceptance_reasons(
    *,
    finite: bool,
    alpha_bound_violations: int,
    alpha_monotonicity_violations: int,
    tool_alpha_nonzero_count: int,
    diagnostics: dict[str, Any],
    acceptance: dict[str, Any],
) -> list[dict[str, Any]]:
    reasons: list[dict[str, Any]] = []
    if bool(acceptance["finite_field_required"]) and not finite:
        reasons.append({"code": "nonfinite_field", "observed": False})
    if alpha_bound_violations > int(
        acceptance["alpha_bound_violation_count_max"]
    ):
        reasons.append(
            {
                "code": "alpha_bound_violation",
                "observed": alpha_bound_violations,
                "maximum": int(
                    acceptance["alpha_bound_violation_count_max"]
                ),
            }
        )
    if tool_alpha_nonzero_count > int(
        acceptance["tool_alpha_nonzero_count_max"]
    ):
        reasons.append(
            {
                "code": "tool_alpha_nonzero",
                "observed": tool_alpha_nonzero_count,
                "maximum": int(acceptance["tool_alpha_nonzero_count_max"]),
            }
        )
    if alpha_monotonicity_violations > int(
        acceptance["alpha_monotonicity_violation_count_max"]
    ):
        reasons.append(
            {
                "code": "alpha_monotonicity_violation",
                "observed": alpha_monotonicity_violations,
                "maximum": int(
                    acceptance["alpha_monotonicity_violation_count_max"]
                ),
            }
        )
    if bool(acceptance["all_coupling_steps_converged_required"]) and not bool(
        diagnostics.get("all_coupling_steps_converged", True)
    ):
        reasons.append(
            {
                "code": "coupling_not_converged",
                "observed": diagnostics.get("all_coupling_steps_converged"),
                "required": True,
            }
        )
    for diagnostic, threshold in ACCEPTANCE_DIAGNOSTIC_THRESHOLDS:
        try:
            observed = float(diagnostics.get(diagnostic, np.nan))
        except (TypeError, ValueError):
            observed = np.nan
        maximum = float(acceptance[threshold])
        if not np.isfinite(observed) or observed > maximum:
            reasons.append(
                {
                    "code": diagnostic,
                    "observed": _json_safe(observed),
                    "maximum": maximum,
                }
            )
    return reasons


def _run_case(
    definition: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Worker entry point returning arrays and structured terminal metadata."""

    started = time.perf_counter()
    try:
        grid = _condition_grid(definition, config)
        settings = config["tiers"][definition["tier"]]
        solver_config = config["solver"]
        result = simulate_cure_2d(
            np.asarray(_time_axis(config), dtype=np.float64),
            np.asarray(definition["air_temperature_K"], dtype=np.float64),
            grid=grid,
            boundaries=RobinBoundaries2D(
                float(definition["bottom_h_W_m2_K"]),
                np.asarray(definition["top_h_W_m2_K"], dtype=np.float64),
                float(definition["left_h_W_m2_K"]),
                float(definition["right_h_W_m2_K"]),
            ),
            maximum_step_s=float(settings["maximum_step_s"]),
            coupling_tolerance_K=float(solver_config["coupling_tolerance_K"]),
            maximum_coupling_iterations=int(
                solver_config["maximum_coupling_iterations"]
            ),
        )
        temperature_K = result.temperature_K.astype(np.float32)
        alpha = result.alpha.astype(np.float32)
        alpha_difference = np.diff(alpha, axis=0)
        alpha_bound_violations = int(
            np.count_nonzero(
                (alpha < -1.0e-12) | (alpha > 1.0 + 1.0e-12)
            )
        )
        alpha_monotonicity_violations = int(
            np.count_nonzero(alpha_difference < -1.0e-12)
        )
        tool_alpha_nonzero_count = int(
            np.count_nonzero(alpha[:, ~grid.composite_mask] != 0.0)
        )
        finite = bool(
            np.all(np.isfinite(temperature_K)) and np.all(np.isfinite(alpha))
        )
        diagnostics = asdict(result.diagnostics)
        reasons = _acceptance_reasons(
            finite=finite,
            alpha_bound_violations=alpha_bound_violations,
            alpha_monotonicity_violations=alpha_monotonicity_violations,
            tool_alpha_nonzero_count=tool_alpha_nonzero_count,
            diagnostics=diagnostics,
            acceptance=config["acceptance"],
        )
        return {
            "case_id": int(definition["case_id"]),
            "status": "passed" if not reasons else "failed_acceptance",
            "failure": (
                None
                if not reasons
                else {
                    "kind": "acceptance",
                    "stage": "post_solve_validation",
                    "reasons": reasons,
                }
            ),
            "temperature_K": temperature_K,
            "alpha": alpha,
            "solver": {
                **_json_safe(diagnostics),
                "finite_fields": finite,
                "alpha_bound_violation_count": alpha_bound_violations,
                "alpha_monotonicity_violation_count": (
                    alpha_monotonicity_violations
                ),
                "tool_alpha_nonzero_count": tool_alpha_nonzero_count,
                "configured_maximum_step_s": float(settings["maximum_step_s"]),
                "configured_coupling_tolerance_K": float(
                    solver_config["coupling_tolerance_K"]
                ),
                "configured_maximum_coupling_iterations": int(
                    solver_config["maximum_coupling_iterations"]
                ),
            },
            "total_case_wall_seconds": time.perf_counter() - started,
        }
    except Exception as error:
        return {
            "case_id": int(definition["case_id"]),
            "status": "solver_exception",
            "failure": {
                "kind": "solver_exception",
                "stage": "simulate_cure_2d",
                "exception_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
            "temperature_K": None,
            "alpha": None,
            "solver": {
                "finite_fields": False,
                "configured_maximum_step_s": float(
                    config["tiers"][definition["tier"]]["maximum_step_s"]
                ),
                "configured_coupling_tolerance_K": float(
                    config["solver"]["coupling_tolerance_K"]
                ),
                "configured_maximum_coupling_iterations": int(
                    config["solver"]["maximum_coupling_iterations"]
                ),
            },
            "total_case_wall_seconds": time.perf_counter() - started,
        }


def _array_shapes(
    plan: dict[str, Any], config: dict[str, Any]
) -> dict[str, tuple[int, ...]]:
    tier = str(plan["tier"])
    settings = config["tiers"][tier]
    count = int(plan["case_count"])
    nt = len(plan["time_s"])
    nx = int(
        round(
            float(config["geometry"]["width_m"])
            / float(settings["spacing_x_m"])
        )
    )
    total_z = float(config["geometry"]["tool_thickness_m"]) + float(
        config["geometry"]["composite_thickness_m"]
    )
    nz = int(round(total_z / float(settings["spacing_z_m"])))
    return {
        "temperature_K": (count, nt, nz, nx),
        "alpha": (count, nt, nz, nx),
        "air_temperature_K": (count, nt),
        "top_h_W_m2_K": (count, nx),
        "time_s": (nt,),
        "x_m": (nx,),
        "z_m": (nz,),
        "composite_mask": (nz, nx),
    }


def _open_or_create_memmap(
    path: Path, shape: tuple[int, ...], dtype: np.dtype[Any]
) -> np.memmap:
    if path.exists():
        array = np.load(path, mmap_mode="r+", allow_pickle=False)
        if array.shape != shape or array.dtype != dtype:
            raise ValueError(
                f"Staging array {path} has {array.shape}/{array.dtype}, "
                f"expected {shape}/{dtype}."
            )
        return array
    creation_path = path.with_name(f".{path.name}.creating")
    if creation_path.exists():
        candidate = np.load(
            creation_path, mmap_mode="r+", allow_pickle=False
        )
        if candidate.shape != shape or candidate.dtype != dtype:
            raise ValueError(
                f"Incomplete staging array {creation_path} has "
                f"{candidate.shape}/{candidate.dtype}, expected {shape}/{dtype}."
            )
        candidate.flush()
        del candidate
        os.replace(creation_path, path)
    else:
        candidate = np.lib.format.open_memmap(
            creation_path, mode="w+", dtype=dtype, shape=shape
        )
        candidate.flush()
        del candidate
        os.replace(creation_path, path)
    return np.load(path, mmap_mode="r+", allow_pickle=False)


def _prepare_staging(
    plan: dict[str, Any],
    config: dict[str, Any],
    staging_root: Path,
) -> dict[str, np.ndarray]:
    staging_root.mkdir(parents=True, exist_ok=True)
    _write_immutable_bytes(
        staging_root / "plan.json", _pretty_json_bytes(plan)
    )
    shapes = _array_shapes(plan, config)
    temperature = _open_or_create_memmap(
        staging_root / "temperature_K.npy", shapes["temperature_K"], np.dtype("float32")
    )
    alpha = _open_or_create_memmap(
        staging_root / "alpha.npy", shapes["alpha"], np.dtype("float32")
    )
    definitions = [entry["definition"] for entry in plan["cases"]]
    small_arrays: dict[str, np.ndarray] = {
        "air_temperature_K": np.asarray(
            [item["air_temperature_K"] for item in definitions], dtype=np.float32
        ),
        "top_h_W_m2_K": np.asarray(
            [item["top_h_W_m2_K"] for item in definitions], dtype=np.float32
        ),
        "time_s": np.asarray(plan["time_s"], dtype=np.float64),
    }
    settings = config["tiers"][plan["tier"]]
    grid = rectangular_tool_composite_grid(
        width_m=float(config["geometry"]["width_m"]),
        spacing_x_m=float(settings["spacing_x_m"]),
        spacing_z_m=float(settings["spacing_z_m"]),
    )
    small_arrays.update(
        {
            "x_m": grid.x_m.astype(np.float64),
            "z_m": grid.z_m.astype(np.float64),
            "composite_mask": grid.composite_mask.astype(np.bool_),
        }
    )
    for name, expected in small_arrays.items():
        path = staging_root / f"{name}.npy"
        if path.exists():
            existing = np.load(path, allow_pickle=False)
            if not np.array_equal(existing, expected):
                raise ValueError(f"Staging input array mismatch: {path}")
        else:
            _atomic_save_npy(path, expected)
    (staging_root / "case_completion").mkdir(exist_ok=True)
    (staging_root / "case_failure_history").mkdir(exist_ok=True)
    return {
        "temperature_K": temperature,
        "alpha": alpha,
        **small_arrays,
    }


def _preflight_resources(
    plan: dict[str, Any],
    config: dict[str, Any],
    staging_root: Path,
    maximum_in_flight: int,
) -> dict[str, Any]:
    shapes = _array_shapes(plan, config)
    float_bytes = np.dtype("float32").itemsize
    estimated_output_bytes = (
        int(np.prod(shapes["temperature_K"]))
        + int(np.prod(shapes["alpha"]))
    ) * float_bytes
    expected_missing = 0
    for name in ("temperature_K", "alpha"):
        final_path = staging_root / f"{name}.npy"
        creating_path = staging_root / f".{name}.npy.creating"
        if not final_path.exists() and not creating_path.exists():
            expected_missing += int(np.prod(shapes[name])) * float_bytes + 4096
    margin = int(config["storage"]["minimum_free_disk_bytes"])
    staging_root.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(staging_root.parent).free
    if free < expected_missing + margin:
        raise OSError(
            "Insufficient free disk for P4 staging: "
            f"free={free}, required={expected_missing + margin}."
        )
    case_result_bytes = (
        int(np.prod(shapes["temperature_K"][1:])) * float_bytes * 2
    )
    available_memory: int | None = None
    if sys.platform == "win32":
        class _MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatus()
        status.length = ctypes.sizeof(_MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            available_memory = int(status.available_physical)
    elif hasattr(os, "sysconf"):
        try:
            available_memory = int(
                os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            )
        except (OSError, ValueError):
            available_memory = None
    estimated_in_flight_memory = case_result_bytes * maximum_in_flight * 3
    minimum_free_memory = int(config["storage"]["minimum_free_memory_bytes"])
    required_memory = max(estimated_in_flight_memory, minimum_free_memory)
    if available_memory is not None and available_memory < required_memory:
        raise MemoryError(
            "Insufficient available memory for bounded P4 generation: "
            f"available={available_memory}, required={required_memory}."
        )
    return {
        "estimated_output_array_bytes": estimated_output_bytes,
        "estimated_case_result_bytes": case_result_bytes,
        "expected_additional_staging_bytes": expected_missing,
        "minimum_free_disk_margin_bytes": margin,
        "free_disk_bytes": free,
        "maximum_in_flight": maximum_in_flight,
        "estimated_in_flight_memory_bytes": estimated_in_flight_memory,
        "minimum_free_memory_bytes": minimum_free_memory,
        "available_memory_bytes": available_memory,
    }


def _completion_path(staging_root: Path, case_id: int) -> Path:
    return staging_root / "case_completion" / f"{case_id:04d}.json"


def _archive_failed_completion(
    staging_root: Path, case_id: int, payload: bytes
) -> None:
    completion = json.loads(payload)
    if completion.get("status") == "passed":
        return
    digest = _sha256_bytes(payload)
    history_path = (
        staging_root
        / "case_failure_history"
        / f"{case_id:04d}"
        / f"{digest}.json"
    )
    _write_immutable_bytes(history_path, payload)


def _case_input_hashes(definition: dict[str, Any]) -> dict[str, str]:
    return {
        "air_temperature_K": _sha256_array(
            np.asarray(definition["air_temperature_K"], dtype=np.float32)
        ),
        "top_h_W_m2_K": _sha256_array(
            np.asarray(definition["top_h_W_m2_K"], dtype=np.float32)
        ),
        "case_definition": _sha256_bytes(_canonical_json_bytes(definition)),
    }


def _commit_case_result(
    *,
    definition: dict[str, Any],
    result: dict[str, Any],
    arrays: dict[str, np.ndarray],
    staging_root: Path,
) -> dict[str, Any]:
    case_id = int(definition["case_id"])
    temperature = arrays["temperature_K"]
    alpha = arrays["alpha"]
    expected_shape = temperature.shape[1:]
    if result["temperature_K"] is None or result["alpha"] is None:
        temperature[case_id] = np.nan
        alpha[case_id] = np.nan
    else:
        result_temperature = np.asarray(result["temperature_K"], dtype=np.float32)
        result_alpha = np.asarray(result["alpha"], dtype=np.float32)
        if result_temperature.shape != expected_shape or result_alpha.shape != expected_shape:
            raise ValueError(
                f"Case {case_id} output shape mismatch: "
                f"{result_temperature.shape}/{result_alpha.shape}, "
                f"expected {expected_shape}."
            )
        temperature[case_id] = result_temperature
        alpha[case_id] = result_alpha
    temperature.flush()
    alpha.flush()
    output_hashes = {
        "temperature_K": _sha256_array(np.asarray(temperature[case_id])),
        "alpha": _sha256_array(np.asarray(alpha[case_id])),
    }
    completion = {
        "schema_version": 2,
        "case_id": case_id,
        "case_key": definition["case_key"],
        "case_definition_sha256": _sha256_bytes(
            _canonical_json_bytes(definition)
        ),
        "status": result["status"],
        "failure": result["failure"],
        "input_slice_sha256": _case_input_hashes(definition),
        "output_slice_sha256": output_hashes,
        "solver": result["solver"],
        "runtime": {
            "total_case_wall_seconds": float(
                result["total_case_wall_seconds"]
            )
        },
    }
    prior_path = _completion_path(staging_root, case_id)
    if prior_path.exists():
        _archive_failed_completion(staging_root, case_id, prior_path.read_bytes())
    _atomic_write_bytes(
        prior_path, _pretty_json_bytes(completion)
    )
    return completion


def _load_completions(
    plan: dict[str, Any],
    arrays: dict[str, np.ndarray],
    staging_root: Path,
    *,
    retry_failed: bool,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    expected_ids = {int(entry["definition"]["case_id"]) for entry in plan["cases"]}
    extra = []
    for path in (staging_root / "case_completion").glob("*.json"):
        try:
            case_id = int(path.stem)
        except ValueError:
            extra.append(path.name)
            continue
        if case_id not in expected_ids:
            extra.append(path.name)
    if extra:
        raise ValueError(f"Unexpected completion records: {sorted(extra)}")
    completed: dict[int, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    for entry in plan["cases"]:
        definition = entry["definition"]
        case_id = int(definition["case_id"])
        path = _completion_path(staging_root, case_id)
        if not path.exists():
            pending.append(definition)
            continue
        completion = json.loads(path.read_text(encoding="utf-8"))
        if completion.get("case_definition_sha256") != entry[
            "case_definition_sha256"
        ]:
            raise ValueError(f"Completion/plan mismatch for case {case_id}.")
        expected_inputs = _case_input_hashes(definition)
        if completion.get("input_slice_sha256") != expected_inputs:
            raise ValueError(f"Input slice hash mismatch for case {case_id}.")
        if retry_failed and completion.get("status") != "passed":
            _archive_failed_completion(staging_root, case_id, path.read_bytes())
            pending.append(definition)
            continue
        actual_outputs = {
            "temperature_K": _sha256_array(
                np.asarray(arrays["temperature_K"][case_id])
            ),
            "alpha": _sha256_array(np.asarray(arrays["alpha"][case_id])),
        }
        if completion.get("output_slice_sha256") != actual_outputs:
            raise ValueError(f"Output slice hash mismatch for case {case_id}.")
        completed[case_id] = completion
    return completed, pending


def _future_failure(definition: dict[str, Any], error: BaseException) -> dict[str, Any]:
    return {
        "case_id": int(definition["case_id"]),
        "status": "worker_exception",
        "failure": {
            "kind": "worker_exception",
            "stage": "future_result",
            "exception_type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            ),
        },
        "temperature_K": None,
        "alpha": None,
        "solver": {"finite_fields": False},
        "total_case_wall_seconds": 0.0,
    }


def _execute_pending(
    *,
    pending_definitions: list[dict[str, Any]],
    config: dict[str, Any],
    arrays: dict[str, np.ndarray],
    staging_root: Path,
    workers: int,
    maximum_in_flight: int,
) -> None:
    if workers < 1:
        raise ValueError("--workers must be at least 1.")
    if maximum_in_flight < workers:
        raise ValueError("--max-in-flight must be at least --workers.")
    iterator = iter(pending_definitions)
    executor = ProcessPoolExecutor(max_workers=workers)
    futures: dict[Future[dict[str, Any]], dict[str, Any]] = {}

    def submit_until_full() -> None:
        while len(futures) < maximum_in_flight:
            try:
                definition = next(iterator)
            except StopIteration:
                break
            future = executor.submit(_run_case, definition, config)
            futures[future] = definition

    try:
        submit_until_full()
        while futures:
            done, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
            for future in done:
                definition = futures.pop(future)
                try:
                    result = future.result()
                except BaseException as error:
                    result = _future_failure(definition, error)
                _commit_case_result(
                    definition=definition,
                    result=result,
                    arrays=arrays,
                    staging_root=staging_root,
                )
            submit_until_full()
    except BaseException:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)


def _metadata_row(
    definition: dict[str, Any], completion: dict[str, Any]
) -> dict[str, Any]:
    solver = {
        key: value
        for key, value in completion["solver"].items()
        if key != "wall_seconds"
    }
    return {
        **{
            key: value
            for key, value in definition.items()
            if key not in {"air_temperature_K", "top_h_W_m2_K"}
        },
        "air_temperature_min_K": float(
            np.min(np.asarray(definition["air_temperature_K"]))
        ),
        "air_temperature_max_K": float(
            np.max(np.asarray(definition["air_temperature_K"]))
        ),
        "case_definition_sha256": completion["case_definition_sha256"],
        "input_slice_sha256": completion["input_slice_sha256"],
        "output_slice_sha256": completion["output_slice_sha256"],
        "status": completion["status"],
        "failure": completion["failure"],
        "solver": solver,
    }


def _acceptance_aggregates(
    completions: dict[int, dict[str, Any]],
    config: dict[str, Any],
    *,
    expected_case_count: int,
) -> tuple[dict[str, Any], dict[str, bool]]:
    acceptance = config["acceptance"]
    solvers = [item["solver"] for item in completions.values()]
    if not solvers:
        raise ValueError("Cannot aggregate acceptance without completed cases.")
    aggregates: dict[str, Any] = {
        diagnostic: max(float(item[diagnostic]) for item in solvers)
        for diagnostic, _ in ACCEPTANCE_DIAGNOSTIC_THRESHOLDS
    }
    aggregates.update(
        {
            "maximum_alpha_bound_violation_count": max(
                int(item["alpha_bound_violation_count"]) for item in solvers
            ),
            "maximum_alpha_monotonicity_violation_count": max(
                int(item["alpha_monotonicity_violation_count"])
                for item in solvers
            ),
            "maximum_tool_alpha_nonzero_count": max(
                int(item["tool_alpha_nonzero_count"]) for item in solvers
            ),
            "all_finite_fields": all(
                bool(item["finite_fields"]) for item in solvers
            ),
            "all_coupling_steps_converged": all(
                bool(item["all_coupling_steps_converged"]) for item in solvers
            ),
        }
    )
    checks = {
        "all_cases_accounted_for": len(completions) == expected_case_count,
        "no_failed_cases": all(
            item.get("status") == "passed" for item in completions.values()
        ),
        "no_silent_failures": True,
        "finite_fields": (
            not bool(acceptance["finite_field_required"])
            or bool(aggregates["all_finite_fields"])
        ),
        "alpha_bounds": int(
            aggregates["maximum_alpha_bound_violation_count"]
        )
        <= int(acceptance["alpha_bound_violation_count_max"]),
        "alpha_monotonicity": int(
            aggregates["maximum_alpha_monotonicity_violation_count"]
        )
        <= int(acceptance["alpha_monotonicity_violation_count_max"]),
        "tool_alpha": int(aggregates["maximum_tool_alpha_nonzero_count"])
        <= int(acceptance["tool_alpha_nonzero_count_max"]),
        "coupling_convergence": (
            not bool(acceptance["all_coupling_steps_converged_required"])
            or bool(aggregates["all_coupling_steps_converged"])
        ),
    }
    diagnostic_check_names = {
        "maximum_abs_energy_residual_W_m3": "local_energy",
        "maximum_relative_global_energy_residual": "global_energy",
        "maximum_interface_flux_imbalance_W": "interface_flux_continuity",
        "maximum_temperature_interface_jump_K": (
            "interface_temperature_continuity"
        ),
        "maximum_robin_flux_imbalance_W": "robin_flux_continuity",
        "maximum_converged_coupling_update_K": "maximum_coupling_update",
    }
    for diagnostic, threshold in ACCEPTANCE_DIAGNOSTIC_THRESHOLDS:
        observed = float(aggregates[diagnostic])
        checks[diagnostic_check_names[diagnostic]] = bool(
            np.isfinite(observed) and observed <= float(acceptance[threshold])
        )
    return aggregates, checks


def _build_final_payloads(
    *,
    plan: dict[str, Any],
    config: dict[str, Any],
    staging_root: Path,
    output_root: Path,
    plan_path: Path,
    metadata_path: Path,
    manifest_path: Path,
    summary_path: Path,
    completions: dict[int, dict[str, Any]],
    generation_wall_seconds: float,
    workers: int,
    maximum_in_flight: int,
    resource_preflight: dict[str, Any],
    project_root: Path = ROOT,
) -> tuple[bytes, bytes, bytes]:
    definitions = [entry["definition"] for entry in plan["cases"]]
    rows = [_metadata_row(item, completions[int(item["case_id"])]) for item in definitions]
    metadata_bytes = b"".join(_canonical_json_bytes(row) for row in rows)
    array_names = (
        "temperature_K",
        "alpha",
        "air_temperature_K",
        "top_h_W_m2_K",
        "time_s",
        "x_m",
        "z_m",
        "composite_mask",
    )
    array_sha256 = {
        name: _sha256_file(staging_root / f"{name}.npy") for name in array_names
    }
    shapes = _array_shapes(plan, config)
    split_map = {
        name: [int(value) for value in ids]
        for name, ids in plan["splits"].items()
    }
    case_definition_hashes = {
        entry["definition"]["case_key"]: entry["case_definition_sha256"]
        for entry in plan["cases"]
    }
    case_artifact_hashes = {
        definition["case_key"]: {
            "input_slice_sha256": completions[int(definition["case_id"])][
                "input_slice_sha256"
            ],
            "output_slice_sha256": completions[int(definition["case_id"])][
                "output_slice_sha256"
            ],
        }
        for definition in definitions
    }
    failure_history: dict[str, list[dict[str, Any]]] = {}
    history_root = staging_root / "case_failure_history"
    for case_directory in sorted(history_root.glob("[0-9][0-9][0-9][0-9]")):
        entries = []
        for path in sorted(case_directory.glob("*.json")):
            payload = path.read_bytes()
            record = json.loads(payload)
            entries.append(
                {
                    "path": (
                        f"case_failure_history/{case_directory.name}/{path.name}"
                    ),
                    "sha256": _sha256_bytes(payload),
                    "status": record.get("status"),
                    "failure": record.get("failure"),
                }
            )
        if entries:
            failure_history[case_directory.name] = entries
    canonical = plan["tier_role"] == "canonical"
    canonical_path = lambda path: _portable_path(  # noqa: E731
        path,
        project_root,
        require_repo_relative=canonical,
    )
    acceptance_aggregates, acceptance_checks = _acceptance_aggregates(
        completions,
        config,
        expected_case_count=int(plan["case_count"]),
    )
    summary = {
        "schema_version": 2,
        "phase": "P4",
        "dataset_id": plan["dataset_id"],
        "tier": plan["tier"],
        "tier_role": plan["tier_role"],
        "case_count": int(plan["case_count"]),
        "passed_case_count": len(completions),
        "failed_case_count": 0,
        "failed_case_ids": [],
        "silent_failure_count": 0,
        "silent_failure_case_ids": [],
        "array_shape_case_time_z_x": list(shapes["temperature_K"]),
        "maximum_step_s": float(
            config["tiers"][plan["tier"]]["maximum_step_s"]
        ),
        **acceptance_aggregates,
        "generation_wall_seconds": generation_wall_seconds,
        "recovered_failure_attempt_count": sum(
            len(entries) for entries in failure_history.values()
        ),
        "workers": workers,
        "maximum_in_flight": maximum_in_flight,
        "resource_preflight": resource_preflight,
        "acceptance_thresholds_prespecified": config["acceptance"],
        "acceptance_checks": acceptance_checks,
        "manifest": canonical_path(manifest_path),
        "metadata": canonical_path(metadata_path),
        "plan_sha256": plan["plan_sha256"],
        "config_path": plan["config_path"],
        "config_sha256": plan["config_sha256"],
        "code_sha256": plan["code_sha256"],
        "provenance": plan["provenance"],
        "array_sha256": array_sha256,
    }
    summary["passed"] = bool(all(summary["acceptance_checks"].values()))
    if not summary["passed"]:
        raise RuntimeError("Final P4 acceptance unexpectedly failed.")
    summary_bytes = _pretty_json_bytes(summary)
    manifest = {
        "schema_version": 2,
        "phase": "P4",
        "dataset_id": plan["dataset_id"],
        "tier": plan["tier"],
        "tier_role": plan["tier_role"],
        "case_count": int(plan["case_count"]),
        "array_shape_case_time_z_x": list(shapes["temperature_K"]),
        "array_artifact_root": canonical_path(output_root),
        "array_sha256": array_sha256,
        "metadata_path": canonical_path(metadata_path),
        "metadata_sha256": _sha256_bytes(metadata_bytes),
        "generation_summary_path": canonical_path(summary_path),
        "generation_summary_sha256": _sha256_bytes(summary_bytes),
        "plan_path": canonical_path(plan_path),
        "plan_sha256": plan["plan_sha256"],
        "config_path": plan["config_path"],
        "config_sha256": plan["config_sha256"],
        "code_sha256": plan["code_sha256"],
        "provenance": plan["provenance"],
        "case_definition_hashes": case_definition_hashes,
        "case_artifact_hashes": case_artifact_hashes,
        "recovered_failure_history": failure_history,
        "splits": split_map,
        "nested_training_budgets": plan["nested_training_budgets"],
        "generation_status": {
            "passed_case_ids": list(range(int(plan["case_count"]))),
            "failed_case_ids": [],
            "silent_failure_case_ids": [],
        },
    }
    return metadata_bytes, _pretty_json_bytes(manifest), summary_bytes


def _publish_payloads(
    *,
    final_root: Path,
    metadata_path: Path,
    manifest_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    metadata_bytes = (final_root / "case_metadata.jsonl").read_bytes()
    manifest_bytes = (final_root / "manifest.json").read_bytes()
    summary_bytes = (final_root / "generation_summary.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    summary = json.loads(summary_bytes)
    if manifest.get("metadata_sha256") != _sha256_bytes(metadata_bytes):
        raise ValueError("Internal P4 metadata checksum does not match its manifest.")
    if manifest.get("generation_summary_sha256") != _sha256_bytes(
        summary_bytes
    ):
        raise ValueError(
            "Internal P4 generation-summary checksum does not match its manifest."
        )
    for field in (
        "dataset_id",
        "tier",
        "tier_role",
        "case_count",
        "array_shape_case_time_z_x",
        "plan_sha256",
        "config_path",
        "config_sha256",
        "code_sha256",
        "provenance",
        "array_sha256",
    ):
        if summary.get(field) != manifest.get(field):
            raise ValueError(
                f"Internal P4 summary/manifest field mismatch: {field}"
            )
    _write_immutable_bytes(metadata_path, metadata_bytes)
    _write_immutable_bytes(manifest_path, manifest_bytes)
    _write_immutable_bytes(summary_path, summary_bytes)
    return summary


def _finalize(
    *,
    plan: dict[str, Any],
    config: dict[str, Any],
    staging_root: Path,
    output_root: Path,
    plan_path: Path,
    metadata_path: Path,
    manifest_path: Path,
    summary_path: Path,
    completions: dict[int, dict[str, Any]],
    generation_wall_seconds: float,
    workers: int,
    maximum_in_flight: int,
    resource_preflight: dict[str, Any],
    project_root: Path = ROOT,
) -> dict[str, Any]:
    metadata_bytes, manifest_bytes, summary_bytes = _build_final_payloads(
        plan=plan,
        config=config,
        staging_root=staging_root,
        output_root=output_root,
        plan_path=plan_path,
        metadata_path=metadata_path,
        manifest_path=manifest_path,
        summary_path=summary_path,
        completions=completions,
        generation_wall_seconds=generation_wall_seconds,
        workers=workers,
        maximum_in_flight=maximum_in_flight,
        resource_preflight=resource_preflight,
        project_root=project_root,
    )
    _atomic_write_bytes(staging_root / "case_metadata.jsonl", metadata_bytes)
    _atomic_write_bytes(staging_root / "manifest.json", manifest_bytes)
    _atomic_write_bytes(staging_root / "generation_summary.json", summary_bytes)
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to replace existing dataset root: {output_root}"
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging_root, output_root)
    return _publish_payloads(
        final_root=output_root,
        metadata_path=metadata_path,
        manifest_path=manifest_path,
        summary_path=summary_path,
    )


def _publish_existing_if_matching(
    *,
    plan: dict[str, Any],
    output_root: Path,
    metadata_path: Path,
    manifest_path: Path,
    summary_path: Path,
) -> dict[str, Any] | None:
    if not output_root.exists():
        return None
    internal_plan = output_root / "plan.json"
    required = (
        internal_plan,
        output_root / "manifest.json",
        output_root / "case_metadata.jsonl",
        output_root / "generation_summary.json",
    )
    if not all(path.is_file() for path in required):
        raise FileExistsError(
            "Existing dataset root is not a matching finalized P4 artifact; "
            f"refusing to modify it: {output_root}"
        )
    existing_plan = json.loads(internal_plan.read_text(encoding="utf-8"))
    if existing_plan.get("plan_sha256") != plan["plan_sha256"]:
        raise FileExistsError(
            f"Existing dataset root belongs to a different plan: {output_root}"
        )
    return _publish_payloads(
        final_root=output_root,
        metadata_path=metadata_path,
        manifest_path=manifest_path,
        summary_path=summary_path,
    )


def _incomplete_summary(
    plan: dict[str, Any],
    completions: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    expected = set(range(int(plan["case_count"])))
    observed = set(completions)
    failed = sorted(
        case_id
        for case_id, item in completions.items()
        if item.get("status") != "passed"
    )
    silent = sorted(expected - observed)
    return {
        "schema_version": 2,
        "phase": "P4",
        "dataset_id": plan["dataset_id"],
        "passed": False,
        "case_count": int(plan["case_count"]),
        "passed_case_count": sum(
            item.get("status") == "passed" for item in completions.values()
        ),
        "failed_case_count": len(failed),
        "failed_case_ids": failed,
        "silent_failure_count": len(silent),
        "silent_failure_case_ids": silent,
        "plan_sha256": plan["plan_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=("debug", "pilot", "core"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--max-in-flight", type=int)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    tier = args.tier
    if not bool(config["tiers"][tier]["generation_enabled"]):
        raise SystemExit(
            f"Tier {tier!r} is a preserved historical exploratory artifact "
            "and cannot be regenerated under this configuration."
        )
    dataset_id = (
        f"p4_2d_{tier}_{config.get('dataset_version', 'v1')}"
    )
    plan_path = args.plan or ROOT / "splits" / f"{dataset_id}_plan.json"
    canonical = config["tiers"][tier]["role"] == "canonical"
    _portable_path(
        plan_path,
        require_repo_relative=canonical,
    )
    if args.plan_only:
        plan = build_case_plan(tier, config_path=config_path, config=config)
        disposition = write_or_verify_plan(plan, plan_path)
        print(
            json.dumps(
                {
                    "dataset_id": dataset_id,
                    "plan": _portable_path(plan_path),
                    "plan_sha256": plan["plan_sha256"],
                    "case_count": plan["case_count"],
                    "disposition": disposition,
                    "labels_generated": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if not plan_path.is_file():
        raise SystemExit(
            "Pre-label plan is required. Run this command first:\n"
            f"  {sys.executable} {Path(__file__).name} --tier {tier} --plan-only"
        )
    plan = load_and_validate_plan(
        plan_path,
        tier=tier,
        config=config,
        config_path=config_path,
    )
    root_template = str(config["storage"]["root"])
    output_root = args.output_root or ROOT / root_template.format(tier=tier)
    metadata_path = args.metadata or ROOT / "outputs" / "tables" / (
        f"{dataset_id}_cases.jsonl"
    )
    manifest_path = args.manifest or ROOT / "splits" / f"{dataset_id}.json"
    summary_path = args.summary or ROOT / "outputs" / "tables" / (
        f"{dataset_id}_generation.json"
    )
    for artifact_path in (
        output_root,
        metadata_path,
        manifest_path,
        summary_path,
    ):
        _portable_path(
            artifact_path,
            require_repo_relative=canonical,
        )
    existing = _publish_existing_if_matching(
        plan=plan,
        output_root=output_root,
        metadata_path=metadata_path,
        manifest_path=manifest_path,
        summary_path=summary_path,
    )
    if existing is not None:
        print(json.dumps(existing, indent=2, sort_keys=True))
        return
    staging_root = output_root.with_name(
        f".{output_root.name}.staging-{plan['plan_sha256'][:12]}"
    )
    workers = int(args.workers)
    maximum_in_flight = int(
        args.max_in_flight
        or workers * int(config["storage"]["maximum_in_flight_per_worker"])
    )
    if workers < 1:
        raise ValueError("--workers must be at least 1.")
    if maximum_in_flight < workers:
        raise ValueError("--max-in-flight must be at least --workers.")
    resource_preflight = _preflight_resources(
        plan, config, staging_root, maximum_in_flight
    )
    arrays = _prepare_staging(plan, config, staging_root)
    completed, pending = _load_completions(
        plan,
        arrays,
        staging_root,
        retry_failed=bool(args.retry_failed),
    )
    started = time.perf_counter()
    if pending:
        _execute_pending(
            pending_definitions=pending,
            config=config,
            arrays=arrays,
            staging_root=staging_root,
            workers=workers,
            maximum_in_flight=maximum_in_flight,
        )
    arrays["temperature_K"].flush()
    arrays["alpha"].flush()
    completed, _ = _load_completions(
        plan, arrays, staging_root, retry_failed=False
    )
    del arrays
    incomplete = _incomplete_summary(plan, completed)
    if (
        incomplete["failed_case_count"] > 0
        or incomplete["silent_failure_count"] > 0
    ):
        _atomic_write_bytes(
            staging_root / "generation_summary.json",
            _pretty_json_bytes(incomplete),
        )
        print(json.dumps(incomplete, indent=2, sort_keys=True))
        raise SystemExit(1)
    summary = _finalize(
        plan=plan,
        config=config,
        staging_root=staging_root,
        output_root=output_root,
        plan_path=plan_path,
        metadata_path=metadata_path,
        manifest_path=manifest_path,
        summary_path=summary_path,
        completions=completed,
        generation_wall_seconds=time.perf_counter() - started,
        workers=workers,
        maximum_in_flight=maximum_in_flight,
        resource_preflight=resource_preflight,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
