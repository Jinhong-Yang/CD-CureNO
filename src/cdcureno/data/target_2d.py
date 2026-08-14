"""Leakage-safe P5 data access for the frozen P4 true-2-D benchmark.

The training API in this module deliberately exposes only the nested target
training budget and the frozen validation cases.  ID-test and OOD labels need
an evaluation-only API and therefore cannot be requested through
``PreparedTarget2DTraining.dataset``.

The first fourteen input channels exactly follow the P3 source checkpoint.
Their normalization is loaded from that checkpoint rather than fitted to any
P4 label.  Six target-only channels follow as a fixed suffix.  Those suffix
channels are zero on a laterally homogeneous F0 case, which makes
``lift_homogeneous_source_input`` the exact network-level restriction input.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from cdcureno.data.normalization import RangeNormalizer
from cdcureno.data.source_1d import INPUT_CHANNELS as SOURCE_INPUT_CHANNELS
from cdcureno.physics import PublicCase1Material
from cdcureno.solvers import (
    LayeredGrid1D,
    RobinBoundaries,
    layered_tool_composite_grid,
    simulate_cure_1d,
)


TARGET_ADAPTER_CHANNELS = (
    "heterogeneity_gated_in_plane_position_normalized",
    "heterogeneity_gated_signed_distance_to_left_boundary_normalized",
    "heterogeneity_gated_signed_distance_to_right_boundary_normalized",
    "top_htc_anomaly_normalized",
    "edge_htc_boundary_map_normalized",
    "lateral_conductivity_scale_delta_normalized",
)
TARGET_INPUT_CHANNELS = SOURCE_INPUT_CHANNELS + TARGET_ADAPTER_CHANNELS
TARGET_OUTPUT_CHANNELS = ("temperature_normalized", "degree_of_cure")

SOURCE_CHANNEL_COUNT = 14
TARGET_CHANNEL_COUNT = 20
COARSE_BASELINE_INTERVALS = 16
COARSE_BASELINE_MAXIMUM_STEP_S = 60.0
COARSE_BASELINE_MAXIMUM_COUPLING_ITERATIONS = 16
EDGE_HTC_REFERENCE_MAX_W_M2_K = 100.0
HOMOGENEITY_ABSOLUTE_TOLERANCE = 1.0e-7
NORMALIZED_COORDINATE_RANGE = (0.0, 1.0)
NORMALIZED_PARAMETER_RANGE = (0.0, 1.0)

# Reader-facing definitions are kept beside the executable channel contract so
# a resolved training configuration can record the exact feature semantics.
TARGET_SUFFIX_NORMALIZATION_DEFINITIONS = {
    "heterogeneity_gated_in_plane_position_normalized": (
        "(2 * x_m / width_m - 1) * lateral_heterogeneity_gate; "
        "exactly zero for F0"
    ),
    "heterogeneity_gated_signed_distance_to_left_boundary_normalized": (
        "(x_m / width_m - 0.5) * lateral_heterogeneity_gate; "
        "centered signed coordinate, exactly zero for F0"
    ),
    "heterogeneity_gated_signed_distance_to_right_boundary_normalized": (
        "((width_m - x_m) / width_m - 0.5) * "
        "lateral_heterogeneity_gate; centered signed coordinate, "
        "exactly zero for F0"
    ),
    "top_htc_anomaly_normalized": (
        "(top_h(x) - spatial_mean(top_h)) / "
        "(source_upper_htc_max - source_upper_htc_min); anomaly is "
        "exactly zero for F0"
    ),
    "edge_htc_boundary_map_normalized": (
        "left_h / 100 on the left boundary, right_h / 100 on the right "
        "boundary, zero in the interior and exactly zero for F0"
    ),
    "lateral_conductivity_scale_delta_normalized": (
        "((kx / base_kx) - (kz / base_kz)) / "
        "(source_conductivity_scale_max - "
        "source_conductivity_scale_min), multiplied by the lateral "
        "heterogeneity gate; lateral-vs-z scale delta, "
        "exactly zero for the current common-multiplier P4 benchmark"
    ),
}

FloatArray = NDArray[np.float64]
BaselineBuilder = Callable[
    [dict[str, Any], FloatArray, FloatArray, dict[str, float]],
    tuple[FloatArray, FloatArray],
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _sha256_canonical_json(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read {label} JSON at {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object.")
    return payload


def _resolve_repo_path(
    project_root: Path,
    value: Any,
    *,
    label: str,
) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(
            f"{label} must be a nonempty repository-relative POSIX path."
        )
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must stay inside the project root.")
    root = project_root.resolve()
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{label} escapes the project root.")
    return resolved


def _finite_range(
    payload: Any,
    *,
    label: str,
) -> RangeNormalizer:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} normalization must be an object.")
    try:
        minimum = float(payload["minimum"])
        maximum = float(payload["maximum"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{label} normalization needs finite minimum and maximum."
        ) from error
    if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum <= minimum:
        raise ValueError(
            f"{label} normalization needs finite maximum greater than minimum."
        )
    return RangeNormalizer(minimum=minimum, maximum=maximum)


@dataclass(frozen=True)
class FrozenSourceNormalization:
    """Validated source-training-only normalization from a P3 checkpoint."""

    air_temperature: RangeNormalizer
    field_temperature: RangeNormalizer
    parameter_bounds: NDArray[np.float64]
    checkpoint_sha256: str
    normalization_sha256: str
    checkpoint_path: str
    source_payload: dict[str, Any]

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "scope": (
                "source_training_only_frozen_for_transfer_comparability"
            ),
            "source_checkpoint": self.checkpoint_path,
            "source_checkpoint_sha256": self.checkpoint_sha256,
            "source_normalization_sha256": self.normalization_sha256,
            "air_temperature": {
                "minimum": self.air_temperature.minimum,
                "maximum": self.air_temperature.maximum,
            },
            "field_temperature": {
                "minimum": self.field_temperature.minimum,
                "maximum": self.field_temperature.maximum,
            },
            "parameter_bounds": self.parameter_bounds.tolist(),
            "input_channels": list(TARGET_INPUT_CHANNELS),
            "source_prefix_channels": list(SOURCE_INPUT_CHANNELS),
            "target_adapter_channels": list(TARGET_ADAPTER_CHANNELS),
            "target_labels_used_to_fit_normalization": False,
            "validation_or_test_labels_used_to_fit_normalization": False,
            "suffix_definitions": dict(TARGET_SUFFIX_NORMALIZATION_DEFINITIONS),
            "edge_htc_reference_max_W_m2_K": (
                EDGE_HTC_REFERENCE_MAX_W_M2_K
            ),
        }

    def normalize_parameter(self, index: int, value: float) -> float:
        lower, upper = self.parameter_bounds[index]
        normalized = (float(value) - lower) / (upper - lower)
        if not np.isfinite(normalized):
            raise ValueError("A target conditioning parameter is not finite.")
        return float(normalized)


def load_source_normalization_contract(
    checkpoint_path: Path,
) -> FrozenSourceNormalization:
    """Load and validate the source checkpoint's transfer normalization."""

    path = checkpoint_path.resolve()
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(f"Cannot load source checkpoint {path}: {error}") from error
    if not isinstance(checkpoint, dict):
        raise ValueError("Source checkpoint must contain a mapping.")
    checkpoint_channels = tuple(checkpoint.get("channel_names", ()))
    if checkpoint_channels != SOURCE_INPUT_CHANNELS:
        raise ValueError(
            "Source checkpoint channel_names do not match the frozen P3 "
            "14-channel prefix."
        )
    normalization = checkpoint.get("normalization")
    if not isinstance(normalization, dict):
        raise ValueError("Source checkpoint has no normalization mapping.")
    if tuple(normalization.get("input_channels", ())) != SOURCE_INPUT_CHANNELS:
        raise ValueError(
            "Source normalization input_channels do not match channel_names."
        )
    if normalization.get("scope") != "training_cases_only":
        raise ValueError(
            "Source normalization must have been fitted on training cases only."
        )
    if normalization.get("held_out_labels_used_for_normalization") is not False:
        raise ValueError(
            "Source checkpoint does not explicitly exclude held-out labels "
            "from normalization."
        )
    causal = normalization.get("causal_baseline")
    if (
        not isinstance(causal, dict)
        or causal.get("method")
        != "coarse_thermochemical_conservative_solver"
        or causal.get("uses_thermochemical_labels") is not False
    ):
        raise ValueError(
            "Source checkpoint must use the label-free coarse "
            "thermochemical baseline."
        )
    if int(normalization.get("time_stride", -1)) != 2:
        raise ValueError("Source checkpoint must use the frozen P3 time_stride=2.")
    parameter_bounds = np.asarray(
        normalization.get("parameter_bounds"), dtype=np.float64
    )
    if (
        parameter_bounds.shape != (6, 2)
        or not np.all(np.isfinite(parameter_bounds))
        or not np.all(parameter_bounds[:, 1] > parameter_bounds[:, 0])
    ):
        raise ValueError(
            "Source normalization parameter_bounds must have finite shape [6,2]."
        )
    parameter_bounds.setflags(write=False)
    return FrozenSourceNormalization(
        air_temperature=_finite_range(
            normalization.get("air_temperature"),
            label="air_temperature",
        ),
        field_temperature=_finite_range(
            normalization.get("field_temperature"),
            label="field_temperature",
        ),
        parameter_bounds=parameter_bounds,
        checkpoint_sha256=_sha256_file(path),
        normalization_sha256=_sha256_canonical_json(normalization),
        checkpoint_path=path.as_posix(),
        source_payload=normalization,
    )


def build_label_free_coarse_1d_baseline(
    definition: dict[str, Any],
    times_s: FloatArray,
    z_m: FloatArray,
    geometry: dict[str, float],
) -> tuple[FloatArray, FloatArray]:
    """Build the P3-compatible 16-interval baseline from pre-label inputs.

    This helper has no path or target-array argument by construction.  It uses
    only the immutable case definition, time/grid coordinates, and material
    model.  The spatially varying top HTC is reduced to its mean, which is the
    shared 1-D-compatible anchor.
    """

    time = np.asarray(times_s, dtype=np.float64)
    target_z = np.asarray(z_m, dtype=np.float64)
    air = np.asarray(definition.get("air_temperature_K"), dtype=np.float64)
    top_h = np.asarray(definition.get("top_h_W_m2_K"), dtype=np.float64)
    if (
        time.ndim != 1
        or target_z.ndim != 1
        or air.shape != time.shape
        or top_h.ndim != 1
        or top_h.size < 1
    ):
        raise ValueError("Coarse baseline conditioning arrays have bad shapes.")
    if not (
        np.all(np.isfinite(time))
        and np.all(np.isfinite(target_z))
        and np.all(np.isfinite(air))
        and np.all(np.isfinite(top_h))
    ):
        raise ValueError("Coarse baseline conditioning must be finite.")
    if time.size < 2 or not np.all(np.diff(time) > 0.0):
        raise ValueError("Coarse baseline times must be strictly increasing.")

    tool_thickness = float(geometry["tool_thickness_m"])
    composite_thickness = float(geometry["composite_thickness_m"])
    total_thickness = tool_thickness + composite_thickness
    if (
        tool_thickness <= 0.0
        or composite_thickness <= 0.0
        or target_z.min() < -1.0e-12
        or target_z.max() > total_thickness + 1.0e-12
    ):
        raise ValueError("Coarse baseline geometry is invalid.")

    material = replace(
        PublicCase1Material(),
        tool_thickness_m=tool_thickness,
        composite_thickness_m=composite_thickness,
    )
    coarse_grid = layered_tool_composite_grid(
        total_thickness / COARSE_BASELINE_INTERVALS,
        material,
    )
    conductivity = coarse_grid.conductivity_W_m_K.copy()
    conductivity[coarse_grid.composite_mask] *= float(
        definition["composite_conductivity_scale"]
    )
    source = coarse_grid.cure_source_J_m3_per_alpha.copy()
    source[coarse_grid.composite_mask] *= float(
        definition["reaction_enthalpy_scale"]
    )
    conditioned_grid = LayeredGrid1D(
        z_m=coarse_grid.z_m,
        control_volume_width_m=coarse_grid.control_volume_width_m,
        composite_mask=coarse_grid.composite_mask,
        density_kg_m3=coarse_grid.density_kg_m3,
        specific_heat_J_kg_K=coarse_grid.specific_heat_J_kg_K,
        conductivity_W_m_K=conductivity,
        cure_source_J_m3_per_alpha=source,
    )
    result = simulate_cure_1d(
        time,
        air,
        grid=conditioned_grid,
        boundaries=RobinBoundaries(
            float(definition["bottom_h_W_m2_K"]),
            float(np.mean(top_h)),
        ),
        maximum_step_s=COARSE_BASELINE_MAXIMUM_STEP_S,
        maximum_coupling_iterations=(
            COARSE_BASELINE_MAXIMUM_COUPLING_ITERATIONS
        ),
    )
    temperature = np.stack(
        [np.interp(target_z, result.z_m, row) for row in result.temperature_K]
    )
    alpha = np.stack(
        [np.interp(target_z, result.z_m, row) for row in result.alpha]
    )
    alpha[:, target_z <= tool_thickness] = 0.0
    return temperature, alpha


def _heterogeneity_gate(
    definition: dict[str, Any],
    top_h: NDArray[np.float64],
) -> float:
    top_anomaly = top_h - float(np.mean(top_h))
    has_top_pattern = bool(
        np.max(np.abs(top_anomaly)) > HOMOGENEITY_ABSOLUTE_TOLERANCE
    )
    has_edge_htc = bool(
        abs(float(definition["left_h_W_m2_K"]))
        > HOMOGENEITY_ABSOLUTE_TOLERANCE
        or abs(float(definition["right_h_W_m2_K"]))
        > HOMOGENEITY_ABSOLUTE_TOLERANCE
    )
    return 1.0 if has_top_pattern or has_edge_htc else 0.0


def build_target_2d_input(
    definition: dict[str, Any],
    times_s: NDArray[np.float64],
    z_m: NDArray[np.float64],
    x_m: NDArray[np.float64],
    composite_mask: NDArray[np.bool_],
    normalization: FrozenSourceNormalization,
    geometry: dict[str, float],
    *,
    baseline_builder: BaselineBuilder = build_label_free_coarse_1d_baseline,
) -> NDArray[np.float32]:
    """Construct one canonical ``[Nt,Nz,Nx,20]`` target input."""

    time = np.asarray(times_s, dtype=np.float64)
    z = np.asarray(z_m, dtype=np.float64)
    x = np.asarray(x_m, dtype=np.float64)
    mask = np.asarray(composite_mask, dtype=np.bool_)
    air = np.asarray(definition.get("air_temperature_K"), dtype=np.float64)
    top_h = np.asarray(definition.get("top_h_W_m2_K"), dtype=np.float64)
    if (
        time.ndim != 1
        or z.ndim != 1
        or x.ndim != 1
        or mask.shape != (z.size, x.size)
        or air.shape != time.shape
        or top_h.shape != x.shape
    ):
        raise ValueError("Target conditioning arrays have inconsistent shapes.")
    if not all(
        np.all(np.isfinite(value))
        for value in (time, z, x, air, top_h)
    ):
        raise ValueError("Target conditioning arrays must be finite.")
    nt, nz, nx = time.size, z.size, x.size
    shape = (nt, nz, nx)
    tool_thickness = float(geometry["tool_thickness_m"])
    composite_thickness = float(geometry["composite_thickness_m"])
    width = float(geometry["width_m"])
    total_thickness = tool_thickness + composite_thickness
    if width <= 0.0 or total_thickness <= 0.0:
        raise ValueError("Target geometry dimensions must be positive.")

    coarse_temperature, coarse_alpha = baseline_builder(
        definition,
        time,
        z,
        {
            "tool_thickness_m": tool_thickness,
            "composite_thickness_m": composite_thickness,
            "width_m": width,
        },
    )
    coarse_temperature = np.asarray(coarse_temperature, dtype=np.float64)
    coarse_alpha = np.asarray(coarse_alpha, dtype=np.float64)
    if coarse_temperature.shape != (nt, nz) or coarse_alpha.shape != (nt, nz):
        raise ValueError("Coarse baseline must return equal [Nt,Nz] fields.")
    if not (
        np.all(np.isfinite(coarse_temperature))
        and np.all(np.isfinite(coarse_alpha))
    ):
        raise ValueError("Coarse baseline returned non-finite fields.")

    air_normalized = normalization.air_temperature.encode(air)
    coarse_temperature_normalized = (
        normalization.field_temperature.encode(coarse_temperature)
    )
    time_normalized = np.linspace(0.0, 1.0, nt, dtype=np.float64)
    z_normalized = z / total_thickness
    signed_interface = z_normalized - tool_thickness / total_thickness
    initial_alpha = np.where(mask, PublicCase1Material().initial_alpha, 0.0)
    parameter_values = (
        tool_thickness,
        composite_thickness,
        float(definition["bottom_h_W_m2_K"]),
        float(np.mean(top_h)),
        float(definition["composite_conductivity_scale"]),
        float(definition["reaction_enthalpy_scale"]),
    )
    parameters = tuple(
        normalization.normalize_parameter(index, value)
        for index, value in enumerate(parameter_values)
    )

    gate = _heterogeneity_gate(definition, top_h)
    x_fraction = x / width
    x_normalized = 2.0 * x_fraction - 1.0
    left_distance = x_fraction - 0.5
    right_distance = (width - x) / width - 0.5
    top_span = (
        normalization.parameter_bounds[3, 1]
        - normalization.parameter_bounds[3, 0]
    )
    top_anomaly = (top_h - float(np.mean(top_h))) / top_span
    top_anomaly[
        np.abs(top_anomaly) <= HOMOGENEITY_ABSOLUTE_TOLERANCE
    ] = 0.0
    edge_map = np.zeros((nz, nx), dtype=np.float64)
    edge_map[:, 0] = (
        float(definition["left_h_W_m2_K"])
        / EDGE_HTC_REFERENCE_MAX_W_M2_K
    )
    edge_map[:, -1] = (
        float(definition["right_h_W_m2_K"])
        / EDGE_HTC_REFERENCE_MAX_W_M2_K
    )
    conductivity_values = np.asarray(
        [
            definition["composite_conductivity_x_W_m_K"],
            definition["base_composite_conductivity_x_W_m_K"],
            definition["composite_conductivity_z_W_m_K"],
            definition["base_composite_conductivity_z_W_m_K"],
        ],
        dtype=np.float64,
    )
    if (
        not np.all(np.isfinite(conductivity_values))
        or conductivity_values[1] <= 0.0
        or conductivity_values[3] <= 0.0
    ):
        raise ValueError("Target conductivity conditioning is invalid.")
    lateral_scale_delta = (
        conductivity_values[0] / conductivity_values[1]
        - conductivity_values[2] / conductivity_values[3]
    )
    if abs(lateral_scale_delta) <= HOMOGENEITY_ABSOLUTE_TOLERANCE:
        lateral_scale_delta = 0.0
    conductivity_span = (
        normalization.parameter_bounds[4, 1]
        - normalization.parameter_bounds[4, 0]
    )

    def broadcast_time(value: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.broadcast_to(value[:, None, None], shape)

    def broadcast_zx(value: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.broadcast_to(value[None, :, :], shape)

    def broadcast_z(value: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.broadcast_to(value[None, :, None], shape)

    def broadcast_x(value: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.broadcast_to(value[None, None, :], shape)

    def broadcast_scalar(value: float) -> NDArray[np.float64]:
        return np.broadcast_to(np.asarray(value), shape)

    channels = (
        broadcast_time(air_normalized),
        np.broadcast_to(coarse_temperature_normalized[:, :, None], shape),
        broadcast_time(time_normalized),
        broadcast_z(z_normalized),
        broadcast_zx(mask.astype(np.float64)),
        broadcast_z(signed_interface),
        broadcast_zx(initial_alpha),
        broadcast_scalar(parameters[0]),
        broadcast_scalar(parameters[1]),
        broadcast_scalar(parameters[2]),
        broadcast_scalar(parameters[3]),
        broadcast_scalar(parameters[4]),
        broadcast_scalar(parameters[5]),
        np.broadcast_to(coarse_alpha[:, :, None], shape),
        broadcast_x(x_normalized * gate),
        broadcast_x(left_distance * gate),
        broadcast_x(right_distance * gate),
        broadcast_x(top_anomaly),
        broadcast_zx(edge_map),
        broadcast_scalar(lateral_scale_delta / conductivity_span * gate),
    )
    result = np.stack(channels, axis=-1).astype(np.float32, copy=False)
    if result.shape != (*shape, TARGET_CHANNEL_COUNT):
        raise AssertionError("Internal target input channel assembly failed.")
    return result


def lift_homogeneous_source_input(
    source_input: torch.Tensor | NDArray[np.floating[Any]],
    nx: int,
) -> torch.Tensor | NDArray[np.floating[Any]]:
    """Extrude ``[...,Nt,Nz,14]`` over x and append six exact-zero channels."""

    if nx < 1:
        raise ValueError("nx must be positive.")
    if source_input.ndim < 3 or source_input.shape[-1] != SOURCE_CHANNEL_COUNT:
        raise ValueError(
            "source_input must end in canonical [Nt,Nz,14] dimensions."
        )
    if isinstance(source_input, torch.Tensor):
        expanded = source_input.unsqueeze(-2).expand(
            *source_input.shape[:-1],
            nx,
            SOURCE_CHANNEL_COUNT,
        )
        zeros = torch.zeros(
            *expanded.shape[:-1],
            len(TARGET_ADAPTER_CHANNELS),
            dtype=source_input.dtype,
            device=source_input.device,
        )
        return torch.cat((expanded, zeros), dim=-1)
    array = np.asarray(source_input)
    expanded = np.broadcast_to(
        np.expand_dims(array, axis=-2),
        (*array.shape[:-1], nx, SOURCE_CHANNEL_COUNT),
    )
    zeros = np.zeros(
        (*expanded.shape[:-1], len(TARGET_ADAPTER_CHANNELS)),
        dtype=array.dtype,
    )
    return np.concatenate((expanded, zeros), axis=-1)


def stack_target_fields(
    temperature: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Stack separate fields into canonical ``[...,2]`` target order."""

    if temperature.shape != alpha.shape:
        raise ValueError("Temperature and alpha target shapes must match.")
    return torch.stack((temperature, alpha), dim=-1)


@dataclass(frozen=True)
class _FrozenTargetContract:
    project_root: Path
    artifact_root: Path
    split_manifest_path: Path
    source_manifest_path: Path
    plan_path: Path
    split_manifest: dict[str, Any]
    source_manifest: dict[str, Any]
    plan: dict[str, Any]
    definitions: dict[int, dict[str, Any]]
    geometry: dict[str, float]
    array_paths: dict[str, Path]
    checksums: dict[str, Any]


def _validate_integer_ids(
    values: Any,
    *,
    label: str,
    allow_empty: bool = False,
) -> list[int]:
    if not isinstance(values, list):
        raise ValueError(f"{label} must be a JSON list.")
    if not values and not allow_empty:
        raise ValueError(f"{label} must not be empty.")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError(f"{label} must contain integer case IDs.")
    ids = [int(value) for value in values]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label} contains duplicate case IDs.")
    if any(value < 0 for value in ids):
        raise ValueError(f"{label} contains a negative case ID.")
    return ids


def _validate_target_contract(
    split_manifest_path: Path,
    *,
    project_root: Path,
    artifact_root: Path | None,
    verify_array_checksums: bool,
) -> _FrozenTargetContract:
    split_path = split_manifest_path.resolve()
    split = _read_json(split_path, label="target ID split manifest")
    if split.get("schema_version") != 2 or split.get("role") != "target_id":
        raise PermissionError(
            "P5 training construction accepts only the frozen role=target_id "
            "manifest; test/OOD manifests require evaluation-only code."
        )
    source_path = _resolve_repo_path(
        project_root,
        split.get("source_manifest"),
        label="source_manifest",
    )
    expected_source_sha = split.get("source_manifest_sha256")
    if not isinstance(expected_source_sha, str) or (
        _sha256_file(source_path) != expected_source_sha
    ):
        raise ValueError("Frozen target split source-manifest checksum mismatch.")
    source = _read_json(source_path, label="P4 source manifest")
    if source.get("schema_version") != 2:
        raise ValueError("P4 source manifest must use schema_version=2.")
    if split.get("dataset_id") != source.get("dataset_id"):
        raise ValueError("Target split and source manifest dataset_id differ.")
    if split.get("dataset_array_sha256") != source.get("array_sha256"):
        raise ValueError("Target split is not bound to the source array hashes.")
    if split.get("dataset_metadata_sha256") != source.get("metadata_sha256"):
        raise ValueError("Target split is not bound to source metadata.")
    if split.get("dataset_plan_sha256") != source.get("plan_sha256"):
        raise ValueError("Target split is not bound to the pre-label plan.")

    plan_path = _resolve_repo_path(
        project_root,
        source.get("plan_path"),
        label="plan_path",
    )
    plan = _read_json(plan_path, label="P4 pre-label plan")
    plan_sha = plan.get("plan_sha256")
    if plan_sha != source.get("plan_sha256"):
        raise ValueError("Pre-label plan semantic checksum does not match source.")
    plan_without_sha = {
        key: value for key, value in plan.items() if key != "plan_sha256"
    }
    if _sha256_canonical_json(plan_without_sha) != plan_sha:
        raise ValueError("Pre-label plan semantic checksum is invalid.")
    if (
        plan.get("plan_role") != "pre_label_case_plan"
        or plan.get("dataset_id") != source.get("dataset_id")
    ):
        raise ValueError("Source plan is not the canonical pre-label case plan.")

    split_payload = split.get("splits")
    if not isinstance(split_payload, dict) or set(split_payload) != {
        "train_pool",
        "validation",
        "test",
    }:
        raise ValueError(
            "Target ID splits must be exactly train_pool, validation, and test."
        )
    train_pool = _validate_integer_ids(
        split_payload["train_pool"], label="train_pool"
    )
    validation = _validate_integer_ids(
        split_payload["validation"], label="validation"
    )
    test = _validate_integer_ids(split_payload["test"], label="test")
    split_sets = (set(train_pool), set(validation), set(test))
    if (
        split_sets[0] & split_sets[1]
        or split_sets[0] & split_sets[2]
        or split_sets[1] & split_sets[2]
    ):
        raise ValueError("Target ID split case IDs overlap.")
    source_splits = source.get("splits")
    plan_splits = plan.get("splits")
    if not isinstance(source_splits, dict) or not isinstance(plan_splits, dict):
        raise ValueError("Source manifest and plan need split mappings.")
    expected = {
        "train_pool": source_splits.get("train"),
        "validation": source_splits.get("validation"),
        "test": source_splits.get("id_test"),
    }
    if split_payload != expected:
        raise ValueError("Target ID splits differ from the canonical source split.")
    if (
        plan_splits.get("train") != train_pool
        or plan_splits.get("validation") != validation
        or plan_splits.get("id_test") != test
    ):
        raise ValueError("Target ID splits differ from the pre-label plan.")

    declared_budgets = source.get("nested_training_budgets")
    if not isinstance(declared_budgets, list) or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in declared_budgets
    ):
        raise ValueError("Source nested training budgets are invalid.")
    budget_payload = split.get("nested_training_budgets")
    if not isinstance(budget_payload, dict) or set(budget_payload) != {
        str(value) for value in declared_budgets
    }:
        raise ValueError("Target split budget keys differ from the source.")
    previous: list[int] = []
    for budget in sorted(declared_budgets):
        ids = _validate_integer_ids(
            budget_payload[str(budget)],
            label=f"nested_training_budgets[{budget}]",
        )
        if len(ids) != budget or ids != train_pool[:budget]:
            raise ValueError(
                f"Budget {budget} must be the exact train_pool prefix."
            )
        if previous and ids[: len(previous)] != previous:
            raise ValueError("Target label budgets are not exactly nested.")
        previous = ids

    cases = plan.get("cases")
    case_count = source.get("case_count")
    if (
        not isinstance(case_count, int)
        or not isinstance(cases, list)
        or len(cases) != case_count
    ):
        raise ValueError("P4 plan case count is inconsistent.")
    definitions: dict[int, dict[str, Any]] = {}
    for position, entry in enumerate(cases):
        if not isinstance(entry, dict) or not isinstance(
            entry.get("definition"), dict
        ):
            raise ValueError("Every plan case needs a definition.")
        definition = entry["definition"]
        case_id = definition.get("case_id")
        if (
            isinstance(case_id, bool)
            or not isinstance(case_id, int)
            or case_id != position
            or case_id in definitions
        ):
            raise ValueError(
                "Plan case IDs must be unique contiguous array row indices."
            )
        definitions[case_id] = definition
    if set(train_pool + validation + test) - set(definitions):
        raise ValueError("Target ID split references a missing plan case.")
    source_definition_hashes = source.get("case_definition_hashes")
    split_definition_hashes = split.get("case_definition_hashes")
    if (
        not isinstance(source_definition_hashes, dict)
        or not isinstance(split_definition_hashes, dict)
    ):
        raise ValueError("Frozen manifests need case-definition hash mappings.")
    expected_id_hashes: dict[str, str] = {}
    for case_id, definition in definitions.items():
        case_key = definition.get("case_key")
        if not isinstance(case_key, str) or not case_key:
            raise ValueError("Every plan definition needs a nonempty case_key.")
        expected_hash = _sha256_canonical_json(definition)
        if source_definition_hashes.get(case_key) != expected_hash:
            raise ValueError(
                f"Source case-definition hash mismatch for case {case_id}."
            )
        if case_id in split_sets[0] | split_sets[1] | split_sets[2]:
            expected_id_hashes[case_key] = expected_hash
    if split_definition_hashes != expected_id_hashes:
        raise ValueError(
            "Target ID case-definition hashes differ from the pre-label plan."
        )
    generation_status = source.get("generation_status")
    if (
        not isinstance(generation_status, dict)
        or generation_status.get("failed_case_ids") != []
        or generation_status.get("silent_failure_case_ids") != []
        or set(generation_status.get("passed_case_ids", ()))
        != set(definitions)
    ):
        raise ValueError("P4 generation status does not pass every plan case.")

    shape = source.get("array_shape_case_time_z_x")
    if (
        not isinstance(shape, list)
        or len(shape) != 4
        or any(isinstance(value, bool) or not isinstance(value, int) for value in shape)
        or shape[0] != case_count
    ):
        raise ValueError("P4 canonical array shape is invalid.")
    root = (
        artifact_root.resolve()
        if artifact_root is not None
        else _resolve_repo_path(
            project_root,
            source.get("array_artifact_root"),
            label="array_artifact_root",
        )
    )
    required_shapes = {
        "air_temperature_K": (shape[0], shape[1]),
        "alpha": tuple(shape),
        "composite_mask": (shape[2], shape[3]),
        "temperature_K": tuple(shape),
        "time_s": (shape[1],),
        "top_h_W_m2_K": (shape[0], shape[3]),
        "x_m": (shape[3],),
        "z_m": (shape[2],),
    }
    required_dtypes = {
        "air_temperature_K": np.dtype("float32"),
        "alpha": np.dtype("float32"),
        "composite_mask": np.dtype("bool"),
        "temperature_K": np.dtype("float32"),
        "time_s": np.dtype("float64"),
        "top_h_W_m2_K": np.dtype("float32"),
        "x_m": np.dtype("float64"),
        "z_m": np.dtype("float64"),
    }
    array_hashes = source.get("array_sha256")
    if not isinstance(array_hashes, dict) or set(array_hashes) != set(
        required_shapes
    ):
        raise ValueError("P4 source array checksum mapping is incomplete.")
    array_paths: dict[str, Path] = {}
    for name, expected_shape in required_shapes.items():
        path = root / f"{name}.npy"
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ValueError(f"Cannot open P4 array {path}: {error}") from error
        if array.shape != expected_shape:
            raise ValueError(f"{name} shape differs from the frozen manifest.")
        if array.dtype != required_dtypes[name]:
            raise ValueError(f"{name} dtype differs from the frozen contract.")
        del array
        if verify_array_checksums and _sha256_file(path) != array_hashes[name]:
            raise ValueError(f"{name} file checksum differs from the manifest.")
        array_paths[name] = path

    # These small exogenous arrays are checked against the pre-label plan.
    planned_air = np.asarray(
        [
            definitions[case_id]["air_temperature_K"]
            for case_id in range(case_count)
        ],
        dtype=np.float32,
    )
    planned_top = np.asarray(
        [
            definitions[case_id]["top_h_W_m2_K"]
            for case_id in range(case_count)
        ],
        dtype=np.float32,
    )
    stored_air = np.load(
        array_paths["air_temperature_K"], mmap_mode="r", allow_pickle=False
    )
    stored_top = np.load(
        array_paths["top_h_W_m2_K"], mmap_mode="r", allow_pickle=False
    )
    if not np.array_equal(stored_air, planned_air) or not np.array_equal(
        stored_top, planned_top
    ):
        raise ValueError("Stored P4 conditioning differs from the pre-label plan.")
    del stored_air, stored_top

    resolved = plan.get("resolved_config")
    geometry_payload = (
        resolved.get("geometry") if isinstance(resolved, dict) else None
    )
    if not isinstance(geometry_payload, dict):
        raise ValueError("P4 plan has no resolved geometry.")
    geometry = {
        key: float(geometry_payload[key])
        for key in (
            "width_m",
            "tool_thickness_m",
            "composite_thickness_m",
        )
    }
    if any(not np.isfinite(value) or value <= 0.0 for value in geometry.values()):
        raise ValueError("P4 resolved geometry must be finite and positive.")
    stored_time = np.load(
        array_paths["time_s"], mmap_mode="r", allow_pickle=False
    )
    if not np.array_equal(
        stored_time,
        np.asarray(plan.get("time_s"), dtype=np.float64),
    ):
        raise ValueError("Stored P4 time grid differs from the pre-label plan.")
    stored_z = np.load(array_paths["z_m"], mmap_mode="r", allow_pickle=False)
    stored_x = np.load(array_paths["x_m"], mmap_mode="r", allow_pickle=False)
    stored_mask = np.load(
        array_paths["composite_mask"], mmap_mode="r", allow_pickle=False
    )
    if (
        stored_z.min() < 0.0
        or stored_z.max()
        > geometry["tool_thickness_m"]
        + geometry["composite_thickness_m"]
        or stored_x.min() < 0.0
        or stored_x.max() > geometry["width_m"]
        or not np.array_equal(
            stored_mask,
            np.broadcast_to(
                (stored_z > geometry["tool_thickness_m"])[:, None],
                stored_mask.shape,
            ),
        )
    ):
        raise ValueError("Stored P4 coordinates or material mask are invalid.")
    del stored_time, stored_z, stored_x, stored_mask

    return _FrozenTargetContract(
        project_root=project_root.resolve(),
        artifact_root=root,
        split_manifest_path=split_path,
        source_manifest_path=source_path,
        plan_path=plan_path,
        split_manifest=split,
        source_manifest=source,
        plan=plan,
        definitions=definitions,
        geometry=geometry,
        array_paths=array_paths,
        checksums={
            "target_id_manifest_sha256": _sha256_file(split_path),
            "source_manifest_sha256": expected_source_sha,
            "pre_label_plan_file_sha256": _sha256_file(plan_path),
            "pre_label_plan_semantic_sha256": plan_sha,
            "dataset_array_sha256": dict(array_hashes),
            "array_files_verified": bool(verify_array_checksums),
        },
    )


class Target2DTrainingDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
):
    """On-demand mmap-backed train or validation subset."""

    def __init__(
        self,
        *,
        split_name: str,
        case_ids: list[int],
        contract: _FrozenTargetContract,
        normalization: FrozenSourceNormalization,
        baseline_builder: BaselineBuilder,
    ) -> None:
        if split_name not in {"train", "validation"}:
            raise PermissionError(
                "Training datasets may expose only train or validation labels."
            )
        self.split_name = split_name
        self.case_ids = tuple(case_ids)
        self.channel_names = TARGET_INPUT_CHANNELS
        self.output_channel_names = TARGET_OUTPUT_CHANNELS
        self.normalization = normalization.metadata
        self._array_paths = dict(contract.array_paths)
        self._definitions = {
            case_id: contract.definitions[case_id] for case_id in case_ids
        }
        self._geometry = dict(contract.geometry)
        self._source_normalization = normalization
        self._baseline_builder = baseline_builder
        # Input and label mmaps are intentionally opened by separate methods.
        # Physics-only training can therefore construct inputs without even
        # opening, indexing, or retaining a handle to a fine-label array.
        self._input_arrays: dict[str, NDArray[Any]] | None = None
        self._label_arrays: dict[str, NDArray[Any]] | None = None
        self._baseline_cache: dict[
            int, tuple[NDArray[np.float64], NDArray[np.float64]]
        ] = {}
        self._accessed_case_ids: set[int] = set()
        self._label_access_attempted_case_ids: set[int] = set()
        self._input_only_accessed_case_ids: set[int] = set()
        self._opened_input_array_names: set[str] = set()
        self._opened_label_array_names: set[str] = set()
        self._lock = threading.Lock()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_input_arrays"] = None
        state["_label_arrays"] = None
        state["_baseline_cache"] = {}
        state["_accessed_case_ids"] = set()
        state["_label_access_attempted_case_ids"] = set()
        state["_input_only_accessed_case_ids"] = set()
        state["_opened_input_array_names"] = set()
        state["_opened_label_array_names"] = set()
        state["_lock"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self.case_ids)

    @property
    def accessed_case_ids(self) -> tuple[int, ...]:
        """Case IDs whose fine temperature/alpha values were returned."""

        with self._lock:
            return tuple(sorted(self._accessed_case_ids))

    @property
    def label_access_attempted_case_ids(self) -> tuple[int, ...]:
        """Case IDs for which the label-bearing accessor was invoked."""

        with self._lock:
            return tuple(sorted(self._label_access_attempted_case_ids))

    @property
    def input_only_accessed_case_ids(self) -> tuple[int, ...]:
        with self._lock:
            return tuple(sorted(self._input_only_accessed_case_ids))

    @property
    def opened_label_array_names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._opened_label_array_names))

    @property
    def opened_input_array_names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._opened_input_array_names))

    def _open_input_arrays(self) -> dict[str, NDArray[Any]]:
        if self._input_arrays is None:
            names = tuple(
                name
                for name in self._array_paths
                if name not in {"temperature_K", "alpha"}
            )
            opened = {
                name: np.load(
                    self._array_paths[name],
                    mmap_mode="r",
                    allow_pickle=False,
                )
                for name in names
            }
            with self._lock:
                self._input_arrays = opened
                self._opened_input_array_names.update(names)
        return self._input_arrays

    def _open_label_arrays(self) -> dict[str, NDArray[Any]]:
        if self._label_arrays is None:
            names = ("temperature_K", "alpha")
            opened = {
                name: np.load(
                    self._array_paths[name],
                    mmap_mode="r",
                    allow_pickle=False,
                )
                for name in names
            }
            with self._lock:
                self._label_arrays = opened
                self._opened_label_array_names.update(names)
        return self._label_arrays

    def _resolved_position(self, index: int) -> tuple[int, int]:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise TypeError("Target dataset index must be an integer.")
        position = int(index)
        if position < 0:
            position += len(self.case_ids)
        if position < 0 or position >= len(self.case_ids):
            raise IndexError("Target dataset index is out of range.")
        case_id = self.case_ids[position]
        if case_id not in self._definitions:
            raise PermissionError("Case ID is outside this frozen dataset subset.")
        return position, case_id

    def _input_for_case(self, case_id: int) -> torch.Tensor:
        arrays = self._open_input_arrays()
        time = np.asarray(arrays["time_s"], dtype=np.float64)
        z = np.asarray(arrays["z_m"], dtype=np.float64)
        x = np.asarray(arrays["x_m"], dtype=np.float64)
        mask = np.asarray(arrays["composite_mask"], dtype=np.bool_)
        baseline = self._baseline(case_id, time, z)

        def use_cached_baseline(
            definition: dict[str, Any],
            times_s: FloatArray,
            z_m: FloatArray,
            geometry: dict[str, float],
        ) -> tuple[FloatArray, FloatArray]:
            del definition, times_s, z_m, geometry
            return baseline

        inputs = build_target_2d_input(
            self._definitions[case_id],
            time,
            z,
            x,
            mask,
            self._source_normalization,
            self._geometry,
            baseline_builder=use_cached_baseline,
        )
        return torch.from_numpy(inputs)

    def input_only_item(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return label-free model inputs and case ID.

        This method never calls ``_open_label_arrays``.  The separate audit
        sets make a later terminal receipt able to prove that the
        label-bearing accessor was never attempted for a physics-only run.
        """

        _position, case_id = self._resolved_position(index)
        inputs = self._input_for_case(case_id)
        with self._lock:
            self._input_only_accessed_case_ids.add(case_id)
        return inputs, torch.tensor(case_id, dtype=torch.long)

    def _baseline(
        self,
        case_id: int,
        time: NDArray[np.float64],
        z: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        with self._lock:
            cached = self._baseline_cache.get(case_id)
        if cached is not None:
            return cached
        baseline = self._baseline_builder(
            self._definitions[case_id],
            time,
            z,
            self._geometry,
        )
        frozen = (
            np.asarray(baseline[0], dtype=np.float64),
            np.asarray(baseline[1], dtype=np.float64),
        )
        with self._lock:
            self._baseline_cache.setdefault(case_id, frozen)
            return self._baseline_cache[case_id]

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _position, case_id = self._resolved_position(index)
        with self._lock:
            self._label_access_attempted_case_ids.add(case_id)
        inputs = self._input_for_case(case_id)
        arrays = self._open_label_arrays()
        # Fine labels are first interpreted here, after the case ID has passed
        # the frozen subset check.  No fitting or input construction uses them.
        temperature_raw = np.array(
            arrays["temperature_K"][case_id],
            dtype=np.float32,
            copy=True,
        )
        alpha = np.array(
            arrays["alpha"][case_id],
            dtype=np.float32,
            copy=True,
        )
        temperature = self._source_normalization.field_temperature.encode(
            temperature_raw
        ).astype(np.float32, copy=False)
        with self._lock:
            self._accessed_case_ids.add(case_id)
        return (
            inputs,
            torch.from_numpy(np.array(temperature, copy=True)),
            torch.from_numpy(alpha),
            torch.tensor(case_id, dtype=torch.long),
        )


@dataclass(frozen=True)
class PreparedTarget2DTraining:
    """Prepared, frozen P5 train/validation access contract."""

    train_dataset: Target2DTrainingDataset
    validation_dataset: Target2DTrainingDataset
    label_budget: int
    train_case_ids: tuple[int, ...]
    validation_case_ids: tuple[int, ...]
    normalization: dict[str, Any]
    checksums: dict[str, Any]
    channel_names: tuple[str, ...] = TARGET_INPUT_CHANNELS
    output_channel_names: tuple[str, ...] = TARGET_OUTPUT_CHANNELS

    def dataset(self, split: str) -> Target2DTrainingDataset:
        if split == "train":
            return self.train_dataset
        if split == "validation":
            return self.validation_dataset
        raise PermissionError(
            "P5 training data exposes only 'train' and 'validation'. "
            "ID-test and OOD labels require an evaluation-only loader."
        )

    @property
    def accessed_case_ids(self) -> dict[str, tuple[int, ...]]:
        return {
            "train": self.train_dataset.accessed_case_ids,
            "validation": self.validation_dataset.accessed_case_ids,
        }

    @property
    def label_access_audit(self) -> dict[str, dict[str, Any]]:
        """Return split-specific fine-label attempt/open/access evidence."""

        return {
            split: {
                "attempted_case_ids": list(
                    dataset.label_access_attempted_case_ids
                ),
                "accessed_case_ids": list(dataset.accessed_case_ids),
                "opened_label_array_names": list(
                    dataset.opened_label_array_names
                ),
                "input_only_accessed_case_ids": list(
                    dataset.input_only_accessed_case_ids
                ),
            }
            for split, dataset in (
                ("train", self.train_dataset),
                ("validation", self.validation_dataset),
            )
        }


def prepare_target_2d_training(
    split_manifest: Path,
    source_checkpoint: Path,
    *,
    label_budget: int,
    project_root: Path | None = None,
    artifact_root: Path | None = None,
    verify_array_checksums: bool = True,
    baseline_builder: BaselineBuilder = build_label_free_coarse_1d_baseline,
) -> PreparedTarget2DTraining:
    """Prepare the only label-bearing data path allowed during P5 training."""

    split_path = split_manifest.resolve()
    root = (
        project_root.resolve()
        if project_root is not None
        else split_path.parent.parent.resolve()
    )
    contract = _validate_target_contract(
        split_path,
        project_root=root,
        artifact_root=artifact_root,
        verify_array_checksums=verify_array_checksums,
    )
    if isinstance(label_budget, bool) or not isinstance(label_budget, int):
        raise ValueError("label_budget must be an integer.")
    budget_key = str(label_budget)
    budgets = contract.split_manifest["nested_training_budgets"]
    if budget_key not in budgets:
        raise ValueError(
            f"label_budget={label_budget} is not one of "
            f"{sorted(int(value) for value in budgets)}."
        )
    train_case_ids = [int(value) for value in budgets[budget_key]]
    validation_case_ids = [
        int(value)
        for value in contract.split_manifest["splits"]["validation"]
    ]
    source_normalization = load_source_normalization_contract(
        source_checkpoint
    )
    checksums = dict(contract.checksums)
    checksums.update(
        {
            "source_checkpoint_sha256": (
                source_normalization.checkpoint_sha256
            ),
            "source_normalization_sha256": (
                source_normalization.normalization_sha256
            ),
        }
    )
    train_dataset = Target2DTrainingDataset(
        split_name="train",
        case_ids=train_case_ids,
        contract=contract,
        normalization=source_normalization,
        baseline_builder=baseline_builder,
    )
    validation_dataset = Target2DTrainingDataset(
        split_name="validation",
        case_ids=validation_case_ids,
        contract=contract,
        normalization=source_normalization,
        baseline_builder=baseline_builder,
    )
    return PreparedTarget2DTraining(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        label_budget=label_budget,
        train_case_ids=tuple(train_case_ids),
        validation_case_ids=tuple(validation_case_ids),
        normalization=source_normalization.metadata,
        checksums=checksums,
    )
