"""Preparation of parameterized conservative-solver cases for P3 pretraining."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import TensorDataset

from cdcureno.data.joint_case1 import causal_exponential_smoothing
from cdcureno.data.normalization import RangeNormalizer


INPUT_CHANNELS = (
    "air_temperature_normalized",
    "causal_physics_temperature_baseline_normalized",
    "time_normalized",
    "through_thickness_position_normalized",
    "composite_mask",
    "signed_distance_to_interface_normalized",
    "initial_degree_of_cure",
    "tool_thickness_normalized",
    "composite_thickness_normalized",
    "lower_htc_normalized",
    "upper_htc_normalized",
    "composite_conductivity_scale_normalized",
    "heat_of_reaction_scale_normalized",
    "low_fidelity_degree_of_cure",
)
PARAMETER_BOUNDS = np.array(
    [
        [0.010, 0.040],
        [0.010, 0.040],
        [40.0, 100.0],
        [80.0, 160.0],
        [0.8, 1.2],
        [0.9, 1.1],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class PreparedSource1D:
    inputs: torch.Tensor
    temperature: torch.Tensor
    alpha: torch.Tensor
    case_ids: torch.Tensor
    family_ids: torch.Tensor
    splits: dict[str, list[int]]
    held_out_families: dict[str, list[int]]
    normalization: dict[str, Any]
    channel_names: tuple[str, ...] = INPUT_CHANNELS

    def dataset(self, split: str) -> TensorDataset:
        if split in self.splits:
            ids = self.splits[split]
        elif split in self.held_out_families:
            ids = self.held_out_families[split]
        else:
            raise KeyError(f"Unknown source split: {split}")
        indices = torch.tensor(ids, dtype=torch.long)
        return TensorDataset(
            self.inputs[indices],
            self.temperature[indices],
            self.alpha[indices],
            self.case_ids[indices],
            self.family_ids[indices],
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_source_1d(
    data_path: Path,
    split_manifest: Path,
    *,
    smoothing: float = 0.2,
    time_stride: int = 2,
) -> PreparedSource1D:
    """Load source arrays with train-only target normalization."""

    if time_stride < 1:
        raise ValueError("time_stride must be positive.")
    manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
    if _sha256(data_path) != manifest["array_sha256"]:
        raise ValueError("Source array checksum does not match its manifest.")
    with np.load(data_path) as arrays:
        air_full = np.asarray(arrays["air_temperature_K"], dtype=np.float64)
        temperature_full = np.asarray(arrays["temperature_K"], dtype=np.float64)
        inert_temperature_full = (
            np.asarray(arrays["inert_temperature_K"], dtype=np.float64)
            if "inert_temperature_K" in arrays
            else None
        )
        coarse_temperature_full = (
            np.asarray(arrays["coarse_temperature_K"], dtype=np.float64)
            if "coarse_temperature_K" in arrays
            else None
        )
        coarse_alpha_full = (
            np.asarray(arrays["coarse_alpha"], dtype=np.float64)
            if "coarse_alpha" in arrays
            else None
        )
        alpha_full = np.asarray(arrays["alpha"], dtype=np.float64)
        composite_mask = np.asarray(
            arrays["composite_mask"], dtype=np.float64
        )
        signed_distance = np.asarray(
            arrays["signed_distance_fraction"], dtype=np.float64
        )
        parameters = np.asarray(arrays["parameters"], dtype=np.float64)
        case_ids = np.asarray(arrays["case_ids"], dtype=np.int64)
        family_ids = np.asarray(arrays["family_ids"], dtype=np.int64)
    time_indices = np.arange(0, air_full.shape[1], time_stride)
    if time_indices[-1] != air_full.shape[1] - 1:
        time_indices = np.append(time_indices, air_full.shape[1] - 1)
    air = air_full[:, time_indices]
    temperature = temperature_full[:, time_indices]
    alpha = alpha_full[:, time_indices]
    if temperature.shape != alpha.shape or temperature.ndim != 3:
        raise ValueError("Temperature and alpha must be equal [case,time,z] arrays.")
    case_count, time_count, space_count = temperature.shape
    if (
        air.shape != (case_count, time_count)
        or composite_mask.shape != (case_count, space_count)
        or signed_distance.shape != (case_count, space_count)
        or parameters.shape != (case_count, 6)
    ):
        raise ValueError("Source conditioning arrays have inconsistent shapes.")
    split_payload = manifest["splits"]
    splits = {
        key: [int(value) for value in split_payload[key]]
        for key in ("train", "validation", "in_family_test")
    }
    held_out = {
        name: [int(value) for value in values]
        for name, values in split_payload["held_out_family_test"].items()
    }
    all_ids = [
        *splits["train"],
        *splits["validation"],
        *splits["in_family_test"],
        *(case_id for ids in held_out.values() for case_id in ids),
    ]
    if sorted(all_ids) != list(range(case_count)):
        raise ValueError("Source manifest must partition every case exactly once.")
    train_ids = np.asarray(splits["train"], dtype=np.int64)
    air_normalizer = RangeNormalizer.fit(air[train_ids])
    temperature_normalizer = RangeNormalizer.fit(temperature[train_ids])
    baseline = (
        coarse_temperature_full[:, time_indices]
        if coarse_temperature_full is not None
        else (
            inert_temperature_full[:, time_indices]
            if inert_temperature_full is not None
            else causal_exponential_smoothing(air, smoothing=smoothing)
        )
    )
    coarse_alpha = (
        coarse_alpha_full[:, time_indices]
        if coarse_alpha_full is not None
        else np.zeros_like(alpha)
    )
    air_normalized = air_normalizer.encode(air)
    baseline_normalized = temperature_normalizer.encode(baseline)
    temperature_normalized = temperature_normalizer.encode(temperature)
    parameter_normalized = (
        parameters - PARAMETER_BOUNDS[:, 0]
    ) / (PARAMETER_BOUNDS[:, 1] - PARAMETER_BOUNDS[:, 0])
    if np.any(parameter_normalized < -1.0e-6) or np.any(
        parameter_normalized > 1.0 + 1.0e-6
    ):
        raise ValueError("Source parameter lies outside its frozen design range.")
    parameter_normalized = np.clip(parameter_normalized, 0.0, 1.0)
    time_coordinate = np.linspace(0.0, 1.0, time_count, dtype=np.float64)
    space_coordinate = np.linspace(0.0, 1.0, space_count, dtype=np.float64)
    shape = (case_count, time_count, space_count)
    baseline_field = (
        baseline_normalized
        if baseline_normalized.ndim == 3
        else np.broadcast_to(baseline_normalized[:, :, None], shape)
    )
    inputs = np.stack(
        [
            np.broadcast_to(air_normalized[:, :, None], shape),
            baseline_field,
            np.broadcast_to(time_coordinate[None, :, None], shape),
            np.broadcast_to(space_coordinate[None, None, :], shape),
            np.broadcast_to(composite_mask[:, None, :], shape),
            np.broadcast_to(signed_distance[:, None, :], shape),
            np.broadcast_to(alpha[:, :1, :], shape),
            *[
                np.broadcast_to(parameter_normalized[:, None, index, None], shape)
                for index in range(parameter_normalized.shape[1])
            ],
            coarse_alpha,
        ],
        axis=-1,
    )
    normalization = {
        "scope": "training_cases_only",
        "time_stride": time_stride,
        "time_indices": time_indices.tolist(),
        "air_temperature": {
            "minimum": air_normalizer.minimum,
            "maximum": air_normalizer.maximum,
        },
        "field_temperature": {
            "minimum": temperature_normalizer.minimum,
            "maximum": temperature_normalizer.maximum,
        },
        "parameter_bounds": PARAMETER_BOUNDS.tolist(),
        "input_channels": list(INPUT_CHANNELS),
        "held_out_labels_used_for_normalization": False,
        "causal_baseline": {
            "method": (
                "zero_reaction_conservative_conduction"
                if coarse_temperature_full is None
                and inert_temperature_full is not None
                else (
                    "coarse_thermochemical_conservative_solver"
                    if coarse_temperature_full is not None
                    else "exponential_smoothing"
                )
            ),
            "uses_thermochemical_labels": False,
        },
    }
    return PreparedSource1D(
        inputs=torch.from_numpy(inputs.astype(np.float32)),
        temperature=torch.from_numpy(temperature_normalized.astype(np.float32)),
        alpha=torch.from_numpy(alpha.astype(np.float32)),
        case_ids=torch.from_numpy(case_ids),
        family_ids=torch.from_numpy(family_ids),
        splits=splits,
        held_out_families=held_out,
        normalization=normalization,
    )
