"""Canonical joint-field preparation for the public ResFNO Case1 data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import TensorDataset

from cdcureno.data.normalization import RangeNormalizer


CASE_COUNT = 200
SPACE_COUNT = 51
TIME_COUNT = 223
TOOL_LAST_INDEX = 20
INPUT_CHANNELS = (
    "air_temperature_normalized",
    "causal_temperature_baseline_normalized",
    "time_normalized",
    "through_thickness_position_normalized",
    "composite_mask",
    "signed_distance_to_interface_normalized",
    "initial_degree_of_cure",
)


@dataclass(frozen=True)
class PreparedJointCase1:
    """In-memory complete-case tensors in canonical `[B,Nt,Nz,C]` order."""

    inputs: torch.Tensor
    temperature: torch.Tensor
    alpha: torch.Tensor
    case_ids: torch.Tensor
    splits: dict[str, list[int]]
    normalization: dict[str, Any]
    channel_names: tuple[str, ...] = INPUT_CHANNELS

    def dataset(self, split: str) -> TensorDataset:
        if split not in self.splits:
            raise KeyError(f"Unknown split {split!r}; expected one of {tuple(self.splits)}")
        ids = torch.tensor(self.splits[split], dtype=torch.long)
        return TensorDataset(
            self.inputs[ids],
            self.temperature[ids],
            self.alpha[ids],
            self.case_ids[ids],
        )


def causal_exponential_smoothing(
    values: np.ndarray, smoothing: float = 0.2
) -> np.ndarray:
    """Low-pass time series without using any future samples."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"Expected [case,time] values, got shape {array.shape}.")
    if not 0.0 < smoothing <= 1.0:
        raise ValueError("smoothing must be in (0,1].")
    filtered = np.empty_like(array)
    filtered[:, 0] = array[:, 0]
    for time_index in range(1, array.shape[1]):
        filtered[:, time_index] = (
            smoothing * array[:, time_index]
            + (1.0 - smoothing) * filtered[:, time_index - 1]
        )
    return filtered


def _load_splits(path: Path) -> dict[str, list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    splits = payload["splits"]
    observed = [
        case_id
        for split in ("train", "validation", "test")
        for case_id in splits[split]
    ]
    if sorted(observed) != list(range(CASE_COUNT)):
        raise ValueError("Joint Case1 manifest must partition all 200 cases exactly once.")
    return splits


def prepare_joint_case1(
    data_path: Path,
    split_manifest: Path,
    smoothing: float = 0.2,
) -> PreparedJointCase1:
    """Load, validate, normalize, and orient Case1 for joint model training."""

    arrays = sio.loadmat(data_path)
    temperature_zt = np.asarray(arrays["dataT"], dtype=np.float64)
    alpha_zt = np.asarray(arrays["dataA"], dtype=np.float64)
    air = np.asarray(arrays["dataTair"], dtype=np.float64)
    expected_field_shape = (CASE_COUNT, SPACE_COUNT, TIME_COUNT)
    if temperature_zt.shape != expected_field_shape:
        raise ValueError(
            f"dataT expected {expected_field_shape}, got {temperature_zt.shape}."
        )
    if alpha_zt.shape != expected_field_shape:
        raise ValueError(
            f"dataA expected {expected_field_shape}, got {alpha_zt.shape}."
        )
    if air.shape != (CASE_COUNT, TIME_COUNT):
        raise ValueError(
            f"dataTair expected {(CASE_COUNT, TIME_COUNT)}, got {air.shape}."
        )
    if not all(np.isfinite(values).all() for values in (temperature_zt, alpha_zt, air)):
        raise ValueError("Case1 contains non-finite values.")
    if alpha_zt.min() < 0.0 or alpha_zt.max() > 1.0:
        raise ValueError("Degree of cure must lie within [0,1].")

    splits = _load_splits(split_manifest)
    train_ids = np.asarray(splits["train"], dtype=int)
    air_normalizer = RangeNormalizer.fit(air[train_ids])
    temperature_normalizer = RangeNormalizer.fit(temperature_zt[train_ids])
    causal_baseline = causal_exponential_smoothing(air, smoothing=smoothing)

    temperature = np.transpose(temperature_zt, (0, 2, 1))
    alpha = np.transpose(alpha_zt, (0, 2, 1))
    air_normalized = air_normalizer.encode(air)
    baseline_normalized = temperature_normalizer.encode(causal_baseline)
    temperature_normalized = temperature_normalizer.encode(temperature)

    time_coordinate = np.linspace(0.0, 1.0, TIME_COUNT, dtype=np.float64)
    space_coordinate = np.linspace(0.0, 1.0, SPACE_COUNT, dtype=np.float64)
    position_mm = np.arange(SPACE_COUNT, dtype=np.float64)
    composite_mask = (position_mm > TOOL_LAST_INDEX).astype(np.float64)
    signed_distance = (position_mm - TOOL_LAST_INDEX) / (SPACE_COUNT - 1)
    alpha_initial = alpha[:, 0, :]

    shape = (CASE_COUNT, TIME_COUNT, SPACE_COUNT)
    inputs = np.stack(
        [
            np.broadcast_to(air_normalized[:, :, None], shape),
            np.broadcast_to(baseline_normalized[:, :, None], shape),
            np.broadcast_to(time_coordinate[None, :, None], shape),
            np.broadcast_to(space_coordinate[None, None, :], shape),
            np.broadcast_to(composite_mask[None, None, :], shape),
            np.broadcast_to(signed_distance[None, None, :], shape),
            np.broadcast_to(alpha_initial[:, None, :], shape),
        ],
        axis=-1,
    )
    normalization = {
        "scope": "train_cases_only",
        "train_case_ids": splits["train"],
        "air_temperature": {
            "minimum": air_normalizer.minimum,
            "maximum": air_normalizer.maximum,
        },
        "field_temperature": {
            "minimum": temperature_normalizer.minimum,
            "maximum": temperature_normalizer.maximum,
        },
        "alpha": {"minimum": 0.0, "maximum": 1.0, "transform": "identity"},
        "causal_baseline": {
            "method": "exponential_smoothing",
            "smoothing": smoothing,
        },
        "axis_order": ["case", "time", "through_thickness_position", "channel"],
        "input_channels": list(INPUT_CHANNELS),
        "interface": {
            "tool_indices": [0, TOOL_LAST_INDEX],
            "composite_indices": [TOOL_LAST_INDEX + 1, SPACE_COUNT - 1],
            "source": "external/ResFNO/Step1_main.py",
        },
    }
    return PreparedJointCase1(
        inputs=torch.from_numpy(inputs.astype(np.float32)),
        temperature=torch.from_numpy(temperature_normalized.astype(np.float32)),
        alpha=torch.from_numpy(alpha.astype(np.float32)),
        case_ids=torch.arange(CASE_COUNT, dtype=torch.long),
        splits=splits,
        normalization=normalization,
    )
