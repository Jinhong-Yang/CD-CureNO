from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

from cdcureno.data.joint_case1 import (
    INPUT_CHANNELS,
    causal_exponential_smoothing,
    prepare_joint_case1,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _prepared():
    return prepare_joint_case1(
        PROJECT_ROOT / "external" / "ResFNO" / "data" / "Case1.mat",
        PROJECT_ROOT / "splits" / "legacy_case1_corrected_matched_v1.json",
    )


def test_joint_case1_uses_canonical_axis_order_and_complete_case_splits() -> None:
    prepared = _prepared()
    assert prepared.inputs.shape == (200, 223, 51, len(INPUT_CHANNELS))
    assert prepared.temperature.shape == (200, 223, 51)
    assert prepared.alpha.shape == (200, 223, 51)
    assert len(prepared.dataset("train")) == 50
    assert len(prepared.dataset("validation")) == 25
    assert len(prepared.dataset("test")) == 125
    arrays = sio.loadmat(PROJECT_ROOT / "external" / "ResFNO" / "data" / "Case1.mat")
    expected_alpha = torch.from_numpy(
        arrays["dataA"][0].T.astype(np.float32)
    )
    assert torch.equal(prepared.alpha[0], expected_alpha)


def test_joint_case1_material_interface_and_initial_condition_are_explicit() -> None:
    prepared = _prepared()
    mask = prepared.inputs[0, :, :, 4]
    assert torch.count_nonzero(mask[:, :21]) == 0
    assert torch.all(mask[:, 21:] == 1)
    assert torch.equal(prepared.inputs[:, 0, :, 6], prepared.alpha[:, 0, :])
    assert torch.count_nonzero(prepared.alpha[:, :, :21]) == 0


def test_joint_case1_normalizers_are_fit_on_train_cases_only() -> None:
    prepared = _prepared()
    arrays = sio.loadmat(PROJECT_ROOT / "external" / "ResFNO" / "data" / "Case1.mat")
    train = np.arange(50)
    assert prepared.normalization["scope"] == "train_cases_only"
    assert prepared.normalization["air_temperature"]["minimum"] == float(
        arrays["dataTair"][train].min()
    )
    assert prepared.normalization["air_temperature"]["maximum"] == float(
        arrays["dataTair"][train].max()
    )
    assert prepared.normalization["field_temperature"]["minimum"] == float(
        arrays["dataT"][train].min()
    )
    assert prepared.normalization["field_temperature"]["maximum"] == float(
        arrays["dataT"][train].max()
    )


def test_causal_smoothing_is_invariant_to_future_perturbations() -> None:
    values = np.arange(24, dtype=float).reshape(2, 12)
    perturbed = values.copy()
    perturbed[:, 7:] += 1000.0
    original_filtered = causal_exponential_smoothing(values)
    perturbed_filtered = causal_exponential_smoothing(perturbed)
    np.testing.assert_array_equal(original_filtered[:, :7], perturbed_filtered[:, :7])
