from pathlib import Path

import numpy as np
import torch

from cdcureno.training.joint import (
    JointTrainConfig,
    compute_joint_field_metrics,
    joint_loss_components,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _config() -> JointTrainConfig:
    return JointTrainConfig(
        experiment="source_joint_causal",
        data_path=PROJECT_ROOT / "external" / "ResFNO" / "data" / "Case1.mat",
        split_manifest=PROJECT_ROOT
        / "splits"
        / "legacy_case1_corrected_matched_v1.json",
        output_root=PROJECT_ROOT / "outputs" / "runs",
        project_root=PROJECT_ROOT,
        epochs=1,
        minimum_epochs=1,
    ).validated()


def test_joint_config_rejects_zero_total_loss_weight() -> None:
    config = _config()
    try:
        JointTrainConfig(
            **{
                **config.__dict__,
                "temperature_weight": 0.0,
                "alpha_weight": 0.0,
                "gradient_weight": 0.0,
            }
        ).validated()
    except ValueError as error:
        assert "At least one loss weight" in str(error)
    else:
        raise AssertionError("Zero total loss weight was accepted.")


def test_joint_loss_is_complete_case_and_differentiable() -> None:
    prediction_temperature = torch.ones(2, 5, 4, requires_grad=True)
    prediction_alpha = torch.full((2, 5, 4), 0.2, requires_grad=True)
    outputs = {
        "temperature": prediction_temperature,
        "alpha": prediction_alpha,
    }
    target_temperature = torch.full((2, 5, 4), 1.2)
    target_alpha = torch.full((2, 5, 4), 0.3)
    mask = torch.ones(2, 5, 4)
    components = joint_loss_components(
        outputs, target_temperature, target_alpha, mask
    )
    total = sum(components.values())
    total.backward()
    assert all(torch.isfinite(value) for value in components.values())
    assert prediction_temperature.grad is not None
    assert prediction_alpha.grad is not None


def test_joint_field_metrics_are_zero_for_exact_physical_predictions() -> None:
    case_ids = np.array([75, 76])
    temperature = np.linspace(293.0, 450.0, 2 * 7 * 51).reshape(2, 7, 51)
    alpha = np.zeros((2, 7, 51))
    alpha[:, :, 21:] = np.linspace(0.05, 0.9, 7)[None, :, None]
    summary, frame = compute_joint_field_metrics(
        case_ids, temperature, temperature, alpha, alpha
    )
    assert summary["field_relative_l2_mean"] == 0.0
    assert summary["alpha_relative_l2_mean"] == 0.0
    assert summary["spatial_gradient_relative_l2_mean"] == 0.0
    assert summary["alpha_monotonic_violation_count"] == 0
    assert np.array_equal(frame["case_id"].to_numpy(), case_ids)
