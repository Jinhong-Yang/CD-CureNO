from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from cdcureno.data.target_2d import (
    TARGET_INPUT_CHANNELS,
    PreparedTarget2DTraining,
)
from cdcureno.models.target_operators import AxisFactorized2DOperator
from cdcureno.training.target_2d import (
    PARAMETER_GROUP_RATIOS,
    TargetPilotTrainConfig,
    _resolve_device,
    build_parameter_groups,
    evaluate_restriction_validation,
    homogeneous_restriction_loss,
    target_loss_components,
    train_target_pilot,
)


def _small_model() -> AxisFactorized2DOperator:
    torch.manual_seed(9)
    model = AxisFactorized2DOperator(
        source_channel_names=TARGET_INPUT_CHANNELS[:14],
        new_channel_names=TARGET_INPUT_CHANNELS[14:],
        width=4,
        depth=1,
        modes_time=3,
        modes_z=2,
        modes_x=2,
        lateral_rank=2,
        adapter_seed=9,
    )
    model.set_transfer_stage("T2")
    return model


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device resolution regression"
)
def test_unindexed_cuda_is_resolved_to_explicit_device_zero() -> None:
    device = _resolve_device("cuda")
    assert device == torch.device("cuda", 0)
    assert str(device) == "cuda:0"


def test_parameter_groups_are_exhaustive_and_use_frozen_ratios() -> None:
    model = _small_model()
    groups, report = build_parameter_groups(
        model, base_learning_rate=2.0e-3, weight_decay=1.0e-4
    )
    assert [group["group_name"] for group in groups] == list(
        PARAMETER_GROUP_RATIOS
    )
    assert [group["lr"] for group in groups] == [
        2.0e-3,
        1.0e-3,
        2.0e-4,
    ]
    names = [
        name
        for group_name in PARAMETER_GROUP_RATIOS
        for name in report[group_name]["parameter_names"]
    ]
    assert len(names) == len(set(names))
    assert set(names) == {name for name, _ in model.named_parameters()}
    assert report["total_parameter_count"] == sum(
        parameter.numel() for parameter in model.parameters()
    )
    assert "lift.geometry.weight" in report["adapter"]["parameter_names"]
    assert "head.cure_rate.0.weight" in report[
        "lift_and_heads"
    ]["parameter_names"]
    assert "blocks.0.spatial.weight" in report[
        "shared_core"
    ]["parameter_names"]


def test_target_loss_has_explicit_x_z_gradients_and_composite_mask() -> None:
    temperature = torch.tensor(
        [[[[0.0, 1.0], [1.0, 3.0]], [[0.0, 2.0], [2.0, 5.0]]]]
    )
    alpha = 0.25 + 0.1 * temperature
    mask = torch.ones_like(temperature)
    prediction_temperature = temperature.clone()
    prediction_temperature[..., 1, 1] += 1.0
    prediction_alpha = alpha.clone()
    prediction_alpha[..., 1, 1] += 0.05
    components = target_loss_components(
        {
            "temperature": prediction_temperature,
            "alpha": prediction_alpha,
        },
        temperature,
        alpha,
        mask,
    )
    assert set(components) == {
        "temperature",
        "alpha",
        "gradient_x",
        "gradient_z",
    }
    assert all(float(value) > 0.0 for value in components.values())

    tool_only_mask = mask.clone()
    tool_only_mask[..., 1, 1] = 0.0
    ignored = target_loss_components(
        {
            "temperature": prediction_temperature,
            "alpha": prediction_alpha,
        },
        temperature,
        alpha,
        tool_only_mask,
    )
    assert float(ignored["temperature"]) == 0.0
    assert float(ignored["alpha"]) == 0.0


def test_homogeneous_restriction_batch_is_label_free_and_x_invariant() -> None:
    model = _small_model().eval()
    generator = torch.Generator().manual_seed(4)
    source = torch.rand(2, 5, 4, 14, generator=generator)
    source[..., 4] = 1.0
    loss, fields = homogeneous_restriction_loss(model, source, nx=3)
    assert float(loss.detach()) <= 1.0e-12
    assert float(fields["temperature"].detach()) <= 1.0e-12
    assert float(fields["alpha"].detach()) <= 1.0e-12


def test_selected_model_restriction_audit_enforces_frozen_thresholds() -> None:
    model = _small_model().eval()
    source = torch.rand(4, 5, 4, 14, generator=torch.Generator().manual_seed(7))
    source[..., 4] = 1.0
    passing = evaluate_restriction_validation(
        model,
        source,
        nx=40,
        case_count=4,
        device=torch.device("cpu"),
    )
    assert passing["passed"]
    assert all(passing["contract_checks"].values())

    def violate_lateral_invariance(_module, _inputs, outputs):
        # Deliberately break the output contract. Amplifying high-pass weights
        # cannot guarantee leakage from an exactly homogeneous FFT input.
        broken = dict(outputs)
        broken["temperature"] = outputs["temperature"].clone()
        broken["temperature"][..., 0] += 1.0
        return broken

    hook = model.register_forward_hook(violate_lateral_invariance)
    try:
        failing = evaluate_restriction_validation(
            model,
            source,
            nx=40,
            case_count=4,
            device=torch.device("cpu"),
        )
    finally:
        hook.remove()
    assert not failing["passed"]
    assert not failing["contract_checks"]["temperature_maximum_lateral_range"]
    assert failing["contract_checks"]["temperature_is_finite"]
    assert failing["thresholds"] == passing["thresholds"]


class _SyntheticTargetDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
):
    def __init__(self, case_ids: tuple[int, ...], *, seed: int) -> None:
        self.case_ids = case_ids
        self._accessed: set[int] = set()
        generator = torch.Generator().manual_seed(seed)
        count, nt, nz, nx = len(case_ids), 5, 4, 3
        inputs = torch.rand(
            count, nt, nz, nx, len(TARGET_INPUT_CHANNELS), generator=generator
        )
        inputs[..., 2] = torch.linspace(0.0, 1.0, nt)[None, :, None, None]
        inputs[..., 3] = torch.linspace(0.0, 1.0, nz)[None, None, :, None]
        inputs[..., 4] = 1.0
        inputs[..., 6] = 0.05
        inputs[..., 14:] = 0.0
        z = torch.linspace(0.0, 1.0, nz)[None, None, :, None]
        x = torch.linspace(-1.0, 1.0, nx)[None, None, None, :]
        t = torch.linspace(0.0, 1.0, nt)[None, :, None, None]
        temperature = 0.2 + 0.3 * t + 0.1 * z + 0.05 * x
        temperature = temperature.expand(count, -1, -1, -1).clone()
        alpha = (
            0.05 + 0.3 * t + 0.02 * z + 0.01 * x
        ).expand(count, -1, -1, -1).clone()
        self.inputs = inputs
        self.temperature = temperature
        self.alpha = alpha

    @property
    def accessed_case_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._accessed))

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        case_id = self.case_ids[index]
        self._accessed.add(case_id)
        return (
            self.inputs[index],
            self.temperature[index],
            self.alpha[index],
            torch.tensor(case_id, dtype=torch.long),
        )


def _write_small_model_config(path: Path) -> None:
    path.write_text(
        """
schema_version: 1
family: axis_factorized_2d
temporal_family: spectral_noncausal
causal: false
axis_order: [batch, time, z, x, channel]
output_order: [batch, time, z, x, field]
source_channel_names:
  - air_temperature_normalized
  - causal_physics_temperature_baseline_normalized
  - time_normalized
  - through_thickness_position_normalized
  - composite_mask
  - signed_distance_to_interface_normalized
  - initial_degree_of_cure
  - tool_thickness_normalized
  - composite_thickness_normalized
  - lower_htc_normalized
  - upper_htc_normalized
  - composite_conductivity_scale_normalized
  - heat_of_reaction_scale_normalized
  - low_fidelity_degree_of_cure
new_channel_names:
  - heterogeneity_gated_in_plane_position_normalized
  - heterogeneity_gated_signed_distance_to_left_boundary_normalized
  - heterogeneity_gated_signed_distance_to_right_boundary_normalized
  - top_htc_anomaly_normalized
  - edge_htc_boundary_map_normalized
  - lateral_conductivity_scale_delta_normalized
width: 4
depth: 1
modes_time: 3
modes_z: 2
modes_x: 2
lateral_rank: 2
""".lstrip(),
        encoding="utf-8",
    )


def _synthetic_prepared() -> PreparedTarget2DTraining:
    train_ids = tuple(range(8))
    validation_ids = (256, 257)
    train = _SyntheticTargetDataset(train_ids, seed=21)
    validation = _SyntheticTargetDataset(validation_ids, seed=22)
    return PreparedTarget2DTraining(
        train_dataset=train,  # type: ignore[arg-type]
        validation_dataset=validation,  # type: ignore[arg-type]
        label_budget=8,
        train_case_ids=train_ids,
        validation_case_ids=validation_ids,
        normalization={
            "field_temperature": {"minimum": 293.0, "maximum": 523.0}
        },
        checksums={"synthetic": "fixed"},
    )


def _synthetic_config(tmp_path: Path, run_id: str) -> TargetPilotTrainConfig:
    model_config = tmp_path / "target.yaml"
    if not model_config.exists():
        _write_small_model_config(model_config)
    files = {}
    for name in (
        "target_split.json",
        "source.pt",
        "inflated.pt",
        "source_data.npz",
        "source_split.json",
    ):
        path = tmp_path / name
        if not path.exists():
            path.write_bytes(name.encode("utf-8"))
        files[name] = path
    parameter_count = sum(
        parameter.numel() for parameter in _small_model().parameters()
    )
    return TargetPilotTrainConfig(
        project_root=Path(__file__).resolve().parents[2],
        target_split_manifest=files["target_split.json"],
        source_checkpoint=files["source.pt"],
        inflated_checkpoint=files["inflated.pt"],
        target_model_config=model_config,
        source_data_path=files["source_data.npz"],
        source_split_manifest=files["source_split.json"],
        output_root=tmp_path / "runs",
        resource_profile_path=tmp_path / "resource.json",
        run_id=run_id,
        method="scratch_ffno",
        label_budget=8,
        seed=0,
        width=4,
        depth=1,
        modes_time=3,
        modes_z=2,
        modes_x=2,
        lateral_rank=2,
        expected_parameter_count=parameter_count,
        epochs=2,
        minimum_epochs=2,
        early_stopping_patience=2,
        effective_batch_size=2,
        preflight_candidate_batch_sizes=(1,),
        micro_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=2.0e-3,
        restriction_weight=0.1,
        device="cpu",
        num_threads=1,
        verify_array_checksums=False,
        require_resource_profile=False,
    ).validated(require_resolved_resources=True)


def test_synthetic_pause_resume_restores_full_training_state(
    tmp_path: Path,
) -> None:
    virtual_generator = torch.Generator().manual_seed(18)
    virtual_inputs = torch.rand(
        8, 5, 4, 14, generator=virtual_generator
    )
    virtual_inputs[..., 4] = 1.0
    coordinates = (
        np.linspace(0.0, 0.003, 4),
        np.linspace(0.0, 0.002, 3),
    )

    resumed_config = _synthetic_config(tmp_path, "synthetic-resume")
    paused = train_target_pilot(
        resumed_config,
        session_epoch_limit=1,
        prepared_override=_synthetic_prepared(),
        virtual_inputs_override=virtual_inputs,
        coordinates_override=coordinates,
    )
    assert paused["status"] == "paused"
    last_path = (
        resumed_config.output_root
        / resumed_config.run_id
        / "checkpoints"
        / "last.pt"
    )
    paused_checkpoint = torch.load(
        last_path, map_location="cpu", weights_only=False
    )
    for key in (
        "model",
        "optimizer",
        "scheduler",
        "train_generator_state",
        "virtual_generator_state",
        "train_sampler_state",
        "virtual_sampler_state",
        "torch_rng_state",
        "python_rng_state",
        "numpy_rng_state",
    ):
        assert key in paused_checkpoint

    completed = train_target_pilot(
        replace(resumed_config, resume=True),
        prepared_override=_synthetic_prepared(),
        virtual_inputs_override=virtual_inputs,
        coordinates_override=coordinates,
    )
    assert completed["status"] == "completed"
    assert completed["selection"]["target_test_or_ood_labels_used"] is False
    assert completed["target_label_access_audit"]["id_test"] == []
    assert completed["target_label_access_audit"]["ood"] == []
    assert completed["restriction_validation"]["passed"]
    assert (
        completed["restriction_validation"]["provenance"][
            "target_labels_used"
        ]
        is False
    )

    continuous_config = _synthetic_config(tmp_path, "synthetic-continuous")
    continuous = train_target_pilot(
        continuous_config,
        prepared_override=_synthetic_prepared(),
        virtual_inputs_override=virtual_inputs,
        coordinates_override=coordinates,
    )
    assert continuous["status"] == "completed"
    resumed_last = torch.load(
        last_path, map_location="cpu", weights_only=False
    )
    continuous_last = torch.load(
        continuous_config.output_root
        / continuous_config.run_id
        / "checkpoints"
        / "last.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert resumed_last["epoch"] == continuous_last["epoch"] == 2
    for name, tensor in resumed_last["model"].items():
        assert torch.equal(tensor, continuous_last["model"][name]), name

    metrics_path = (
        resumed_config.output_root / resumed_config.run_id / "metrics.json"
    )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["training"]["gradient_accumulation_steps"] == 2
    assert set(metrics["auditable_artifacts"]) == {
        "history",
        "validation_metrics_per_case",
        "validation_predictions",
        "restriction_validation",
    }
    predictions_path = (
        resumed_config.output_root
        / resumed_config.run_id
        / "predictions"
        / "validation_best.npz"
    )
    with np.load(predictions_path) as predictions:
        assert predictions["composite_mask"].shape == (4, 3)
        assert predictions["x_m"].shape == (3,)
        assert predictions["z_m"].shape == (4,)
    assert (
        metrics["validation_metrics"][
            "temperature_relative_l2_K_composite_mean"
        ]
        >= 0.0
    )
