from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from cdcureno.data.source_1d import INPUT_CHANNELS as SOURCE_INPUT_CHANNELS
from cdcureno.data.target_2d import (
    EDGE_HTC_REFERENCE_MAX_W_M2_K,
    TARGET_ADAPTER_CHANNELS,
    TARGET_INPUT_CHANNELS,
    build_label_free_coarse_1d_baseline,
    lift_homogeneous_source_input,
    prepare_target_2d_training,
    stack_target_fields,
)


def _canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _source_normalization() -> dict[str, Any]:
    return {
        "scope": "training_cases_only",
        "time_stride": 2,
        "time_indices": [0, 2, 4],
        "air_temperature": {"minimum": 293.0, "maximum": 480.0},
        "field_temperature": {"minimum": 293.0, "maximum": 525.0},
        "parameter_bounds": [
            [0.01, 0.04],
            [0.01, 0.04],
            [40.0, 100.0],
            [80.0, 160.0],
            [0.8, 1.2],
            [0.9, 1.1],
        ],
        "input_channels": list(SOURCE_INPUT_CHANNELS),
        "held_out_labels_used_for_normalization": False,
        "causal_baseline": {
            "method": "coarse_thermochemical_conservative_solver",
            "uses_thermochemical_labels": False,
        },
    }


def _definition(case_id: int, split: str) -> dict[str, Any]:
    scale = 0.9 + 0.02 * case_id
    if case_id == 0:
        difficulty = "F0"
        top_h = [120.0] * 5
        left_h = 0.0
        right_h = 0.0
    else:
        difficulty = "F1_smooth"
        top_h = [100.0, 110.0, 120.0, 130.0, 140.0]
        left_h = 25.0
        right_h = 50.0
    base_x = 4.6724
    base_z = 0.6369803
    return {
        "case_id": case_id,
        "case_key": f"fixture-{case_id:04d}",
        "split": split,
        "difficulty_family": difficulty,
        "air_temperature_K": [293.0, 330.0 + case_id, 293.0],
        "top_h_W_m2_K": top_h,
        "bottom_h_W_m2_K": 60.0 + case_id,
        "left_h_W_m2_K": left_h,
        "right_h_W_m2_K": right_h,
        "base_composite_conductivity_x_W_m_K": base_x,
        "base_composite_conductivity_z_W_m_K": base_z,
        "composite_conductivity_x_W_m_K": base_x * scale,
        "composite_conductivity_z_W_m_K": base_z * scale,
        "composite_conductivity_scale": scale,
        "reaction_enthalpy_scale": 0.95 + 0.01 * case_id,
    }


def _fake_baseline(
    definition: dict[str, Any],
    times_s: np.ndarray,
    z_m: np.ndarray,
    geometry: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    del geometry
    air = np.asarray(definition["air_temperature_K"], dtype=np.float64)
    temperature = np.broadcast_to(air[:, None], (times_s.size, z_m.size)).copy()
    alpha = np.zeros_like(temperature)
    alpha[:, z_m > 0.02] = 0.05
    return temperature, alpha


def _build_fixture(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "repo"
    split_root = root / "splits"
    artifact_root = root / "data" / "processed" / "fixture"
    artifact_root.mkdir(parents=True)
    case_count, nt, nz, nx = 8, 3, 4, 5
    splits = {
        "train": [0, 1, 2],
        "validation": [3],
        "id_test": [4],
        "cycle_ood": [5],
        "htc_ood": [6],
        "pattern_ood": [],
        "combined_ood": [7],
    }
    case_to_split = {
        case_id: name for name, ids in splits.items() for case_id in ids
    }
    definitions = [
        _definition(case_id, case_to_split[case_id])
        for case_id in range(case_count)
    ]
    case_definition_hashes = {
        definition["case_key"]: hashlib.sha256(
            _canonical_bytes(definition)
        ).hexdigest()
        for definition in definitions
    }
    plan = {
        "schema_version": 2,
        "phase": "P4",
        "dataset_id": "p4_fixture",
        "plan_role": "pre_label_case_plan",
        "case_count": case_count,
        "nested_training_budgets": [1, 2, 3],
        "splits": splits,
        "time_s": [0.0, 60.0, 120.0],
        "resolved_config": {
            "geometry": {
                "width_m": 0.20,
                "tool_thickness_m": 0.02,
                "composite_thickness_m": 0.03,
            }
        },
        "cases": [{"definition": definition} for definition in definitions],
    }
    plan["plan_sha256"] = hashlib.sha256(
        _canonical_bytes(plan)
    ).hexdigest()
    plan_path = split_root / "p4_fixture_plan.json"
    _write_json(plan_path, plan)

    time = np.array([0.0, 60.0, 120.0], dtype=np.float64)
    z = np.array([0.00625, 0.01875, 0.03125, 0.04375], dtype=np.float64)
    x = np.array([0.02, 0.06, 0.10, 0.14, 0.18], dtype=np.float64)
    mask = np.broadcast_to((z > 0.02)[:, None], (nz, nx)).copy()
    air = np.asarray(
        [definition["air_temperature_K"] for definition in definitions],
        dtype=np.float32,
    )
    top_h = np.asarray(
        [definition["top_h_W_m2_K"] for definition in definitions],
        dtype=np.float32,
    )
    temperature = np.empty((case_count, nt, nz, nx), dtype=np.float32)
    alpha = np.zeros_like(temperature)
    for case_id in range(case_count):
        temperature[case_id] = 300.0 + case_id
        alpha[case_id, 1:, mask] = 0.1 + case_id * 0.01
    # If target-label fitting occurred, this held-out value would alter it.
    temperature[4:] = 10_000.0
    arrays = {
        "air_temperature_K": air,
        "alpha": alpha,
        "composite_mask": mask,
        "temperature_K": temperature,
        "time_s": time,
        "top_h_W_m2_K": top_h,
        "x_m": x,
        "z_m": z,
    }
    array_hashes: dict[str, str] = {}
    for name, value in arrays.items():
        path = artifact_root / f"{name}.npy"
        np.save(path, value, allow_pickle=False)
        array_hashes[name] = _sha256(path)

    source_manifest = {
        "schema_version": 2,
        "dataset_id": "p4_fixture",
        "case_count": case_count,
        "array_artifact_root": "data/processed/fixture",
        "array_shape_case_time_z_x": [case_count, nt, nz, nx],
        "array_sha256": array_hashes,
        "case_definition_hashes": case_definition_hashes,
        "metadata_sha256": "a" * 64,
        "plan_path": "splits/p4_fixture_plan.json",
        "plan_sha256": plan["plan_sha256"],
        "nested_training_budgets": [1, 2, 3],
        "splits": splits,
        "generation_status": {
            "passed_case_ids": list(range(case_count)),
            "failed_case_ids": [],
            "silent_failure_case_ids": [],
        },
    }
    source_path = split_root / "p4_fixture.json"
    _write_json(source_path, source_manifest)
    id_manifest = {
        "schema_version": 2,
        "phase": "P4",
        "role": "target_id",
        "dataset_id": "p4_fixture",
        "source_manifest": "splits/p4_fixture.json",
        "source_manifest_sha256": _sha256(source_path),
        "dataset_array_sha256": array_hashes,
        "dataset_metadata_sha256": "a" * 64,
        "dataset_plan_sha256": plan["plan_sha256"],
        "case_definition_hashes": {
            definitions[case_id]["case_key"]: case_definition_hashes[
                definitions[case_id]["case_key"]
            ]
            for case_id in [0, 1, 2, 3, 4]
        },
        "splits": {
            "train_pool": [0, 1, 2],
            "validation": [3],
            "test": [4],
        },
        "nested_training_budgets": {
            "1": [0],
            "2": [0, 1],
            "3": [0, 1, 2],
        },
    }
    id_path = split_root / "2d_id_v1.json"
    _write_json(id_path, id_manifest)
    checkpoint_path = root / "outputs" / "source.pt"
    checkpoint_path.parent.mkdir(parents=True)
    torch.save(
        {
            "channel_names": SOURCE_INPUT_CHANNELS,
            "normalization": _source_normalization(),
            "model": {},
        },
        checkpoint_path,
    )
    return {
        "root": root,
        "artifact": artifact_root,
        "source": source_path,
        "split": id_path,
        "checkpoint": checkpoint_path,
    }


def test_prepares_exact_nested_budget_and_frozen_source_normalization(
    tmp_path: Path,
) -> None:
    paths = _build_fixture(tmp_path)
    prepared = prepare_target_2d_training(
        paths["split"],
        paths["checkpoint"],
        label_budget=2,
        project_root=paths["root"],
        baseline_builder=_fake_baseline,
    )
    assert prepared.train_case_ids == (0, 1)
    assert prepared.validation_case_ids == (3,)
    assert len(prepared.dataset("train")) == 2
    assert len(prepared.dataset("validation")) == 1
    assert prepared.channel_names == TARGET_INPUT_CHANNELS
    assert prepared.channel_names[:14] == SOURCE_INPUT_CHANNELS
    assert prepared.channel_names[14:] == TARGET_ADAPTER_CHANNELS
    assert prepared.normalization["scope"] == (
        "source_training_only_frozen_for_transfer_comparability"
    )
    assert prepared.normalization["target_labels_used_to_fit_normalization"] is False
    assert (
        prepared.normalization[
            "validation_or_test_labels_used_to_fit_normalization"
        ]
        is False
    )

    inputs, temperature, alpha, case_id = prepared.dataset("train")[1]
    assert inputs.shape == (3, 4, 5, 20)
    assert temperature.shape == alpha.shape == (3, 4, 5)
    assert case_id.item() == 1
    assert temperature[0, 0, 0].item() == pytest.approx(
        (301.0 - 293.0) / (525.0 - 293.0)
    )
    assert prepared.accessed_case_ids == {
        "train": (1,),
        "validation": (),
    }
    assert prepared.checksums["array_files_verified"] is True
    assert prepared.checksums["source_checkpoint_sha256"] == _sha256(
        paths["checkpoint"]
    )


def test_all_six_target_channels_are_exact_zero_for_f0_and_maps_are_bound(
    tmp_path: Path,
) -> None:
    paths = _build_fixture(tmp_path)
    prepared = prepare_target_2d_training(
        paths["split"],
        paths["checkpoint"],
        label_budget=2,
        project_root=paths["root"],
        baseline_builder=_fake_baseline,
    )
    homogeneous = prepared.dataset("train")[0][0].numpy()
    assert np.count_nonzero(homogeneous[..., 14:]) == 0
    assert np.all(homogeneous[..., 10] == pytest.approx(0.5))

    heterogeneous = prepared.dataset("train")[1][0].numpy()
    top_channel = heterogeneous[0, 0, :, 17]
    assert top_channel == pytest.approx(
        np.array([-20.0, -10.0, 0.0, 10.0, 20.0]) / 80.0
    )
    edge = heterogeneous[0, :, :, 18]
    assert np.all(
        edge[:, 0] == pytest.approx(25.0 / EDGE_HTC_REFERENCE_MAX_W_M2_K)
    )
    assert np.all(
        edge[:, -1] == pytest.approx(50.0 / EDGE_HTC_REFERENCE_MAX_W_M2_K)
    )
    assert np.count_nonzero(edge[:, 1:-1]) == 0
    # The benchmark applies one common conductivity multiplier to kx and kz.
    assert np.count_nonzero(heterogeneous[..., 19]) == 0


def test_training_path_rejects_test_ood_and_noncanonical_budgets(
    tmp_path: Path,
) -> None:
    paths = _build_fixture(tmp_path)
    prepared = prepare_target_2d_training(
        paths["split"],
        paths["checkpoint"],
        label_budget=1,
        project_root=paths["root"],
        baseline_builder=_fake_baseline,
    )
    with pytest.raises(PermissionError, match="evaluation-only"):
        prepared.dataset("test")
    with pytest.raises(PermissionError, match="evaluation-only"):
        prepared.dataset("combined_ood")
    with pytest.raises(ValueError, match="not one of"):
        prepare_target_2d_training(
            paths["split"],
            paths["checkpoint"],
            label_budget=4,
            project_root=paths["root"],
            baseline_builder=_fake_baseline,
        )

    payload = json.loads(paths["split"].read_text(encoding="utf-8"))
    payload["role"] = "combined_ood"
    ood_path = paths["root"] / "splits" / "2d_combined_ood_v1.json"
    _write_json(ood_path, payload)
    with pytest.raises(PermissionError, match="role=target_id"):
        prepare_target_2d_training(
            ood_path,
            paths["checkpoint"],
            label_budget=1,
            project_root=paths["root"],
            baseline_builder=_fake_baseline,
        )


def test_rejects_resigned_non_nested_budget_and_checkpoint_channel_drift(
    tmp_path: Path,
) -> None:
    paths = _build_fixture(tmp_path)
    payload = json.loads(paths["split"].read_text(encoding="utf-8"))
    payload["nested_training_budgets"]["2"] = [1, 0]
    bad_split = paths["root"] / "splits" / "bad_budget.json"
    _write_json(bad_split, payload)
    with pytest.raises(ValueError, match="exact train_pool prefix"):
        prepare_target_2d_training(
            bad_split,
            paths["checkpoint"],
            label_budget=1,
            project_root=paths["root"],
            baseline_builder=_fake_baseline,
        )

    bad_checkpoint = paths["root"] / "outputs" / "bad_source.pt"
    normalization = _source_normalization()
    torch.save(
        {
            "channel_names": tuple(reversed(SOURCE_INPUT_CHANNELS)),
            "normalization": normalization,
        },
        bad_checkpoint,
    )
    with pytest.raises(ValueError, match="channel_names"):
        prepare_target_2d_training(
            paths["split"],
            bad_checkpoint,
            label_budget=1,
            project_root=paths["root"],
            baseline_builder=_fake_baseline,
        )


def test_label_free_baseline_has_no_fine_label_file_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = _definition(0, "train")
    loaded_paths: list[str] = []
    original_load = np.load

    def recording_load(*args: Any, **kwargs: Any) -> Any:
        loaded_paths.append(str(args[0]))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(np, "load", recording_load)
    temperature, alpha = build_label_free_coarse_1d_baseline(
        definition,
        np.array([0.0, 60.0, 120.0], dtype=np.float64),
        np.array([0.005, 0.015, 0.025, 0.035, 0.045], dtype=np.float64),
        {
            "width_m": 0.20,
            "tool_thickness_m": 0.02,
            "composite_thickness_m": 0.03,
        },
    )
    assert temperature.shape == alpha.shape == (3, 5)
    assert loaded_paths == []
    assert np.count_nonzero(alpha[:, :2]) == 0


@pytest.mark.parametrize("kind", ["numpy", "torch"])
def test_homogeneous_lift_preserves_prefix_and_appends_exact_zeros(
    kind: str,
) -> None:
    source_numpy = np.arange(2 * 3 * 4 * 14, dtype=np.float32).reshape(
        2, 3, 4, 14
    )
    source = torch.from_numpy(source_numpy) if kind == "torch" else source_numpy
    lifted = lift_homogeneous_source_input(source, nx=5)
    lifted_numpy = lifted.numpy() if isinstance(lifted, torch.Tensor) else lifted
    assert lifted_numpy.shape == (2, 3, 4, 5, 20)
    assert np.array_equal(
        lifted_numpy[..., :14],
        np.broadcast_to(source_numpy[..., None, :], (2, 3, 4, 5, 14)),
    )
    assert np.count_nonzero(lifted_numpy[..., 14:]) == 0

    temperature = torch.zeros(2, 3, 4, 5)
    alpha = torch.ones_like(temperature)
    stacked = stack_target_fields(temperature, alpha)
    assert stacked.shape == (2, 3, 4, 5, 2)
    assert torch.equal(stacked[..., 0], temperature)
    assert torch.equal(stacked[..., 1], alpha)
