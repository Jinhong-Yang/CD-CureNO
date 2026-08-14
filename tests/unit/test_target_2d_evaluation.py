from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import cdcureno.data.target_2d_evaluation as evaluation
from cdcureno.data.source_1d import INPUT_CHANNELS as SOURCE_INPUT_CHANNELS
from cdcureno.data.target_2d_evaluation import (
    FrozenEvaluationManifestSpec,
    materialize_target_2d_label_shard,
    prepare_target_2d_evaluation,
    target_2d_evaluation_dry_run,
    validate_target_2d_evaluation_metadata,
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


def _semantic_array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(_canonical_bytes(list(contiguous.shape)))
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_canonical_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(payload))


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _normalization() -> dict[str, Any]:
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
    held_out = split not in {"train", "validation", "id_test"}
    top_h = 180.0 if held_out else 120.0
    bottom_h = 110.0 if held_out else 70.0
    scale = 0.9 + 0.02 * case_id
    base_x = 4.6724
    base_z = 0.6369803
    return {
        "case_id": case_id,
        "case_key": f"fixture-{case_id:04d}",
        "split": split,
        "difficulty_family": "F2_left" if held_out else "F0",
        "cycle_family": "smart_cure" if held_out else "single_hold",
        "air_temperature_K": [293.0, 350.0 + case_id, 293.0],
        "top_h_W_m2_K": [top_h] * 5,
        "bottom_h_W_m2_K": bottom_h,
        "left_h_W_m2_K": 80.0 if held_out else 0.0,
        "right_h_W_m2_K": 0.0,
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
    temperature = np.broadcast_to(
        air[:, None], (times_s.size, z_m.size)
    ).copy()
    alpha = np.zeros_like(temperature)
    alpha[:, z_m > 0.02] = 0.05
    return temperature, alpha


def _build_fixture(
    tmp_path: Path,
    *,
    role: str = "cycle_ood",
) -> dict[str, Any]:
    root = tmp_path / "repo"
    split_root = root / "splits"
    artifact_root = root / "data" / "processed" / "fixture"
    artifact_root.mkdir(parents=True)
    case_count, nt, nz, nx = 8, 3, 4, 5
    splits = {
        "train": [0, 1],
        "validation": [2],
        "id_test": [3],
        "cycle_ood": [4],
        "htc_ood": [5],
        "pattern_ood": [6],
        "combined_ood": [7],
    }
    case_to_split = {
        case_id: split
        for split, case_ids in splits.items()
        for case_id in case_ids
    }
    definitions = [
        _definition(case_id, case_to_split[case_id])
        for case_id in range(case_count)
    ]
    definition_hashes = {
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
        "nested_training_budgets": [1, 2],
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
    plan["plan_sha256"] = hashlib.sha256(_canonical_bytes(plan)).hexdigest()
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
    # Poisoned held-out values make accidental normalization fitting obvious.
    temperature[3:] += 10_000.0
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

    metadata_path = root / "outputs" / "tables" / "fixture_cases.jsonl"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(
        "\n".join(
            json.dumps({"case_id": case_id, "status": "passed"})
            for case_id in range(case_count)
        )
        + "\n",
        encoding="utf-8",
    )
    source = {
        "schema_version": 2,
        "phase": "P4",
        "dataset_id": "p4_fixture",
        "case_count": case_count,
        "array_artifact_root": "data/processed/fixture",
        "array_shape_case_time_z_x": [case_count, nt, nz, nx],
        "array_sha256": array_hashes,
        "case_artifact_hashes": {
            definition["case_key"]: {
                "input_slice_sha256": {
                    "case_definition": definition_hashes[
                        definition["case_key"]
                    ]
                },
                "output_slice_sha256": {
                    "temperature_K": _semantic_array_sha256(
                        temperature[definition["case_id"]]
                    ),
                    "alpha": _semantic_array_sha256(
                        alpha[definition["case_id"]]
                    ),
                },
            }
            for definition in definitions
        },
        "case_definition_hashes": definition_hashes,
        "metadata_path": "outputs/tables/fixture_cases.jsonl",
        "metadata_sha256": _sha256(metadata_path),
        "plan_path": "splits/p4_fixture_plan.json",
        "plan_sha256": plan["plan_sha256"],
        "nested_training_budgets": [1, 2],
        "splits": splits,
        "generation_status": {
            "passed_case_ids": list(range(case_count)),
            "failed_case_ids": [],
            "silent_failure_case_ids": [],
        },
    }
    source_path = split_root / "p4_fixture.json"
    _write_json(source_path, source)

    if role == "target_id":
        manifest_name = "fixture_id.json"
        source_split_key = "id_test"
        evaluation_ids = (3,)
        manifest_hash_ids = (0, 1, 2, 3)
        manifest = {
            "schema_version": 2,
            "phase": "P4",
            "role": role,
            "dataset_id": "p4_fixture",
            "source_manifest": "splits/p4_fixture.json",
            "source_manifest_sha256": _sha256(source_path),
            "dataset_array_sha256": array_hashes,
            "dataset_metadata_sha256": source["metadata_sha256"],
            "dataset_plan_sha256": plan["plan_sha256"],
            "case_definition_hashes": {
                definitions[case_id]["case_key"]: definition_hashes[
                    definitions[case_id]["case_key"]
                ]
                for case_id in manifest_hash_ids
            },
            "splits": {
                "train_pool": [0, 1],
                "validation": [2],
                "test": [3],
            },
            "nested_training_budgets": {"1": [0], "2": [0, 1]},
        }
    else:
        manifest_name = f"fixture_{role}.json"
        source_split_key = role
        evaluation_ids = tuple(splits[role])
        manifest = {
            "schema_version": 2,
            "phase": "P4",
            "role": role,
            "dataset_id": "p4_fixture",
            "source_manifest": "splits/p4_fixture.json",
            "source_manifest_sha256": _sha256(source_path),
            "dataset_array_sha256": array_hashes,
            "dataset_metadata_sha256": source["metadata_sha256"],
            "dataset_plan_sha256": plan["plan_sha256"],
            "case_definition_hashes": {
                definitions[case_id]["case_key"]: definition_hashes[
                    definitions[case_id]["case_key"]
                ]
                for case_id in evaluation_ids
            },
            "case_ids": list(evaluation_ids),
        }
    manifest_path = split_root / manifest_name
    _write_json(manifest_path, manifest)
    spec = FrozenEvaluationManifestSpec(
        name=f"fixture_{role}",
        relative_path=f"splits/{manifest_name}",
        manifest_sha256=_sha256(manifest_path),
        role=role,
        case_ids=evaluation_ids,
        source_split_key=source_split_key,
        source_manifest_relative_path="splits/p4_fixture.json",
        source_manifest_sha256=_sha256(source_path),
        plan_file_sha256=_sha256(plan_path),
    ).validated()

    stage = "stage1" if role == "target_id" else "stage2"
    split_name = source_split_key
    control = root / "outputs" / "p6_v2" / "control" / "release_v1"
    order = (
        1
        if stage == "stage1"
        else (
            ("cycle_ood", "htc_ood", "pattern_ood", "combined_ood").index(
                split_name
            )
            + 1
        )
    )
    split_start = (
        control
        / f"{stage}_split_{order:02d}_{split_name}_started.json"
    )
    release_pending = control / f"{stage}_pending.json"
    pending_unsigned = {
        "schema_version": 1,
        "phase": "P6V2",
        "release_id": evaluation.P6_V2_RELEASE_ID,
        "artifact_role": "p6_v2_release_stage_pending",
        "stage": stage,
        "status": "started_write_once_hard_interruption_resume_only",
        "started_at": "2026-07-26T00:00:00+00:00",
        "prelabel_payload_sha256": "a" * 64,
        "population_sha256": "b" * 64,
        "hard_interruption_resume_allowed": True,
        "retry_or_overwrite_allowed": False,
    }
    _write_canonical_json(
        release_pending,
        {
            **pending_unsigned,
            "pending_payload_sha256": hashlib.sha256(
                _canonical_bytes(pending_unsigned)
            ).hexdigest(),
        },
    )
    split_start_unsigned = {
        "schema_version": 1,
        "phase": "P6V2",
        "release_id": evaluation.P6_V2_RELEASE_ID,
        "artifact_role": "p6_v2_release_split_start",
        "stage": stage,
        "split": split_name,
        "split_order": order,
        "status": "started_write_once",
        "started_at": "2026-07-26T00:00:01+00:00",
        "prelabel_payload_sha256": "a" * 64,
        "population_sha256": "b" * 64,
        "pending_transaction": _artifact(release_pending, root),
        "hard_interruption_resume_allowed": True,
        "retry_or_overwrite_allowed": False,
    }
    _write_canonical_json(
        split_start,
        {
            **split_start_unsigned,
            "split_start_payload_sha256": hashlib.sha256(
                _canonical_bytes(split_start_unsigned)
            ).hexdigest(),
        },
    )
    stage1_reference: dict[str, Any] | None = None
    if stage == "stage2":
        stage1_receipt = control / "stage1_receipt.json"
        _write_canonical_json(
            stage1_receipt,
            {
                "artifact_role": "p6_v2_release_stage_receipt",
                "stage": "stage1",
                "status": "passed",
                "prelabel_payload_sha256": "a" * 64,
                "population_sha256": "b" * 64,
            },
        )
        stage1_reference = _artifact(stage1_receipt, root)
    authorization_unsigned = {
        "schema_version": 1,
        "artifact_role": "p6_v2_release_label_shard_authorization",
        "stage": stage,
        "split": split_name,
        "prelabel_payload_sha256": "a" * 64,
        "population_sha256": "b" * 64,
        "implementation_sha256": "c" * 64,
        "split_start": _artifact(split_start, root),
        "stage1_receipt": stage1_reference,
    }
    authorization = {
        **authorization_unsigned,
        "authorization_payload_sha256": hashlib.sha256(
            _canonical_bytes(authorization_unsigned)
        ).hexdigest(),
    }
    access_session_unsigned = {
        "schema_version": 1,
        "artifact_role": "p6_v2_release_label_access_session",
        "stage": stage,
        "split": split_name,
        "invocation_kind": "initial_release_stage_invocation",
        "adapter_resume_interrupted": False,
        "split_start": authorization["split_start"],
        "release_resume_attempt": None,
        "prelabel_payload_sha256": "a" * 64,
        "population_sha256": "b" * 64,
        "session_authority_sha256": authorization["split_start"]["sha256"],
    }
    initial_access_session = {
        **access_session_unsigned,
        "session_payload_sha256": hashlib.sha256(
            _canonical_bytes(access_session_unsigned)
        ).hexdigest(),
    }

    checkpoint_path = root / "outputs" / "source.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "channel_names": SOURCE_INPUT_CHANNELS,
            "normalization": _normalization(),
            "model": {},
        },
        checkpoint_path,
    )
    return {
        "root": root,
        "artifact": artifact_root,
        "source": source_path,
        "plan": plan_path,
        "manifest": manifest_path,
        "spec": spec,
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "stage": stage,
        "split_name": split_name,
        "authorization": authorization,
        "initial_access_session": initial_access_session,
        "stage1_reference": stage1_reference,
        "release_pending": release_pending,
        "split_start": split_start,
        "prepare_count": 0,
        "shard_root": root / "outputs" / "synthetic_label_shards",
    }


def _resumed_access_session(paths: dict[str, Any]) -> dict[str, Any]:
    control = paths["split_start"].parent
    attempts = sorted(control.glob(f"{paths['stage']}_resume_attempt_*.json"))
    attempt_number = len(attempts) + 1
    resume_unsigned = {
        "schema_version": 1,
        "phase": "P6V2",
        "release_id": evaluation.P6_V2_RELEASE_ID,
        "artifact_role": "p6_v2_release_hard_interruption_resume_attempt",
        "stage": paths["stage"],
        "attempt_number": attempt_number,
        "status": "hard_interruption_resume_requested",
        "requested_at": f"2026-07-26T00:01:{attempt_number:02d}+00:00",
        "prelabel_payload_sha256": "a" * 64,
        "prelabel_authority_commit_sha": "d" * 64,
        "adapter_implementation_sha256": "c" * 64,
        "population_sha256": "b" * 64,
        "pending_transaction": _artifact(paths["release_pending"], paths["root"]),
        "stage1_receipt": paths["stage1_reference"],
        "completed_prefix_splits": [],
        "completed_prefix_split_receipts": [],
        "partial_split": paths["split_name"],
        "resume_interrupted_adapter_call_required": True,
        "hard_interruption_resume_allowed": True,
        "hard_interruption_resume_policy": dict(
            evaluation.P6_V2_HARD_INTERRUPTION_RESUME_POLICY
        ),
        "outcome_interpreted": False,
        "retry_or_overwrite_allowed": False,
    }
    resume = {
        **resume_unsigned,
        "resume_attempt_payload_sha256": hashlib.sha256(
            _canonical_bytes(resume_unsigned)
        ).hexdigest(),
    }
    resume_path = (
        control
        / f"{paths['stage']}_resume_attempt_{attempt_number:04d}.json"
    )
    _write_canonical_json(resume_path, resume)
    resume_reference = _artifact(resume_path, paths["root"])
    session_unsigned = {
        "schema_version": 1,
        "artifact_role": "p6_v2_release_label_access_session",
        "stage": paths["stage"],
        "split": paths["split_name"],
        "invocation_kind": "hard_interruption_resume_stage_invocation",
        "adapter_resume_interrupted": True,
        "split_start": paths["authorization"]["split_start"],
        "release_resume_attempt": resume_reference,
        "prelabel_payload_sha256": "a" * 64,
        "population_sha256": "b" * 64,
        "session_authority_sha256": resume_reference["sha256"],
    }
    return {
        **session_unsigned,
        "session_payload_sha256": hashlib.sha256(
            _canonical_bytes(session_unsigned)
        ).hexdigest(),
    }


def _prepare(paths: dict[str, Any]):
    contract = validate_target_2d_evaluation_metadata(
        paths["manifest"],
        project_root=paths["root"],
        manifest_spec=paths["spec"],
    )
    access_session = (
        paths["initial_access_session"]
        if paths["prepare_count"] == 0
        else _resumed_access_session(paths)
    )
    paths["prepare_count"] += 1
    shard = materialize_target_2d_label_shard(
        contract,
        stage=paths["stage"],
        split=paths["split_name"],
        authorization=paths["authorization"],
        access_session=access_session,
        shard_root=paths["shard_root"],
    )
    return prepare_target_2d_evaluation(
        paths["manifest"],
        paths["checkpoint"],
        expected_source_checkpoint_sha256=paths["checkpoint_sha256"],
        project_root=paths["root"],
        manifest_spec=paths["spec"],
        baseline_builder=_fake_baseline,
        label_shard=shard,
    )


def test_dry_run_validates_metadata_without_indexing_label_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _build_fixture(tmp_path)
    original_load = np.load
    label_index_attempts: list[Any] = []
    label_load_calls: list[Path] = []

    class HeaderOnly:
        def __init__(self, array: np.ndarray) -> None:
            self.shape = array.shape
            self.dtype = array.dtype

        def __getitem__(self, index: Any) -> Any:
            label_index_attempts.append(index)
            raise AssertionError("Dry-run indexed a held-out label.")

    def guarded_load(path: Any, *args: Any, **kwargs: Any) -> Any:
        array = original_load(path, *args, **kwargs)
        if Path(path).stem in {"temperature_K", "alpha"}:
            label_load_calls.append(Path(path))
            return HeaderOnly(array)
        return array

    monkeypatch.setattr(evaluation.np, "load", guarded_load)
    result = target_2d_evaluation_dry_run(
        paths["manifest"],
        paths["checkpoint"],
        expected_source_checkpoint_sha256=paths["checkpoint_sha256"],
        project_root=paths["root"],
        manifest_spec=paths["spec"],
        baseline_builder=_fake_baseline,
    )
    assert result["passed"] is True
    assert result["case_ids"] == [4]
    assert result["access_audit"]["attempted_case_ids"] == []
    assert result["access_audit"]["completed_case_ids"] == []
    assert label_index_attempts == []
    assert label_load_calls == []


@pytest.mark.parametrize(
    ("role", "expected_case_id"),
    [("target_id", 3), ("cycle_ood", 4), ("combined_ood", 7)],
)
def test_prepares_exact_id_and_ood_roles_with_frozen_source_normalization(
    tmp_path: Path,
    role: str,
    expected_case_id: int,
) -> None:
    paths = _build_fixture(tmp_path, role=role)
    prepared = _prepare(paths)
    assert prepared.case_ids == (expected_case_id,)
    assert prepared.required_num_workers == 0
    assert prepared.normalization["field_temperature"] == {
        "minimum": 293.0,
        "maximum": 525.0,
    }
    assert prepared.normalization[
        "target_labels_used_to_fit_normalization"
    ] is False
    assert prepared.access_audit["attempted_case_ids"] == []


def test_actual_access_is_audited_and_ood_conditioning_is_not_clipped(
    tmp_path: Path,
) -> None:
    paths = _build_fixture(tmp_path)
    prepared = _prepare(paths)
    inputs, temperature, alpha, case_id = prepared.dataset[0]
    assert case_id.item() == 4
    assert inputs.shape == (3, 4, 5, 20)
    assert temperature.shape == alpha.shape == (3, 4, 5)
    assert temperature[0, 0, 0].item() == pytest.approx(10_304.0)
    # lower/top HTC are deliberately beyond the frozen source ranges.
    assert torch.all(inputs[..., 9] > 1.0)
    assert torch.all(inputs[..., 10] > 1.0)
    assert prepared.access_audit["attempted_case_ids"] == [4]
    assert prepared.access_audit["completed_case_ids"] == [4]


def test_input_only_and_physics_snapshot_never_open_labels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _build_fixture(tmp_path)
    prepared = _prepare(paths)

    def forbidden_labels() -> dict[str, Any]:
        raise AssertionError("label arrays must remain unopened")

    monkeypatch.setattr(
        prepared.dataset,
        "_open_label_arrays",
        forbidden_labels,
    )
    inputs, case_id = prepared.dataset.input_only_item(0)
    physics = prepared.dataset.label_free_physics_inputs()
    assert case_id.item() == 4
    assert inputs.shape == (3, 4, 5, 20)
    assert tuple(physics.definitions) == (4,)
    assert physics.time_s.shape == (3,)
    assert physics.z_m.shape == (4,)
    assert physics.x_m.shape == (5,)
    assert physics.composite_mask.shape == (4, 5)
    assert physics.time_s.flags.writeable is False
    assert physics.composite_mask.flags.writeable is False
    assert prepared.access_audit["attempted_case_ids"] == []
    assert prepared.access_audit["completed_case_ids"] == []


def test_attempt_is_recorded_before_a_failing_label_slice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _build_fixture(tmp_path)
    prepared = _prepare(paths)

    class FailingLabels:
        def __getitem__(self, index: Any) -> Any:
            raise OSError(f"synthetic label read failed at {index}")

    prepared.dataset._open_label_arrays()
    monkeypatch.setattr(
        prepared.dataset,
        "_open_label_arrays",
        lambda: {"temperature_K": FailingLabels(), "alpha": FailingLabels()},
    )
    with pytest.raises(OSError, match="synthetic label read failed"):
        prepared.dataset[0]
    assert prepared.access_audit["attempted_case_ids"] == [4]
    assert prepared.access_audit["completed_case_ids"] == []


def test_worker_process_access_is_rejected_before_label_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _build_fixture(tmp_path)
    prepared = _prepare(paths)
    monkeypatch.setattr(evaluation, "get_worker_info", lambda: object())
    with pytest.raises(RuntimeError, match="num_workers=0"):
        prepared.dataset[0]
    assert prepared.access_audit["attempted_case_ids"] == []
    assert prepared.access_audit["completed_case_ids"] == []


def test_stage1_reads_only_id_ranges_and_dataset_never_opens_monolith(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _build_fixture(tmp_path, role="target_id")
    prepared = _prepare(paths)
    evidence = prepared.access_audit["label_shard_evidence"]
    receipt_path = paths["root"].joinpath(
        *Path(evidence["access_receipt"]["path"]).parts
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    for name in ("temperature_K", "alpha"):
        source = receipt["source_containers"][name]
        ranges = source["authorized_case_ranges"]
        assert [item["global_case_id"] for item in ranges] == [3]
        assert source["whole_source_container_hashed"] is False
        assert source["source_container_mmap_used"] is False
        assert source["unauthorized_case_bytes_read"] is False
        row_bytes = ranges[0]["bytes"]
        data_offset = source["header"]["data_offset"]
        forbidden_ood = [
            data_offset + 4 * row_bytes,
            data_offset + 8 * row_bytes,
        ]
        observed = ranges[0]["source_byte_range"]
        assert max(observed[0], forbidden_ood[0]) >= min(
            observed[1], forbidden_ood[1]
        )

    original_load = evaluation.np.load
    opened: list[Path] = []

    def tracking_load(path: Any, *args: Any, **kwargs: Any) -> Any:
        opened.append(Path(path).resolve())
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(evaluation.np, "load", tracking_load)
    prepared.dataset[0]
    monoliths = {
        (paths["artifact"] / "temperature_K.npy").resolve(),
        (paths["artifact"] / "alpha.npy").resolve(),
    }
    assert not monoliths.intersection(opened)
    assert {
        Path(item["path"]).name
        for item in receipt["label_shards"].values()
    } == {"temperature_K.npy", "alpha.npy"}


def test_stage2_requires_matching_stage1_authority_before_source_open(
    tmp_path: Path,
) -> None:
    paths = _build_fixture(tmp_path, role="cycle_ood")
    contract = validate_target_2d_evaluation_metadata(
        paths["manifest"],
        project_root=paths["root"],
        manifest_spec=paths["spec"],
    )
    unsigned = {
        key: value
        for key, value in paths["authorization"].items()
        if key != "authorization_payload_sha256"
    }
    unsigned["stage1_receipt"] = None
    invalid = {
        **unsigned,
        "authorization_payload_sha256": hashlib.sha256(
            _canonical_bytes(unsigned)
        ).hexdigest(),
    }
    with pytest.raises(PermissionError, match="Stage-2"):
        materialize_target_2d_label_shard(
            contract,
            stage="stage2",
            split="cycle_ood",
            authorization=invalid,
            access_session=paths["initial_access_session"],
            shard_root=paths["shard_root"],
        )
    assert not list(paths["shard_root"].rglob("source_access.jsonl"))


class _SyntheticProcessDeath(BaseException):
    pass


def test_interrupted_exact_row_is_adopted_without_source_range_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _build_fixture(tmp_path, role="cycle_ood")
    contract = validate_target_2d_evaluation_metadata(
        paths["manifest"],
        project_root=paths["root"],
        manifest_spec=paths["spec"],
    )
    original = evaluation._SourceRangeSession.complete_read
    triggered = False

    def interrupt_after_durable_row(
        self: Any,
        *,
        purpose: str,
        recovered: bool = False,
        **kwargs: Any,
    ) -> None:
        nonlocal triggered
        if (
            purpose == "authorized_case_label_slice"
            and not recovered
            and not triggered
        ):
            triggered = True
            raise _SyntheticProcessDeath("synthetic post-row interruption")
        original(
            self,
            purpose=purpose,
            recovered=recovered,
            **kwargs,
        )

    with monkeypatch.context() as scoped:
        scoped.setattr(
            evaluation._SourceRangeSession,
            "complete_read",
            interrupt_after_durable_row,
        )
        with pytest.raises(_SyntheticProcessDeath):
            materialize_target_2d_label_shard(
                contract,
                stage=paths["stage"],
                split=paths["split_name"],
                authorization=paths["authorization"],
                access_session=paths["initial_access_session"],
                shard_root=paths["shard_root"],
            )
    shard = materialize_target_2d_label_shard(
        contract,
        stage=paths["stage"],
        split=paths["split_name"],
        authorization=paths["authorization"],
        access_session=paths["initial_access_session"],
        shard_root=paths["shard_root"],
    )
    assert shard.receipt["source_access_summary"][
        "recovered_without_source_replay_completion_count"
    ] == 1
    assert shard.receipt["source_access_summary"][
        "source_range_replay_count"
    ] == 0
    assert shard.resume_attempt is not None
    source_records = [
        json.loads(line)
        for line in shard.source_journal_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    case_attempts = [
        item
        for item in source_records
        if item["event"] == "source_range_read_attempt"
        and item["purpose"] == "authorized_case_label_slice"
        and item["label_name"] == "temperature_K"
    ]
    assert len(case_attempts) == 1


def test_partial_staging_before_any_source_read_is_cleaned_and_rebuilt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _build_fixture(tmp_path, role="target_id")
    contract = validate_target_2d_evaluation_metadata(
        paths["manifest"],
        project_root=paths["root"],
        manifest_spec=paths["spec"],
    )
    original = evaluation._create_empty_npy
    triggered = False

    def partial_then_die(path: Path, **kwargs: Any) -> int:
        nonlocal triggered
        if not triggered:
            triggered = True
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"partial synthetic staging NPY")
            raise _SyntheticProcessDeath("synthetic staging interruption")
        return original(path, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(evaluation, "_create_empty_npy", partial_then_die)
        with pytest.raises(_SyntheticProcessDeath):
            materialize_target_2d_label_shard(
                contract,
                stage=paths["stage"],
                split=paths["split_name"],
                authorization=paths["authorization"],
                access_session=paths["initial_access_session"],
                shard_root=paths["shard_root"],
            )
    shard = materialize_target_2d_label_shard(
        contract,
        stage=paths["stage"],
        split=paths["split_name"],
        authorization=paths["authorization"],
        access_session=paths["initial_access_session"],
        shard_root=paths["shard_root"],
    )
    assert shard.receipt_path.is_file()
    assert shard.resume_attempt is not None


def test_receipt_cannot_be_adopted_without_prior_slice_journal(
    tmp_path: Path,
) -> None:
    paths = _build_fixture(tmp_path, role="target_id")
    prepared = _prepare(paths)
    shard = prepared.dataset._label_shard
    assert shard is not None
    shard.slice_journal_path.unlink()
    contract = validate_target_2d_evaluation_metadata(
        paths["manifest"],
        project_root=paths["root"],
        manifest_spec=paths["spec"],
    )
    with pytest.raises(PermissionError, match="slice-access journal"):
        materialize_target_2d_label_shard(
            contract,
            stage=paths["stage"],
            split=paths["split_name"],
            authorization=paths["authorization"],
            access_session=paths["initial_access_session"],
            shard_root=paths["shard_root"],
        )


def test_tampered_adopted_shard_fails_on_journaled_slice_without_monolith_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _build_fixture(tmp_path, role="target_id")
    prepared = _prepare(paths)
    shard = prepared.dataset._label_shard
    assert shard is not None
    target = shard.array_paths["alpha"]
    with target.open("r+b", buffering=0) as stream:
        stream.seek(-1, 2)
        current = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([current[0] ^ 0x01]))
        stream.flush()

    def forbidden_source_open(self: Any) -> Any:
        raise AssertionError("monolithic source must not reopen for adoption")

    monkeypatch.setattr(
        evaluation._SourceRangeSession,
        "open",
        forbidden_source_open,
    )
    resumed = _prepare(paths)
    assert resumed.access_audit["attempted_case_ids"] == []
    assert resumed.access_audit["completed_case_ids"] == []
    with pytest.raises(PermissionError, match="shard slice hash differs"):
        resumed.dataset[0]
    audit = resumed.access_audit
    assert audit["shard_label_slice_read_attempt_count"] == 2
    assert audit["shard_label_slice_read_completion_count"] == 1
    assert audit["shard_label_slice_unresolved_attempt_count"] == 1


def test_shard_slice_replay_is_durably_rejected_before_payload_read(
    tmp_path: Path,
) -> None:
    paths = _build_fixture(tmp_path, role="cycle_ood")
    prepared = _prepare(paths)
    prepared.dataset[0]
    with pytest.raises(PermissionError, match="replay is forbidden"):
        prepared.dataset[0]
    audit = prepared.access_audit
    assert audit["shard_label_slice_read_attempt_count"] == 2
    assert audit["shard_label_slice_read_completion_count"] == 2
    assert audit["shard_label_slice_replay_rejection_count"] == 1


def test_forged_access_session_is_rejected_before_shard_writes(
    tmp_path: Path,
) -> None:
    paths = _build_fixture(tmp_path, role="target_id")
    contract = validate_target_2d_evaluation_metadata(
        paths["manifest"],
        project_root=paths["root"],
        manifest_spec=paths["spec"],
    )
    unsigned = {
        key: value
        for key, value in paths["initial_access_session"].items()
        if key != "session_payload_sha256"
    }
    unsigned["session_authority_sha256"] = "e" * 64
    forged = {
        **unsigned,
        "session_payload_sha256": hashlib.sha256(
            _canonical_bytes(unsigned)
        ).hexdigest(),
    }
    with pytest.raises(PermissionError, match="authority differs"):
        materialize_target_2d_label_shard(
            contract,
            stage=paths["stage"],
            split=paths["split_name"],
            authorization=paths["authorization"],
            access_session=forged,
            shard_root=paths["shard_root"],
        )
    assert not paths["shard_root"].exists()


def test_resume_session_must_match_manager_partial_split_flag(
    tmp_path: Path,
) -> None:
    paths = _build_fixture(tmp_path, role="cycle_ood")
    contract = validate_target_2d_evaluation_metadata(
        paths["manifest"],
        project_root=paths["root"],
        manifest_spec=paths["spec"],
    )
    resumed = _resumed_access_session(paths)
    unsigned = {
        key: value
        for key, value in resumed.items()
        if key != "session_payload_sha256"
    }
    unsigned["adapter_resume_interrupted"] = False
    mismatched = {
        **unsigned,
        "session_payload_sha256": hashlib.sha256(
            _canonical_bytes(unsigned)
        ).hexdigest(),
    }
    with pytest.raises(PermissionError, match="authority differs"):
        materialize_target_2d_label_shard(
            contract,
            stage=paths["stage"],
            split=paths["split_name"],
            authorization=paths["authorization"],
            access_session=mismatched,
            shard_root=paths["shard_root"],
        )
    assert not paths["shard_root"].exists()


def test_access_session_registration_is_unique_and_precedes_label_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _build_fixture(tmp_path, role="target_id")
    prepared = _prepare(paths)
    shard = prepared.dataset._label_shard
    assert shard is not None

    def forbidden_open() -> Any:
        raise OSError("synthetic label-open interruption")

    monkeypatch.setattr(prepared.dataset, "_open_label_arrays", forbidden_open)
    with pytest.raises(OSError, match="label-open interruption"):
        prepared.dataset[0]
    records = [
        json.loads(line)
        for line in shard.slice_journal_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [record["event"] for record in records] == [
        "shard_label_access_session_registered"
    ]
    assert records[0]["label_access_session"] == (
        paths["initial_access_session"]
    )

    journal = evaluation._DurableJournal(
        shard.slice_journal_path,
        journal_id=(
            f"target2d-shard-slice:{paths['stage']}:"
            f"{paths['split_name']}"
        ),
    )
    journal.append(
        "shard_label_access_session_registered",
        label_access_session=paths["initial_access_session"],
    )
    with pytest.raises(PermissionError, match="more than once"):
        _ = prepared.access_audit


def test_authorized_resume_gets_one_new_exact_slice_read_session(
    tmp_path: Path,
) -> None:
    paths = _build_fixture(tmp_path, role="cycle_ood")
    first = _prepare(paths)
    for index in range(len(first.dataset)):
        first.dataset[index]
    initial = first.access_audit
    assert initial["label_access_session_kind"] == (
        "initial_release_stage_invocation"
    )
    assert initial["shard_label_slice_read_attempt_count"] == (
        2 * len(first.case_ids)
    )

    resumed = _prepare(paths)
    before = resumed.access_audit
    assert before["attempted_case_ids"] == []
    assert before["completed_case_ids"] == []
    assert before["label_access_session_kind"] == (
        "hard_interruption_resume_stage_invocation"
    )
    assert before["prior_complete_access_session_count"] == 1
    for index in range(len(resumed.dataset)):
        resumed.dataset[index]
    result = evaluation.validate_target_2d_label_shard_evidence(
        resumed.access_audit["label_shard_evidence"],
        project_root=paths["root"],
        stage=paths["stage"],
        split=paths["split_name"],
        expected_case_ids=resumed.case_ids,
        require_complete_slice_reads=True,
    )
    assert result["access_session_satisfaction"] == (
        "current_authorized_invocation"
    )
    assert result["cumulative_access_session_count"] == 2
    assert result["complete_access_session_count"] == 2
    assert result["shard_slice_attempt_count"] == 2 * len(
        resumed.case_ids
    )
    assert result["cumulative_shard_slice_attempt_count"] == 4 * len(
        resumed.case_ids
    )


def test_resume_recovers_after_prior_unresolved_slice_without_hiding_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _build_fixture(tmp_path, role="target_id")
    first = _prepare(paths)
    original = evaluation._semantic_array_sha256
    interrupted = False

    def interrupt_after_payload_read(*args: Any, **kwargs: Any) -> str:
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise _SyntheticProcessDeath(
                "synthetic slice hash interruption"
            )
        return original(*args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(
            evaluation,
            "_semantic_array_sha256",
            interrupt_after_payload_read,
        )
        with pytest.raises(
            _SyntheticProcessDeath,
            match="slice hash interruption",
        ):
            first.dataset[0]

    resumed = _prepare(paths)
    for index in range(len(resumed.dataset)):
        resumed.dataset[index]
    result = evaluation.validate_target_2d_label_shard_evidence(
        resumed.access_audit["label_shard_evidence"],
        project_root=paths["root"],
        stage=paths["stage"],
        split=paths["split_name"],
        expected_case_ids=resumed.case_ids,
        require_complete_slice_reads=True,
    )
    assert result["access_session_satisfaction"] == (
        "current_authorized_invocation"
    )
    assert result[
        "cumulative_shard_slice_unresolved_attempt_count"
    ] == 1
    assert result["complete_access_session_count"] == 1


def test_complete_prior_session_can_be_adopted_without_payload_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _build_fixture(tmp_path, role="target_id")
    first = _prepare(paths)
    for index in range(len(first.dataset)):
        first.dataset[index]
    shard = first.dataset._label_shard
    assert shard is not None
    forbidden = {
        path.resolve() for path in shard.array_paths.values()
    }
    original_hash = evaluation._sha256_file
    original_path_open = Path.open

    def guarded_hash(path: Path) -> str:
        if path.resolve() in forbidden:
            raise AssertionError(
                "complete-session adoption hashed a label-shard payload"
            )
        return original_hash(path)

    def guarded_open(
        path: Path,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if path.resolve() in forbidden:
            raise AssertionError(
                "complete-session adoption reopened a label-shard payload"
            )
        return original_path_open(path, *args, **kwargs)

    monkeypatch.setattr(evaluation, "_sha256_file", guarded_hash)
    monkeypatch.setattr(Path, "open", guarded_open)
    resumed = _prepare(paths)
    resumed_shard = resumed.dataset._label_shard
    assert resumed_shard is not None
    journal_before = resumed_shard.slice_journal_path.read_bytes()
    adopted = resumed.dataset.adopt_prior_complete_label_access()
    assert resumed_shard.slice_journal_path.read_bytes() == journal_before
    audit = resumed.access_audit
    assert audit["attempted_case_ids"] == list(resumed.case_ids)
    assert audit["completed_case_ids"] == list(resumed.case_ids)
    assert audit["adopted_from_access_session_sha256"] == adopted
    assert audit["label_access_session_satisfaction"] == (
        "prior_complete_session_adopted_without_payload_reopen"
    )
    result = evaluation.validate_target_2d_label_shard_evidence(
        audit["label_shard_evidence"],
        project_root=paths["root"],
        stage=paths["stage"],
        split=paths["split_name"],
        expected_case_ids=resumed.case_ids,
        require_complete_slice_reads=True,
        allow_prior_complete_session_adoption=True,
    )
    assert result["access_session_satisfaction"] == (
        "prior_complete_invocation_adopted"
    )


def test_downstream_evidence_validation_never_reopens_or_hashes_shards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _build_fixture(tmp_path, role="target_id")
    prepared = _prepare(paths)
    prepared.dataset[0]
    shard = prepared.dataset._label_shard
    assert shard is not None
    forbidden = {
        path.resolve() for path in shard.array_paths.values()
    }
    original_hash = evaluation._sha256_file

    def guarded_hash(path: Path) -> str:
        if path.resolve() in forbidden:
            raise AssertionError("downstream reopened/hashed a shard payload")
        return original_hash(path)

    monkeypatch.setattr(evaluation, "_sha256_file", guarded_hash)
    result = evaluation.validate_target_2d_label_shard_evidence(
        prepared.access_audit["label_shard_evidence"],
        project_root=paths["root"],
        stage=paths["stage"],
        split=paths["split_name"],
        expected_case_ids=prepared.case_ids,
        require_complete_slice_reads=True,
    )
    assert result[
        "shard_label_payloads_reopened_or_hashed_by_validation"
    ] is False


@pytest.mark.parametrize(
    "mutation",
    ["manifest", "source", "plan", "array"],
)
def test_rejects_manifest_source_plan_and_array_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    paths = _build_fixture(tmp_path)
    if mutation == "manifest":
        paths["manifest"].write_bytes(
            paths["manifest"].read_bytes() + b"\n"
        )
        message = "Evaluation manifest SHA256"
    elif mutation == "source":
        paths["source"].write_bytes(paths["source"].read_bytes() + b"\n")
        message = "source manifest SHA256"
    elif mutation == "plan":
        paths["plan"].write_bytes(paths["plan"].read_bytes() + b"\n")
        message = "plan file SHA256"
    else:
        path = paths["artifact"] / "temperature_K.npy"
        values = np.load(path, allow_pickle=False)
        values[4, 0, 0, 0] += 1.0
        np.save(path, values, allow_pickle=False)
        contract = validate_target_2d_evaluation_metadata(
            paths["manifest"],
            project_root=paths["root"],
            manifest_spec=paths["spec"],
        )
        assert contract.checksums[
            "monolithic_label_containers_hashed_during_metadata_validation"
        ] is False
        with pytest.raises(ValueError, match="slice hash differs"):
            materialize_target_2d_label_shard(
                contract,
                stage=paths["stage"],
                split=paths["split_name"],
                authorization=paths["authorization"],
                access_session=paths["initial_access_session"],
                shard_root=paths["shard_root"],
            )
        return
    with pytest.raises(ValueError, match=message):
        validate_target_2d_evaluation_metadata(
            paths["manifest"],
            project_root=paths["root"],
            manifest_spec=paths["spec"],
        )


def test_rejects_arbitrary_ids_wrong_role_and_train_ids(
    tmp_path: Path,
) -> None:
    paths = _build_fixture(tmp_path)
    payload = json.loads(paths["manifest"].read_text(encoding="utf-8"))

    payload["case_ids"] = [5]
    _write_json(paths["manifest"], payload)
    resigned = replace(paths["spec"], manifest_sha256=_sha256(paths["manifest"]))
    with pytest.raises(PermissionError, match="pre-registered order"):
        validate_target_2d_evaluation_metadata(
            paths["manifest"],
            project_root=paths["root"],
            manifest_spec=resigned,
        )

    payload["case_ids"] = [4]
    payload["role"] = "htc_ood"
    _write_json(paths["manifest"], payload)
    resigned = replace(paths["spec"], manifest_sha256=_sha256(paths["manifest"]))
    with pytest.raises(ValueError, match="role"):
        validate_target_2d_evaluation_metadata(
            paths["manifest"],
            project_root=paths["root"],
            manifest_spec=resigned,
        )

    payload["case_ids"] = [0]
    payload["role"] = "cycle_ood"
    payload["case_definition_hashes"] = {}
    _write_json(paths["manifest"], payload)
    resigned = replace(
        paths["spec"],
        manifest_sha256=_sha256(paths["manifest"]),
        case_ids=(0,),
    )
    with pytest.raises(PermissionError, match="train or validation"):
        validate_target_2d_evaluation_metadata(
            paths["manifest"],
            project_root=paths["root"],
            manifest_spec=resigned,
        )


@pytest.mark.parametrize(
    ("role", "schema_version", "status"),
    [
        ("geometry_ood", 2, "deferred_until_F3"),
        (
            "external_validation",
            1,
            "deferred_until_P7_compatibility_audit",
        ),
    ],
)
def test_rejects_empty_deferred_geometry_and_external_manifests(
    tmp_path: Path,
    role: str,
    schema_version: int,
    status: str,
) -> None:
    root = tmp_path / "repo"
    path = root / "splits" / f"fixture_{role}.json"
    payload = {
        "schema_version": schema_version,
        "phase": "P4",
        "role": role,
        "status": status,
        "case_ids": [],
    }
    _write_json(path, payload)
    spec = FrozenEvaluationManifestSpec(
        name=f"fixture_{role}",
        relative_path=f"splits/{path.name}",
        manifest_sha256=_sha256(path),
        role=role,
        case_ids=(),
        source_split_key=role,
        source_manifest_relative_path="splits/unused.json",
        source_manifest_sha256="0" * 64,
        plan_file_sha256="0" * 64,
    ).validated()
    with pytest.raises(PermissionError, match="Deferred"):
        validate_target_2d_evaluation_metadata(
            path,
            project_root=root,
            manifest_spec=spec,
        )


def test_rejects_checkpoint_hash_drift_before_dataset_construction(
    tmp_path: Path,
) -> None:
    paths = _build_fixture(tmp_path)
    with pytest.raises(ValueError, match="checkpoint SHA256"):
        prepare_target_2d_evaluation(
            paths["manifest"],
            paths["checkpoint"],
            expected_source_checkpoint_sha256="0" * 64,
            project_root=paths["root"],
            manifest_spec=paths["spec"],
            baseline_builder=_fake_baseline,
        )


def test_canonical_repository_cannot_override_frozen_manifest_registry(
    tmp_path: Path,
) -> None:
    paths = _build_fixture(tmp_path)
    canonical_source = (
        paths["root"] / "splits" / "p4_2d_core_v1.json"
    )
    _write_json(canonical_source, {})
    with pytest.raises(PermissionError, match="cannot override"):
        validate_target_2d_evaluation_metadata(
            paths["manifest"],
            project_root=paths["root"],
            manifest_spec=paths["spec"],
        )
