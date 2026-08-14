from __future__ import annotations

import builtins
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import cdcureno.data.target_2d_training_shard as shard
from cdcureno.data.source_1d import INPUT_CHANNELS as SOURCE_INPUT_CHANNELS


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


def _definition(case_id: int, split: str) -> dict[str, Any]:
    base_x = 4.6724
    base_z = 0.6369803
    return {
        "case_id": case_id,
        "case_key": f"fixture-{case_id:04d}",
        "split": split,
        "difficulty_family": "F0",
        "air_temperature_K": [293.0, 350.0, 293.0],
        "top_h_W_m2_K": [120.0, 120.0],
        "bottom_h_W_m2_K": 60.0,
        "left_h_W_m2_K": 0.0,
        "right_h_W_m2_K": 0.0,
        "base_composite_conductivity_x_W_m_K": base_x,
        "base_composite_conductivity_z_W_m_K": base_z,
        "composite_conductivity_x_W_m_K": base_x,
        "composite_conductivity_z_W_m_K": base_z,
        "composite_conductivity_scale": 1.0,
        "reaction_enthalpy_scale": 1.0,
    }


def _normalization() -> dict[str, Any]:
    return {
        "scope": "training_cases_only",
        "time_stride": 2,
        "time_indices": [0, 2],
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


def _build_fixture(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "repo"
    split_root = root / "splits"
    source_root = root / "data" / "processed" / "p4_fixture"
    source_root.mkdir(parents=True)
    case_count, nt, nz, nx = 300, 3, 2, 2
    splits = {
        "train": list(range(256)),
        "validation": list(range(256, 288)),
        "id_test": list(range(288, 300)),
        "cycle_ood": [],
        "htc_ood": [],
        "pattern_ood": [],
        "combined_ood": [],
    }
    split_by_case = {
        case_id: name
        for name, case_ids in splits.items()
        for case_id in case_ids
    }
    definitions = [
        _definition(case_id, split_by_case[case_id])
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
        "nested_training_budgets": [8, 16, 32, 64, 128, 256],
        "splits": splits,
        "time_s": [0.0, 60.0, 120.0],
        "resolved_config": {
            "geometry": {
                "width_m": 0.20,
                "tool_thickness_m": 0.02,
                "composite_thickness_m": 0.03,
            }
        },
        "cases": [
            {
                "definition": definition,
                "case_definition_sha256": definition_hashes[
                    definition["case_key"]
                ],
            }
            for definition in definitions
        ],
    }
    plan["plan_sha256"] = hashlib.sha256(
        _canonical_bytes(plan)
    ).hexdigest()
    plan_path = split_root / "p4_fixture_plan.json"
    _write_json(plan_path, plan)

    time = np.asarray(plan["time_s"], dtype=np.float64)
    x = np.array([0.05, 0.15], dtype=np.float64)
    z = np.array([0.0125, 0.0375], dtype=np.float64)
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
    alpha = np.empty_like(temperature)
    for case_id in range(case_count):
        temperature[case_id] = 300.0 + case_id
        alpha[case_id] = case_id / 1000.0
    temperature[288:] = 10_000.0
    alpha[288:] = 0.999
    arrays = {
        "temperature_K": temperature,
        "alpha": alpha,
        "air_temperature_K": air,
        "top_h_W_m2_K": top_h,
        "time_s": time,
        "x_m": x,
        "z_m": z,
        "composite_mask": mask,
    }
    array_hashes: dict[str, str] = {}
    for name, values in arrays.items():
        path = source_root / f"{name}.npy"
        np.save(path, values, allow_pickle=False)
        array_hashes[name] = _sha256(path)

    core = {
        "schema_version": 2,
        "phase": "P4",
        "dataset_id": "p4_fixture",
        "case_count": case_count,
        "array_artifact_root": "data/processed/p4_fixture",
        "array_shape_case_time_z_x": [case_count, nt, nz, nx],
        "array_sha256": array_hashes,
        "metadata_sha256": "a" * 64,
        "plan_path": "splits/p4_fixture_plan.json",
        "plan_sha256": plan["plan_sha256"],
        "nested_training_budgets": [8, 16, 32, 64, 128, 256],
        "case_definition_hashes": definition_hashes,
        "splits": splits,
    }
    core_path = split_root / "p4_fixture.json"
    _write_json(core_path, core)
    id_manifest = {
        "schema_version": 2,
        "phase": "P4",
        "role": "target_id",
        "dataset_id": "p4_fixture",
        "source_manifest": "splits/p4_fixture.json",
        "source_manifest_sha256": _sha256(core_path),
        "dataset_array_sha256": array_hashes,
        "dataset_metadata_sha256": "a" * 64,
        "dataset_plan_sha256": plan["plan_sha256"],
        "case_definition_hashes": definition_hashes,
        "splits": {
            "train_pool": splits["train"],
            "validation": splits["validation"],
            "test": splits["id_test"],
        },
        "nested_training_budgets": {
            str(budget): list(range(budget))
            for budget in (8, 16, 32, 64, 128, 256)
        },
    }
    id_path = split_root / "2d_id_fixture.json"
    _write_json(id_path, id_manifest)
    checkpoint = root / "outputs" / "source.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save(
        {
            "channel_names": SOURCE_INPUT_CHANNELS,
            "normalization": _normalization(),
            "model": {},
        },
        checkpoint,
    )
    return {
        "root": root,
        "source_root": source_root,
        "core": core_path,
        "id": id_path,
        "plan": plan_path,
        "checkpoint": checkpoint,
        "output_root": (
            root / "data" / "processed" / "p4_2d_train_validation_v1"
        ),
        "output_manifest": (
            root / "splits" / "p4_2d_train_validation_shard_v1.json"
        ),
    }


def _build(paths: dict[str, Path], **kwargs: Any) -> dict[str, Any]:
    return shard.build_training_shard(
        project_root=paths["root"],
        core_manifest_path=paths["core"],
        id_manifest_path=paths["id"],
        output_manifest_path=paths["output_manifest"],
        output_root=paths["output_root"],
        **kwargs,
    )


def test_builder_indexes_monolithic_labels_only_with_first_288_slice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _build_fixture(tmp_path)
    prohibited_hashes = {
        (paths["source_root"] / f"{name}.npy").resolve()
        for name in shard.LABEL_ARRAYS
    }
    original_hash = shard._sha256_file

    def guarded_hash(path: Path) -> str:
        if path.resolve() in prohibited_hashes:
            raise AssertionError("Builder hashed a monolithic label file")
        return original_hash(path)

    original_load = shard.np.load
    label_slices: dict[str, list[Any]] = {
        name: [] for name in shard.LABEL_ARRAYS
    }

    class SliceGuard:
        def __init__(self, name: str, array: np.ndarray) -> None:
            self.name = name
            self.array = array
            self.shape = array.shape
            self.dtype = array.dtype

        def __getitem__(self, index: Any) -> Any:
            label_slices[self.name].append(index)
            assert index == slice(None, 288, None)
            return self.array[index]

    def guarded_load(path: Any, *args: Any, **kwargs: Any) -> Any:
        array = original_load(path, *args, **kwargs)
        resolved = Path(path).resolve()
        if resolved in prohibited_hashes:
            return SliceGuard(resolved.stem, array)
        return array

    monkeypatch.setattr(shard, "_sha256_file", guarded_hash)
    monkeypatch.setattr(shard.np, "load", guarded_load)
    result = _build(paths)

    assert result["phase"] == "P6V2"
    assert result["case_axis_policy"]["included_case_ids"] == list(
        range(288)
    )
    assert result["case_axis_policy"][
        "id_test_or_ood_label_values_copied"
    ] is False
    assert result["source_binding"][
        "monolithic_label_file_sha256_recomputed"
    ] is False
    assert label_slices == {
        "temperature_K": [slice(None, 288, None)],
        "alpha": [slice(None, 288, None)],
    }
    for name in shard.SHARD_ARRAYS:
        record = result["arrays"][name]
        assert len(record["sha256"]) == 64
        assert record["bytes"] > 0
    temperature = original_load(
        paths["output_root"] / "temperature_K.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    alpha = original_load(
        paths["output_root"] / "alpha.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    assert temperature.shape[0] == alpha.shape[0] == 288
    assert np.all(temperature[-1] == 587.0)
    assert np.all(alpha[-1] == pytest.approx(0.287))
    assert not any(paths["output_root"].parent.glob(".*.staging-*"))


def test_build_is_no_overwrite_and_verify_existing_is_read_only(
    tmp_path: Path,
) -> None:
    paths = _build_fixture(tmp_path)
    _build(paths)
    before = paths["output_manifest"].read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _build(paths)
    verified = _build(paths, verify_existing=True)
    assert verified["status"] == "verified_existing"
    assert verified["case_count"] == 288
    assert paths["output_manifest"].read_bytes() == before

    with (paths["output_root"] / "time_s.npy").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="byte count differs"):
        _build(paths, verify_existing=True)


def test_build_failure_leaves_no_partial_shard_or_staging_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _build_fixture(tmp_path)
    real_write = shard._write_npy_from_view
    writes = 0

    def fail_after_first_write(*args: Any, **kwargs: Any) -> None:
        nonlocal writes
        real_write(*args, **kwargs)
        writes += 1
        if writes == 1:
            raise OSError("injected shard write failure")

    monkeypatch.setattr(shard, "_write_npy_from_view", fail_after_first_write)
    with pytest.raises(OSError, match="injected shard write failure"):
        _build(paths)
    assert not paths["output_root"].exists()
    assert not paths["output_manifest"].exists()
    assert not any(paths["output_root"].parent.glob(".*.staging-*"))


def test_loader_never_opens_monolithic_label_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _build_fixture(tmp_path)
    _build(paths)
    prohibited = {
        (paths["source_root"] / f"{name}.npy").resolve()
        for name in shard.LABEL_ARRAYS
    }
    attempts: list[Path] = []
    original_np_load = shard.np.load
    original_open = builtins.open
    original_path_open = Path.open

    def check_path(value: Any) -> None:
        if isinstance(value, (str, os.PathLike)):
            resolved = Path(value).resolve()
            if resolved in prohibited:
                attempts.append(resolved)
                raise AssertionError(
                    "Training-shard loader opened a monolithic label file"
                )

    def guarded_np_load(path: Any, *args: Any, **kwargs: Any) -> Any:
        check_path(path)
        return original_np_load(path, *args, **kwargs)

    def guarded_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        check_path(file)
        return original_open(file, *args, **kwargs)

    def guarded_path_open(
        self: Path, *args: Any, **kwargs: Any
    ) -> Any:
        check_path(self)
        return original_path_open(self, *args, **kwargs)

    monkeypatch.setattr(shard.np, "load", guarded_np_load)
    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(Path, "open", guarded_path_open)
    expected_manifest_sha = _sha256(paths["output_manifest"])
    prepared = shard.prepare_target_2d_training_shard(
        paths["output_manifest"],
        paths["checkpoint"],
        label_budget=8,
        project_root=paths["root"],
        expected_manifest_sha256=expected_manifest_sha,
        baseline_builder=_fake_baseline,
    )
    assert prepared.train_case_ids == tuple(range(8))
    assert prepared.validation_case_ids == tuple(range(256, 288))
    assert prepared.checksums[
        "monolithic_label_files_opened_by_loader"
    ] is False
    train_item = prepared.dataset("train")[0]
    validation_item = prepared.dataset("validation")[0]
    assert train_item[-1].item() == 0
    assert validation_item[-1].item() == 256
    assert attempts == []
    with pytest.raises(PermissionError, match="evaluation-only"):
        prepared.dataset("test")


def test_loader_rejects_any_case_policy_expansion_into_id_test(
    tmp_path: Path,
) -> None:
    paths = _build_fixture(tmp_path)
    _build(paths)
    payload = json.loads(paths["output_manifest"].read_text(encoding="utf-8"))
    payload["case_axis_policy"]["included_case_ids"].append(288)
    _write_json(paths["output_manifest"], payload)
    with pytest.raises(PermissionError, match="exactly cases 0--287"):
        shard.validate_training_shard(
            paths["output_manifest"],
            project_root=paths["root"],
        )
