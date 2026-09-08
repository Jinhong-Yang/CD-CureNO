from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import freeze_p4_splits as freezer  # noqa: E402
import generate_p4_2d as p4  # noqa: E402


def _tiny_core_config(path: Path) -> dict:
    config = copy.deepcopy(p4.load_config())
    config["time"]["duration_s"] = 120.0
    config["time"]["output_interval_s"] = 120.0
    config["geometry"]["width_m"] = 0.02
    config["tiers"]["core"].update(
        {
            "case_count": 8,
            "spacing_x_m": 0.01,
            "spacing_z_m": 0.025,
            "split_counts": {
                "train": 2,
                "validation": 1,
                "id_test": 1,
                "cycle_ood": 1,
                "htc_ood": 1,
                "pattern_ood": 1,
                "combined_ood": 1,
            },
            "nested_training_budgets": [1, 2],
        }
    )
    path.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    return p4.load_config(path)


def _build_tiny_finalized_core(tmp_path: Path) -> dict[str, Path]:
    project_root = tmp_path / "repo"
    for relative in p4._code_hashes(ROOT):
        source = ROOT / relative
        destination = project_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    config_path = project_root / "configs" / "data" / "p4.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config = _tiny_core_config(config_path)
    plan = p4.build_case_plan(
        "core",
        config_path=config_path,
        config=config,
        project_root=project_root,
    )
    plan["provenance"]["git_commit_sha"] = "a" * 40
    plan["plan_sha256"] = p4._sha256_bytes(
        p4._canonical_json_bytes(
            {key: value for key, value in plan.items() if key != "plan_sha256"}
        )
    )
    plan_path = project_root / "splits" / "p4_2d_core_v1_plan.json"
    p4.write_or_verify_plan(plan, plan_path)
    staging = project_root / "data" / "processed" / ".core-staging"
    arrays = p4._prepare_staging(plan, config, staging)
    shape = arrays["temperature_K"].shape[1:]
    composite_mask = np.asarray(arrays["composite_mask"])
    for entry in plan["cases"]:
        definition = entry["definition"]
        case_id = int(definition["case_id"])
        temperature = np.full(
            shape, 300.0 + case_id, dtype=np.float32
        )
        alpha = np.zeros(shape, dtype=np.float32)
        alpha[1:, composite_mask] = 0.1 + case_id * 0.01
        result = {
            "case_id": case_id,
            "status": "passed",
            "failure": None,
            "temperature_K": temperature,
            "alpha": alpha,
            "solver": {
                "finite_fields": True,
                "alpha_bound_violation_count": 0,
                "alpha_monotonicity_violation_count": 0,
                "tool_alpha_nonzero_count": 0,
                "maximum_abs_energy_residual_W_m3": 0.0,
                "maximum_relative_global_energy_residual": 0.0,
                "maximum_interface_flux_imbalance_W": 0.0,
                "maximum_temperature_interface_jump_K": 0.0,
                "maximum_robin_flux_imbalance_W": 0.0,
                "all_coupling_steps_converged": True,
                "maximum_converged_coupling_update_K": 0.0,
                "substeps": 1,
                "maximum_coupling_iterations": 1,
                "configured_maximum_step_s": 10.0,
                "configured_coupling_tolerance_K": 1.0e-9,
                "configured_maximum_coupling_iterations": 16,
            },
            "total_case_wall_seconds": 0.01,
        }
        p4._commit_case_result(
            definition=definition,
            result=result,
            arrays=arrays,
            staging_root=staging,
        )
    completions, pending = p4._load_completions(
        plan, arrays, staging, retry_failed=False
    )
    assert not pending
    del arrays

    final_root = project_root / "data" / "processed" / "core"
    metadata_path = (
        project_root / "outputs" / "tables" / "core_cases.jsonl"
    )
    manifest_path = project_root / "splits" / "p4_2d_core_v1.json"
    summary_path = (
        project_root / "outputs" / "tables" / "core_generation.json"
    )
    p4._finalize(
        plan=plan,
        config=config,
        staging_root=staging,
        output_root=final_root,
        plan_path=plan_path,
        metadata_path=metadata_path,
        manifest_path=manifest_path,
        summary_path=summary_path,
        completions=completions,
        generation_wall_seconds=0.1,
        workers=1,
        maximum_in_flight=1,
        resource_preflight={"test_fixture": True},
        project_root=project_root,
    )
    source_path = project_root / "splits" / "custom" / "source.json"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        json.dumps(
            {
                "dataset_id": "source",
                "array_sha256": "source-hash",
                "splits": {"train": [0]},
            }
        ),
        encoding="utf-8",
    )
    phase_path = project_root / "splits" / "custom" / "phase.json"
    phase_path.write_text(
        json.dumps({"version": "phase", "splits": {"train": [0]}}),
        encoding="utf-8",
    )
    return {
        "manifest": manifest_path,
        "summary": summary_path,
        "metadata": metadata_path,
        "plan": plan_path,
        "config": config_path,
        "artifact_root": final_root,
        "source": source_path,
        "phase": phase_path,
        "frozen": project_root / "splits" / "frozen",
        "project_root": project_root,
    }


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_bytes(p4._pretty_json_bytes(payload))


def _rewrite_summary_and_manifest(
    paths: dict[str, Path],
    summary: dict,
    manifest: dict,
) -> None:
    _write_json(paths["summary"], summary)
    manifest["generation_summary_sha256"] = p4._sha256_file(paths["summary"])
    _write_json(paths["manifest"], manifest)


def test_freezer_validates_checksums_and_writes_nested_immutable_manifests(
    tmp_path: Path,
) -> None:
    paths = _build_tiny_finalized_core(tmp_path)
    result = freezer.freeze_manifests(
        core_path=paths["manifest"],
        source_path=paths["source"],
        phase_a_path=paths["phase"],
        output_root=paths["frozen"],
        project_root=paths["project_root"],
    )
    assert result["nested_training_budgets"] == ["1", "2"]
    target = json.loads(
        (paths["frozen"] / "2d_id_v1.json").read_text(encoding="utf-8")
    )
    assert target["nested_training_budgets"]["1"] == target["splits"][
        "train_pool"
    ][:1]
    assert target["nested_training_budgets"]["2"] == target["splits"][
        "train_pool"
    ][:2]
    manifest = _read_json(paths["manifest"])
    assert manifest["generation_summary_sha256"] == p4._sha256_file(
        paths["summary"]
    )
    assert target["source_manifest"] == "splits/p4_2d_core_v1.json"
    assert target["source_manifest_sha256"] == p4._sha256_file(
        paths["manifest"]
    )
    source_alias = _read_json(paths["frozen"] / "source_1d_v1.json")
    assert source_alias["source_manifest"] == "splits/custom/source.json"
    assert source_alias["source_manifest_sha256"] == p4._sha256_file(
        paths["source"]
    )
    phase_alias = _read_json(paths["frozen"] / "phase_a_v1.json")
    assert phase_alias["source_manifest"] == "splits/custom/phase.json"
    assert phase_alias["source_manifest_sha256"] == p4._sha256_file(
        paths["phase"]
    )

    # Byte-identical replay is allowed.
    freezer.freeze_manifests(
        core_path=paths["manifest"],
        source_path=paths["source"],
        phase_a_path=paths["phase"],
        output_root=paths["frozen"],
        project_root=paths["project_root"],
    )
    frozen_path = paths["frozen"] / "2d_id_v1.json"
    original = frozen_path.read_bytes()
    frozen_path.write_bytes(b"not-the-frozen-manifest\n")
    with pytest.raises(FileExistsError, match="non-identical"):
        freezer.freeze_manifests(
            core_path=paths["manifest"],
            source_path=paths["source"],
            phase_a_path=paths["phase"],
            output_root=paths["frozen"],
            project_root=paths["project_root"],
        )
    assert frozen_path.read_bytes() == b"not-the-frozen-manifest\n"
    frozen_path.write_bytes(original)


def test_freezer_rejects_checksum_mismatch_and_overlapping_split(
    tmp_path: Path,
) -> None:
    paths = _build_tiny_finalized_core(tmp_path)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))

    bad_checksum = copy.deepcopy(manifest)
    bad_checksum["array_sha256"]["temperature_K"] = "0" * 64
    bad_checksum_path = paths["project_root"] / "splits" / "bad_checksum.json"
    bad_checksum_path.write_text(json.dumps(bad_checksum), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        freezer.validate_core_manifest(
            bad_checksum_path, project_root=paths["project_root"]
        )

    overlapping = copy.deepcopy(manifest)
    overlapping["splits"]["validation"] = [
        overlapping["splits"]["train"][0]
    ]
    overlapping_path = paths["project_root"] / "splits" / "overlapping.json"
    overlapping_path.write_text(json.dumps(overlapping), encoding="utf-8")
    with pytest.raises(ValueError, match="overlaps"):
        freezer.validate_core_manifest(
            overlapping_path, project_root=paths["project_root"]
        )


def test_freezer_binds_generation_summary_bytes(
    tmp_path: Path,
) -> None:
    paths = _build_tiny_finalized_core(tmp_path)
    summary = _read_json(paths["summary"])
    summary["workers"] = 99
    _write_json(paths["summary"], summary)
    with pytest.raises(ValueError, match="generation-summary checksum"):
        freezer.validate_core_manifest(
            paths["manifest"], project_root=paths["project_root"]
        )


@pytest.mark.parametrize("target", ["core_code", "core_provenance", "summary_code"])
def test_freezer_binds_plan_core_summary_and_current_environment(
    tmp_path: Path,
    target: str,
) -> None:
    paths = _build_tiny_finalized_core(tmp_path)
    manifest = _read_json(paths["manifest"])
    summary = _read_json(paths["summary"])
    if target == "core_code":
        manifest["code_sha256"] = {"tampered": "0" * 64}
        _write_json(paths["manifest"], manifest)
        message = "code_sha256 mismatch"
    elif target == "core_provenance":
        manifest["provenance"]["python_version"] = "0.0"
        _write_json(paths["manifest"], manifest)
        message = "provenance mismatch"
    else:
        summary["code_sha256"] = {"tampered": "0" * 64}
        _rewrite_summary_and_manifest(paths, summary, manifest)
        message = "summary/pre-label plan code_sha256 mismatch"
    with pytest.raises(ValueError, match=message):
        freezer.validate_core_manifest(
            paths["manifest"], project_root=paths["project_root"]
        )


@pytest.mark.parametrize("drift", ["code", "config"])
def test_freezer_rejects_current_code_or_config_drift(
    tmp_path: Path,
    drift: str,
) -> None:
    paths = _build_tiny_finalized_core(tmp_path)
    if drift == "code":
        code_path = paths["project_root"] / "scripts" / "generate_p4_2d.py"
        code_path.write_bytes(code_path.read_bytes() + b"\n# drift\n")
        message = "scientific design differs"
    else:
        paths["config"].write_bytes(
            paths["config"].read_bytes() + b"\n# drift\n"
        )
        message = "authoritative-config checksum"
    with pytest.raises(ValueError, match=message):
        freezer.validate_core_manifest(
            paths["manifest"], project_root=paths["project_root"]
        )


@pytest.mark.parametrize(
    ("diagnostic", "threshold"),
    list(p4.ACCEPTANCE_DIAGNOSTIC_THRESHOLDS),
)
def test_freezer_recomputes_every_row_acceptance_diagnostic(
    tmp_path: Path,
    diagnostic: str,
    threshold: str,
) -> None:
    paths = _build_tiny_finalized_core(tmp_path)
    manifest = _read_json(paths["manifest"])
    rows = [
        json.loads(line)
        for line in paths["metadata"].read_text(encoding="utf-8").splitlines()
    ]
    acceptance = p4.load_config(paths["config"])["acceptance"]
    rows[0]["solver"][diagnostic] = float(
        np.nextafter(float(acceptance[threshold]), np.inf)
    )
    paths["metadata"].write_bytes(
        b"".join(p4._canonical_json_bytes(row) for row in rows)
    )
    manifest["metadata_sha256"] = p4._sha256_file(paths["metadata"])
    _write_json(paths["manifest"], manifest)
    with pytest.raises(ValueError, match="fails independently recomputed"):
        freezer.validate_core_manifest(
            paths["manifest"], project_root=paths["project_root"]
        )


def test_freezer_recomputes_summary_acceptance_aggregates(
    tmp_path: Path,
) -> None:
    paths = _build_tiny_finalized_core(tmp_path)
    manifest = _read_json(paths["manifest"])
    summary = _read_json(paths["summary"])
    summary["maximum_abs_energy_residual_W_m3"] = 1.0e-12
    _rewrite_summary_and_manifest(paths, summary, manifest)
    with pytest.raises(ValueError, match="summary aggregate"):
        freezer.validate_core_manifest(
            paths["manifest"], project_root=paths["project_root"]
        )


def test_freezer_independently_rejects_nonzero_tool_alpha(
    tmp_path: Path,
) -> None:
    paths = _build_tiny_finalized_core(tmp_path)
    manifest = _read_json(paths["manifest"])
    summary = _read_json(paths["summary"])
    plan = _read_json(paths["plan"])
    rows = [
        json.loads(line)
        for line in paths["metadata"].read_text(encoding="utf-8").splitlines()
    ]
    alpha_path = paths["artifact_root"] / "alpha.npy"
    alpha = np.load(alpha_path, allow_pickle=False)
    mask = np.load(
        paths["artifact_root"] / "composite_mask.npy", allow_pickle=False
    )
    tool_index = tuple(np.argwhere(~mask)[0])
    alpha[(0, 1, *tool_index)] = 0.01
    np.save(alpha_path, alpha, allow_pickle=False)
    case_alpha_hash = p4._sha256_array(np.asarray(alpha[0]))
    case_key = plan["cases"][0]["definition"]["case_key"]
    manifest["case_artifact_hashes"][case_key]["output_slice_sha256"][
        "alpha"
    ] = case_alpha_hash
    rows[0]["output_slice_sha256"]["alpha"] = case_alpha_hash
    rows[0]["solver"]["tool_alpha_nonzero_count"] = 1
    paths["metadata"].write_bytes(
        b"".join(p4._canonical_json_bytes(row) for row in rows)
    )
    manifest["metadata_sha256"] = p4._sha256_file(paths["metadata"])
    alpha_hash = p4._sha256_file(alpha_path)
    manifest["array_sha256"]["alpha"] = alpha_hash
    summary["array_sha256"]["alpha"] = alpha_hash
    _rewrite_summary_and_manifest(paths, summary, manifest)
    with pytest.raises(ValueError, match="fails independently recomputed"):
        freezer.validate_core_manifest(
            paths["manifest"], project_root=paths["project_root"]
        )


@pytest.mark.parametrize(
    "array_name",
    [
        "air_temperature_K",
        "top_h_W_m2_K",
        "time_s",
        "x_m",
        "z_m",
        "composite_mask",
    ],
)
def test_freezer_recreates_auxiliary_arrays_even_after_resealing(
    tmp_path: Path,
    array_name: str,
) -> None:
    paths = _build_tiny_finalized_core(tmp_path)
    manifest = _read_json(paths["manifest"])
    summary = _read_json(paths["summary"])
    array_path = paths["artifact_root"] / f"{array_name}.npy"
    array = np.load(array_path, allow_pickle=False)
    if array.dtype == np.dtype("bool"):
        array.flat[0] = not bool(array.flat[0])
    else:
        array.flat[0] += 0.123
    np.save(array_path, array, allow_pickle=False)
    new_hash = p4._sha256_file(array_path)
    manifest["array_sha256"][array_name] = new_hash
    summary["array_sha256"][array_name] = new_hash
    _rewrite_summary_and_manifest(paths, summary, manifest)
    with pytest.raises(ValueError, match="auxiliary array content mismatch"):
        freezer.validate_core_manifest(
            paths["manifest"], project_root=paths["project_root"]
        )


@pytest.mark.parametrize(
    ("array_name", "mutation", "message"),
    [
        ("x_m", "dtype", "dtype mismatch"),
        ("time_s", "shape", "shape mismatch"),
    ],
)
def test_freezer_enforces_auxiliary_shape_and_dtype(
    tmp_path: Path,
    array_name: str,
    mutation: str,
    message: str,
) -> None:
    paths = _build_tiny_finalized_core(tmp_path)
    manifest = _read_json(paths["manifest"])
    summary = _read_json(paths["summary"])
    array_path = paths["artifact_root"] / f"{array_name}.npy"
    array = np.load(array_path, allow_pickle=False)
    changed = array.astype(np.float32) if mutation == "dtype" else array[:-1]
    np.save(array_path, changed, allow_pickle=False)
    new_hash = p4._sha256_file(array_path)
    manifest["array_sha256"][array_name] = new_hash
    summary["array_sha256"][array_name] = new_hash
    _rewrite_summary_and_manifest(paths, summary, manifest)
    with pytest.raises(ValueError, match=message):
        freezer.validate_core_manifest(
            paths["manifest"], project_root=paths["project_root"]
        )


@pytest.mark.parametrize("mutation", ["zero_budget", "split_count"])
def test_freezer_enforces_exact_positive_budgets_and_split_counts(
    tmp_path: Path,
    mutation: str,
) -> None:
    paths = _build_tiny_finalized_core(tmp_path)
    manifest = _read_json(paths["manifest"])
    if mutation == "zero_budget":
        manifest["nested_training_budgets"] = [0, 2]
        message = "positive and exactly match"
    else:
        moved = manifest["splits"]["validation"].pop()
        manifest["splits"]["train"].append(moved)
        message = "count mismatch"
    _write_json(paths["manifest"], manifest)
    with pytest.raises(ValueError, match=message):
        freezer.validate_core_manifest(
            paths["manifest"], project_root=paths["project_root"]
        )


@pytest.mark.parametrize(
    "bad_path",
    [
        r"splits\p4_2d_core_v1_plan.json",
        "splits/../splits/p4_2d_core_v1_plan.json",
        "C:/absolute/p4_2d_core_v1_plan.json",
        "c:/absolute/p4_2d_core_v1_plan.json",
        "C:relative/p4_2d_core_v1_plan.json",
        "C:",
        "/absolute/p4_2d_core_v1_plan.json",
        "//server/share/p4_2d_core_v1_plan.json",
        r"\\server\share\p4_2d_core_v1_plan.json",
        "./splits/p4_2d_core_v1_plan.json",
        "splits//p4_2d_core_v1_plan.json",
    ],
)
def test_freezer_rejects_noncanonical_core_paths(
    tmp_path: Path,
    bad_path: str,
) -> None:
    paths = _build_tiny_finalized_core(tmp_path)
    manifest = _read_json(paths["manifest"])
    manifest["plan_path"] = bad_path
    _write_json(paths["manifest"], manifest)
    with pytest.raises(ValueError, match="repository-relative|POSIX|normalized"):
        freezer.validate_core_manifest(
            paths["manifest"], project_root=paths["project_root"]
        )
