from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from cdcureno.solvers import SolverDiagnostics2D


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import generate_p4_2d as p4  # noqa: E402


def test_core_plan_is_deterministic_and_uses_independent_named_dimensions() -> None:
    config = p4.load_config()
    first = p4.build_case_plan("core", config=config)
    second = p4.build_case_plan("core", config=config)
    assert p4._canonical_json_bytes(first) == p4._canonical_json_bytes(second)
    assert first["design_dimensions"] == list(p4.REQUIRED_DESIGN_DIMENSIONS)
    assert len(set(first["design_dimensions"])) == len(first["design_dimensions"])

    definitions = [entry["definition"] for entry in first["cases"]]
    for definition in definitions:
        assert set(definition["raw_design_coordinates"]) == set(
            p4.REQUIRED_DESIGN_DIMENSIONS
        )
        assert definition["realized_top_h_mean_W_m2_K"] == pytest.approx(
            definition["requested_top_h_center_W_m2_K"], abs=1.0e-12
        )
        if definition["difficulty_family"] in {"F0", "F2_left", "F2_both"}:
            assert definition["realized_top_h_amplitude_W_m2_K"] == 0.0
            assert definition["realized_top_h_frequency_per_width"] == 0

    single = [
        item["raw_design_coordinates"]["cycle_p0"]
        for item in definitions
        if item["cycle_family"] == "single_hold"
    ]
    double = [
        item["raw_design_coordinates"]["cycle_p0"]
        for item in definitions
        if item["cycle_family"] == "two_hold"
    ]
    assert max(single) > 0.35
    assert min(double) < 0.35

    for budget in (8, 16, 32):
        subset = definitions[:budget]
        assert {item["cycle_family"] for item in subset} == {
            "single_hold",
            "two_hold",
        }
        assert {item["difficulty_family"] for item in subset} == {
            "F0",
            "F1_smooth",
            "F1_piecewise",
            "F2_left",
            "F2_both",
        }


def test_piecewise_pattern_records_true_cycles_per_width() -> None:
    config = p4.load_config()
    plan = p4.build_case_plan("core", config=config)
    piecewise = next(
        entry["definition"]
        for entry in plan["cases"]
        if entry["definition"]["difficulty_family"] == "F1_piecewise"
    )
    top = np.asarray(piecewise["top_h_W_m2_K"])
    center = float(piecewise["requested_top_h_center_W_m2_K"])
    states = top > center
    circular_transition_count = int(np.count_nonzero(states != np.roll(states, 1)))
    assert circular_transition_count == (
        2 * int(piecewise["requested_top_h_frequency_per_width"])
    )
    assert 0.5 * np.ptp(top) == pytest.approx(
        piecewise["requested_top_h_amplitude_W_m2_K"]
    )


def test_staging_resume_does_not_truncate_completed_case(tmp_path: Path) -> None:
    config = p4.load_config()
    plan = p4.build_case_plan("debug", config=config)
    staging = tmp_path / "staging"
    arrays = p4._prepare_staging(plan, config, staging)
    definition = plan["cases"][0]["definition"]
    shape = arrays["temperature_K"].shape[1:]
    result = {
        "case_id": 0,
        "status": "passed",
        "failure": None,
        "temperature_K": np.full(shape, 333.0, dtype=np.float32),
        "alpha": np.full(shape, 0.2, dtype=np.float32),
        "solver": {
            "finite_fields": True,
            "alpha_bound_violation_count": 0,
            "alpha_monotonicity_violation_count": 0,
            "maximum_abs_energy_residual_W_m3": 0.0,
            "maximum_relative_global_energy_residual": 0.0,
        },
        "total_case_wall_seconds": 0.1,
    }
    completion = p4._commit_case_result(
        definition=definition,
        result=result,
        arrays=arrays,
        staging_root=staging,
    )
    expected_hash = completion["output_slice_sha256"]["temperature_K"]
    del arrays

    resumed_arrays = p4._prepare_staging(plan, config, staging)
    completed, pending = p4._load_completions(
        plan, resumed_arrays, staging, retry_failed=False
    )
    assert set(completed) == {0}
    assert len(pending) == 7
    assert p4._sha256_array(
        np.asarray(resumed_arrays["temperature_K"][0])
    ) == expected_hash


def test_solver_exception_has_structured_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = p4.load_config()
    definition = p4.build_case_plan("debug", config=config)["cases"][0][
        "definition"
    ]

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("intentional test failure")

    monkeypatch.setattr(p4, "simulate_cure_2d", fail)
    result = p4._run_case(definition, config)
    assert result["status"] == "solver_exception"
    assert result["failure"]["kind"] == "solver_exception"
    assert result["failure"]["stage"] == "simulate_cure_2d"
    assert result["failure"]["exception_type"] == "RuntimeError"
    assert "intentional test failure" in result["failure"]["message"]
    assert "Traceback" in result["failure"]["traceback"]


def test_retry_archives_prior_failure_record(tmp_path: Path) -> None:
    config = p4.load_config()
    plan = p4.build_case_plan("debug", config=config)
    staging = tmp_path / "staging"
    arrays = p4._prepare_staging(plan, config, staging)
    definition = plan["cases"][0]["definition"]
    failure = {
        "case_id": 0,
        "status": "worker_exception",
        "failure": {
            "kind": "worker_exception",
            "stage": "future_result",
            "message": "transient",
        },
        "temperature_K": None,
        "alpha": None,
        "solver": {"finite_fields": False},
        "total_case_wall_seconds": 0.0,
    }
    p4._commit_case_result(
        definition=definition,
        result=failure,
        arrays=arrays,
        staging_root=staging,
    )
    completed, pending = p4._load_completions(
        plan, arrays, staging, retry_failed=True
    )
    assert 0 not in completed
    assert pending[0]["case_id"] == 0
    history = list((staging / "case_failure_history" / "0000").glob("*.json"))
    assert len(history) == 1
    assert json.loads(history[0].read_text())["status"] == "worker_exception"

    shape = arrays["temperature_K"].shape[1:]
    passed = {
        "case_id": 0,
        "status": "passed",
        "failure": None,
        "temperature_K": np.full(shape, 300.0, dtype=np.float32),
        "alpha": np.zeros(shape, dtype=np.float32),
        "solver": {
            "finite_fields": True,
            "alpha_bound_violation_count": 0,
            "alpha_monotonicity_violation_count": 0,
            "maximum_abs_energy_residual_W_m3": 0.0,
            "maximum_relative_global_energy_residual": 0.0,
        },
        "total_case_wall_seconds": 0.1,
    }
    p4._commit_case_result(
        definition=definition,
        result=passed,
        arrays=arrays,
        staging_root=staging,
    )
    assert len(list((staging / "case_failure_history" / "0000").glob("*.json"))) == 1


def test_nonfinite_diagnostic_is_a_serializable_acceptance_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = p4.load_config()
    definition = p4.build_case_plan("debug", config=config)["cases"][0][
        "definition"
    ]
    shape = (
        len(p4._time_axis(config)),
        10,
        10,
    )
    fake = SimpleNamespace(
        temperature_K=np.full(shape, 300.0),
        alpha=np.zeros(shape),
        diagnostics=SolverDiagnostics2D(
            maximum_abs_energy_residual_W_m3=np.nan,
            maximum_relative_global_energy_residual=0.0,
            maximum_interface_flux_imbalance_W=0.0,
            maximum_temperature_interface_jump_K=0.0,
            maximum_robin_flux_imbalance_W=0.0,
            all_coupling_steps_converged=True,
            maximum_converged_coupling_update_K=0.0,
            substeps=1,
            maximum_coupling_iterations=1,
            wall_seconds=0.0,
        ),
    )
    monkeypatch.setattr(p4, "simulate_cure_2d", lambda *args, **kwargs: fake)
    result = p4._run_case(definition, config)
    assert result["status"] == "failed_acceptance"
    assert result["failure"]["reasons"][0]["observed"] == "nan"
    p4._pretty_json_bytes(
        {"failure": result["failure"], "solver": result["solver"]}
    )


def test_immutable_write_accepts_replay_but_rejects_replacement(
    tmp_path: Path,
) -> None:
    path = tmp_path / "frozen.json"
    assert p4._write_immutable_bytes(path, b"one\n") == "created"
    assert p4._write_immutable_bytes(path, b"one\n") == "verified_existing"
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        p4._write_immutable_bytes(path, b"two\n")
    assert path.read_bytes() == b"one\n"


def test_plan_validation_rebuild_rejects_resigned_scientific_tamper() -> None:
    config = p4.load_config()
    plan = p4.build_case_plan("core", config=config)
    tampered = copy.deepcopy(plan)
    entry = tampered["cases"][0]
    entry["definition"]["reaction_enthalpy_scale"] += 0.001
    entry["case_definition_sha256"] = p4._sha256_bytes(
        p4._canonical_json_bytes(entry["definition"])
    )
    tampered["plan_sha256"] = p4._sha256_bytes(
        p4._canonical_json_bytes(
            {
                key: value
                for key, value in tampered.items()
                if key != "plan_sha256"
            }
        )
    )
    with pytest.raises(ValueError, match="scientific design differs"):
        p4._validate_plan(
            tampered,
            tier="core",
            config=config,
            config_path=p4.DEFAULT_CONFIG,
        )


def test_plan_validation_allows_head_to_advance_after_plan_freeze() -> None:
    config = p4.load_config()
    plan = p4.build_case_plan("core", config=config)
    plan["provenance"]["git_commit_sha"] = "b" * 40
    plan["plan_sha256"] = p4._sha256_bytes(
        p4._canonical_json_bytes(
            {key: value for key, value in plan.items() if key != "plan_sha256"}
        )
    )
    p4._validate_plan(
        plan,
        tier="core",
        config=config,
        config_path=p4.DEFAULT_CONFIG,
    )


def test_acceptance_reasons_fail_closed_for_every_new_invariant() -> None:
    config = p4.load_config()
    acceptance = config["acceptance"]
    diagnostics = {
        diagnostic: 0.0
        for diagnostic, _ in p4.ACCEPTANCE_DIAGNOSTIC_THRESHOLDS
    }
    diagnostics["all_coupling_steps_converged"] = True
    assert not p4._acceptance_reasons(
        finite=True,
        alpha_bound_violations=0,
        alpha_monotonicity_violations=0,
        tool_alpha_nonzero_count=0,
        diagnostics=diagnostics,
        acceptance=acceptance,
    )

    for diagnostic, threshold in p4.ACCEPTANCE_DIAGNOSTIC_THRESHOLDS:
        failing = dict(diagnostics)
        failing[diagnostic] = np.nextafter(
            float(acceptance[threshold]), np.inf
        )
        reasons = p4._acceptance_reasons(
            finite=True,
            alpha_bound_violations=0,
            alpha_monotonicity_violations=0,
            tool_alpha_nonzero_count=0,
            diagnostics=failing,
            acceptance=acceptance,
        )
        assert any(item["code"] == diagnostic for item in reasons)

        missing = dict(diagnostics)
        del missing[diagnostic]
        reasons = p4._acceptance_reasons(
            finite=True,
            alpha_bound_violations=0,
            alpha_monotonicity_violations=0,
            tool_alpha_nonzero_count=0,
            diagnostics=missing,
            acceptance=acceptance,
        )
        assert any(item["code"] == diagnostic for item in reasons)

    tool_reasons = p4._acceptance_reasons(
        finite=True,
        alpha_bound_violations=0,
        alpha_monotonicity_violations=0,
        tool_alpha_nonzero_count=1,
        diagnostics=diagnostics,
        acceptance=acceptance,
    )
    assert any(item["code"] == "tool_alpha_nonzero" for item in tool_reasons)
    unconverged = dict(diagnostics)
    unconverged["all_coupling_steps_converged"] = False
    coupling_reasons = p4._acceptance_reasons(
        finite=True,
        alpha_bound_violations=0,
        alpha_monotonicity_violations=0,
        tool_alpha_nonzero_count=0,
        diagnostics=unconverged,
        acceptance=acceptance,
    )
    assert any(item["code"] == "coupling_not_converged" for item in coupling_reasons)


def test_run_case_rejects_nonzero_tool_alpha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = p4.load_config()
    definition = p4.build_case_plan("debug", config=config)["cases"][0][
        "definition"
    ]
    grid = p4._condition_grid(definition, config)
    shape = (len(p4._time_axis(config)), *grid.shape)
    alpha = np.zeros(shape, dtype=np.float64)
    tool_index = tuple(np.argwhere(~grid.composite_mask)[0])
    alpha[(0, *tool_index)] = 0.1
    fake = SimpleNamespace(
        temperature_K=np.full(shape, 300.0),
        alpha=alpha,
        diagnostics=SolverDiagnostics2D(
            maximum_abs_energy_residual_W_m3=0.0,
            maximum_relative_global_energy_residual=0.0,
            maximum_interface_flux_imbalance_W=0.0,
            maximum_temperature_interface_jump_K=0.0,
            maximum_robin_flux_imbalance_W=0.0,
            all_coupling_steps_converged=True,
            maximum_converged_coupling_update_K=0.0,
            substeps=1,
            maximum_coupling_iterations=1,
            wall_seconds=0.0,
        ),
    )
    monkeypatch.setattr(p4, "simulate_cure_2d", lambda *args, **kwargs: fake)
    result = p4._run_case(definition, config)
    assert result["status"] == "failed_acceptance"
    assert any(
        item["code"] == "tool_alpha_nonzero"
        for item in result["failure"]["reasons"]
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("budget", 0, "strictly positive"),
        ("budget", True, "only integers"),
        ("finite_field_required", False, "invariant acceptance"),
        (
            "all_coupling_steps_converged_required",
            False,
            "invariant acceptance",
        ),
        ("tool_alpha_nonzero_count_max", 1, "invariant acceptance"),
    ],
)
def test_config_rejects_weakened_core_acceptance_or_budget(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    payload = yaml.safe_load(p4.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    if field == "budget":
        payload["tiers"]["core"]["nested_training_budgets"][0] = value
    else:
        payload["acceptance"][field] = value
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        p4.load_config(path)


def test_publish_revalidates_internal_manifest_bindings(tmp_path: Path) -> None:
    final_root = tmp_path / "final"
    final_root.mkdir()
    metadata_bytes = b'{"case_id":0}\n'
    common = {
        "dataset_id": "fixture",
        "tier": "core",
        "tier_role": "canonical",
        "case_count": 1,
        "array_shape_case_time_z_x": [1, 2, 1, 1],
        "plan_sha256": "a" * 64,
        "config_path": "configs/data/fixture.yaml",
        "config_sha256": "b" * 64,
        "code_sha256": {"solver.py": "c" * 64},
        "provenance": {"git_commit_sha": "d" * 40},
        "array_sha256": {"temperature_K": "e" * 64},
    }
    summary = dict(common)
    summary_bytes = p4._pretty_json_bytes(summary)
    manifest = {
        **common,
        "metadata_sha256": p4._sha256_bytes(metadata_bytes),
        "generation_summary_sha256": p4._sha256_bytes(summary_bytes),
    }
    (final_root / "case_metadata.jsonl").write_bytes(metadata_bytes)
    (final_root / "manifest.json").write_bytes(
        p4._pretty_json_bytes(manifest)
    )
    (final_root / "generation_summary.json").write_bytes(summary_bytes)

    published = p4._publish_payloads(
        final_root=final_root,
        metadata_path=tmp_path / "published" / "metadata.jsonl",
        manifest_path=tmp_path / "published" / "manifest.json",
        summary_path=tmp_path / "published" / "summary.json",
    )
    assert published == summary

    resigned_summary = {**summary, "case_count": 2}
    resigned_summary_bytes = p4._pretty_json_bytes(resigned_summary)
    resigned_manifest = {
        **manifest,
        "generation_summary_sha256": p4._sha256_bytes(
            resigned_summary_bytes
        ),
    }
    (final_root / "manifest.json").write_bytes(
        p4._pretty_json_bytes(resigned_manifest)
    )
    (final_root / "generation_summary.json").write_bytes(
        resigned_summary_bytes
    )
    with pytest.raises(ValueError, match="summary/manifest field mismatch"):
        p4._publish_payloads(
            final_root=final_root,
            metadata_path=tmp_path / "rejected" / "metadata.jsonl",
            manifest_path=tmp_path / "rejected" / "manifest.json",
            summary_path=tmp_path / "rejected" / "summary.json",
        )
