from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cdcureno.evaluation.p5_pilot import (
    ALPHA_COLUMN,
    FROZEN_INPUTS,
    PEAK_COLUMN,
    PER_CASE_COLUMNS,
    PRIMARY_COLUMN,
    PilotRunEvidence,
    decide_pilot_gate,
    recompute_validation_cases,
)


def _evidence(
    method: str,
    budget: int,
    *,
    temperature: float,
    alpha: float,
    peak: float,
) -> PilotRunEvidence:
    rows = []
    for case_id in range(256, 288):
        row = {"split": "validation", "case_id": case_id}
        row.update({column: 1.0 for column in PER_CASE_COLUMNS})
        row[PRIMARY_COLUMN] = temperature
        row[ALPHA_COLUMN] = alpha
        row[PEAK_COLUMN] = peak
        rows.append(row)
    frame = pd.DataFrame(rows)
    model = {
        "family": "axis_factorized_2d",
        "parameter_count": 100,
        "trainable_parameter_count": 100,
        "transfer_stage": "T2",
    }
    parameter_groups = {"total_parameter_count": 100}
    training = {
        "configured_epochs": 120,
        "micro_batch_size": 2,
        "gradient_accumulation_steps": 2,
        "effective_batch_size": 4,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "mixed_precision": False,
        "device": "cuda:0",
    }
    access = {
        "train": list(range(budget)),
        "validation": list(range(256, 288)),
        "id_test": [],
        "ood": [],
    }
    metrics = {
        "run_id": f"{method}-{budget}",
        "git_sha": "a" * 40,
        "implementation_sha256": "b" * 64,
        "model": model,
        "parameter_groups": parameter_groups,
        "loss_weights": {"temperature_relative_l2": 1.0},
        "training": training,
        "resource_profile": {"passed": True},
        "input_checksums": {
            name: {"sha256": value[1]}
            for name, value in FROZEN_INPUTS.items()
        },
        "target_label_access_audit": access,
        "selection": {"target_test_or_ood_labels_used": False},
        "restriction_validation": {
            "passed": True,
            "contract_checks": {"frozen_thresholds": True},
        },
        "initialization": {
            "method": method,
            "source_checkpoint_weights_loaded": (
                method == "restriction_transfer_ffno"
            ),
            "inflation_verification_passed": (
                method == "restriction_transfer_ffno"
            ),
            "inflated_checkpoint_sha256": FROZEN_INPUTS[
                "inflated_checkpoint"
            ][1],
        },
    }
    resolved = {
        "scientific_config": {
            "method": method,
            "run_id": f"{method}-{budget}",
            "label_budget": budget,
            "resume": False,
            "epochs": 120,
        }
    }
    targets = {
        "temperature_target_K": np.zeros((1,), dtype=np.float32),
        "alpha_target": np.zeros((1,), dtype=np.float32),
        "composite_mask": np.ones((1,), dtype=bool),
        "x_m": np.zeros((1,), dtype=np.float64),
        "z_m": np.zeros((1,), dtype=np.float64),
    }
    return PilotRunEvidence(
        run_dir=Path(f"/{method}-{budget}"),
        metrics=metrics,
        resolved_config=resolved,
        cases=frame,
        targets=targets,
        recomputed_summary={},
    )


def _restriction_gate() -> dict[str, object]:
    return {
        "schema_version": 2,
        "phase": "P5",
        "gate": "canonical_inflation_and_source_to_target_restriction",
        "dry_run": False,
        "passed": True,
        "uses_target_temperature_or_cure_labels": False,
        "source_checkpoint_sha256": FROZEN_INPUTS["source_checkpoint"][1],
        "target_checkpoint_sha256": FROZEN_INPUTS[
            "inflated_checkpoint"
        ][1],
        "target_config_sha256": FROZEN_INPUTS["target_model_config"][1],
        "p4_id_split_sha256": FROZEN_INPUTS["target_id_manifest"][1],
        "actual_p3_validation": {"passed": True},
        "actual_p4_validation_f0": {"passed": True},
        "inflation_integrity": {
            "schema_version": 1,
            "passed": True,
            "deterministic_seed": 20260726,
            "state_tensor_count": 61,
            "verification": "deterministic_full_state_recreation",
            "source_checkpoint_sha256": FROZEN_INPUTS[
                "source_checkpoint"
            ][1],
            "target_config_sha256": FROZEN_INPUTS[
                "target_model_config"
            ][1],
            "checks": {"full_state_recreated": True},
        },
    }


def test_p5_pilot_gate_applies_all_preregistered_conditions() -> None:
    runs = {
        ("scratch_ffno", 8): _evidence(
            "scratch_ffno", 8, temperature=0.10, alpha=0.05, peak=2.0
        ),
        ("restriction_transfer_ffno", 8): _evidence(
            "restriction_transfer_ffno",
            8,
            temperature=0.09,
            alpha=0.052,
            peak=2.1,
        ),
        ("scratch_ffno", 16): _evidence(
            "scratch_ffno", 16, temperature=0.08, alpha=0.04, peak=1.5
        ),
        ("restriction_transfer_ffno", 16): _evidence(
            "restriction_transfer_ffno",
            16,
            temperature=0.07,
            alpha=0.042,
            peak=1.6,
        ),
    }
    report = decide_pilot_gate(
        runs,
        restriction_inflation_gate=_restriction_gate(),
    )
    assert report["passed"]
    assert all(report["checks"].values())


def test_p5_pilot_gate_preserves_a_negative_result() -> None:
    runs = {
        ("scratch_ffno", 8): _evidence(
            "scratch_ffno", 8, temperature=0.10, alpha=0.05, peak=2.0
        ),
        ("restriction_transfer_ffno", 8): _evidence(
            "restriction_transfer_ffno",
            8,
            temperature=0.11,
            alpha=0.05,
            peak=2.0,
        ),
        ("scratch_ffno", 16): _evidence(
            "scratch_ffno", 16, temperature=0.08, alpha=0.04, peak=1.5
        ),
        ("restriction_transfer_ffno", 16): _evidence(
            "restriction_transfer_ffno",
            16,
            temperature=0.0796,
            alpha=0.04,
            peak=1.5,
        ),
    }
    report = decide_pilot_gate(
        runs,
        restriction_inflation_gate=_restriction_gate(),
    )
    assert not report["passed"]
    assert report["decision"] == "failed_preserved"
    assert not report["checks"]["transfer_temperature_lower_at_budget_8"]
    assert not report["checks"][
        "budget_16_relative_improvement_at_least_1_percent"
    ]


def test_recompute_validation_cases_accepts_static_2d_material_mask(
    tmp_path: Path,
) -> None:
    case_ids = np.arange(256, 288, dtype=np.int64)
    shape = (len(case_ids), 3, 2, 3)
    temperature_target = np.full(shape, 300.0, dtype=np.float32)
    alpha_target = np.full(shape, 0.5, dtype=np.float32)
    predictions_path = tmp_path / "validation_best.npz"
    np.savez_compressed(
        predictions_path,
        case_ids=case_ids,
        temperature_prediction_K=temperature_target.copy(),
        temperature_target_K=temperature_target,
        alpha_prediction=alpha_target.copy(),
        alpha_target=alpha_target,
        composite_mask=np.asarray(
            [[False, False, False], [True, True, True]],
            dtype=np.bool_,
        ),
        x_m=np.asarray([0.0, 0.1, 0.2], dtype=np.float64),
        z_m=np.asarray([0.0, 0.1], dtype=np.float64),
    )

    frame, targets = recompute_validation_cases(predictions_path)

    assert len(frame) == len(case_ids)
    assert targets["composite_mask"].shape == (2, 3)
    assert np.allclose(frame[list(PER_CASE_COLUMNS)].to_numpy(), 0.0)
