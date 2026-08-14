import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import cdcureno.training.causal_source_1d as causal_training
from cdcureno.legacy.audit import sha256_file
from cdcureno.models.joint_operators import (
    CausalFactorizedOperator,
    parameter_count,
)
from cdcureno.training.causal_source_1d import (
    COMPUTE_ACCOUNTING_FIELDS,
    CausalSourceTrainConfig,
    causal_source_compute_timing_from_history,
    train_causal_source,
    validate_causal_source_compute_evidence,
)


def _write_synthetic_source(root: Path) -> tuple[Path, Path]:
    case_count = 8
    time_count = 7
    space_count = 7
    time = np.linspace(0.0, 1.0, time_count, dtype=np.float32)
    position = np.linspace(0.0, 1.0, space_count, dtype=np.float32)
    air = (
        293.0
        + 45.0 * time[None, :]
        + np.arange(case_count, dtype=np.float32)[:, None]
    )
    temperature = (
        air[:, :, None]
        + 8.0 * time[None, :, None] * position[None, None, :]
    )
    composite_mask = np.zeros((case_count, space_count), dtype=np.float32)
    composite_mask[:, 3:] = 1.0
    alpha = (
        composite_mask[:, None, :]
        * (0.05 + 0.45 * time[None, :, None])
    ).astype(np.float32)
    coarse_temperature = (temperature - 0.75).astype(np.float32)
    coarse_alpha = np.maximum(alpha - 0.01, 0.0).astype(np.float32)
    signed_distance = np.broadcast_to(
        position[None, :] - 0.4, (case_count, space_count)
    ).astype(np.float32)
    parameters = np.tile(
        np.array(
            [0.02, 0.03, 70.0, 120.0, 1.0, 1.0], dtype=np.float32
        ),
        (case_count, 1),
    )
    data_path = root / "synthetic_source.npz"
    np.savez_compressed(
        data_path,
        air_temperature_K=air.astype(np.float32),
        temperature_K=temperature.astype(np.float32),
        coarse_temperature_K=coarse_temperature,
        coarse_alpha=coarse_alpha,
        alpha=alpha,
        composite_mask=composite_mask,
        signed_distance_fraction=signed_distance,
        parameters=parameters,
        case_ids=np.arange(case_count, dtype=np.int64),
        family_ids=np.array([0, 0, 0, 0, 1, 2, 2, 3], dtype=np.int64),
    )
    manifest = {
        "schema_version": 1,
        "dataset_id": "synthetic_causal_source",
        "array_sha256": sha256_file(data_path),
        "splits": {
            "train": [0, 1, 2],
            "validation": [3],
            "in_family_test": [4],
            "held_out_family_test": {
                "smart_cure": [5, 6],
                "three_hold": [7],
            },
        },
    }
    split_path = root / "synthetic_split.json"
    split_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return data_path, split_path


def _small_training_config(
    tmp_path: Path,
    *,
    run_id: str,
    seed: int,
    epochs: int = 3,
) -> CausalSourceTrainConfig:
    data_path, split_path = _write_synthetic_source(tmp_path)
    project_root = Path(__file__).resolve().parents[2]
    expected_parameters = parameter_count(
        CausalFactorizedOperator(
            input_channels=14,
            width=4,
            depth=2,
            modes_space=3,
        )
    )
    return CausalSourceTrainConfig(
        data_path=data_path,
        split_manifest=split_path,
        output_root=tmp_path / "runs",
        project_root=project_root,
        run_id=run_id,
        expected_data_sha256=sha256_file(data_path),
        expected_split_sha256=sha256_file(split_path),
        seed=seed,
        time_stride=1,
        width=4,
        depth=2,
        modes_space=3,
        expected_parameter_count=expected_parameters,
        epochs=epochs,
        minimum_epochs=epochs,
        early_stopping_patience=epochs,
        batch_size=2,
        device="cpu",
        num_threads=1,
        causality_case_count=1,
    )


def test_synthetic_causal_source_run_pauses_resumes_and_audits(
    tmp_path: Path,
) -> None:
    data_path, split_path = _write_synthetic_source(tmp_path)
    project_root = Path(__file__).resolve().parents[2]
    expected_parameters = parameter_count(
        CausalFactorizedOperator(
            input_channels=14,
            width=4,
            depth=2,
            modes_space=3,
        )
    )
    config = CausalSourceTrainConfig(
        data_path=data_path,
        split_manifest=split_path,
        output_root=tmp_path / "runs",
        project_root=project_root,
        run_id="synthetic-causal-source",
        expected_data_sha256=sha256_file(data_path),
        expected_split_sha256=sha256_file(split_path),
        seed=7,
        time_stride=1,
        width=4,
        depth=2,
        modes_space=3,
        expected_parameter_count=expected_parameters,
        epochs=2,
        minimum_epochs=2,
        early_stopping_patience=2,
        batch_size=2,
        device="cpu",
        num_threads=1,
        causality_case_count=1,
        causality_cutoff_fractions=(0.25, 0.5, 0.75),
    )

    paused = train_causal_source(config, session_epoch_limit=1)
    run_dir = config.output_root / "synthetic-causal-source"
    assert paused == {
        "status": "paused",
        "run_id": "synthetic-causal-source",
        "last_epoch": 1,
        "held_out_labels_evaluated": False,
    }
    assert not (run_dir / "metrics_per_case.parquet").exists()
    assert not (run_dir / "DONE").exists()

    metrics = train_causal_source(replace(config, resume=True))

    assert metrics["status"] == "completed"
    assert metrics["selection"]["split"] == "validation"
    assert metrics["selection"]["held_out_labels_used"] is False
    assert metrics["model"]["structurally_causal"] is True
    assert metrics["model"]["temporal_receptive_field"] == 7
    assert metrics["causality_passed"] is True
    assert metrics["recorded_session_count"] == 2
    assert metrics["resume_session_count"] == 1
    assert (run_dir / "DONE").is_file()
    assert not (run_dir / "FAILED").exists()

    for relative in (
        "config_resolved.json",
        "data_checksums.json",
        "implementation_runtime.json",
        "environment.json",
        "git_state.txt",
        "history.parquet",
        "metrics_per_case.parquet",
        "causality.json",
        "metrics.json",
        "checkpoints/best.pt",
        "checkpoints/last.pt",
    ):
        assert (run_dir / relative).is_file(), relative
    history = pd.read_parquet(run_dir / "history.parquet")
    per_case = pd.read_parquet(run_dir / "metrics_per_case.parquet")
    assert history["epoch"].tolist() == [1, 2]
    assert history["optimizer_update_count"].tolist() == [2, 2]
    assert history["candidate_count"].tolist() == [1, 1]
    assert set(history["optimizer_device_synchronization"]) == {
        "not_applicable_cpu"
    }
    assert set(history["candidate_device_synchronization"]) == {
        "not_applicable_cpu"
    }
    assert history["resume_invocation_count"].tolist() == [0, 1]
    assert history["controlled_resume_invocation_count"].tolist() == [0, 1]
    assert history["unobserved_interruption_resume_count"].tolist() == [0, 0]
    assert history["timing_complete"].tolist() == [True, True]
    assert set(per_case["split"]) == {
        "in_family_test",
        "smart_cure",
        "three_hold",
    }
    assert len(per_case) == 4
    checkpoint = torch.load(
        run_dir / "checkpoints" / "last.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["epoch"] == 2
    assert checkpoint["selection_split"] == "validation"
    assert checkpoint["held_out_labels_used_for_selection"] is False
    assert checkpoint["cuda_rng_state_all"] is None
    assert {
        "optimizer",
        "scheduler",
        "generator_state",
        "torch_rng_state",
        "python_rng_state",
        "numpy_rng_state",
        "input_checksums",
        "implementation_runtime_binding",
        "authoritative_history",
        "best_snapshot",
        "compute_timing",
        "compute_accounting",
    }.issubset(checkpoint)
    recomputed = causal_source_compute_timing_from_history(
        history,
        expected_parameter_count=expected_parameters,
        expected_optimizer_updates_per_epoch=2,
        device_type="cpu",
        expected_resume_evidence={
            "resume_invocation_count": 1,
            "controlled_resume_invocation_count": 1,
            "unobserved_interruption_resume_count": 0,
            "unobserved_failed_attempt_overhead_possible": False,
            "timing_complete": True,
        },
    )
    assert checkpoint["compute_timing"] == recomputed
    assert metrics["compute_timing"] == recomputed
    assert metrics["compute_accounting"] == {
        key: recomputed[key] for key in COMPUTE_ACCOUNTING_FIELDS
    }
    done = json.loads((run_dir / "DONE").read_text(encoding="utf-8"))
    assert done["compute_timing"] == recomputed
    assert done["compute_accounting"] == metrics["compute_accounting"]
    validated = validate_causal_source_compute_evidence(
        run_dir,
        project_root=project_root,
        expected_run_id="synthetic-causal-source",
        expected_seed=7,
        allow_external_run=True,
    )
    assert validated["passed"] is True
    assert validated["compute_timing"] == recomputed


def test_causal_source_resume_rejects_changed_implementation(
    tmp_path: Path,
) -> None:
    data_path, split_path = _write_synthetic_source(tmp_path)
    project_root = Path(__file__).resolve().parents[2]
    expected_parameters = parameter_count(
        CausalFactorizedOperator(
            input_channels=14,
            width=4,
            depth=2,
            modes_space=3,
        )
    )
    config = CausalSourceTrainConfig(
        data_path=data_path,
        split_manifest=split_path,
        output_root=tmp_path / "runs",
        project_root=project_root,
        run_id="changed-git-causal-source",
        expected_data_sha256=sha256_file(data_path),
        expected_split_sha256=sha256_file(split_path),
        seed=9,
        time_stride=1,
        width=4,
        depth=2,
        modes_space=3,
        expected_parameter_count=expected_parameters,
        epochs=2,
        minimum_epochs=2,
        early_stopping_patience=2,
        batch_size=2,
        device="cpu",
        num_threads=1,
        causality_case_count=1,
    )
    train_causal_source(config, session_epoch_limit=1)
    last_path = (
        config.output_root
        / "changed-git-causal-source"
        / "checkpoints"
        / "last.pt"
    )
    checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
    checkpoint["git_sha"] = "0" * 40
    torch.save(checkpoint, last_path)

    with pytest.raises(ValueError, match="Git SHA differs"):
        train_causal_source(replace(config, resume=True))


def test_causal_source_resume_rejects_mutated_best_checkpoint(
    tmp_path: Path,
) -> None:
    config = _small_training_config(
        tmp_path,
        run_id="mutated-best-causal-source",
        seed=13,
        epochs=2,
    )
    train_causal_source(config, session_epoch_limit=1)
    best_path = (
        config.output_root
        / str(config.run_id)
        / "checkpoints"
        / "best.pt"
    )
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    first_key = next(iter(best["model"]))
    best["model"][first_key] = best["model"][first_key].clone()
    best["model"][first_key].view(-1)[0] += 1.0
    torch.save(best, best_path)

    with pytest.raises(ValueError, match="best.pt"):
        train_causal_source(replace(config, resume=True))


def test_causal_source_resume_rejects_mocked_implementation_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _small_training_config(
        tmp_path,
        run_id="implementation-drift-causal-source",
        seed=17,
        epochs=2,
    )
    train_causal_source(config, session_epoch_limit=1)
    original = causal_training._implementation_file_manifest

    def drifted_manifest(project_root: Path) -> list[dict[str, object]]:
        rows = [dict(row) for row in original(project_root)]
        rows[0]["sha256"] = "0" * 64
        return rows

    monkeypatch.setattr(
        causal_training,
        "_implementation_file_manifest",
        drifted_manifest,
    )
    with pytest.raises(ValueError, match="binding differs"):
        train_causal_source(replace(config, resume=True))


@pytest.mark.parametrize("history_state", ["missing", "stale"])
def test_causal_source_resume_repairs_interrupted_derived_artifacts(
    tmp_path: Path,
    history_state: str,
) -> None:
    config = _small_training_config(
        tmp_path,
        run_id=f"repair-{history_state}-causal-source",
        seed=23 if history_state == "missing" else 29,
        epochs=3,
    )
    train_causal_source(config, session_epoch_limit=2)
    run_dir = config.output_root / str(config.run_id)
    history_path = run_dir / "history.parquet"
    if history_state == "missing":
        history_path.unlink()
        (run_dir / "checkpoints" / "best.pt").unlink()
    else:
        history = pd.read_parquet(history_path)
        history.iloc[:1].to_parquet(history_path, index=False)

    metrics = train_causal_source(replace(config, resume=True))

    assert metrics["status"] == "completed"
    repaired = pd.read_parquet(history_path)
    assert repaired["epoch"].tolist() == [1, 2, 3]
    last = torch.load(
        run_dir / "checkpoints" / "last.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert [row["epoch"] for row in last["authoritative_history"]] == [
        1,
        2,
        3,
    ]
    best = torch.load(
        run_dir / "checkpoints" / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert best["artifact_generation"] == 3


def test_source_timing_history_rejects_wrong_optimizer_count(
    tmp_path: Path,
) -> None:
    config = _small_training_config(
        tmp_path,
        run_id="wrong-count-causal-source",
        seed=31,
        epochs=1,
    )
    train_causal_source(config)
    history = pd.read_parquet(
        config.output_root
        / str(config.run_id)
        / "history.parquet"
    )
    history.loc[0, "optimizer_update_count"] = 3

    with pytest.raises(ValueError, match="timing identity differs"):
        causal_source_compute_timing_from_history(
            history,
            expected_parameter_count=int(
                history.loc[0, "parameter_count"]
            ),
            expected_optimizer_updates_per_epoch=2,
            device_type="cpu",
        )


def test_source_terminal_validator_rejects_tampered_metrics(
    tmp_path: Path,
) -> None:
    config = _small_training_config(
        tmp_path,
        run_id="tampered-metrics-causal-source",
        seed=37,
        epochs=1,
    )
    train_causal_source(config)
    run_dir = config.output_root / str(config.run_id)
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["compute_accounting"]["optimizer_update_count"] += 1
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="artifact binding differs"):
        validate_causal_source_compute_evidence(
            run_dir,
            project_root=config.project_root,
            expected_run_id=str(config.run_id),
            expected_seed=config.seed,
            allow_external_run=True,
        )


def test_source_terminal_validator_rejects_live_implementation_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _small_training_config(
        tmp_path,
        run_id="implementation-drift-causal-source",
        seed=39,
        epochs=1,
    )
    train_causal_source(config)
    run_dir = config.output_root / str(config.run_id)
    live_manifest = causal_training._implementation_file_manifest(
        config.project_root
    )
    drifted_manifest = [dict(row) for row in live_manifest]
    drifted_manifest[0]["sha256"] = "f" * 64
    monkeypatch.setattr(
        causal_training,
        "_implementation_file_manifest",
        lambda _root: drifted_manifest,
    )

    with pytest.raises(ValueError, match="implementation differs"):
        validate_causal_source_compute_evidence(
            run_dir,
            project_root=config.project_root,
            expected_run_id=str(config.run_id),
            expected_seed=config.seed,
            allow_external_run=True,
        )


def test_uncontrolled_resume_marks_process_overhead_incomplete(
    tmp_path: Path,
) -> None:
    config = _small_training_config(
        tmp_path,
        run_id="uncontrolled-resume-causal-source",
        seed=41,
        epochs=2,
    )
    train_causal_source(config, session_epoch_limit=1)
    run_dir = config.output_root / str(config.run_id)
    (run_dir / "STATUS.json").write_text(
        json.dumps(
            {
                "status": "running",
                "timestamp": "simulated-process-death",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    metrics = train_causal_source(replace(config, resume=True))

    timing = metrics["compute_timing"]
    assert timing["resume_invocation_count"] == 1
    assert timing["controlled_resume_invocation_count"] == 0
    assert timing["unobserved_interruption_resume_count"] == 1
    assert timing["unobserved_failed_attempt_overhead_possible"] is True
    assert timing["failed_attempt_overhead_fully_observed"] is False
    assert timing["timing_complete"] is False
    assert metrics["compute_accounting"]["timing_complete"] is False
    validated = validate_causal_source_compute_evidence(
        run_dir,
        project_root=config.project_root,
        expected_run_id=str(config.run_id),
        expected_seed=config.seed,
        allow_external_run=True,
    )
    assert validated["compute_timing"] == timing


def test_resume_rejects_tampered_checkpoint_compute_repeat(
    tmp_path: Path,
) -> None:
    config = _small_training_config(
        tmp_path,
        run_id="tampered-compute-checkpoint-source",
        seed=43,
        epochs=2,
    )
    train_causal_source(config, session_epoch_limit=1)
    last_path = (
        config.output_root
        / str(config.run_id)
        / "checkpoints"
        / "last.pt"
    )
    checkpoint = torch.load(
        last_path,
        map_location="cpu",
        weights_only=False,
    )
    checkpoint["compute_accounting"]["optimizer_update_count"] += 1
    torch.save(checkpoint, last_path)

    with pytest.raises(
        ValueError,
        match="compute accounting differs from its history",
    ):
        train_causal_source(replace(config, resume=True))


def test_resume_after_terminal_checkpoint_updates_only_resume_evidence(
    tmp_path: Path,
) -> None:
    config = _small_training_config(
        tmp_path,
        run_id="terminal-checkpoint-reentry-source",
        seed=47,
        epochs=1,
    )
    first = train_causal_source(config)
    run_dir = config.output_root / str(config.run_id)
    sessions_path = run_dir / "run_sessions.jsonl"
    session_rows = [
        json.loads(line)
        for line in sessions_path.read_text(encoding="utf-8").splitlines()
    ]
    sessions_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in session_rows
            if row["event"] != "session_completed"
        ),
        encoding="utf-8",
    )
    for name in (
        "DONE",
        "metrics.json",
        "metrics_per_case.parquet",
        "causality.json",
    ):
        (run_dir / name).unlink()
    (run_dir / "STATUS.json").write_text(
        json.dumps(
            {"status": "running", "timestamp": "simulated-finalization-crash"},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    resumed = train_causal_source(replace(config, resume=True))

    assert resumed["compute_accounting"]["optimizer_update_count"] == (
        first["compute_accounting"]["optimizer_update_count"]
    )
    assert resumed["compute_accounting"]["candidate_count"] == (
        first["compute_accounting"]["candidate_count"]
    )
    assert resumed["compute_timing"]["resume_invocation_count"] == 1
    assert resumed["compute_timing"][
        "unobserved_interruption_resume_count"
    ] == 1
    assert resumed["compute_timing"]["timing_complete"] is False
    history = pd.read_parquet(run_dir / "history.parquet")
    assert history["epoch"].tolist() == [1]
    assert history["resume_invocation_count"].tolist() == [1]
    validate_causal_source_compute_evidence(
        run_dir,
        project_root=config.project_root,
        expected_run_id=str(config.run_id),
        expected_seed=config.seed,
        allow_external_run=True,
    )
