import json
from pathlib import Path

import pandas as pd

from cdcureno.evaluation.run_verification import verify_run
from cdcureno.training.joint import JointTrainConfig, train_joint


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_one_epoch_joint_run_writes_and_verifies_complete_artifacts(
    tmp_path: Path,
) -> None:
    config = JointTrainConfig(
        experiment="source_joint_causal",
        data_path=PROJECT_ROOT / "external" / "ResFNO" / "data" / "Case1.mat",
        split_manifest=PROJECT_ROOT
        / "splits"
        / "legacy_case1_corrected_matched_v1.json",
        output_root=tmp_path,
        project_root=PROJECT_ROOT,
        epochs=1,
        batch_size=10,
        width=4,
        depth=1,
        modes_space=3,
        seed=919,
        device="cpu",
        num_threads=1,
        minimum_epochs=1,
        early_stopping_patience=1,
        run_id="integration_joint_seed919",
        register_result=False,
    )
    metrics = train_joint(config)
    run_dir = tmp_path / "integration_joint_seed919"

    assert metrics["phase"] == "P2"
    assert metrics["test"]["case_count"] == 125
    assert metrics["normalization"]["scope"] == "train_cases_only"
    assert metrics["peak_process_rss_bytes"] is not None
    for relative in (
        "config_resolved.yaml",
        "environment.txt",
        "git_state.txt",
        "data_checksums.json",
        "STARTED_AT",
        "run_sessions.jsonl",
        "stdout.log",
        "metrics.json",
        "metrics_per_case.parquet",
        "history.parquet",
        "checkpoints/best.pt",
        "checkpoints/last.pt",
        "predictions/test_predictions.npz",
        "DONE",
    ):
        assert (run_dir / relative).is_file(), relative

    history = pd.read_parquet(run_dir / "history.parquet")
    cases = pd.read_parquet(run_dir / "metrics_per_case.parquet")
    assert len(history) == 1
    assert len(cases) == 125
    assert cases["case_id"].is_unique
    assert json.loads((run_dir / "metrics.json").read_text())["selected_epoch"] == 0
    assert metrics["recorded_session_count"] == 1
    assert metrics["resume_session_count"] == 0
    assert metrics["summed_epoch_seconds"] > 0
    assert metrics["final_session_wall_seconds"] >= metrics["summed_epoch_seconds"]
    verification = verify_run(run_dir)
    assert verification["passed"]
    assert all(verification["physical_constraints"].values())
