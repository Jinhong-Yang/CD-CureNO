import json
from pathlib import Path

import pandas as pd

from cdcureno.legacy.training import LegacyTrainConfig, train_legacy


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_one_epoch_corrected_run_writes_complete_audit_trail(tmp_path: Path) -> None:
    config = LegacyTrainConfig(
        experiment="legacy_resfno_corrected",
        data_path=PROJECT_ROOT / "external" / "ResFNO" / "data" / "Case1.mat",
        split_manifest=PROJECT_ROOT
        / "splits"
        / "legacy_case1_corrected_v1.json",
        output_root=tmp_path,
        project_root=PROJECT_ROOT,
        location=35,
        task="T",
        epochs=1,
        seed=101,
        device="cpu",
        run_id="integration_corrected_seed101",
        register_result=False,
    )
    metrics = train_legacy(config)
    run_dir = tmp_path / "integration_corrected_seed101"

    assert metrics["experiment"] == "legacy_resfno_corrected"
    assert metrics["test"]["case_count"] == 125
    assert metrics["normalization"]["normalization_scope"] == "train_cases_only"
    for relative in (
        "config_resolved.yaml",
        "environment.txt",
        "git_state.txt",
        "data_checksums.json",
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
