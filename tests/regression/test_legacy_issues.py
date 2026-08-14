from pathlib import Path

import pytest

from cdcureno.legacy.audit import analyze_legacy_issues


REPO_ROOT = Path(__file__).resolve().parents[2] / "external" / "ResFNO"


@pytest.fixture(scope="module")
def issues() -> dict[str, object]:
    return {issue.issue_id: issue for issue in analyze_legacy_issues(REPO_ROOT)}


def test_l1_positionwise_models_confirmed(issues: dict[str, object]) -> None:
    assert issues["L1"].status == "confirmed"


def test_l2_presplit_normalization_confirmed(issues: dict[str, object]) -> None:
    assert issues["L2"].status == "confirmed"


def test_l3_unused_smoothness_loss_confirmed(issues: dict[str, object]) -> None:
    assert issues["L3"].status == "confirmed"


def test_l4_alpha_normalizer_failure_confirmed(issues: dict[str, object]) -> None:
    assert issues["L4"].status == "confirmed"


def test_l5_batch_overwrite_metric_confirmed(issues: dict[str, object]) -> None:
    assert issues["L5"].status == "confirmed"


def test_l6_predict_signature_mismatch_confirmed(issues: dict[str, object]) -> None:
    assert issues["L6"].status == "confirmed"


def test_l7_one_output_vs_51_plot_inputs_confirmed(issues: dict[str, object]) -> None:
    assert issues["L7"].status == "confirmed"


def test_l8_unpinned_noncore_requirements_confirmed(
    issues: dict[str, object],
) -> None:
    assert issues["L8"].status == "confirmed"


def test_l9_missing_documented_mat_path_confirmed(issues: dict[str, object]) -> None:
    assert issues["L9"].status == "confirmed"
