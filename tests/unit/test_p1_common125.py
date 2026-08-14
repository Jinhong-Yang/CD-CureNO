from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import verify_p1_common125 as correction


def _write_saved_field(
    root: Path,
    *,
    name: str,
    experiment: str,
    case_ids: list[int],
    errors: dict[int, float],
) -> Path:
    field_dir = root / "outputs" / "runs" / "_fields" / name
    table_dir = root / "outputs" / "tables"
    field_dir.mkdir(parents=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    ids = np.asarray(case_ids, dtype=np.int64)
    time = np.arange(4, dtype=np.float32)
    input_air = ids[:, None].astype(np.float32) + time[None, :]
    target = (
        300.0
        + ids[:, None, None].astype(np.float32)
        + np.zeros((len(ids), 51, 4), dtype=np.float32)
    )
    prediction = target.copy()
    for index, case_id in enumerate(case_ids):
        prediction[index] += np.float32(errors[case_id])

    field_relative = correction._relative_l2(prediction, target)
    tool_relative = correction._relative_l2(
        prediction[:, :21], target[:, :21]
    )
    composite_relative = correction._relative_l2(
        prediction[:, 21:], target[:, 21:]
    )
    metrics = pd.DataFrame(
        {
            "case_id": ids,
            "field_relative_l2": field_relative,
            "tool_relative_l2": tool_relative,
            "composite_relative_l2": composite_relative,
        }
    )
    metrics_path = field_dir / "metrics_per_case.parquet"
    field_path = field_dir / "temperature_field.npz"
    metrics.to_parquet(metrics_path, index=False)
    np.savez_compressed(
        field_path,
        case_ids=ids,
        input_air=input_air,
        prediction=prediction,
        target=target,
    )
    summary = {
        "experiment": experiment,
        "seed": 1,
        "case_count": len(ids),
        "field_relative_l2_mean": float(np.mean(field_relative)),
        "tool_relative_l2_mean": float(np.mean(tool_relative)),
        "composite_relative_l2_mean": float(
            np.mean(composite_relative)
        ),
        "field_artifact": field_path.relative_to(root).as_posix(),
        "per_case_metrics": metrics_path.relative_to(root).as_posix(),
    }
    summary_path = table_dir / f"{name}_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary_path


def test_build_correction_intersects_and_compares_only_aligned_cases(
    tmp_path: Path,
) -> None:
    exact_summary = _write_saved_field(
        tmp_path,
        name="exact",
        experiment="legacy_resfno_exact",
        case_ids=[1, 2, 3],
        errors={1: 1.0, 2: 2.0, 3: 4.0},
    )
    corrected_summary = _write_saved_field(
        tmp_path,
        name="corrected",
        experiment="legacy_resfno_corrected",
        case_ids=[2, 3],
        errors={2: 1.0, 3: 8.0},
    )

    result = correction.build_correction(
        tmp_path,
        exact_summary_path=exact_summary,
        corrected_summary_path=corrected_summary,
        expected_common_case_count=2,
    )

    population = result["population_audit"]
    assert population["common_case_ids"] == [2, 3]
    assert population["exact_only_case_ids"] == [1]
    assert population["corrected_only_case_ids"] == []
    assert population["whole_population_direct_comparison_performed"] is False
    assert population["paired_alignment_asserted_before_each_metric"] is True
    for values in result["metrics"].values():
        assert values["paired_case_count"] == 2
        assert values["corrected_better_case_count"] == 1
        assert values["exact_better_case_count"] == 1
        assert values["tie_case_count"] == 0
    for source in result["source_artifacts"].values():
        for artifact in source.values():
            assert len(artifact["sha256"]) == 64
            assert artifact["bytes"] > 0


def test_direct_comparison_rejects_different_or_reordered_case_ids() -> None:
    exact = pd.DataFrame(
        {"case_id": [1, 2], "field_relative_l2": [0.1, 0.2]}
    )
    corrected = pd.DataFrame(
        {"case_id": [2, 1], "field_relative_l2": [0.1, 0.2]}
    )
    with pytest.raises(ValueError, match="identical ordered case IDs"):
        correction.compare_aligned_metric(
            exact, corrected, "field_relative_l2"
        )


def test_build_correction_requires_versioned_common_case_count(
    tmp_path: Path,
) -> None:
    exact_summary = _write_saved_field(
        tmp_path,
        name="exact",
        experiment="legacy_resfno_exact",
        case_ids=[1, 2],
        errors={1: 1.0, 2: 2.0},
    )
    corrected_summary = _write_saved_field(
        tmp_path,
        name="corrected",
        experiment="legacy_resfno_corrected",
        case_ids=[2],
        errors={2: 1.0},
    )
    with pytest.raises(ValueError, match="Expected 2 common cases, found 1"):
        correction.build_correction(
            tmp_path,
            exact_summary_path=exact_summary,
            corrected_summary_path=corrected_summary,
            expected_common_case_count=2,
        )


def test_persist_is_no_overwrite_and_has_read_only_verification(
    tmp_path: Path,
) -> None:
    output = tmp_path / "correction.json"
    payload = {"schema_version": 2, "passed": True}
    correction.persist_correction(payload, output)
    correction.persist_correction(payload, output, verify_existing=True)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        correction.persist_correction(payload, output)

    output.write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="differs from recomputed"):
        correction.persist_correction(
            payload, output, verify_existing=True
        )


def test_frozen_common125_correction_records_retracted_comparison() -> None:
    project_root = Path(__file__).resolve().parents[2]
    path = (
        project_root
        / "outputs"
        / "tables"
        / "p1_common125_correction_v2.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    population = payload["population_audit"]
    assert population["exact_case_count"] == 150
    assert population["corrected_case_count"] == 125
    assert population["common_case_ids"] == list(range(75, 200))
    assert population["exact_only_case_ids"] == list(range(50, 75))
    assert population["corrected_only_case_ids"] == []

    expected = {
        "field_relative_l2": (
            0.0020070583559572697,
            0.0020275022834539413,
            46,
        ),
        "composite_relative_l2": (
            0.0018733012257143855,
            0.0019000263419002295,
            44,
        ),
        "tool_relative_l2": (
            0.0021625016815960407,
            0.0021782591938972473,
            52,
        ),
    }
    for metric, (
        exact_mean,
        corrected_mean,
        corrected_better,
    ) in expected.items():
        values = payload["metrics"][metric]
        assert values["paired_case_count"] == 125
        assert values["exact_mean"] == exact_mean
        assert values["corrected_mean"] == corrected_mean
        assert values["corrected_better_case_count"] == corrected_better
        assert values["corrected_relative_improvement_percent"] < 0
