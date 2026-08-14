from pathlib import Path

import json
import numpy as np
import pandas as pd

from cdcureno.evaluation.run_verification import verify_run


def test_verify_run_recomputes_saved_metrics(tmp_path: Path) -> None:
    prediction = np.array([[1.0, 3.0], [2.0, 5.0]], dtype=np.float32)
    target = np.array([[1.0, 2.0], [3.0, 5.0]], dtype=np.float32)
    difference = prediction - target
    relative = np.linalg.norm(difference, axis=1) / np.linalg.norm(target, axis=1)
    frame = pd.DataFrame(
        {
            "split": ["test", "test"],
            "case_id": [10, 11],
            "relative_l2": relative,
            "mae": np.mean(np.abs(difference), axis=1),
            "rmse": np.sqrt(np.mean(difference**2, axis=1)),
            "linf": np.max(np.abs(difference), axis=1),
            "peak_value_error": np.max(prediction, axis=1) - np.max(target, axis=1),
            "time_to_peak_index_error": np.argmax(prediction, axis=1)
            - np.argmax(target, axis=1),
        }
    )
    test_summary = {
        "relative_l2_mean": float(frame["relative_l2"].mean()),
        "relative_l2_median": float(frame["relative_l2"].median()),
        "mae_mean": float(frame["mae"].mean()),
        "rmse_mean": float(frame["rmse"].mean()),
        "linf_max": float(frame["linf"].max()),
        "peak_value_error_mae": float(frame["peak_value_error"].abs().mean()),
        "time_to_peak_index_error_mae": float(
            frame["time_to_peak_index_error"].abs().mean()
        ),
    }
    (tmp_path / "predictions").mkdir()
    np.savez_compressed(
        tmp_path / "predictions" / "test_predictions.npz",
        case_ids=np.array([10, 11]),
        prediction=prediction,
        target=target,
    )
    frame.to_parquet(tmp_path / "metrics_per_case.parquet", index=False)
    (tmp_path / "metrics.json").write_text(
        json.dumps({"run_id": "synthetic", "test": test_summary})
    )
    (tmp_path / "DONE").write_text("done\n")

    assert verify_run(tmp_path)["passed"]
