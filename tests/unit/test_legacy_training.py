from pathlib import Path

import scipy.io as sio
import torch

from cdcureno.legacy.training import (
    LegacyTrainConfig,
    _relative_l2_per_case,
    prepare_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _config(experiment: str, task: str = "T") -> LegacyTrainConfig:
    manifest = (
        "legacy_case1_exact_v1.json"
        if experiment == "legacy_resfno_exact"
        else "legacy_case1_corrected_v1.json"
    )
    return LegacyTrainConfig(
        experiment=experiment,  # type: ignore[arg-type]
        data_path=PROJECT_ROOT / "external" / "ResFNO" / "data" / "Case1.mat",
        split_manifest=PROJECT_ROOT / "splits" / manifest,
        output_root=PROJECT_ROOT / "outputs" / "runs",
        project_root=PROJECT_ROOT,
        task=task,  # type: ignore[arg-type]
        epochs=1,
        register_result=False,
    ).validated()


def test_exact_configuration_forces_plain_l2_objective() -> None:
    assert _config("legacy_resfno_exact").smoothness_weight == 0.0


def test_corrected_normalization_statistics_are_train_only() -> None:
    config = _config("legacy_resfno_corrected")
    _, x_normalizer, y_normalizer, metadata = prepare_data(config)
    arrays = sio.loadmat(config.data_path)
    train_ids = torch.tensor(
        __import__("json").loads(config.split_manifest.read_text())["splits"]["train"]
    )
    expected_x = torch.from_numpy(arrays["dataTair"].astype("float32"))[train_ids]
    expected_y = torch.from_numpy(
        arrays["dataT"][:, config.location, :].astype("float32")
    )[train_ids]
    assert float(x_normalizer.minimum) == float(torch.min(expected_x))
    assert float(x_normalizer.maximum) == float(torch.max(expected_x))
    assert y_normalizer is not None
    assert float(y_normalizer.minimum) == float(torch.min(expected_y))
    assert float(y_normalizer.maximum) == float(torch.max(expected_y))
    assert metadata["normalization_scope"] == "train_cases_only"


def test_corrected_alpha_path_does_not_require_temperature_normalizer() -> None:
    prepared, _, y_normalizer, metadata = prepare_data(
        _config("legacy_resfno_corrected", task="A")
    )
    assert y_normalizer is None
    assert torch.equal(prepared["y_encoded"], prepared["y_physical"])
    assert metadata["y"] is None


def test_per_case_metric_is_batch_partition_independent() -> None:
    prediction = torch.arange(60, dtype=torch.float32).reshape(6, 10)
    target = prediction + 1.0
    whole = _relative_l2_per_case(prediction, target)
    partitioned = torch.cat(
        [
            _relative_l2_per_case(prediction[:2], target[:2]),
            _relative_l2_per_case(prediction[2:5], target[2:5]),
            _relative_l2_per_case(prediction[5:], target[5:]),
        ]
    )
    assert torch.equal(whole, partitioned)
