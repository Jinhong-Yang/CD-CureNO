import json
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

from cdcureno.data.normalization import RangeNormalizer
from cdcureno.data.splits import validate_disjoint_complete_case_split
from cdcureno.legacy.audit import (
    EXPECTED_CASE1_SHAPES,
    deterministic_cpu_smoke,
    load_legacy_module,
    model_metadata,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT / "external" / "ResFNO"


def test_public_case1_keys_and_shapes() -> None:
    arrays = sio.loadmat(REPO_ROOT / "data" / "Case1.mat")
    observed = {key: tuple(arrays[key].shape) for key in EXPECTED_CASE1_SHAPES}
    assert observed == EXPECTED_CASE1_SHAPES


def test_legacy_parameter_count() -> None:
    assert model_metadata(REPO_ROOT)["parameter_count"] == 295_617


def test_saved_checkpoint_round_trip_equivalence() -> None:
    legacy = load_legacy_module(REPO_ROOT)
    state = torch.load(
        REPO_ROOT / "logs" / "ResFNO_T_X35_net_params.pkl",
        map_location="cpu",
        weights_only=True,
    )
    first = legacy.FNO1d(16, 64, "T").cpu().eval()
    second = legacy.FNO1d(16, 64, "T").cpu().eval()
    first.load_state_dict(state)
    second.load_state_dict(state)
    x = torch.linspace(0.0, 1.0, 223).reshape(1, 223, 1)
    with torch.no_grad():
        assert torch.equal(first(x), second(x))


def test_legacy_one_location_output_shape() -> None:
    legacy = load_legacy_module(REPO_ROOT)
    model = legacy.FNO1d(16, 64, "T").cpu().eval()
    with torch.no_grad():
        output = model(torch.zeros(2, 223, 1))
    assert tuple(output.shape) == (2, 223, 1)
    assert bool(torch.isfinite(output).all())


def test_exact_split_indices() -> None:
    payload = json.loads(
        (PROJECT_ROOT / "splits" / "legacy_case1_exact_v1.json").read_text(
            encoding="utf-8"
        )
    )
    split = payload["splits"]
    assert split["train"] == list(range(50))
    assert split["validation"] == []
    assert split["test"] == list(range(50, 200))
    validate_disjoint_complete_case_split(split, list(range(200)))


def test_corrected_manifest_has_no_case_leakage() -> None:
    payload = json.loads(
        (PROJECT_ROOT / "splits" / "legacy_case1_corrected_v1.json").read_text(
            encoding="utf-8"
        )
    )
    split = payload["splits"]
    validate_disjoint_complete_case_split(split, list(range(200)))
    assert [len(split[name]) for name in ("train", "validation", "test")] == [
        50,
        25,
        125,
    ]


def test_corrected_matched_manifest_controls_training_cases() -> None:
    payload = json.loads(
        (
            PROJECT_ROOT
            / "splits"
            / "legacy_case1_corrected_matched_v1.json"
        ).read_text(encoding="utf-8")
    )
    split = payload["splits"]
    validate_disjoint_complete_case_split(split, list(range(200)))
    assert split["train"] == list(range(50))
    assert split["validation"] == list(range(50, 75))
    assert split["test"] == list(range(75, 200))


def test_corrected_normalizer_is_train_only() -> None:
    arrays = sio.loadmat(REPO_ROOT / "data" / "Case1.mat")
    payload = json.loads(
        (PROJECT_ROOT / "splits" / "legacy_case1_corrected_v1.json").read_text(
            encoding="utf-8"
        )
    )
    train_ids = payload["splits"]["train"]
    normalizer = RangeNormalizer.fit(arrays["dataTair"][train_ids])
    assert normalizer.minimum == float(np.min(arrays["dataTair"][train_ids]))
    assert normalizer.maximum == float(np.max(arrays["dataTair"][train_ids]))


def test_deterministic_legacy_cpu_training_smoke() -> None:
    smoke = deterministic_cpu_smoke(REPO_ROOT)
    assert smoke["passed"]
