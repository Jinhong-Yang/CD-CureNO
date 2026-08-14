from pathlib import Path

import numpy as np

from cdcureno.data.source_1d import INPUT_CHANNELS, prepare_source_1d


ROOT = Path(__file__).resolve().parents[2]


def test_prepared_source_uses_training_only_normalization() -> None:
    data_path = ROOT / "data" / "processed" / "p3_source_1d_v1.npz"
    if not data_path.exists():
        return
    prepared = prepare_source_1d(
        data_path,
        ROOT / "splits" / "p3_source_1d_v1.json",
        time_stride=4,
    )
    assert prepared.inputs.shape == (400, 57, 51, len(INPUT_CHANNELS))
    assert prepared.temperature.shape == (400, 57, 51)
    assert len(prepared.splits["train"]) == 160
    assert set(prepared.held_out_families) == {"smart_cure", "three_hold"}
    assert not prepared.normalization["held_out_labels_used_for_normalization"]
    assert np.isfinite(prepared.inputs.numpy()).all()

