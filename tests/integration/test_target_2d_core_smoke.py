from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cdcureno.data.target_2d import prepare_target_2d_training


ROOT = Path(__file__).resolve().parents[2]
CORE_ROOT = ROOT / "data" / "processed" / "p4_2d_core_v1"
SOURCE_CHECKPOINT = (
    ROOT / "outputs" / "runs" / "p3-source-factorized-v4-seed0" / "best.pt"
)


@pytest.mark.skipif(
    not CORE_ROOT.exists() or not SOURCE_CHECKPOINT.exists(),
    reason="Ignored canonical P4 arrays or P3 source checkpoint are unavailable.",
)
def test_real_core_training_contract_and_one_f0_case_smoke() -> None:
    prepared = prepare_target_2d_training(
        ROOT / "splits" / "2d_id_v1.json",
        SOURCE_CHECKPOINT,
        label_budget=8,
        project_root=ROOT,
        # P4 gate already verifies every immutable array hash.  This smoke test
        # checks headers/bindings without re-reading the two 459 MB label files.
        verify_array_checksums=False,
    )
    assert prepared.train_case_ids == tuple(range(8))
    assert prepared.validation_case_ids == tuple(range(256, 288))
    assert prepared.accessed_case_ids == {"train": (), "validation": ()}

    # Frozen case 1 is an exact-extrusion F0 example.
    inputs, temperature, alpha, case_id = prepared.dataset("train")[1]
    assert inputs.shape == (112, 50, 40, 20)
    assert temperature.shape == alpha.shape == (112, 50, 40)
    assert case_id.item() == 1
    assert np.count_nonzero(inputs.numpy()[..., 14:]) == 0
    assert prepared.accessed_case_ids == {
        "train": (1,),
        "validation": (),
    }
