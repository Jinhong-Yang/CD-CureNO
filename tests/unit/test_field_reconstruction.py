import numpy as np

from cdcureno.evaluation.field_reconstruction import _relative_l2


def test_field_relative_l2_aggregates_complete_cases() -> None:
    target = np.ones((2, 3, 4), dtype=np.float32)
    prediction = target.copy()
    prediction[0, 0, 0] = 2.0
    values = _relative_l2(prediction, target)
    assert values.shape == (2,)
    assert values[0] > 0
    assert values[1] == 0
