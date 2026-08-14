import numpy as np
import pytest

from cdcureno.data.normalization import RangeNormalizer


def test_train_only_fit_ignores_heldout_extrema() -> None:
    train = np.array([0.0, 1.0, 2.0])
    heldout = np.array([-100.0, 100.0])
    normalizer = RangeNormalizer.fit(train)

    assert normalizer.minimum == 0.0
    assert normalizer.maximum == 2.0
    assert np.allclose(normalizer.encode(heldout), [-50.0, 50.0])


def test_range_normalizer_round_trip() -> None:
    values = np.array([2.0, 3.0, 5.0])
    normalizer = RangeNormalizer.fit(values)
    assert np.allclose(normalizer.decode(normalizer.encode(values)), values)


def test_constant_training_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="maximum"):
        RangeNormalizer.fit(np.ones(4))
