"""Train-only range normalization.

The legacy repository fits global extrema before splitting. This module keeps
fitting separate from encoding so callers must explicitly provide training
data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RangeNormalizer:
    """Affine min/max normalizer fitted to an explicitly supplied array."""

    minimum: float
    maximum: float
    low: float = 0.0
    high: float = 1.0

    @classmethod
    def fit(
        cls, train_values: np.ndarray, low: float = 0.0, high: float = 1.0
    ) -> "RangeNormalizer":
        values = np.asarray(train_values)
        if values.size == 0:
            raise ValueError("Cannot fit a normalizer on an empty training array.")
        minimum = float(np.nanmin(values))
        maximum = float(np.nanmax(values))
        if not np.isfinite(minimum) or not np.isfinite(maximum):
            raise ValueError("Training values contain no finite normalization range.")
        if maximum <= minimum:
            raise ValueError("Training maximum must be greater than training minimum.")
        return cls(minimum=minimum, maximum=maximum, low=low, high=high)

    @property
    def scale(self) -> float:
        return (self.high - self.low) / (self.maximum - self.minimum)

    def encode(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values)
        return (array - self.minimum) * self.scale + self.low

    def decode(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values)
        return (array - self.low) / self.scale + self.minimum
