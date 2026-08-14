"""Complete-case split manifest helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def validate_disjoint_complete_case_split(
    split: Mapping[str, Sequence[int]], expected_case_ids: Sequence[int]
) -> None:
    """Validate that train/validation/test contain every case exactly once."""

    required = {"train", "validation", "test"}
    if set(split) != required:
        raise ValueError(f"Split keys must be exactly {sorted(required)}.")

    sets = {name: set(ids) for name, ids in split.items()}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = sets[left] & sets[right]
        if overlap:
            raise ValueError(f"{left}/{right} overlap: {sorted(overlap)}")

    observed = sets["train"] | sets["validation"] | sets["test"]
    expected = set(expected_case_ids)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"Split coverage mismatch; missing={missing}, extra={extra}")
