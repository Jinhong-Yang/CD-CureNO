import pytest

from cdcureno.data.splits import validate_disjoint_complete_case_split


def test_complete_case_split_is_disjoint_and_exhaustive() -> None:
    validate_disjoint_complete_case_split(
        {"train": [0, 1], "validation": [2], "test": [3, 4]},
        [0, 1, 2, 3, 4],
    )


def test_overlapping_case_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="overlap"):
        validate_disjoint_complete_case_split(
            {"train": [0, 1], "validation": [1, 2], "test": [3]},
            [0, 1, 2, 3],
        )
