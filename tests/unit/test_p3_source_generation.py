import numpy as np

from scripts.generate_p3_source_1d import (
    FAMILIES,
    _make_cycle,
    _parameter_record,
)


def test_source_cycle_families_are_deterministic_and_well_formed() -> None:
    first = np.random.default_rng(20260723)
    second = np.random.default_rng(20260723)
    for family in FAMILIES:
        cycle_a = _make_cycle(family, first)
        cycle_b = _make_cycle(family, second)
        assert np.array_equal(cycle_a, cycle_b)
        assert cycle_a.shape == (223,)
        assert cycle_a[0] == 293.0
        assert cycle_a[-1] == 293.0
        assert 293.0 <= cycle_a.min() <= cycle_a.max() <= 480.0


def test_conditioning_record_stays_within_frozen_ranges() -> None:
    record = _parameter_record(
        0, "single_hold", np.random.default_rng(20260723)
    )
    assert 0.010 <= record["tool_thickness_m"] <= 0.040
    assert 0.010 <= record["composite_thickness_m"] <= 0.040
    assert 40.0 <= record["lower_h_W_m2_K"] <= 100.0
    assert 80.0 <= record["upper_h_W_m2_K"] <= 160.0
    assert 0.8 <= record["composite_conductivity_scale"] <= 1.2
    assert 0.9 <= record["heat_of_reaction_scale"] <= 1.1
