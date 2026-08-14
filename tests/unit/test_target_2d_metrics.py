from __future__ import annotations

import numpy as np
import pytest

from cdcureno.evaluation.target_2d_metrics import (
    compute_target_2d_case_metrics,
    first_threshold_crossing_time,
    target_2d_case_metric_columns,
)
from cdcureno.physics.target_2d_residuals import Target2DPhysicsContext


def _context(
    *,
    density: float = 2.0,
    specific_heat: float = 5.0,
    source: float = 0.0,
) -> Target2DPhysicsContext:
    return Target2DPhysicsContext(
        density_kg_m3=density,
        specific_heat_J_kg_K=specific_heat,
        conductivity_x_W_m_K=1.0,
        conductivity_z_W_m_K=1.0,
        base_cure_source_J_m3_per_alpha=source,
        air_temperature_K=300.0,
        initial_temperature_K=300.0,
    )


def _grid() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time = np.array([0.0, 1.0, 3.0])
    z = np.array([0.0, 1.0, 3.0])
    x = np.array([0.0, 2.0, 5.0])
    mask = np.array(
        [
            [False, False, False],
            [True, True, True],
            [True, True, True],
        ]
    )
    return time, z, x, mask


def _uniform_fields(
    temperature_K: float = 300.0,
    alpha: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    temperature = np.full((3, 3, 3), temperature_K, dtype=np.float64)
    degree = np.full((3, 3, 3), alpha, dtype=np.float64)
    return temperature, degree


def test_public_metric_schema_is_complete_and_branch_invariant() -> None:
    time, z, x, mask = _grid()
    temperature, alpha = _uniform_fields(
        temperature_K=300.0,
        alpha=0.2,
    )
    expected = target_2d_case_metric_columns()
    assert len(expected) == 185
    assert expected == tuple(sorted(set(expected)))

    finite = compute_target_2d_case_metrics(
        temperature,
        temperature,
        alpha,
        alpha,
        mask,
        time,
        z,
        x,
        _context(),
    )
    nonfinite_temperature = temperature.copy()
    nonfinite_temperature[1, 1, 1] = np.nan
    nonfinite_alpha = alpha.copy()
    nonfinite_alpha[1, 1, 1] = np.nan
    nonfinite = compute_target_2d_case_metrics(
        nonfinite_temperature,
        temperature,
        nonfinite_alpha,
        alpha,
        mask,
        time,
        z,
        x,
        _context(),
    )
    no_tool = compute_target_2d_case_metrics(
        temperature,
        temperature,
        alpha,
        alpha,
        np.ones_like(mask, dtype=np.bool_),
        time,
        z,
        x,
        _context(),
    )
    assert tuple(sorted(finite)) == expected
    assert tuple(sorted(nonfinite)) == expected
    assert tuple(sorted(no_tool)) == expected


def test_primary_relative_l2_uses_explicit_physical_quadrature() -> None:
    time, z, x, mask = _grid()
    tt, zz, xx = np.meshgrid(time, z, x, indexing="ij")
    truth_temperature = 300.0 + 2.0 * tt + zz + 0.5 * xx
    temperature_error = 0.5 + 0.25 * tt + 0.1 * zz
    predicted_temperature = truth_temperature + temperature_error
    truth_alpha = 0.1 + 0.05 * tt + 0.01 * zz + 0.005 * xx
    predicted_alpha = truth_alpha + 0.01

    metrics = compute_target_2d_case_metrics(
        predicted_temperature,
        truth_temperature,
        predicted_alpha,
        truth_alpha,
        mask,
        time,
        z,
        x,
        _context(source=100.0),
    )

    time_weight = np.array([0.5, 1.5, 1.0])
    z_width = np.array([1.0, 1.5, 2.0])
    x_width = np.array([2.0, 2.5, 3.0])
    weight = (
        time_weight[:, None, None]
        * z_width[None, :, None]
        * x_width[None, None, :]
    )
    composite = np.broadcast_to(mask, truth_temperature.shape)
    expected = np.sqrt(
        np.sum(weight[composite] * temperature_error[composite] ** 2)
        / np.sum(weight[composite] * truth_temperature[composite] ** 2)
    )
    expected_mae = np.sum(
        weight[composite] * np.abs(temperature_error[composite])
    ) / np.sum(weight[composite])

    assert metrics["temperature_relative_l2_K_composite"] == pytest.approx(
        expected, rel=1.0e-14
    )
    assert metrics["temperature_mae_K_composite"] == pytest.approx(
        expected_mae, rel=1.0e-14
    )
    assert metrics["temperature_tool_region_defined"] is True
    assert metrics["temperature_mae_K_tool"] is not None
    assert metrics["metric_dtype"] == "float64"
    assert metrics["required_guardrail_metrics_finite"] is True


def test_constant_truth_marks_range_nrmse_undefined_and_uses_gradient_fallback() -> None:
    time, z, x, mask = _grid()
    truth_temperature, truth_alpha = _uniform_fields(
        temperature_K=300.0, alpha=0.4
    )
    _, zz, xx = np.meshgrid(time, z, x, indexing="ij")
    predicted_temperature = truth_temperature + 2.0 * xx + 3.0 * zz
    predicted_alpha = truth_alpha + 0.01 * xx + 0.02 * zz

    metrics = compute_target_2d_case_metrics(
        predicted_temperature,
        truth_temperature,
        predicted_alpha,
        truth_alpha,
        mask,
        time,
        z,
        x,
        _context(),
    )

    assert metrics["temperature_nrmse_range_composite"] is None
    assert metrics["temperature_nrmse_range_composite_undefined"] is True
    assert metrics["alpha_nrmse_range_composite"] is None
    assert metrics["alpha_nrmse_range_composite_undefined"] is True
    assert metrics["temperature_relative_l2_K_composite"] is not None
    assert metrics["temperature_gradient_x_relative_l2"] is None
    assert (
        metrics["temperature_gradient_x_metric_mode"]
        == "absolute_rmse_fallback"
    )
    assert metrics["temperature_gradient_x_absolute_rmse_K_per_m"] == (
        pytest.approx(2.0)
    )
    assert metrics["alpha_gradient_z_relative_l2"] is None
    assert metrics["alpha_gradient_z_absolute_rmse_per_m"] == pytest.approx(
        0.02
    )


def test_nonzero_truth_gradients_use_relative_l2() -> None:
    time, z, x, mask = _grid()
    _, zz, xx = np.meshgrid(time, z, x, indexing="ij")
    truth_temperature = 300.0 + 2.0 * xx + 3.0 * zz
    predicted_temperature = truth_temperature + xx + 1.5 * zz
    truth_alpha = 0.2 + 0.02 * xx + 0.03 * zz
    predicted_alpha = truth_alpha + 0.01 * xx + 0.015 * zz

    metrics = compute_target_2d_case_metrics(
        predicted_temperature,
        truth_temperature,
        predicted_alpha,
        truth_alpha,
        mask,
        time,
        z,
        x,
        _context(),
    )

    assert metrics["temperature_gradient_x_relative_l2"] == pytest.approx(
        0.5
    )
    assert metrics["temperature_gradient_z_relative_l2"] == pytest.approx(
        0.5
    )
    assert metrics["temperature_gradient_x_metric_mode"] == "relative_l2"
    assert metrics["alpha_gradient_x_relative_l2"] == pytest.approx(0.5)
    assert metrics["alpha_gradient_z_relative_l2"] == pytest.approx(0.5)


def test_peak_time_uses_first_tie_and_alpha_crossings_are_interpolated() -> None:
    time = np.array([0.0, 10.0, 30.0])
    z = np.array([0.0, 1.0, 2.0])
    x = np.array([0.0, 1.0, 2.0])
    mask = np.ones((3, 3), dtype=bool)
    truth_temperature = np.broadcast_to(
        np.array([300.0, 310.0, 310.0])[:, None, None],
        (3, 3, 3),
    ).copy()
    predicted_temperature = np.broadcast_to(
        np.array([300.0, 310.0, 311.0])[:, None, None],
        (3, 3, 3),
    ).copy()
    truth_alpha = np.broadcast_to(
        np.array([0.70, 0.85, 0.96])[:, None, None],
        (3, 3, 3),
    ).copy()
    predicted_alpha = np.broadcast_to(
        np.array([0.70, 0.80, 0.90])[:, None, None],
        (3, 3, 3),
    ).copy()

    metrics = compute_target_2d_case_metrics(
        predicted_temperature,
        truth_temperature,
        predicted_alpha,
        truth_alpha,
        mask,
        time,
        z,
        x,
        _context(),
    )

    assert metrics["time_to_peak_truth_index"] == 1
    assert metrics["time_to_peak_prediction_index"] == 2
    assert metrics["time_to_peak_error_s"] == 20.0
    assert metrics["peak_temperature_absolute_error_K"] == 1.0
    assert metrics["time_to_alpha_0p8_truth_s"] == pytest.approx(
        20.0 / 3.0
    )
    assert metrics["time_to_alpha_0p8_prediction_s"] == 10.0
    assert metrics["time_to_alpha_0p8_error_s"] == pytest.approx(
        10.0 / 3.0
    )
    assert metrics["time_to_alpha_0p9_prediction_s"] == 30.0
    assert metrics["time_to_alpha_0p95_prediction_s"] is None
    assert metrics["time_to_alpha_0p95_prediction_censored"] is True
    assert metrics["time_to_alpha_0p95_prediction_censor_time_s"] == 30.0
    assert metrics["time_to_alpha_0p95_error_s"] is None


def test_alpha_violation_counts_use_strict_one_e_minus_seven_tolerance() -> None:
    time, z, x, mask = _grid()
    truth_temperature, truth_alpha = _uniform_fields(
        temperature_K=300.0, alpha=0.5
    )
    predicted_alpha = truth_alpha.copy()
    predicted_alpha[:, 1, 0] = -2.0e-7
    predicted_alpha[:, 1, 1] = 1.0 + 2.0e-7
    predicted_alpha[:, 1, 2] = np.array(
        [0.5, 0.5 - 5.0e-8, 0.5 - 2.0e-7]
    )

    metrics = compute_target_2d_case_metrics(
        truth_temperature,
        truth_temperature,
        predicted_alpha,
        truth_alpha,
        mask,
        time,
        z,
        x,
        _context(),
    )

    assert metrics["alpha_lower_bound_violation_count"] == 3
    assert metrics["alpha_upper_bound_violation_count"] == 3
    assert metrics["alpha_bound_violation_count"] == 6
    assert metrics["alpha_monotonicity_violation_count"] == 1
    assert metrics["alpha_lower_bound_excursion"] == pytest.approx(2.0e-7)
    assert metrics["alpha_upper_bound_excursion"] == pytest.approx(2.0e-7)
    assert metrics["alpha_largest_negative_increment"] == pytest.approx(
        1.5e-7
    )


def test_nonfinite_prediction_returns_audit_flags_and_keeps_safe_metrics() -> None:
    time, z, x, mask = _grid()
    truth_temperature, truth_alpha = _uniform_fields(
        temperature_K=300.0, alpha=0.4
    )
    predicted_alpha = truth_alpha.copy()
    predicted_alpha[-1, 1, 1] = np.nan

    metrics = compute_target_2d_case_metrics(
        truth_temperature,
        truth_temperature,
        predicted_alpha,
        truth_alpha,
        mask,
        time,
        z,
        x,
        _context(),
    )

    assert metrics["prediction_finite"] is False
    assert metrics["alpha_prediction_nonfinite_count"] == 1
    assert metrics["alpha_prediction_nonfinite_count_composite"] == 1
    assert metrics["temperature_relative_l2_K_composite"] == 0.0
    assert metrics["peak_temperature_absolute_error_K"] == 0.0
    assert metrics["alpha_mae_composite"] is None
    assert metrics["final_alpha_mae_composite"] is None
    assert metrics["dimensionless_global_energy_residual"] is None
    assert metrics["energy_residual_finite"] is False
    assert metrics["prediction_physics_status"] == "nonfinite_prediction"
    assert metrics["time_to_alpha_0p8_prediction_censored"] is None


def test_dimensionless_global_energy_residual_has_known_fixed_reference() -> None:
    time, z, x, mask = _grid()
    truth_temperature, alpha = _uniform_fields(
        temperature_K=300.0, alpha=0.0
    )
    predicted_temperature = np.broadcast_to(
        np.array([300.0, 301.0, 303.0])[:, None, None],
        truth_temperature.shape,
    ).copy()

    metrics = compute_target_2d_case_metrics(
        predicted_temperature,
        truth_temperature,
        alpha,
        alpha,
        mask,
        time,
        z,
        x,
        _context(density=2.0, specific_heat=5.0, source=0.0),
    )

    # rho*Cp is 10 J/m3/K.  The two integrated storage residuals are
    # 10 and 20 J/m3, while each fixed reference is 10*100 = 1000 J/m3.
    assert metrics["dimensionless_global_energy_residual"] == pytest.approx(
        0.015, rel=1.0e-14
    )
    assert metrics["truth_dimensionless_global_energy_residual"] == 0.0
    assert metrics["energy_residual_finite"] is True
    assert (
        metrics["energy_residual_denominator_convention"]
        == "physical_volume_weighted_fixed_reference_energy_density"
    )
    assert (
        metrics["energy_residual_term_magnitude_denominator_available"]
        is False
    )


def test_threshold_helper_is_deterministic_and_does_not_invent_censor_time() -> None:
    crossing, censored = first_threshold_crossing_time(
        [0.0, 2.0, 5.0],
        [0.8, 0.8, 0.9],
        0.8,
    )
    assert crossing == 0.0
    assert censored is False

    crossing, censored = first_threshold_crossing_time(
        [0.0, 2.0, 5.0],
        [0.1, 0.2, 0.3],
        0.8,
    )
    assert crossing is None
    assert censored is True


@pytest.mark.parametrize("threshold", [0.8, 0.9, 0.95])
def test_threshold_helper_admits_exactly_one_downward_ulp(
    threshold: float,
) -> None:
    predecessor = np.nextafter(np.float64(threshold), -np.inf)
    two_down = np.nextafter(predecessor, -np.inf)

    crossing, censored = first_threshold_crossing_time(
        [0.0, 2.0, 5.0],
        [0.0, predecessor, predecessor],
        threshold,
    )
    assert crossing == 2.0
    assert censored is False

    crossing, censored = first_threshold_crossing_time(
        [0.0, 2.0, 5.0],
        [0.0, two_down, two_down],
        threshold,
    )
    assert crossing is None
    assert censored is True

    crossing, censored = first_threshold_crossing_time(
        [0.0, 10.0, 30.0],
        [0.7, threshold, threshold],
        threshold,
    )
    assert crossing == 10.0
    assert censored is False


def test_case_metrics_record_exact_registered_crossing_predecessors() -> None:
    time, z, x, mask = _grid()
    temperature, alpha = _uniform_fields(
        temperature_K=300.0,
        alpha=np.nextafter(np.float64(0.9), -np.inf),
    )
    metrics = compute_target_2d_case_metrics(
        temperature,
        temperature,
        alpha,
        alpha,
        mask,
        time,
        z,
        x,
        _context(),
    )

    assert (
        metrics["alpha_crossing_threshold_comparison_rule"]
        == "immediate_float64_predecessor"
    )
    assert metrics["time_to_alpha_0p9_registered_threshold"] == 0.9
    assert metrics["time_to_alpha_0p9_comparison_threshold"] == np.nextafter(
        np.float64(0.9), -np.inf
    )
    assert metrics["time_to_alpha_0p9_truth_attained"] is True
    assert metrics["time_to_alpha_0p9_prediction_attained"] is True
    assert metrics["time_to_alpha_0p9_truth_s"] == 0.0
    assert metrics["time_to_alpha_0p9_prediction_s"] == 0.0


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ("truth_nan", "temperature_truth_K"),
        ("bad_time", "strictly increasing"),
        ("bad_mask", "zeros and ones"),
        ("shape", "equal \\[T,Z,X\\] shapes"),
    ],
)
def test_invalid_truth_shape_coordinates_and_mask_raise(
    change: str,
    match: str,
) -> None:
    time, z, x, mask = _grid()
    temperature, alpha = _uniform_fields(
        temperature_K=300.0, alpha=0.2
    )
    prediction_temperature = temperature.copy()
    truth_temperature = temperature.copy()
    prediction_alpha = alpha.copy()
    truth_alpha = alpha.copy()
    if change == "truth_nan":
        truth_temperature[0, 0, 0] = np.nan
    elif change == "bad_time":
        time = np.array([0.0, 2.0, 1.0])
    elif change == "bad_mask":
        mask = mask.astype(np.float64)
        mask[0, 0] = 0.5
    elif change == "shape":
        prediction_alpha = prediction_alpha[:, :, :-1]

    with pytest.raises(ValueError, match=match):
        compute_target_2d_case_metrics(
            prediction_temperature,
            truth_temperature,
            prediction_alpha,
            truth_alpha,
            mask,
            time,
            z,
            x,
            _context(),
        )
