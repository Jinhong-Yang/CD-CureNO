"""Float64 complete-case metrics for physical P6 target fields.

The public evaluator operates on one saved case at a time.  Grid cells and
time samples are repeated measurements within that case, so this module never
pools cells from different cases.  Temperature is supplied in kelvin and both
temperature and degree of cure use the canonical ``[T,Z,X]`` order.

Field norms use explicit time and finite-volume spatial quadrature.  Physics
diagnostics are delegated to :mod:`cdcureno.physics.target_2d_residuals`; the
global energy diagnostic reported here is a physical-volume aggregate of its
raw conservative residual with an explicit fixed-reference denominator.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import numpy as np
import torch

from cdcureno.physics.target_2d_residuals import (
    Target2DPhysicsContext,
    Target2DResidualResult,
    target_2d_physics_residuals,
)


ALPHA_VIOLATION_TOLERANCE = 1.0e-7
RANGE_UNDEFINED_THRESHOLD = 1.0e-12
GRADIENT_TRUTH_NORM_FLOOR = 1.0e-12
DEFAULT_ALPHA_CROSSING_THRESHOLDS = (0.8, 0.9, 0.95)


def _as_float64_array(name: str, value: Any, ndim: int) -> np.ndarray:
    original = np.asarray(value)
    if np.iscomplexobj(original):
        raise ValueError(f"{name} must be real-valued.")
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric array.") from error
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions.")
    return array


def _coordinate(name: str, value: Any, count: int) -> np.ndarray:
    coordinate = _as_float64_array(name, value, 1)
    if coordinate.shape != (count,):
        raise ValueError(f"{name} must have shape [{count}].")
    if not np.all(np.isfinite(coordinate)):
        raise ValueError(f"{name} must contain only finite values.")
    if not np.all(np.diff(coordinate) > 0.0):
        raise ValueError(f"{name} must be strictly increasing.")
    return coordinate


def _static_mask(value: Any, shape: tuple[int, int]) -> np.ndarray:
    original = np.asarray(value)
    if original.ndim != 2 or original.shape != shape:
        raise ValueError(f"composite_mask must have shape {shape}.")
    if np.iscomplexobj(original):
        raise ValueError("composite_mask must be real-valued.")
    try:
        numeric = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("composite_mask must be numeric or boolean.") from error
    if not np.all(np.isfinite(numeric)):
        raise ValueError("composite_mask must contain only finite values.")
    if not np.all((numeric == 0.0) | (numeric == 1.0)):
        raise ValueError("composite_mask must contain only zeros and ones.")
    mask = numeric.astype(np.bool_)
    if not np.any(mask):
        raise ValueError("composite_mask must contain at least one cell.")
    return mask


def _control_volume_widths(coordinate: np.ndarray) -> np.ndarray:
    """Match the cell-centred P4/P6 finite-volume width convention."""

    gaps = np.diff(coordinate)
    if coordinate.size == 2:
        return np.array([gaps[0], gaps[0]], dtype=np.float64)
    return np.concatenate(
        (
            gaps[:1],
            0.5 * (gaps[:-1] + gaps[1:]),
            gaps[-1:],
        )
    )


def _trapezoidal_weights(time_s: np.ndarray) -> np.ndarray:
    gaps = np.diff(time_s)
    weights = np.empty_like(time_s)
    weights[0] = 0.5 * gaps[0]
    weights[-1] = 0.5 * gaps[-1]
    if time_s.size > 2:
        weights[1:-1] = 0.5 * (gaps[:-1] + gaps[1:])
    return weights


def _weighted_mean_absolute(value: np.ndarray, weight: np.ndarray) -> float:
    total = float(np.sum(weight, dtype=np.float64))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Quadrature weights must have a positive finite sum.")
    normalized = weight / total
    result = float(np.sum(normalized * np.abs(value), dtype=np.float64))
    return result


def _weighted_rms(value: np.ndarray, weight: np.ndarray) -> float:
    """Return a scale-safe physical-quadrature RMS."""

    total = float(np.sum(weight, dtype=np.float64))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Quadrature weights must have a positive finite sum.")
    scale = float(np.max(np.abs(value)))
    if scale == 0.0:
        return 0.0
    if not np.isfinite(scale):
        return float("nan")
    normalized = value / scale
    mean_square = float(
        np.sum((weight / total) * normalized * normalized, dtype=np.float64)
    )
    return float(scale * np.sqrt(mean_square))


def _weighted_l2_norm(value: np.ndarray, weight: np.ndarray) -> float:
    """Return a scale-safe unnormalised weighted L2 norm."""

    total = float(np.sum(weight, dtype=np.float64))
    rms = _weighted_rms(value, weight)
    return float(rms * np.sqrt(total))


def _field_metric_names(
    field: str,
    region: str,
) -> dict[str, str]:
    if field == "temperature":
        return {
            "relative": f"temperature_relative_l2_K_{region}",
            "mae": f"temperature_mae_K_{region}",
            "rmse": f"temperature_rmse_K_{region}",
            "nrmse": f"temperature_nrmse_range_{region}",
            "linf": f"temperature_linf_K_{region}",
            "range": f"temperature_truth_range_K_{region}",
        }
    if field == "alpha":
        return {
            "relative": f"alpha_relative_l2_{region}",
            "mae": f"alpha_mae_{region}",
            "rmse": f"alpha_rmse_{region}",
            "nrmse": f"alpha_nrmse_range_{region}",
            "linf": f"alpha_linf_{region}",
            "range": f"alpha_truth_range_{region}",
        }
    raise ValueError(f"Unknown field {field!r}.")


def _field_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    sample_weight: np.ndarray,
    region_mask: np.ndarray,
    *,
    field: str,
    region: str,
) -> dict[str, Any]:
    names = _field_metric_names(field, region)
    full_mask = np.broadcast_to(region_mask, truth.shape)
    region_defined = bool(np.any(full_mask))
    output: dict[str, Any] = {
        f"{field}_{region}_region_defined": region_defined,
        f"{field}_prediction_finite_{region}": False,
        f"{field}_{region}_metrics_evaluable": False,
        names["relative"]: None,
        f"{names['relative']}_undefined": True,
        names["mae"]: None,
        names["rmse"]: None,
        names["nrmse"]: None,
        f"{names['nrmse']}_undefined": True,
        names["linf"]: None,
        names["range"]: None,
    }
    if not region_defined:
        output[f"{field}_{region}_metric_status"] = "region_absent"
        return output

    selected_truth = truth[full_mask]
    selected_prediction = prediction[full_mask]
    selected_weight = sample_weight[full_mask]
    truth_range = float(
        np.max(selected_truth) - np.min(selected_truth)
    )
    output[names["range"]] = truth_range
    output[f"{names['nrmse']}_undefined"] = bool(
        truth_range <= RANGE_UNDEFINED_THRESHOLD
    )

    truth_rms = _weighted_rms(selected_truth, selected_weight)
    output[f"{names['relative']}_undefined"] = bool(
        truth_rms <= RANGE_UNDEFINED_THRESHOLD
    )
    prediction_finite = bool(np.all(np.isfinite(selected_prediction)))
    output[f"{field}_prediction_finite_{region}"] = prediction_finite
    if not prediction_finite:
        output[f"{field}_{region}_metric_status"] = (
            "nonfinite_prediction_in_region"
        )
        return output

    error = selected_prediction - selected_truth
    mae = _weighted_mean_absolute(error, selected_weight)
    rmse = _weighted_rms(error, selected_weight)
    linf = float(np.max(np.abs(error)))
    output[names["mae"]] = mae
    output[names["rmse"]] = rmse
    output[names["linf"]] = linf
    if not output[f"{names['relative']}_undefined"]:
        output[names["relative"]] = float(rmse / truth_rms)
    if not output[f"{names['nrmse']}_undefined"]:
        output[names["nrmse"]] = float(rmse / truth_range)
    output[f"{field}_{region}_metrics_evaluable"] = True
    output[f"{field}_{region}_metric_status"] = "ok"
    return output


def _alpha_audit(
    alpha_prediction: np.ndarray,
    composite_mask: np.ndarray,
    tolerance: float,
) -> dict[str, Any]:
    values = alpha_prediction[:, composite_mask]
    finite = np.isfinite(values)
    finite_values = values[finite]
    nonfinite_count = int(values.size - np.count_nonzero(finite))
    lower_count = int(
        np.count_nonzero(finite & (values < -tolerance))
    )
    upper_count = int(
        np.count_nonzero(finite & (values > 1.0 + tolerance))
    )

    with np.errstate(over="ignore", invalid="ignore"):
        increments = np.diff(values, axis=0)
    finite_increments = np.isfinite(increments)
    monotonicity_count = int(
        np.count_nonzero(
            finite_increments & (increments < -tolerance)
        )
    )
    increment_nonfinite_count = int(
        increments.size - np.count_nonzero(finite_increments)
    )

    if finite_values.size:
        minimum = float(np.min(finite_values))
        maximum = float(np.max(finite_values))
        lower_excursion = float(max(0.0, -minimum))
        upper_excursion = float(max(0.0, maximum - 1.0))
    else:
        minimum = None
        maximum = None
        lower_excursion = None
        upper_excursion = None

    finite_increment_values = increments[finite_increments]
    if finite_increment_values.size:
        minimum_increment = float(np.min(finite_increment_values))
        largest_negative_increment = float(
            max(0.0, -minimum_increment)
        )
    else:
        minimum_increment = None
        largest_negative_increment = None

    bound_count = lower_count + upper_count
    return {
        "alpha_violation_tolerance": float(tolerance),
        "alpha_prediction_nonfinite_count_composite": nonfinite_count,
        "alpha_bound_audit_complete": nonfinite_count == 0,
        "alpha_lower_bound_violation_count": lower_count,
        "alpha_upper_bound_violation_count": upper_count,
        "alpha_bound_violation_count": bound_count,
        "alpha_bounds_satisfied": bool(
            nonfinite_count == 0 and bound_count == 0
        ),
        "alpha_minimum_composite": minimum,
        "alpha_maximum_composite": maximum,
        "alpha_lower_bound_excursion": lower_excursion,
        "alpha_upper_bound_excursion": upper_excursion,
        "alpha_largest_bound_excursion": (
            None
            if lower_excursion is None or upper_excursion is None
            else max(lower_excursion, upper_excursion)
        ),
        "alpha_increment_nonfinite_count_composite": (
            increment_nonfinite_count
        ),
        "alpha_monotonicity_audit_complete": (
            increment_nonfinite_count == 0
        ),
        "alpha_monotonicity_violation_count": monotonicity_count,
        "alpha_monotonicity_satisfied": bool(
            increment_nonfinite_count == 0
            and monotonicity_count == 0
        ),
        "alpha_minimum_increment_composite": minimum_increment,
        "alpha_largest_negative_increment": largest_negative_increment,
    }


def first_threshold_crossing_time(
    time_s: Any,
    trajectory: Any,
    threshold: float,
) -> tuple[float | None, bool]:
    """Return the first linearly interpolated crossing and censor flag.

    The search is deterministic: an initial attainment returns the initial
    time, and a tie or repeated attainment uses the first qualifying sample.
    The attainment comparator is exactly the immediate float64 predecessor of
    ``threshold``.  This admits a spatial mean that is one ULP below the
    registered decimal value without introducing a wider numerical tolerance.
    Interpolation still targets the registered threshold; when the qualifying
    sample is the predecessor itself, that sample time is the crossing.  A
    trajectory that never reaches the predecessor returns ``(None, True)``;
    the final time is deliberately not substituted for the missing crossing.
    """

    time = _as_float64_array("time_s", time_s, 1)
    values = _as_float64_array("trajectory", trajectory, 1)
    if time.shape != values.shape or time.size < 2:
        raise ValueError(
            "time_s and trajectory must have equal lengths of at least two."
        )
    if not np.all(np.isfinite(time)) or not np.all(np.diff(time) > 0.0):
        raise ValueError("time_s must be finite and strictly increasing.")
    if not np.all(np.isfinite(values)):
        raise ValueError("trajectory must contain only finite values.")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite.")
    comparison_threshold = float(
        np.nextafter(np.float64(threshold), np.float64(-np.inf))
    )
    if not np.isfinite(comparison_threshold):
        raise ValueError(
            "threshold must have a finite immediate float64 predecessor."
        )

    attained = np.flatnonzero(values >= comparison_threshold)
    if attained.size == 0:
        return None, True
    index = int(attained[0])
    if index == 0:
        return float(time[0]), False
    lower_value = float(values[index - 1])
    upper_value = float(values[index])
    if upper_value < float(threshold):
        # The only admitted sub-threshold value is the immediate predecessor.
        # It attains at its saved sample time; extrapolation beyond that sample
        # would negate the registered one-ULP attainment rule.
        return float(time[index]), False
    if upper_value == lower_value:
        return float(time[index]), False
    fraction = (float(threshold) - lower_value) / (
        upper_value - lower_value
    )
    fraction = float(np.clip(fraction, 0.0, 1.0))
    crossing = time[index - 1] + fraction * (
        time[index] - time[index - 1]
    )
    return float(crossing), False


def _threshold_slug(threshold: float) -> str:
    return format(float(threshold), ".12g").replace("-", "m").replace(".", "p")


def _crossing_metrics(
    alpha_prediction: np.ndarray,
    alpha_truth: np.ndarray,
    composite_mask: np.ndarray,
    spatial_weight: np.ndarray,
    time_s: np.ndarray,
    thresholds: Sequence[float],
) -> dict[str, Any]:
    selected_weight = spatial_weight[composite_mask]
    normalized_weight = selected_weight / np.sum(selected_weight)
    truth_trajectory = np.sum(
        alpha_truth[:, composite_mask] * normalized_weight[None, :],
        axis=1,
        dtype=np.float64,
    )
    prediction_values = alpha_prediction[:, composite_mask]
    prediction_evaluable = bool(
        np.all(np.isfinite(prediction_values))
    )
    prediction_trajectory = (
        np.sum(
            prediction_values * normalized_weight[None, :],
            axis=1,
            dtype=np.float64,
        )
        if prediction_evaluable
        else None
    )

    output: dict[str, Any] = {
        "alpha_crossing_trajectory_definition": (
            "composite_volume_weighted_mean"
        ),
        "alpha_crossing_threshold_comparison_rule": (
            "immediate_float64_predecessor"
        ),
        "alpha_crossing_prediction_evaluable": prediction_evaluable,
    }
    for threshold in thresholds:
        slug = _threshold_slug(threshold)
        base = f"time_to_alpha_{slug}"
        output[f"{base}_registered_threshold"] = float(threshold)
        output[f"{base}_comparison_threshold"] = float(
            np.nextafter(np.float64(threshold), np.float64(-np.inf))
        )
        truth_time, truth_censored = first_threshold_crossing_time(
            time_s, truth_trajectory, threshold
        )
        output[f"{base}_truth_s"] = truth_time
        output[f"{base}_truth_censored"] = truth_censored
        output[f"{base}_truth_attained"] = not truth_censored
        output[f"{base}_truth_censor_time_s"] = (
            float(time_s[-1]) if truth_censored else None
        )
        if prediction_trajectory is None:
            output[f"{base}_prediction_s"] = None
            output[f"{base}_prediction_censored"] = None
            output[f"{base}_prediction_attained"] = None
            output[f"{base}_prediction_censor_time_s"] = None
            output[f"{base}_error_s"] = None
            output[f"{base}_absolute_error_s"] = None
            output[f"{base}_comparison_status"] = (
                "nonfinite_prediction"
            )
            continue
        prediction_time, prediction_censored = (
            first_threshold_crossing_time(
                time_s, prediction_trajectory, threshold
            )
        )
        output[f"{base}_prediction_s"] = prediction_time
        output[f"{base}_prediction_censored"] = prediction_censored
        output[f"{base}_prediction_attained"] = (
            not prediction_censored
        )
        output[f"{base}_prediction_censor_time_s"] = (
            float(time_s[-1]) if prediction_censored else None
        )
        if truth_censored or prediction_censored:
            output[f"{base}_error_s"] = None
            output[f"{base}_absolute_error_s"] = None
            output[f"{base}_comparison_status"] = "censored"
        else:
            assert truth_time is not None and prediction_time is not None
            error = float(prediction_time - truth_time)
            output[f"{base}_error_s"] = error
            output[f"{base}_absolute_error_s"] = abs(error)
            output[f"{base}_comparison_status"] = "observed"
    return output


def _gradient_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    composite_mask: np.ndarray,
    time_weight: np.ndarray,
    z_width: np.ndarray,
    x_width: np.ndarray,
    coordinate: np.ndarray,
    *,
    field: str,
    axis: str,
    truth_norm_floor: float,
) -> dict[str, Any]:
    if axis == "x":
        prediction_gradient = np.diff(prediction, axis=2) / np.diff(
            coordinate
        )[None, None, :]
        truth_gradient = np.diff(truth, axis=2) / np.diff(coordinate)[
            None, None, :
        ]
        face_mask = composite_mask[:, 1:] & composite_mask[:, :-1]
        face_width = 0.5 * (x_width[1:] + x_width[:-1])
        weight = (
            time_weight[:, None, None]
            * z_width[None, :, None]
            * face_width[None, None, :]
        )
    elif axis == "z":
        prediction_gradient = np.diff(prediction, axis=1) / np.diff(
            coordinate
        )[None, :, None]
        truth_gradient = np.diff(truth, axis=1) / np.diff(coordinate)[
            None, :, None
        ]
        face_mask = composite_mask[1:, :] & composite_mask[:-1, :]
        face_width = 0.5 * (z_width[1:] + z_width[:-1])
        weight = (
            time_weight[:, None, None]
            * face_width[None, :, None]
            * x_width[None, None, :]
        )
    else:
        raise ValueError("axis must be 'x' or 'z'.")

    prefix = f"{field}_gradient_{axis}"
    unit_suffix = "K_per_m" if field == "temperature" else "per_m"
    full_mask = np.broadcast_to(face_mask, truth_gradient.shape)
    face_count = int(np.count_nonzero(face_mask))
    output: dict[str, Any] = {
        f"{prefix}_face_count": face_count,
        f"{prefix}_prediction_finite": False,
        f"{prefix}_truth_weighted_l2_norm": None,
        f"{prefix}_relative_l2": None,
        f"{prefix}_relative_l2_undefined": True,
        f"{prefix}_absolute_rmse_{unit_suffix}": None,
        f"{prefix}_linf_{unit_suffix}": None,
        f"{prefix}_metric_value": None,
        f"{prefix}_metric_mode": "undefined_no_composite_faces",
    }
    if not np.any(full_mask):
        return output

    selected_truth = truth_gradient[full_mask]
    selected_prediction = prediction_gradient[full_mask]
    selected_weight = weight[full_mask]
    truth_norm = _weighted_l2_norm(selected_truth, selected_weight)
    output[f"{prefix}_truth_weighted_l2_norm"] = truth_norm
    prediction_finite = bool(
        np.all(np.isfinite(selected_prediction))
    )
    output[f"{prefix}_prediction_finite"] = prediction_finite
    if not prediction_finite:
        output[f"{prefix}_metric_mode"] = "nonfinite_prediction"
        return output

    error = selected_prediction - selected_truth
    absolute_rmse = _weighted_rms(error, selected_weight)
    linf = float(np.max(np.abs(error)))
    output[f"{prefix}_absolute_rmse_{unit_suffix}"] = absolute_rmse
    output[f"{prefix}_linf_{unit_suffix}"] = linf
    if truth_norm <= truth_norm_floor:
        output[f"{prefix}_metric_value"] = absolute_rmse
        output[f"{prefix}_metric_mode"] = "absolute_rmse_fallback"
        return output

    error_norm = _weighted_l2_norm(error, selected_weight)
    relative = float(error_norm / truth_norm)
    output[f"{prefix}_relative_l2"] = relative
    output[f"{prefix}_relative_l2_undefined"] = False
    output[f"{prefix}_metric_value"] = relative
    output[f"{prefix}_metric_mode"] = "relative_l2"
    return output


def _context_field_zx(
    value: Any,
    shape: tuple[int, int],
    name: str,
) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        original = value.detach().cpu().numpy()
    else:
        original = np.asarray(value)
    if np.iscomplexobj(original):
        raise ValueError(f"{name} must be real-valued.")
    array = np.asarray(original, dtype=np.float64)
    if array.ndim == 0:
        return np.full(shape, float(array), dtype=np.float64)
    if array.shape == (1,):
        return np.full(shape, float(array[0]), dtype=np.float64)
    if array.shape == shape:
        return array.copy()
    if array.shape == (1, *shape):
        return array[0].copy()
    if array.shape == (1, 1, 1):
        return np.full(shape, float(array[0, 0, 0]), dtype=np.float64)
    raise ValueError(
        f"{name} cannot be resolved as a scalar or [Z,X] case field."
    )


def _fixed_energy_reference(
    context: Target2DPhysicsContext,
    composite_mask: np.ndarray,
    temperature_scale_K: float,
) -> np.ndarray:
    shape = composite_mask.shape
    density = _context_field_zx(
        context.density_kg_m3, shape, "density_kg_m3"
    )
    specific_heat = _context_field_zx(
        context.specific_heat_J_kg_K,
        shape,
        "specific_heat_J_kg_K",
    )
    base_source = _context_field_zx(
        context.base_cure_source_J_m3_per_alpha,
        shape,
        "base_cure_source_J_m3_per_alpha",
    )
    reaction_scale = _context_field_zx(
        context.reaction_enthalpy_scale,
        shape,
        "reaction_enthalpy_scale",
    )
    reference = (
        density * specific_heat * temperature_scale_K
        + base_source
        * reaction_scale
        * composite_mask.astype(np.float64)
    )
    if not np.all(np.isfinite(reference)) or np.any(reference <= 0.0):
        raise ValueError(
            "The fixed energy reference must be finite and positive."
        )
    return reference


def _energy_metrics(
    residual: Target2DResidualResult,
    spatial_weight: np.ndarray,
    fixed_reference: np.ndarray,
    *,
    truth: bool,
) -> dict[str, Any]:
    raw = (
        residual.energy_raw_J_m3.detach()
        .cpu()
        .numpy()
        .astype(np.float64, copy=False)[0]
    )
    dimensionless = (
        residual.energy_dimensionless.detach()
        .cpu()
        .numpy()
        .astype(np.float64, copy=False)[0]
    )
    finite = bool(
        np.all(np.isfinite(raw))
        and np.all(np.isfinite(dimensionless))
    )
    prefix = "truth_" if truth else ""
    output: dict[str, Any] = {
        f"{prefix}energy_residual_finite": finite,
        f"{prefix}dimensionless_global_energy_residual": None,
        f"{prefix}dimensionless_energy_residual_mean_absolute": None,
        f"{prefix}dimensionless_energy_residual_rmse": None,
        f"{prefix}dimensionless_energy_residual_linf": None,
    }
    if not finite:
        return output

    interval_count = raw.shape[0]
    weights = np.broadcast_to(spatial_weight, raw.shape)
    denominator = float(
        interval_count
        * np.sum(
            spatial_weight * fixed_reference,
            dtype=np.float64,
        )
    )
    numerator = float(
        np.sum(weights * np.abs(raw), dtype=np.float64)
    )
    if (
        not np.isfinite(denominator)
        or denominator <= 0.0
        or not np.isfinite(numerator)
    ):
        return output

    output[f"{prefix}dimensionless_global_energy_residual"] = float(
        numerator / denominator
    )
    output[
        f"{prefix}dimensionless_energy_residual_mean_absolute"
    ] = _weighted_mean_absolute(dimensionless, weights)
    output[f"{prefix}dimensionless_energy_residual_rmse"] = (
        _weighted_rms(dimensionless, weights)
    )
    output[f"{prefix}dimensionless_energy_residual_linf"] = float(
        np.max(np.abs(dimensionless))
    )
    return output


def _physics_residuals(
    temperature_K: np.ndarray,
    alpha: np.ndarray,
    time_s: np.ndarray,
    z_m: np.ndarray,
    x_m: np.ndarray,
    composite_mask: np.ndarray,
    context: Target2DPhysicsContext,
    temperature_scale_K: float,
) -> Target2DResidualResult:
    with torch.no_grad():
        return target_2d_physics_residuals(
            torch.as_tensor(
                temperature_K[None, ...], dtype=torch.float64
            ),
            torch.as_tensor(alpha[None, ...], dtype=torch.float64),
            torch.as_tensor(time_s, dtype=torch.float64),
            torch.as_tensor(z_m, dtype=torch.float64),
            torch.as_tensor(x_m, dtype=torch.float64),
            torch.as_tensor(composite_mask, dtype=torch.bool),
            context,
            temperature_scale_K=temperature_scale_K,
        )


def compute_target_2d_case_metrics(
    temperature_prediction_K: Any,
    temperature_truth_K: Any,
    alpha_prediction: Any,
    alpha_truth: Any,
    composite_mask: Any,
    time_s: Any,
    z_m: Any,
    x_m: Any,
    physics_context: Target2DPhysicsContext,
    *,
    alpha_tolerance: float = ALPHA_VIOLATION_TOLERANCE,
    gradient_truth_norm_floor: float = GRADIENT_TRUTH_NORM_FLOOR,
    alpha_crossing_thresholds: Sequence[float] = (
        DEFAULT_ALPHA_CROSSING_THRESHOLDS
    ),
    energy_temperature_scale_K: float = 100.0,
) -> dict[str, Any]:
    """Compute audited physical-unit metrics for one complete target case.

    Invalid truth, coordinates, masks, shapes, or physics context raise
    ``ValueError``.  Nonfinite prediction values do not raise: raw audit flags
    are returned and only metrics whose required prediction region is finite
    are evaluated.
    """

    temperature_prediction = _as_float64_array(
        "temperature_prediction_K", temperature_prediction_K, 3
    )
    temperature_truth = _as_float64_array(
        "temperature_truth_K", temperature_truth_K, 3
    )
    predicted_alpha = _as_float64_array(
        "alpha_prediction", alpha_prediction, 3
    )
    true_alpha = _as_float64_array("alpha_truth", alpha_truth, 3)
    shape = temperature_truth.shape
    if (
        temperature_prediction.shape != shape
        or predicted_alpha.shape != shape
        or true_alpha.shape != shape
    ):
        raise ValueError(
            "All temperature and alpha fields must have equal [T,Z,X] shapes."
        )
    time_count, z_count, x_count = shape
    if min(shape) < 2:
        raise ValueError("Complete-case fields require T, Z, and X >= 2.")

    time = _coordinate("time_s", time_s, time_count)
    z_coordinate = _coordinate("z_m", z_m, z_count)
    x_coordinate = _coordinate("x_m", x_m, x_count)
    mask = _static_mask(composite_mask, (z_count, x_count))
    if not np.all(np.isfinite(temperature_truth)):
        raise ValueError("temperature_truth_K must contain only finite values.")
    if np.any(temperature_truth <= 0.0):
        raise ValueError("temperature_truth_K must be positive kelvin.")
    if not np.all(np.isfinite(true_alpha)):
        raise ValueError("alpha_truth must contain only finite values.")
    if not isinstance(physics_context, Target2DPhysicsContext):
        raise ValueError(
            "physics_context must be a Target2DPhysicsContext instance."
        )
    for name, value, allow_zero in (
        ("alpha_tolerance", alpha_tolerance, True),
        (
            "gradient_truth_norm_floor",
            gradient_truth_norm_floor,
            True,
        ),
        (
            "energy_temperature_scale_K",
            energy_temperature_scale_K,
            False,
        ),
    ):
        if not np.isfinite(value) or (
            value < 0.0 if allow_zero else value <= 0.0
        ):
            qualifier = "nonnegative" if allow_zero else "positive"
            raise ValueError(f"{name} must be finite and {qualifier}.")

    thresholds = tuple(float(value) for value in alpha_crossing_thresholds)
    if (
        not thresholds
        or not np.all(np.isfinite(thresholds))
        or any(value < 0.0 or value > 1.0 for value in thresholds)
        or any(
            right <= left
            for left, right in zip(thresholds, thresholds[1:])
        )
    ):
        raise ValueError(
            "alpha_crossing_thresholds must be finite, strictly increasing, "
            "and within [0,1]."
        )

    time_weight = _trapezoidal_weights(time)
    z_width = _control_volume_widths(z_coordinate)
    x_width = _control_volume_widths(x_coordinate)
    spatial_weight = z_width[:, None] * x_width[None, :]
    sample_weight = (
        time_weight[:, None, None] * spatial_weight[None, :, :]
    )

    temperature_finite = np.isfinite(temperature_prediction)
    alpha_finite = np.isfinite(predicted_alpha)
    temperature_nonfinite_count = int(
        temperature_prediction.size - np.count_nonzero(temperature_finite)
    )
    alpha_nonfinite_count = int(
        predicted_alpha.size - np.count_nonzero(alpha_finite)
    )
    temperature_nonpositive_count = int(
        np.count_nonzero(
            temperature_finite & (temperature_prediction <= 0.0)
        )
    )
    output: dict[str, Any] = {
        "prediction_finite": bool(
            temperature_nonfinite_count == 0
            and alpha_nonfinite_count == 0
        ),
        "temperature_prediction_finite": (
            temperature_nonfinite_count == 0
        ),
        "alpha_prediction_finite": alpha_nonfinite_count == 0,
        "temperature_prediction_nonfinite_count": (
            temperature_nonfinite_count
        ),
        "alpha_prediction_nonfinite_count": alpha_nonfinite_count,
        "prediction_nonfinite_count": (
            temperature_nonfinite_count + alpha_nonfinite_count
        ),
        "temperature_prediction_nonpositive_count": (
            temperature_nonpositive_count
        ),
        "quadrature_definition": (
            "trapezoidal_time_x_cell_centered_finite_volume"
        ),
        "metric_dtype": "float64",
    }
    output.update(
        _field_metrics(
            temperature_prediction,
            temperature_truth,
            sample_weight,
            mask,
            field="temperature",
            region="composite",
        )
    )
    output.update(
        _field_metrics(
            temperature_prediction,
            temperature_truth,
            sample_weight,
            ~mask,
            field="temperature",
            region="tool",
        )
    )
    output.update(
        _field_metrics(
            predicted_alpha,
            true_alpha,
            sample_weight,
            mask,
            field="alpha",
            region="composite",
        )
    )
    output.update(
        _field_metrics(
            predicted_alpha,
            true_alpha,
            sample_weight,
            ~mask,
            field="alpha",
            region="tool",
        )
    )
    output.update(_alpha_audit(predicted_alpha, mask, alpha_tolerance))

    composite_temperature_finite = bool(
        np.all(np.isfinite(temperature_prediction[:, mask]))
    )
    if composite_temperature_finite:
        truth_peak_trace = np.max(temperature_truth[:, mask], axis=1)
        prediction_peak_trace = np.max(
            temperature_prediction[:, mask], axis=1
        )
        truth_peak_index = int(np.argmax(truth_peak_trace))
        prediction_peak_index = int(np.argmax(prediction_peak_trace))
        truth_peak = float(truth_peak_trace[truth_peak_index])
        prediction_peak = float(
            prediction_peak_trace[prediction_peak_index]
        )
        peak_error = prediction_peak - truth_peak
        peak_time_error = (
            float(time[prediction_peak_index])
            - float(time[truth_peak_index])
        )
        output.update(
            {
                "peak_temperature_truth_K": truth_peak,
                "peak_temperature_prediction_K": prediction_peak,
                "peak_temperature_error_K": peak_error,
                "peak_temperature_absolute_error_K": abs(peak_error),
                "time_to_peak_truth_index": truth_peak_index,
                "time_to_peak_prediction_index": prediction_peak_index,
                "time_to_peak_truth_s": float(time[truth_peak_index]),
                "time_to_peak_prediction_s": float(
                    time[prediction_peak_index]
                ),
                "time_to_peak_error_s": peak_time_error,
                "time_to_peak_absolute_error_s": abs(peak_time_error),
                "time_to_peak_tie_rule": "first_grid_sample",
            }
        )
    else:
        output.update(
            {
                "peak_temperature_truth_K": float(
                    np.max(temperature_truth[:, mask])
                ),
                "peak_temperature_prediction_K": None,
                "peak_temperature_error_K": None,
                "peak_temperature_absolute_error_K": None,
                "time_to_peak_truth_index": int(
                    np.argmax(np.max(temperature_truth[:, mask], axis=1))
                ),
                "time_to_peak_prediction_index": None,
                "time_to_peak_truth_s": float(
                    time[
                        int(
                            np.argmax(
                                np.max(
                                    temperature_truth[:, mask], axis=1
                                )
                            )
                        )
                    ]
                ),
                "time_to_peak_prediction_s": None,
                "time_to_peak_error_s": None,
                "time_to_peak_absolute_error_s": None,
                "time_to_peak_tie_rule": "first_grid_sample",
            }
        )

    final_prediction = predicted_alpha[-1, mask]
    final_truth = true_alpha[-1, mask]
    final_weight = spatial_weight[mask]
    if np.all(np.isfinite(final_prediction)):
        output["final_alpha_mae_composite"] = _weighted_mean_absolute(
            final_prediction - final_truth,
            final_weight,
        )
        output["final_alpha_mae_composite_evaluable"] = True
    else:
        output["final_alpha_mae_composite"] = None
        output["final_alpha_mae_composite_evaluable"] = False

    output.update(
        _crossing_metrics(
            predicted_alpha,
            true_alpha,
            mask,
            spatial_weight,
            time,
            thresholds,
        )
    )
    for field, prediction, truth in (
        ("temperature", temperature_prediction, temperature_truth),
        ("alpha", predicted_alpha, true_alpha),
    ):
        output.update(
            _gradient_metrics(
                prediction,
                truth,
                mask,
                time_weight,
                z_width,
                x_width,
                x_coordinate,
                field=field,
                axis="x",
                truth_norm_floor=gradient_truth_norm_floor,
            )
        )
        output.update(
            _gradient_metrics(
                prediction,
                truth,
                mask,
                time_weight,
                z_width,
                x_width,
                z_coordinate,
                field=field,
                axis="z",
                truth_norm_floor=gradient_truth_norm_floor,
            )
        )
    output["maximum_lateral_gradient_error_K_per_m"] = output[
        "temperature_gradient_x_linf_K_per_m"
    ]

    fixed_reference = _fixed_energy_reference(
        physics_context, mask, energy_temperature_scale_K
    )
    truth_residual = _physics_residuals(
        temperature_truth,
        true_alpha,
        time,
        z_coordinate,
        x_coordinate,
        mask,
        physics_context,
        energy_temperature_scale_K,
    )
    truth_energy = _energy_metrics(
        truth_residual,
        spatial_weight,
        fixed_reference,
        truth=True,
    )
    if not truth_energy["truth_energy_residual_finite"]:
        raise ValueError(
            "Truth/context produced a nonfinite physical residual."
        )
    output.update(truth_energy)
    output.update(
        {
            "energy_residual_denominator_convention": (
                "physical_volume_weighted_fixed_reference_energy_density"
            ),
            "energy_residual_fixed_reference_definition": (
                "rho*Cp*temperature_scale_K"
                "+composite*base_cure_source*reaction_enthalpy_scale"
            ),
            "energy_residual_temperature_scale_K": float(
                energy_temperature_scale_K
            ),
            "energy_residual_term_magnitude_denominator_available": False,
        }
    )

    physics_prediction_evaluable = bool(
        output["prediction_finite"]
        and temperature_nonpositive_count == 0
    )
    output["prediction_physics_evaluable"] = (
        physics_prediction_evaluable
    )
    if physics_prediction_evaluable:
        prediction_residual = _physics_residuals(
            temperature_prediction,
            predicted_alpha,
            time,
            z_coordinate,
            x_coordinate,
            mask,
            physics_context,
            energy_temperature_scale_K,
        )
        output.update(
            _energy_metrics(
                prediction_residual,
                spatial_weight,
                fixed_reference,
                truth=False,
            )
        )
        output["prediction_physics_status"] = (
            "ok"
            if output["energy_residual_finite"]
            else "nonfinite_derived_residual"
        )
    else:
        output.update(
            {
                "energy_residual_finite": False,
                "dimensionless_global_energy_residual": None,
                "dimensionless_energy_residual_mean_absolute": None,
                "dimensionless_energy_residual_rmse": None,
                "dimensionless_energy_residual_linf": None,
                "prediction_physics_status": (
                    "nonfinite_prediction"
                    if not output["prediction_finite"]
                    else "nonpositive_temperature_prediction"
                ),
            }
        )

    required_keys = (
        "temperature_relative_l2_K_composite",
        "peak_temperature_absolute_error_K",
        "final_alpha_mae_composite",
        "dimensionless_global_energy_residual",
    )
    required_values = [output[key] for key in required_keys]
    output["required_guardrail_metrics_finite"] = bool(
        all(
            value is not None
            and np.isfinite(float(value))
            and float(value) >= 0.0
            for value in required_values
        )
    )
    output["required_guardrail_metric_missing_count"] = int(
        sum(
            value is None or not np.isfinite(float(value))
            for value in required_values
        )
    )
    return output


@lru_cache(maxsize=1)
def target_2d_case_metric_columns() -> tuple[str, ...]:
    """Return the complete default case-metric schema without real labels.

    The schema is derived through the public implementation on a fixed
    synthetic case.  Release code can therefore prove that it persisted every
    value returned by the frozen callable without importing a model-specific
    adapter or touching an evaluation dataset.
    """

    shape = (3, 2, 2)
    temperature = np.full(shape, 300.0, dtype=np.float64)
    temperature[1:] += 1.0
    alpha = np.empty(shape, dtype=np.float64)
    alpha[0] = 0.0
    alpha[1] = 0.1
    alpha[2] = 0.2
    context = Target2DPhysicsContext(
        density_kg_m3=1_500.0,
        specific_heat_J_kg_K=1_000.0,
        conductivity_x_W_m_K=1.0,
        conductivity_z_W_m_K=1.0,
        base_cure_source_J_m3_per_alpha=1.0e8,
        air_temperature_K=(300.0, 300.0, 300.0),
    )
    metrics = compute_target_2d_case_metrics(
        temperature,
        temperature,
        alpha,
        alpha,
        np.asarray(((True, True), (False, False)), dtype=np.bool_),
        np.asarray((0.0, 120.0, 240.0), dtype=np.float64),
        np.asarray((0.0, 0.01), dtype=np.float64),
        np.asarray((0.0, 0.10), dtype=np.float64),
        context,
    )
    columns = tuple(sorted(metrics))
    if len(columns) != len(set(columns)):
        raise AssertionError("Target case-metric schema contains duplicates.")
    return columns


# Stable descriptive aliases for callers that prefer noun- or verb-led APIs.
target_2d_case_metrics = compute_target_2d_case_metrics
evaluate_target_2d_case = compute_target_2d_case_metrics


__all__ = [
    "ALPHA_VIOLATION_TOLERANCE",
    "DEFAULT_ALPHA_CROSSING_THRESHOLDS",
    "GRADIENT_TRUTH_NORM_FLOOR",
    "RANGE_UNDEFINED_THRESHOLD",
    "compute_target_2d_case_metrics",
    "evaluate_target_2d_case",
    "first_threshold_crossing_time",
    "target_2d_case_metric_columns",
    "target_2d_case_metrics",
]
