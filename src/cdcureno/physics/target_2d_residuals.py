"""Differentiable physical-consistency residuals for P6 target fields.

The canonical field order is ``[batch, time, z, x]``.  The energy residual is
an interval-integrated, cell-centred finite-volume balance,

``storage + trapezoidal(outward heat) - reaction heat = 0``.

Interior heat rates use harmonic face conductances and boundary heat rates use
the same half-cell-conduction/Robin-convection series resistance as the P4
solver.  All operations remain in PyTorch so losses can backpropagate through
both temperature and degree of cure.

P4 stores fields every 120 seconds although its solver advances with smaller
internal steps.  Therefore these residuals are *snapshot consistency metrics*,
not the exact residuals of every internal solver step.  Their irreducible
discretization floor must be measured on frozen training/validation truth
before loss weights or acceptance thresholds are fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import torch
import torch.nn.functional as functional
from torch import Tensor

from cdcureno.physics.as4_8552 import CureKinetics


TensorLike: TypeAlias = Tensor | float | int | list[float] | tuple[float, ...]


@dataclass(frozen=True)
class Target2DPhysicsContext:
    """Per-case material, boundary, and initial-condition data.

    Material fields accept a scalar, a shared ``[Z,X]`` field, a per-case
    ``[B]`` scalar, or a ``[B,Z,X]`` field.  Boundary coefficients accept a
    scalar, a shared boundary profile, a per-case scalar, or a per-case
    profile.  ``air_temperature_K`` accepts a scalar, ``[T]``, or ``[B,T]``.

    ``base_cure_source_J_m3_per_alpha`` is the unscaled composite reaction
    enthalpy per volume.  It is multiplied by
    ``reaction_enthalpy_scale`` and by ``composite_mask`` inside the residual.
    """

    density_kg_m3: TensorLike
    specific_heat_J_kg_K: TensorLike
    conductivity_x_W_m_K: TensorLike
    conductivity_z_W_m_K: TensorLike
    base_cure_source_J_m3_per_alpha: TensorLike
    air_temperature_K: TensorLike
    lower_h_W_m2_K: TensorLike = 0.0
    upper_h_W_m2_K: TensorLike = 0.0
    left_h_W_m2_K: TensorLike = 0.0
    right_h_W_m2_K: TensorLike = 0.0
    reaction_enthalpy_scale: TensorLike = 1.0
    initial_temperature_K: TensorLike = 293.0


@dataclass(frozen=True)
class Target2DResidualResult:
    """Cellwise residual fields and differentiable scalar mean-square losses."""

    energy_raw_J_m3: Tensor
    energy_dimensionless: Tensor
    cure_kinetics_dimensionless: Tensor
    temperature_initial_dimensionless: Tensor
    energy_mean_square: Tensor
    cure_kinetics_mean_square: Tensor
    temperature_initial_mean_square: Tensor


@dataclass(frozen=True)
class _ResolvedContext:
    density_kg_m3: Tensor
    specific_heat_J_kg_K: Tensor
    conductivity_x_W_m_K: Tensor
    conductivity_z_W_m_K: Tensor
    cure_source_J_m3_per_alpha: Tensor
    air_temperature_K: Tensor
    lower_h_W_m2_K: Tensor
    upper_h_W_m2_K: Tensor
    left_h_W_m2_K: Tensor
    right_h_W_m2_K: Tensor
    initial_temperature_K: Tensor


def _as_tensor(value: TensorLike, reference: Tensor) -> Tensor:
    return torch.as_tensor(
        value,
        dtype=reference.dtype,
        device=reference.device,
    )


def _require_finite(name: str, value: Tensor) -> None:
    if not bool(torch.all(torch.isfinite(value.detach())).item()):
        raise ValueError(f"{name} must contain only finite values.")


def _require_positive(name: str, value: Tensor) -> None:
    _require_finite(name, value)
    if not bool(torch.all(value.detach() > 0.0).item()):
        raise ValueError(f"{name} must contain only positive values.")


def _require_nonnegative(name: str, value: Tensor) -> None:
    _require_finite(name, value)
    if not bool(torch.all(value.detach() >= 0.0).item()):
        raise ValueError(f"{name} must contain only nonnegative values.")


def _field_bzx(
    value: TensorLike,
    reference: Tensor,
    *,
    batch: int,
    z_count: int,
    x_count: int,
    name: str,
) -> Tensor:
    tensor = _as_tensor(value, reference)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1, 1, 1)
    elif tensor.shape == (batch,):
        tensor = tensor[:, None, None]
    elif tensor.shape == (z_count, x_count):
        tensor = tensor[None, :, :]
    elif tensor.shape == (1, z_count, x_count):
        pass
    elif tensor.shape == (batch, 1, 1):
        pass
    elif tensor.shape != (batch, z_count, x_count):
        raise ValueError(
            f"{name} must be scalar, [B], [Z,X], or [B,Z,X]; "
            f"got {tuple(tensor.shape)}."
        )
    try:
        return tensor.expand(batch, z_count, x_count)
    except RuntimeError as error:
        raise ValueError(f"{name} cannot broadcast to [B,Z,X].") from error


def _air_bt(
    value: TensorLike,
    reference: Tensor,
    *,
    batch: int,
    time_count: int,
) -> Tensor:
    tensor = _as_tensor(value, reference)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1, 1)
    elif tensor.shape == (time_count,):
        tensor = tensor[None, :]
    elif tensor.shape == (batch,):
        tensor = tensor[:, None]
    elif tensor.shape not in {
        (1, time_count),
        (batch, 1),
        (batch, time_count),
    }:
        raise ValueError(
            "air_temperature_K must be scalar, [T], [B], or [B,T]; "
            f"got {tuple(tensor.shape)}."
        )
    return tensor.expand(batch, time_count)


def _boundary_ba(
    value: TensorLike,
    reference: Tensor,
    *,
    batch: int,
    axis_count: int,
    name: str,
) -> Tensor:
    tensor = _as_tensor(value, reference)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1, 1)
    elif tensor.ndim == 1:
        if tensor.shape[0] == batch and tensor.shape[0] == axis_count:
            raise ValueError(
                f"{name} has an ambiguous one-dimensional shape; use "
                "[B,1] for per-case scalars or [1,A] for a shared profile."
            )
        if tensor.shape[0] == batch:
            tensor = tensor[:, None]
        elif tensor.shape[0] == axis_count:
            tensor = tensor[None, :]
        else:
            raise ValueError(
                f"{name} one-dimensional length must be B or boundary size."
            )
    elif tensor.shape not in {
        (1, axis_count),
        (batch, 1),
        (batch, axis_count),
    }:
        raise ValueError(
            f"{name} must be scalar, [B], [A], [B,1], or [B,A]; "
            f"got {tuple(tensor.shape)}."
        )
    return tensor.expand(batch, axis_count)


def _coordinate(
    value: TensorLike,
    reference: Tensor,
    *,
    count: int,
    name: str,
) -> Tensor:
    tensor = _as_tensor(value, reference)
    if tensor.shape != (count,):
        raise ValueError(f"{name} must have shape [{count}].")
    _require_finite(name, tensor)
    if not bool(torch.all(torch.diff(tensor.detach()) > 0.0).item()):
        raise ValueError(f"{name} must be strictly increasing.")
    return tensor


def _control_volume_widths(coordinates: Tensor) -> Tensor:
    gaps = torch.diff(coordinates)
    if coordinates.numel() == 2:
        return torch.stack((gaps[0], gaps[0]))
    return torch.cat(
        (
            gaps[:1],
            0.5 * (gaps[:-1] + gaps[1:]),
            gaps[-1:],
        )
    )


def _composite_mask_bzx(
    value: TensorLike,
    reference: Tensor,
    *,
    batch: int,
    z_count: int,
    x_count: int,
) -> Tensor:
    mask = _field_bzx(
        value,
        reference,
        batch=batch,
        z_count=z_count,
        x_count=x_count,
        name="composite_mask",
    )
    _require_finite("composite_mask", mask)
    binary = (mask.detach() == 0.0) | (mask.detach() == 1.0)
    if not bool(torch.all(binary).item()):
        raise ValueError("composite_mask must contain only zeros and ones.")
    if not bool(torch.any(mask.detach() > 0.5).item()):
        raise ValueError("composite_mask must contain at least one cell.")
    return mask > 0.5


def _resolve_context(
    context: Target2DPhysicsContext,
    temperature_K: Tensor,
    composite_mask: Tensor,
) -> _ResolvedContext:
    batch, time_count, z_count, x_count = temperature_K.shape
    field_arguments = {
        "density_kg_m3": context.density_kg_m3,
        "specific_heat_J_kg_K": context.specific_heat_J_kg_K,
        "conductivity_x_W_m_K": context.conductivity_x_W_m_K,
        "conductivity_z_W_m_K": context.conductivity_z_W_m_K,
        "base_cure_source_J_m3_per_alpha": (
            context.base_cure_source_J_m3_per_alpha
        ),
        "reaction_enthalpy_scale": context.reaction_enthalpy_scale,
        "initial_temperature_K": context.initial_temperature_K,
    }
    fields = {
        name: _field_bzx(
            value,
            temperature_K,
            batch=batch,
            z_count=z_count,
            x_count=x_count,
            name=name,
        )
        for name, value in field_arguments.items()
    }
    for name in (
        "density_kg_m3",
        "specific_heat_J_kg_K",
        "conductivity_x_W_m_K",
        "conductivity_z_W_m_K",
        "reaction_enthalpy_scale",
    ):
        _require_positive(name, fields[name])
    _require_nonnegative(
        "base_cure_source_J_m3_per_alpha",
        fields["base_cure_source_J_m3_per_alpha"],
    )
    _require_positive(
        "initial_temperature_K", fields["initial_temperature_K"]
    )
    cure_source = (
        fields["base_cure_source_J_m3_per_alpha"]
        * fields["reaction_enthalpy_scale"]
        * composite_mask.to(dtype=temperature_K.dtype)
    )
    air = _air_bt(
        context.air_temperature_K,
        temperature_K,
        batch=batch,
        time_count=time_count,
    )
    _require_positive("air_temperature_K", air)
    lower = _boundary_ba(
        context.lower_h_W_m2_K,
        temperature_K,
        batch=batch,
        axis_count=x_count,
        name="lower_h_W_m2_K",
    )
    upper = _boundary_ba(
        context.upper_h_W_m2_K,
        temperature_K,
        batch=batch,
        axis_count=x_count,
        name="upper_h_W_m2_K",
    )
    left = _boundary_ba(
        context.left_h_W_m2_K,
        temperature_K,
        batch=batch,
        axis_count=z_count,
        name="left_h_W_m2_K",
    )
    right = _boundary_ba(
        context.right_h_W_m2_K,
        temperature_K,
        batch=batch,
        axis_count=z_count,
        name="right_h_W_m2_K",
    )
    for name, value in (
        ("lower_h_W_m2_K", lower),
        ("upper_h_W_m2_K", upper),
        ("left_h_W_m2_K", left),
        ("right_h_W_m2_K", right),
    ):
        _require_nonnegative(name, value)
    return _ResolvedContext(
        density_kg_m3=fields["density_kg_m3"],
        specific_heat_J_kg_K=fields["specific_heat_J_kg_K"],
        conductivity_x_W_m_K=fields["conductivity_x_W_m_K"],
        conductivity_z_W_m_K=fields["conductivity_z_W_m_K"],
        cure_source_J_m3_per_alpha=cure_source,
        air_temperature_K=air,
        lower_h_W_m2_K=lower,
        upper_h_W_m2_K=upper,
        left_h_W_m2_K=left,
        right_h_W_m2_K=right,
        initial_temperature_K=fields["initial_temperature_K"],
    )


def _harmonic_face_conductance(
    first_k: Tensor,
    second_k: Tensor,
    distance_m: Tensor,
    face_measure_m: Tensor,
) -> Tensor:
    resistance = 0.5 * distance_m / first_k + 0.5 * distance_m / second_k
    return face_measure_m / resistance


def _robin_boundary_conductance(
    h_W_m2_K: Tensor,
    conductivity_W_m_K: Tensor,
    half_cell_distance_m: Tensor,
    face_measure_m: Tensor,
) -> Tensor:
    active = h_W_m2_K > 0.0
    safe_h = torch.where(active, h_W_m2_K, torch.ones_like(h_W_m2_K))
    conductance = face_measure_m / (
        1.0 / safe_h + half_cell_distance_m / conductivity_W_m_K
    )
    return torch.where(active, conductance, torch.zeros_like(conductance))


def _outward_heat_rate_density(
    temperature_K: Tensor,
    air_temperature_K: Tensor,
    context: _ResolvedContext,
    *,
    z_m: Tensor,
    x_m: Tensor,
) -> Tensor:
    """Return outward finite-volume heat rate per cell volume in W m^-3."""

    z_width = _control_volume_widths(z_m)
    x_width = _control_volume_widths(x_m)
    z_gap = torch.diff(z_m)
    x_gap = torch.diff(x_m)
    volume = z_width[:, None] * x_width[None, :]

    x_conductance = _harmonic_face_conductance(
        context.conductivity_x_W_m_K[:, :, :-1],
        context.conductivity_x_W_m_K[:, :, 1:],
        x_gap[None, None, :],
        z_width[None, :, None],
    )
    x_rate = x_conductance[:, None, :, :] * (
        temperature_K[:, :, :, :-1] - temperature_K[:, :, :, 1:]
    )
    outward = functional.pad(x_rate, (0, 1)) - functional.pad(
        x_rate, (1, 0)
    )

    z_conductance = _harmonic_face_conductance(
        context.conductivity_z_W_m_K[:, :-1, :],
        context.conductivity_z_W_m_K[:, 1:, :],
        z_gap[None, :, None],
        x_width[None, None, :],
    )
    z_rate = z_conductance[:, None, :, :] * (
        temperature_K[:, :, :-1, :] - temperature_K[:, :, 1:, :]
    )
    outward = outward + functional.pad(
        z_rate, (0, 0, 0, 1)
    ) - functional.pad(z_rate, (0, 0, 1, 0))

    lower_conductance = _robin_boundary_conductance(
        context.lower_h_W_m2_K,
        context.conductivity_z_W_m_K[:, 0, :],
        0.5 * z_width[0],
        x_width[None, :],
    )
    upper_conductance = _robin_boundary_conductance(
        context.upper_h_W_m2_K,
        context.conductivity_z_W_m_K[:, -1, :],
        0.5 * z_width[-1],
        x_width[None, :],
    )
    left_conductance = _robin_boundary_conductance(
        context.left_h_W_m2_K,
        context.conductivity_x_W_m_K[:, :, 0],
        0.5 * x_width[0],
        z_width[None, :],
    )
    right_conductance = _robin_boundary_conductance(
        context.right_h_W_m2_K,
        context.conductivity_x_W_m_K[:, :, -1],
        0.5 * x_width[-1],
        z_width[None, :],
    )
    air = air_temperature_K[:, :, None]
    lower_rate = lower_conductance[:, None, :] * (
        temperature_K[:, :, 0, :] - air
    )
    upper_rate = upper_conductance[:, None, :] * (
        temperature_K[:, :, -1, :] - air
    )
    left_rate = left_conductance[:, None, :] * (
        temperature_K[:, :, :, 0] - air
    )
    right_rate = right_conductance[:, None, :] * (
        temperature_K[:, :, :, -1] - air
    )
    z_count = temperature_K.shape[2]
    x_count = temperature_K.shape[3]
    outward = outward + functional.pad(
        lower_rate.unsqueeze(2), (0, 0, 0, z_count - 1)
    )
    outward = outward + functional.pad(
        upper_rate.unsqueeze(2), (0, 0, z_count - 1, 0)
    )
    outward = outward + functional.pad(
        left_rate.unsqueeze(3), (0, x_count - 1)
    )
    outward = outward + functional.pad(
        right_rate.unsqueeze(3), (x_count - 1, 0)
    )
    return outward / volume[None, None, :, :]


def cure_rate_per_s_torch(
    temperature_K: Tensor,
    alpha: Tensor,
    parameters: CureKinetics | None = None,
) -> Tensor:
    """Differentiable Hubert--Johnston cure law in inverse seconds."""

    if temperature_K.shape != alpha.shape:
        raise ValueError("temperature_K and alpha must have equal shapes.")
    _require_finite("temperature_K", temperature_K)
    _require_finite("alpha", alpha)
    if not bool(torch.all(temperature_K.detach() > 0.0).item()):
        raise ValueError("Cure kinetics require positive Kelvin temperatures.")
    values = parameters or CureKinetics()
    bounded = torch.clamp(alpha, 0.0, 1.0)
    interior = (bounded > 0.0) & (bounded < 1.0)
    # Fractional powers have infinite derivatives at alpha=0 or 1.  Evaluate
    # their unselected branch on a finite base, then impose the exact physical
    # zero rate at both bounds.  This is especially important in tool cells,
    # where alpha is identically zero and a later mask would otherwise produce
    # the indeterminate autograd product 0 * inf.
    alpha_power_base = torch.where(
        bounded > 0.0, bounded, torch.ones_like(bounded)
    )
    remaining_power_base = torch.where(
        bounded < 1.0, 1.0 - bounded, torch.ones_like(bounded)
    )
    exponent = values.C * (
        bounded - values.C_T_per_K * temperature_K - values.C_0
    )
    denominator = values.denominator_offset + torch.exp(
        torch.clamp(exponent, -80.0, 80.0)
    )
    rate = (
        values.A_per_s
        * torch.exp(
            -values.delta_E_J_per_mol
            / (values.R_J_per_mol_K * temperature_K)
        )
        * torch.pow(alpha_power_base, values.M)
        * torch.pow(remaining_power_base, values.N)
        / denominator
    )
    finite_rate = torch.clamp_min(rate, 0.0)
    return torch.where(interior, finite_rate, torch.zeros_like(finite_rate))


def trapezoidal_energy_residual(
    temperature_K: Tensor,
    alpha: Tensor,
    time_s: TensorLike,
    z_m: TensorLike,
    x_m: TensorLike,
    composite_mask: TensorLike,
    context: Target2DPhysicsContext,
    *,
    temperature_scale_K: float = 100.0,
) -> tuple[Tensor, Tensor]:
    """Return raw and dimensionless interval-integrated FV energy residuals."""

    (
        temperature,
        degree,
        time,
        z_coordinate,
        x_coordinate,
        mask,
        resolved,
    ) = _validate_and_resolve(
        temperature_K,
        alpha,
        time_s,
        z_m,
        x_m,
        composite_mask,
        context,
    )
    if not torch.isfinite(torch.as_tensor(temperature_scale_K)) or (
        temperature_scale_K <= 0.0
    ):
        raise ValueError("temperature_scale_K must be finite and positive.")
    time_step = torch.diff(time)[None, :, None, None]
    outward = _outward_heat_rate_density(
        temperature,
        resolved.air_temperature_K,
        resolved,
        z_m=z_coordinate,
        x_m=x_coordinate,
    )
    rho_cp = (
        resolved.density_kg_m3 * resolved.specific_heat_J_kg_K
    )[:, None, :, :]
    cure_source = resolved.cure_source_J_m3_per_alpha[:, None, :, :]
    storage = rho_cp * (temperature[:, 1:] - temperature[:, :-1])
    boundary_and_conduction = 0.5 * time_step * (
        outward[:, 1:] + outward[:, :-1]
    )
    reaction = cure_source * (degree[:, 1:] - degree[:, :-1])
    raw = storage + boundary_and_conduction - reaction
    reference = (
        resolved.density_kg_m3
        * resolved.specific_heat_J_kg_K
        * float(temperature_scale_K)
        + resolved.cure_source_J_m3_per_alpha
    )
    epsilon = torch.finfo(temperature.dtype).eps
    dimensionless = raw / reference[:, None, :, :].clamp_min(epsilon)
    del mask
    return raw, dimensionless


def cure_kinetics_interval_residual(
    temperature_K: Tensor,
    alpha: Tensor,
    time_s: TensorLike,
    composite_mask: TensorLike,
    *,
    parameters: CureKinetics | None = None,
) -> Tensor:
    """Return the dimensionless trapezoidal cure-kinetics interval residual."""

    if temperature_K.ndim != 4 or alpha.shape != temperature_K.shape:
        raise ValueError(
            "temperature_K and alpha must have equal [B,T,Z,X] shapes."
        )
    batch, time_count, z_count, x_count = temperature_K.shape
    if min(batch, time_count, z_count, x_count) < 1 or time_count < 2:
        raise ValueError("Cure residual needs nonempty fields and T >= 2.")
    _require_finite("temperature_K", temperature_K)
    _require_finite("alpha", alpha)
    time = _coordinate(
        time_s,
        temperature_K,
        count=time_count,
        name="time_s",
    )
    mask = _composite_mask_bzx(
        composite_mask,
        temperature_K,
        batch=batch,
        z_count=z_count,
        x_count=x_count,
    )
    rates = cure_rate_per_s_torch(
        temperature_K,
        alpha,
        parameters=parameters,
    )
    time_step = torch.diff(time)[None, :, None, None]
    residual = (alpha[:, 1:] - alpha[:, :-1]) - 0.5 * time_step * (
        rates[:, 1:] + rates[:, :-1]
    )
    return residual * mask[:, None, :, :].to(dtype=temperature_K.dtype)


def temperature_initial_condition_residual(
    temperature_K: Tensor,
    initial_temperature_K: TensorLike,
    *,
    temperature_scale_K: float = 100.0,
) -> Tensor:
    """Return the dimensionless temperature initial-condition residual."""

    if temperature_K.ndim != 4:
        raise ValueError("temperature_K must have shape [B,T,Z,X].")
    _require_finite("temperature_K", temperature_K)
    if not torch.isfinite(torch.as_tensor(temperature_scale_K)) or (
        temperature_scale_K <= 0.0
    ):
        raise ValueError("temperature_scale_K must be finite and positive.")
    batch, _, z_count, x_count = temperature_K.shape
    initial = _field_bzx(
        initial_temperature_K,
        temperature_K,
        batch=batch,
        z_count=z_count,
        x_count=x_count,
        name="initial_temperature_K",
    )
    _require_positive("initial_temperature_K", initial)
    return (temperature_K[:, 0] - initial) / float(temperature_scale_K)


def _validate_and_resolve(
    temperature_K: Tensor,
    alpha: Tensor,
    time_s: TensorLike,
    z_m: TensorLike,
    x_m: TensorLike,
    composite_mask: TensorLike,
    context: Target2DPhysicsContext,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, _ResolvedContext]:
    if not isinstance(temperature_K, Tensor) or not isinstance(alpha, Tensor):
        raise TypeError("temperature_K and alpha must be torch tensors.")
    if temperature_K.ndim != 4 or alpha.shape != temperature_K.shape:
        raise ValueError(
            "temperature_K and alpha must have equal [B,T,Z,X] shapes."
        )
    if not temperature_K.is_floating_point() or not alpha.is_floating_point():
        raise TypeError("temperature_K and alpha must be floating-point tensors.")
    if temperature_K.device != alpha.device or temperature_K.dtype != alpha.dtype:
        raise ValueError("temperature_K and alpha must share device and dtype.")
    batch, time_count, z_count, x_count = temperature_K.shape
    if (
        batch < 1
        or time_count < 2
        or z_count < 2
        or x_count < 2
    ):
        raise ValueError("Residuals require B >= 1 and T,Z,X >= 2.")
    _require_finite("temperature_K", temperature_K)
    _require_finite("alpha", alpha)
    if not bool(torch.all(temperature_K.detach() > 0.0).item()):
        raise ValueError("temperature_K must contain positive Kelvin values.")
    time = _coordinate(
        time_s,
        temperature_K,
        count=time_count,
        name="time_s",
    )
    z_coordinate = _coordinate(
        z_m,
        temperature_K,
        count=z_count,
        name="z_m",
    )
    x_coordinate = _coordinate(
        x_m,
        temperature_K,
        count=x_count,
        name="x_m",
    )
    mask = _composite_mask_bzx(
        composite_mask,
        temperature_K,
        batch=batch,
        z_count=z_count,
        x_count=x_count,
    )
    resolved = _resolve_context(context, temperature_K, mask)
    return (
        temperature_K,
        alpha,
        time,
        z_coordinate,
        x_coordinate,
        mask,
        resolved,
    )


def _mean_square(value: Tensor, mask: Tensor | None = None) -> Tensor:
    squared = value.square()
    if mask is None:
        return torch.mean(squared)
    weights = mask.to(dtype=value.dtype)
    expanded = weights.expand_as(value)
    denominator = torch.sum(expanded).clamp_min(1.0)
    return torch.sum(squared * expanded) / denominator


def target_2d_physics_residuals(
    temperature_K: Tensor,
    alpha: Tensor,
    time_s: TensorLike,
    z_m: TensorLike,
    x_m: TensorLike,
    composite_mask: TensorLike,
    context: Target2DPhysicsContext,
    *,
    temperature_scale_K: float = 100.0,
    kinetics: CureKinetics | None = None,
) -> Target2DResidualResult:
    """Evaluate all differentiable P6 physical-consistency residuals."""

    raw_energy, energy = trapezoidal_energy_residual(
        temperature_K,
        alpha,
        time_s,
        z_m,
        x_m,
        composite_mask,
        context,
        temperature_scale_K=temperature_scale_K,
    )
    cure = cure_kinetics_interval_residual(
        temperature_K,
        alpha,
        time_s,
        composite_mask,
        parameters=kinetics,
    )
    initial = temperature_initial_condition_residual(
        temperature_K,
        context.initial_temperature_K,
        temperature_scale_K=temperature_scale_K,
    )
    batch, _, z_count, x_count = temperature_K.shape
    mask = _composite_mask_bzx(
        composite_mask,
        temperature_K,
        batch=batch,
        z_count=z_count,
        x_count=x_count,
    )
    interval_mask = mask[:, None, :, :].expand_as(cure)
    return Target2DResidualResult(
        energy_raw_J_m3=raw_energy,
        energy_dimensionless=energy,
        cure_kinetics_dimensionless=cure,
        temperature_initial_dimensionless=initial,
        energy_mean_square=_mean_square(energy),
        cure_kinetics_mean_square=_mean_square(cure, interval_mask),
        temperature_initial_mean_square=_mean_square(initial),
    )
