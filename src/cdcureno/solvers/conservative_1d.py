"""Conservative node-centred finite-volume solver for layered 1-D cure."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import solve_banded

from cdcureno.physics import CureKinetics, PublicCase1Material, cure_rate_per_s


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class LayeredGrid1D:
    z_m: FloatArray
    control_volume_width_m: FloatArray
    composite_mask: NDArray[np.bool_]
    density_kg_m3: FloatArray
    specific_heat_J_kg_K: FloatArray
    conductivity_W_m_K: FloatArray
    cure_source_J_m3_per_alpha: FloatArray


@dataclass(frozen=True)
class RobinBoundaries:
    lower_h_W_m2_K: float
    upper_h_W_m2_K: float


@dataclass(frozen=True)
class SolverDiagnostics:
    maximum_abs_energy_residual_W_m3: float
    maximum_relative_global_energy_residual: float
    substeps: int
    maximum_coupling_iterations: int


@dataclass(frozen=True)
class SolverResult:
    times_s: FloatArray
    z_m: FloatArray
    temperature_K: FloatArray
    alpha: FloatArray
    diagnostics: SolverDiagnostics


def layered_tool_composite_grid(
    spacing_m: float = 0.001,
    material: PublicCase1Material | None = None,
) -> LayeredGrid1D:
    """Build a uniform nodal grid for the configured tool/composite stack."""

    mat = material or PublicCase1Material()
    total = mat.tool_thickness_m + mat.composite_thickness_m
    intervals_float = total / spacing_m
    intervals = int(round(intervals_float))
    if spacing_m <= 0.0 or not np.isclose(intervals_float, intervals):
        raise ValueError("spacing_m must divide the 50 mm public domain.")
    z = np.linspace(0.0, total, intervals + 1, dtype=np.float64)
    widths = np.empty_like(z)
    widths[0] = 0.5 * (z[1] - z[0])
    widths[-1] = 0.5 * (z[-1] - z[-2])
    widths[1:-1] = 0.5 * (z[2:] - z[:-2])
    composite = z > mat.tool_thickness_m
    density = np.where(
        composite, mat.composite_density_kg_m3, mat.tool_density_kg_m3
    )
    cp = np.where(composite, mat.composite_cp_J_kg_K, mat.tool_cp_J_kg_K)
    conductivity = np.where(
        composite, mat.composite_k_W_m_K, mat.tool_k_W_m_K
    )
    source = np.where(composite, mat.cure_source_J_m3_per_alpha, 0.0)
    return LayeredGrid1D(
        z_m=z,
        control_volume_width_m=widths,
        composite_mask=composite,
        density_kg_m3=density,
        specific_heat_J_kg_K=cp,
        conductivity_W_m_K=conductivity,
        cure_source_J_m3_per_alpha=source,
    )


def public_case1_grid(
    spacing_m: float = 0.001,
    material: PublicCase1Material | None = None,
) -> LayeredGrid1D:
    """Build the public 20 mm tool + 30 mm composite nodal grid."""

    return layered_tool_composite_grid(spacing_m, material)


def _harmonic_face_conductance(grid: LayeredGrid1D) -> FloatArray:
    dz = np.diff(grid.z_m)
    left_half = 0.5 * dz
    right_half = 0.5 * dz
    resistance = (
        left_half / grid.conductivity_W_m_K[:-1]
        + right_half / grid.conductivity_W_m_K[1:]
    )
    return 1.0 / resistance


def _rk4_alpha(
    alpha_old: FloatArray,
    temperature_old: FloatArray,
    temperature_new_guess: FloatArray,
    dt_s: float,
    composite_mask: NDArray[np.bool_],
    kinetics: CureKinetics,
) -> FloatArray:
    new_alpha = alpha_old.copy()
    a0 = alpha_old[composite_mask]
    t0 = temperature_old[composite_mask]
    t1 = temperature_new_guess[composite_mask]
    tm = 0.5 * (t0 + t1)
    k1 = cure_rate_per_s(t0, a0, kinetics)
    k2 = cure_rate_per_s(tm, a0 + 0.5 * dt_s * k1, kinetics)
    k3 = cure_rate_per_s(tm, a0 + 0.5 * dt_s * k2, kinetics)
    k4 = cure_rate_per_s(t1, a0 + dt_s * k3, kinetics)
    increment = dt_s * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    new_alpha[composite_mask] = np.clip(a0 + increment, a0, 1.0)
    new_alpha[~composite_mask] = 0.0
    return new_alpha


def _temperature_step(
    grid: LayeredGrid1D,
    boundaries: RobinBoundaries,
    temperature_old: FloatArray,
    alpha_old: FloatArray,
    alpha_new: FloatArray,
    air_temperature_K: float,
    dt_s: float,
    face_conductance: FloatArray,
) -> FloatArray:
    storage = (
        grid.density_kg_m3
        * grid.specific_heat_J_kg_K
        * grid.control_volume_width_m
        / dt_s
    )
    source_per_area = (
        grid.cure_source_J_m3_per_alpha
        * (alpha_new - alpha_old)
        * grid.control_volume_width_m
        / dt_s
    )
    count = grid.z_m.size
    banded = np.zeros((3, count), dtype=np.float64)
    diagonal = storage.copy()
    diagonal[:-1] += face_conductance
    diagonal[1:] += face_conductance
    diagonal[0] += boundaries.lower_h_W_m2_K
    diagonal[-1] += boundaries.upper_h_W_m2_K
    banded[1] = diagonal
    banded[0, 1:] = -face_conductance
    banded[2, :-1] = -face_conductance
    rhs = storage * temperature_old + source_per_area
    rhs[0] += boundaries.lower_h_W_m2_K * air_temperature_K
    rhs[-1] += boundaries.upper_h_W_m2_K * air_temperature_K
    return solve_banded((1, 1), banded, rhs, check_finite=False)


def _energy_residual(
    grid: LayeredGrid1D,
    boundaries: RobinBoundaries,
    temperature_old: FloatArray,
    temperature_new: FloatArray,
    alpha_old: FloatArray,
    alpha_new: FloatArray,
    air_temperature_K: float,
    dt_s: float,
    face_conductance: FloatArray,
) -> tuple[float, float]:
    storage = (
        grid.density_kg_m3
        * grid.specific_heat_J_kg_K
        * (temperature_new - temperature_old)
        / dt_s
    )
    source = (
        grid.cure_source_J_m3_per_alpha * (alpha_new - alpha_old) / dt_s
    )
    outward_per_area = np.zeros_like(temperature_new)
    face_flux_right = face_conductance * (
        temperature_new[:-1] - temperature_new[1:]
    )
    outward_per_area[:-1] += face_flux_right
    outward_per_area[1:] -= face_flux_right
    outward_per_area[0] += boundaries.lower_h_W_m2_K * (
        temperature_new[0] - air_temperature_K
    )
    outward_per_area[-1] += boundaries.upper_h_W_m2_K * (
        temperature_new[-1] - air_temperature_K
    )
    residual = storage + outward_per_area / grid.control_volume_width_m - source
    maximum = float(np.max(np.abs(residual)))
    global_terms = np.array(
        [
            np.sum(storage * grid.control_volume_width_m),
            boundaries.lower_h_W_m2_K
            * (temperature_new[0] - air_temperature_K),
            boundaries.upper_h_W_m2_K
            * (temperature_new[-1] - air_temperature_K),
            -np.sum(source * grid.control_volume_width_m),
        ],
        dtype=np.float64,
    )
    # Near equilibrium, normalization by the almost-zero instantaneous flux
    # is ill-conditioned. A one-kelvin storage-rate scale retains SI meaning
    # and leaves the dimensional local residual as an independent safeguard.
    one_kelvin_storage_rate = float(
        np.sum(
            grid.density_kg_m3
            * grid.specific_heat_J_kg_K
            * grid.control_volume_width_m
        )
        / dt_s
    )
    scale = max(float(np.sum(np.abs(global_terms))), one_kelvin_storage_rate)
    relative = float(abs(np.sum(global_terms)) / scale)
    return maximum, relative


def simulate_cure_1d(
    times_s: ArrayLike,
    air_temperature_K: ArrayLike,
    *,
    grid: LayeredGrid1D | None = None,
    boundaries: RobinBoundaries | None = None,
    kinetics: CureKinetics | None = None,
    initial_temperature_K: float | ArrayLike = 293.0,
    initial_alpha: float = 0.05,
    maximum_step_s: float = 10.0,
    coupling_tolerance_K: float = 1.0e-9,
    maximum_coupling_iterations: int = 8,
) -> SolverResult:
    """Integrate the conservative layered thermochemical system.

    Diffusion is backward Euler, cure is RK4 over a linearly varying temperature
    guess, and source heat is inserted from the same alpha increment used in the
    state update. Picard iterations couple the two updates.
    """

    time = np.asarray(times_s, dtype=np.float64)
    air = np.asarray(air_temperature_K, dtype=np.float64)
    if time.ndim != 1 or air.shape != time.shape or time.size < 2:
        raise ValueError("times_s and air_temperature_K must be equal 1-D arrays.")
    if not np.all(np.diff(time) > 0.0):
        raise ValueError("times_s must be strictly increasing.")
    if maximum_step_s <= 0.0:
        raise ValueError("maximum_step_s must be positive.")
    active_grid = grid or public_case1_grid()
    mat = PublicCase1Material()
    active_boundaries = boundaries or RobinBoundaries(
        mat.lower_h_W_m2_K, mat.upper_h_W_m2_K
    )
    active_kinetics = kinetics or CureKinetics()
    count = active_grid.z_m.size
    if np.ndim(initial_temperature_K) == 0:
        temperature = np.full(count, float(initial_temperature_K))
    else:
        temperature = np.asarray(initial_temperature_K, dtype=np.float64).copy()
        if temperature.shape != (count,):
            raise ValueError("Initial temperature must match the spatial grid.")
    alpha = np.zeros(count, dtype=np.float64)
    alpha[active_grid.composite_mask] = initial_alpha
    temperatures = np.empty((time.size, count), dtype=np.float64)
    alphas = np.empty_like(temperatures)
    temperatures[0] = temperature
    alphas[0] = alpha
    faces = _harmonic_face_conductance(active_grid)
    maximum_local_residual = 0.0
    maximum_global_residual = 0.0
    total_substeps = 0
    used_coupling_iterations = 0

    for output_index in range(1, time.size):
        interval = time[output_index] - time[output_index - 1]
        steps = int(np.ceil(interval / maximum_step_s))
        dt_s = interval / steps
        for substep in range(steps):
            fraction_new = (substep + 1) / steps
            air_new = float(
                air[output_index - 1]
                + fraction_new * (air[output_index] - air[output_index - 1])
            )
            temperature_old = temperature
            alpha_old = alpha
            temperature_guess = temperature_old.copy()
            for coupling_index in range(1, maximum_coupling_iterations + 1):
                alpha_new = _rk4_alpha(
                    alpha_old,
                    temperature_old,
                    temperature_guess,
                    dt_s,
                    active_grid.composite_mask,
                    active_kinetics,
                )
                temperature_new = _temperature_step(
                    active_grid,
                    active_boundaries,
                    temperature_old,
                    alpha_old,
                    alpha_new,
                    air_new,
                    dt_s,
                    faces,
                )
                if np.max(np.abs(temperature_new - temperature_guess)) <= (
                    coupling_tolerance_K
                ):
                    break
                temperature_guess = temperature_new
            else:
                raise RuntimeError(
                    "Thermochemical Picard coupling failed to converge."
                )
            temperature = temperature_new
            alpha = alpha_new
            used_coupling_iterations = max(
                used_coupling_iterations, coupling_index
            )
            local_residual, global_residual = _energy_residual(
                active_grid,
                active_boundaries,
                temperature_old,
                temperature,
                alpha_old,
                alpha,
                air_new,
                dt_s,
                faces,
            )
            maximum_local_residual = max(
                maximum_local_residual, local_residual
            )
            maximum_global_residual = max(
                maximum_global_residual, global_residual
            )
            total_substeps += 1
        temperatures[output_index] = temperature
        alphas[output_index] = alpha

    return SolverResult(
        times_s=time,
        z_m=active_grid.z_m,
        temperature_K=temperatures,
        alpha=alphas,
        diagnostics=SolverDiagnostics(
            maximum_abs_energy_residual_W_m3=maximum_local_residual,
            maximum_relative_global_energy_residual=maximum_global_residual,
            substeps=total_substeps,
            maximum_coupling_iterations=used_coupling_iterations,
        ),
    )
