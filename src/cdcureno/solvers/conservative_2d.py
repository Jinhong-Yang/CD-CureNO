"""Conservative structured finite-volume solver for 2-D composite cure."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import coo_matrix, csc_matrix, diags
from scipy.sparse.linalg import SuperLU, splu

from cdcureno.physics import CureKinetics, PublicCase1Material, cure_rate_per_s


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class LayeredGrid2D:
    """Rectangular cell-centred grid with arrays ordered ``[z, x]``."""

    x_m: FloatArray
    z_m: FloatArray
    control_volume_width_x_m: FloatArray
    control_volume_width_z_m: FloatArray
    composite_mask: NDArray[np.bool_]
    density_kg_m3: FloatArray
    specific_heat_J_kg_K: FloatArray
    conductivity_x_W_m_K: FloatArray
    conductivity_z_W_m_K: FloatArray
    cure_source_J_m3_per_alpha: FloatArray

    @property
    def shape(self) -> tuple[int, int]:
        return self.composite_mask.shape

    @property
    def control_volume_m2(self) -> FloatArray:
        return np.outer(
            self.control_volume_width_z_m, self.control_volume_width_x_m
        )


@dataclass(frozen=True)
class RobinBoundaries2D:
    """Robin coefficients on bottom/top/left/right boundaries in SI units."""

    lower_h_W_m2_K: float | ArrayLike
    upper_h_W_m2_K: float | ArrayLike
    left_h_W_m2_K: float | ArrayLike = 0.0
    right_h_W_m2_K: float | ArrayLike = 0.0


@dataclass(frozen=True)
class SolverDiagnostics2D:
    maximum_abs_energy_residual_W_m3: float
    maximum_relative_global_energy_residual: float
    maximum_interface_flux_imbalance_W: float
    maximum_temperature_interface_jump_K: float
    maximum_robin_flux_imbalance_W: float
    all_coupling_steps_converged: bool
    maximum_converged_coupling_update_K: float
    substeps: int
    maximum_coupling_iterations: int
    wall_seconds: float


@dataclass(frozen=True)
class SolverResult2D:
    times_s: FloatArray
    x_m: FloatArray
    z_m: FloatArray
    temperature_K: FloatArray
    alpha: FloatArray
    diagnostics: SolverDiagnostics2D


@dataclass(frozen=True)
class _DiscreteOperator2D:
    diffusion_and_robin: csc_matrix
    robin_rhs_coefficient: FloatArray
    x_face_conductance_W_K: FloatArray
    z_face_conductance_W_K: FloatArray
    lower_robin_conductance_W_K: FloatArray
    upper_robin_conductance_W_K: FloatArray
    left_robin_conductance_W_K: FloatArray
    right_robin_conductance_W_K: FloatArray


def rectangular_tool_composite_grid(
    *,
    width_m: float = 0.20,
    spacing_x_m: float = 0.01,
    spacing_z_m: float = 0.001,
    material: PublicCase1Material | None = None,
    composite_conductivity_x_scale: float = 1.0,
    composite_conductivity_z_scale: float = 1.0,
) -> LayeredGrid2D:
    """Build the public layered stack extruded over a rectangular width."""

    mat = material or PublicCase1Material()
    total_z = mat.tool_thickness_m + mat.composite_thickness_m
    nx_intervals = int(round(width_m / spacing_x_m))
    nz_intervals = int(round(total_z / spacing_z_m))
    if (
        width_m <= 0.0
        or spacing_x_m <= 0.0
        or spacing_z_m <= 0.0
        or not np.isclose(nx_intervals * spacing_x_m, width_m)
        or not np.isclose(nz_intervals * spacing_z_m, total_z)
    ):
        raise ValueError("Spatial steps must divide the rectangular domain.")
    if composite_conductivity_x_scale <= 0.0:
        raise ValueError("Composite x-conductivity scale must be positive.")
    if composite_conductivity_z_scale <= 0.0:
        raise ValueError("Composite z-conductivity scale must be positive.")

    x = (np.arange(nx_intervals, dtype=np.float64) + 0.5) * spacing_x_m
    z = (np.arange(nz_intervals, dtype=np.float64) + 0.5) * spacing_z_m
    x_widths = np.full(nx_intervals, spacing_x_m, dtype=np.float64)
    z_widths = np.full(nz_intervals, spacing_z_m, dtype=np.float64)
    composite_by_z = z >= mat.tool_thickness_m
    composite = np.broadcast_to(
        composite_by_z[:, None], (z.size, x.size)
    ).copy()

    density = np.where(
        composite, mat.composite_density_kg_m3, mat.tool_density_kg_m3
    )
    cp = np.where(
        composite, mat.composite_cp_J_kg_K, mat.tool_cp_J_kg_K
    )
    conductivity_x = np.where(
        composite,
        mat.composite_longitudinal_k_W_m_K
        * composite_conductivity_x_scale,
        mat.tool_k_W_m_K,
    )
    conductivity_z = np.where(
        composite,
        mat.composite_k_W_m_K * composite_conductivity_z_scale,
        mat.tool_k_W_m_K,
    )
    source = np.where(composite, mat.cure_source_J_m3_per_alpha, 0.0)
    return LayeredGrid2D(
        x_m=x,
        z_m=z,
        control_volume_width_x_m=x_widths,
        control_volume_width_z_m=z_widths,
        composite_mask=composite,
        density_kg_m3=density,
        specific_heat_J_kg_K=cp,
        conductivity_x_W_m_K=conductivity_x,
        conductivity_z_W_m_K=conductivity_z,
        cure_source_J_m3_per_alpha=source,
    )


def _boundary_array(
    value: float | ArrayLike, size: int, name: str
) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(size, float(array), dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must be scalar or have shape ({size},).")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any(array < 0.0):
        raise ValueError(f"{name} must be nonnegative.")
    return array


def _validate_grid_2d(grid: LayeredGrid2D) -> None:
    nz, nx = grid.shape
    if nz < 1 or nx < 1:
        raise ValueError("The 2-D grid must contain at least one cell per axis.")
    if grid.x_m.shape != (nx,) or grid.z_m.shape != (nz,):
        raise ValueError("Grid coordinates must match the material-field shape.")
    if (
        grid.control_volume_width_x_m.shape != (nx,)
        or grid.control_volume_width_z_m.shape != (nz,)
    ):
        raise ValueError("Control-volume widths must match the grid axes.")
    for name, coordinate in (("x_m", grid.x_m), ("z_m", grid.z_m)):
        if not np.all(np.isfinite(coordinate)):
            raise ValueError(f"{name} must contain only finite values.")
        if coordinate.size > 1 and not np.all(np.diff(coordinate) > 0.0):
            raise ValueError(f"{name} must be strictly increasing.")
    for name, width in (
        ("control_volume_width_x_m", grid.control_volume_width_x_m),
        ("control_volume_width_z_m", grid.control_volume_width_z_m),
    ):
        if not np.all(np.isfinite(width)) or np.any(width <= 0.0):
            raise ValueError(f"{name} must contain finite positive values.")
    positive_fields = (
        ("density_kg_m3", grid.density_kg_m3),
        ("specific_heat_J_kg_K", grid.specific_heat_J_kg_K),
        ("conductivity_x_W_m_K", grid.conductivity_x_W_m_K),
        ("conductivity_z_W_m_K", grid.conductivity_z_W_m_K),
    )
    for name, field in positive_fields:
        if field.shape != grid.shape:
            raise ValueError(f"{name} must match the grid shape.")
        if not np.all(np.isfinite(field)) or np.any(field <= 0.0):
            raise ValueError(f"{name} must contain finite positive values.")
    if grid.cure_source_J_m3_per_alpha.shape != grid.shape:
        raise ValueError("cure_source_J_m3_per_alpha must match the grid shape.")
    if (
        not np.all(np.isfinite(grid.cure_source_J_m3_per_alpha))
        or np.any(grid.cure_source_J_m3_per_alpha < 0.0)
    ):
        raise ValueError(
            "cure_source_J_m3_per_alpha must contain finite nonnegative values."
        )


def _harmonic_conductance(
    left_k: FloatArray,
    right_k: FloatArray,
    distance_m: FloatArray,
    face_measure_m: FloatArray,
) -> FloatArray:
    resistance = 0.5 * distance_m / left_k + 0.5 * distance_m / right_k
    return face_measure_m / resistance


def _robin_boundary_conductance(
    h_W_m2_K: FloatArray,
    conductivity_W_m_K: FloatArray,
    half_cell_distance_m: FloatArray,
    face_measure_m: FloatArray,
) -> FloatArray:
    """Combine half-cell conduction and convection as series resistances."""

    conductance = np.zeros_like(h_W_m2_K)
    active = h_W_m2_K > 0.0
    conductance[active] = face_measure_m[active] / (
        1.0 / h_W_m2_K[active]
        + half_cell_distance_m[active] / conductivity_W_m_K[active]
    )
    return conductance


def _assemble_operator(
    grid: LayeredGrid2D, boundaries: RobinBoundaries2D
) -> _DiscreteOperator2D:
    nz, nx = grid.shape
    dx = np.diff(grid.x_m)
    dz = np.diff(grid.z_m)
    x_faces = _harmonic_conductance(
        grid.conductivity_x_W_m_K[:, :-1],
        grid.conductivity_x_W_m_K[:, 1:],
        dx[None, :],
        grid.control_volume_width_z_m[:, None],
    )
    z_faces = _harmonic_conductance(
        grid.conductivity_z_W_m_K[:-1, :],
        grid.conductivity_z_W_m_K[1:, :],
        dz[:, None],
        grid.control_volume_width_x_m[None, :],
    )

    lower_h = _boundary_array(
        boundaries.lower_h_W_m2_K, nx, "lower_h_W_m2_K"
    )
    upper_h = _boundary_array(
        boundaries.upper_h_W_m2_K, nx, "upper_h_W_m2_K"
    )
    left_h = _boundary_array(
        boundaries.left_h_W_m2_K, nz, "left_h_W_m2_K"
    )
    right_h = _boundary_array(
        boundaries.right_h_W_m2_K, nz, "right_h_W_m2_K"
    )

    node_count = nz * nx
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    diagonal = np.zeros((nz, nx), dtype=np.float64)

    def add_symmetric_faces(
        first: NDArray[np.int64],
        second: NDArray[np.int64],
        conductance: FloatArray,
    ) -> None:
        flat_first = first.ravel()
        flat_second = second.ravel()
        flat_g = conductance.ravel()
        rows.extend(np.concatenate([flat_first, flat_second]).tolist())
        columns.extend(np.concatenate([flat_second, flat_first]).tolist())
        values.extend(np.concatenate([-flat_g, -flat_g]).tolist())

    indices = np.arange(node_count, dtype=np.int64).reshape(nz, nx)
    diagonal[:, :-1] += x_faces
    diagonal[:, 1:] += x_faces
    add_symmetric_faces(indices[:, :-1], indices[:, 1:], x_faces)
    diagonal[:-1, :] += z_faces
    diagonal[1:, :] += z_faces
    add_symmetric_faces(indices[:-1, :], indices[1:, :], z_faces)

    lower_robin = _robin_boundary_conductance(
        lower_h,
        grid.conductivity_z_W_m_K[0, :],
        np.full(nx, 0.5 * grid.control_volume_width_z_m[0]),
        grid.control_volume_width_x_m,
    )
    upper_robin = _robin_boundary_conductance(
        upper_h,
        grid.conductivity_z_W_m_K[-1, :],
        np.full(nx, 0.5 * grid.control_volume_width_z_m[-1]),
        grid.control_volume_width_x_m,
    )
    left_robin = _robin_boundary_conductance(
        left_h,
        grid.conductivity_x_W_m_K[:, 0],
        0.5 * grid.control_volume_width_x_m[0]
        * np.ones(nz, dtype=np.float64),
        grid.control_volume_width_z_m,
    )
    right_robin = _robin_boundary_conductance(
        right_h,
        grid.conductivity_x_W_m_K[:, -1],
        0.5 * grid.control_volume_width_x_m[-1]
        * np.ones(nz, dtype=np.float64),
        grid.control_volume_width_z_m,
    )
    robin = np.zeros((nz, nx), dtype=np.float64)
    robin[0, :] += lower_robin
    robin[-1, :] += upper_robin
    robin[:, 0] += left_robin
    robin[:, -1] += right_robin
    diagonal += robin
    flat_indices = indices.ravel()
    rows.extend(flat_indices.tolist())
    columns.extend(flat_indices.tolist())
    values.extend(diagonal.ravel().tolist())
    operator = coo_matrix(
        (values, (rows, columns)), shape=(node_count, node_count)
    ).tocsc()
    return _DiscreteOperator2D(
        diffusion_and_robin=operator,
        robin_rhs_coefficient=robin,
        x_face_conductance_W_K=x_faces,
        z_face_conductance_W_K=z_faces,
        lower_robin_conductance_W_K=lower_robin,
        upper_robin_conductance_W_K=upper_robin,
        left_robin_conductance_W_K=left_robin,
        right_robin_conductance_W_K=right_robin,
    )


def _rk4_alpha_2d(
    alpha_old: FloatArray,
    temperature_old: FloatArray,
    temperature_new_guess: FloatArray,
    dt_s: float,
    composite_mask: NDArray[np.bool_],
    kinetics: CureKinetics,
) -> FloatArray:
    alpha_new = alpha_old.copy()
    a0 = alpha_old[composite_mask]
    t0 = temperature_old[composite_mask]
    t1 = temperature_new_guess[composite_mask]
    tm = 0.5 * (t0 + t1)
    k1 = cure_rate_per_s(t0, a0, kinetics)
    k2 = cure_rate_per_s(tm, a0 + 0.5 * dt_s * k1, kinetics)
    k3 = cure_rate_per_s(tm, a0 + 0.5 * dt_s * k2, kinetics)
    k4 = cure_rate_per_s(t1, a0 + dt_s * k3, kinetics)
    increment = dt_s * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    alpha_new[composite_mask] = np.clip(a0 + increment, a0, 1.0)
    alpha_new[~composite_mask] = 0.0
    return alpha_new


def _factorized_system(
    grid: LayeredGrid2D,
    operator: _DiscreteOperator2D,
    dt_s: float,
) -> tuple[FloatArray, SuperLU]:
    storage = (
        grid.density_kg_m3
        * grid.specific_heat_J_kg_K
        * grid.control_volume_m2
        / dt_s
    )
    system = operator.diffusion_and_robin + diags(storage.ravel())
    return storage, splu(system.tocsc())


def _temperature_step_2d(
    grid: LayeredGrid2D,
    operator: _DiscreteOperator2D,
    temperature_old: FloatArray,
    alpha_old: FloatArray,
    alpha_new: FloatArray,
    air_temperature_K: float,
    dt_s: float,
    external_heat_source_W_m3: FloatArray,
    factorization_cache: dict[float, tuple[FloatArray, SuperLU]],
) -> FloatArray:
    key = round(float(dt_s), 12)
    if key not in factorization_cache:
        factorization_cache[key] = _factorized_system(grid, operator, dt_s)
    storage, factorization = factorization_cache[key]
    source = (
        grid.cure_source_J_m3_per_alpha
        * (alpha_new - alpha_old)
        * grid.control_volume_m2
        / dt_s
        + external_heat_source_W_m3 * grid.control_volume_m2
    )
    rhs = (
        storage * temperature_old
        + source
        + operator.robin_rhs_coefficient * air_temperature_K
    )
    return factorization.solve(rhs.ravel()).reshape(grid.shape)


def _energy_residual_2d(
    grid: LayeredGrid2D,
    operator: _DiscreteOperator2D,
    temperature_old: FloatArray,
    temperature_new: FloatArray,
    alpha_old: FloatArray,
    alpha_new: FloatArray,
    air_temperature_K: float,
    dt_s: float,
    external_heat_source_W_m3: FloatArray,
) -> tuple[float, float]:
    volume = grid.control_volume_m2
    storage = (
        grid.density_kg_m3
        * grid.specific_heat_J_kg_K
        * (temperature_new - temperature_old)
        / dt_s
    )
    source = (
        grid.cure_source_J_m3_per_alpha
        * (alpha_new - alpha_old)
        / dt_s
        + external_heat_source_W_m3
    )
    outward_W = np.zeros(grid.shape, dtype=np.float64)
    x_flux = operator.x_face_conductance_W_K * (
        temperature_new[:, :-1] - temperature_new[:, 1:]
    )
    outward_W[:, :-1] += x_flux
    outward_W[:, 1:] -= x_flux
    z_flux = operator.z_face_conductance_W_K * (
        temperature_new[:-1, :] - temperature_new[1:, :]
    )
    outward_W[:-1, :] += z_flux
    outward_W[1:, :] -= z_flux
    boundary_W = operator.robin_rhs_coefficient * (
        temperature_new - air_temperature_K
    )
    outward_W += boundary_W
    residual = storage + outward_W / volume - source
    maximum = float(np.max(np.abs(residual)))
    global_terms = np.array(
        [
            np.sum(storage * volume),
            np.sum(boundary_W),
            -np.sum(source * volume),
        ],
        dtype=np.float64,
    )
    one_kelvin_storage_rate = float(
        np.sum(grid.density_kg_m3 * grid.specific_heat_J_kg_K * volume)
        / dt_s
    )
    scale = max(float(np.sum(np.abs(global_terms))), one_kelvin_storage_rate)
    relative = float(abs(np.sum(global_terms)) / scale)
    return maximum, relative


def _interface_diagnostics_2d(
    grid: LayeredGrid2D,
    operator: _DiscreteOperator2D,
    temperature_K: FloatArray,
) -> tuple[float, float]:
    """Check assembled interface rates against a resistance reconstruction."""

    maximum_flux_imbalance = 0.0
    maximum_temperature_jump = 0.0

    x_interfaces = (
        grid.composite_mask[:, :-1] != grid.composite_mask[:, 1:]
    )
    if np.any(x_interfaces):
        distance = np.broadcast_to(
            np.diff(grid.x_m)[None, :], operator.x_face_conductance_W_K.shape
        )
        face_measure = np.broadcast_to(
            grid.control_volume_width_z_m[:, None],
            operator.x_face_conductance_W_K.shape,
        )
        first_resistance = (
            0.5
            * distance
            / (grid.conductivity_x_W_m_K[:, :-1] * face_measure)
        )
        second_resistance = (
            0.5
            * distance
            / (grid.conductivity_x_W_m_K[:, 1:] * face_measure)
        )
        assembled_heat_rate = operator.x_face_conductance_W_K * (
            temperature_K[:, :-1] - temperature_K[:, 1:]
        )
        reconstructed_heat_rate = (
            temperature_K[:, :-1] - temperature_K[:, 1:]
        ) / (first_resistance + second_resistance)
        face_from_first = (
            temperature_K[:, :-1]
            - assembled_heat_rate * first_resistance
        )
        face_from_second = (
            temperature_K[:, 1:]
            + assembled_heat_rate * second_resistance
        )
        maximum_flux_imbalance = max(
            maximum_flux_imbalance,
            float(
                np.max(
                    np.abs(
                        assembled_heat_rate - reconstructed_heat_rate
                    )[x_interfaces]
                )
            ),
        )
        maximum_temperature_jump = max(
            maximum_temperature_jump,
            float(
                np.max(
                    np.abs(face_from_first - face_from_second)[x_interfaces]
                )
            ),
        )

    z_interfaces = (
        grid.composite_mask[:-1, :] != grid.composite_mask[1:, :]
    )
    if np.any(z_interfaces):
        distance = np.broadcast_to(
            np.diff(grid.z_m)[:, None], operator.z_face_conductance_W_K.shape
        )
        face_measure = np.broadcast_to(
            grid.control_volume_width_x_m[None, :],
            operator.z_face_conductance_W_K.shape,
        )
        first_resistance = (
            0.5
            * distance
            / (grid.conductivity_z_W_m_K[:-1, :] * face_measure)
        )
        second_resistance = (
            0.5
            * distance
            / (grid.conductivity_z_W_m_K[1:, :] * face_measure)
        )
        assembled_heat_rate = operator.z_face_conductance_W_K * (
            temperature_K[:-1, :] - temperature_K[1:, :]
        )
        reconstructed_heat_rate = (
            temperature_K[:-1, :] - temperature_K[1:, :]
        ) / (first_resistance + second_resistance)
        face_from_first = (
            temperature_K[:-1, :]
            - assembled_heat_rate * first_resistance
        )
        face_from_second = (
            temperature_K[1:, :]
            + assembled_heat_rate * second_resistance
        )
        maximum_flux_imbalance = max(
            maximum_flux_imbalance,
            float(
                np.max(
                    np.abs(
                        assembled_heat_rate - reconstructed_heat_rate
                    )[z_interfaces]
                )
            ),
        )
        maximum_temperature_jump = max(
            maximum_temperature_jump,
            float(
                np.max(
                    np.abs(face_from_first - face_from_second)[z_interfaces]
                )
            ),
        )

    return maximum_flux_imbalance, maximum_temperature_jump


def _robin_flux_imbalance_2d(
    grid: LayeredGrid2D,
    operator: _DiscreteOperator2D,
    boundaries: RobinBoundaries2D,
    temperature_K: FloatArray,
    air_temperature_K: float,
) -> float:
    """Compare assembled Robin heat rates with independent reconstruction."""

    nz, nx = grid.shape
    boundary_data = (
        (
            temperature_K[0, :],
            grid.conductivity_z_W_m_K[0, :],
            np.full(nx, 0.5 * grid.control_volume_width_z_m[0]),
            grid.control_volume_width_x_m,
            _boundary_array(
                boundaries.lower_h_W_m2_K, nx, "lower_h_W_m2_K"
            ),
            operator.lower_robin_conductance_W_K,
        ),
        (
            temperature_K[-1, :],
            grid.conductivity_z_W_m_K[-1, :],
            np.full(nx, 0.5 * grid.control_volume_width_z_m[-1]),
            grid.control_volume_width_x_m,
            _boundary_array(
                boundaries.upper_h_W_m2_K, nx, "upper_h_W_m2_K"
            ),
            operator.upper_robin_conductance_W_K,
        ),
        (
            temperature_K[:, 0],
            grid.conductivity_x_W_m_K[:, 0],
            np.full(nz, 0.5 * grid.control_volume_width_x_m[0]),
            grid.control_volume_width_z_m,
            _boundary_array(
                boundaries.left_h_W_m2_K, nz, "left_h_W_m2_K"
            ),
            operator.left_robin_conductance_W_K,
        ),
        (
            temperature_K[:, -1],
            grid.conductivity_x_W_m_K[:, -1],
            np.full(nz, 0.5 * grid.control_volume_width_x_m[-1]),
            grid.control_volume_width_z_m,
            _boundary_array(
                boundaries.right_h_W_m2_K, nz, "right_h_W_m2_K"
            ),
            operator.right_robin_conductance_W_K,
        ),
    )
    maximum = 0.0
    for (
        cell_temperature,
        conductivity,
        half_distance,
        face_measure,
        h,
        assembled_conductance,
    ) in boundary_data:
        reconstructed_conductance = _robin_boundary_conductance(
            h, conductivity, half_distance, face_measure
        )
        delta_temperature = cell_temperature - air_temperature_K
        assembled_rate = assembled_conductance * delta_temperature
        reconstructed_rate = reconstructed_conductance * delta_temperature
        maximum = max(
            maximum,
            float(np.max(np.abs(assembled_rate - reconstructed_rate))),
        )

    assembled_sum = np.zeros(grid.shape, dtype=np.float64)
    assembled_sum[0, :] += (
        operator.lower_robin_conductance_W_K
        * (temperature_K[0, :] - air_temperature_K)
    )
    assembled_sum[-1, :] += (
        operator.upper_robin_conductance_W_K
        * (temperature_K[-1, :] - air_temperature_K)
    )
    assembled_sum[:, 0] += (
        operator.left_robin_conductance_W_K
        * (temperature_K[:, 0] - air_temperature_K)
    )
    assembled_sum[:, -1] += (
        operator.right_robin_conductance_W_K
        * (temperature_K[:, -1] - air_temperature_K)
    )
    combined_rate = operator.robin_rhs_coefficient * (
        temperature_K - air_temperature_K
    )
    maximum = max(maximum, float(np.max(np.abs(combined_rate - assembled_sum))))
    return maximum


def simulate_cure_2d(
    times_s: ArrayLike,
    air_temperature_K: ArrayLike,
    *,
    grid: LayeredGrid2D | None = None,
    boundaries: RobinBoundaries2D | None = None,
    kinetics: CureKinetics | None = None,
    initial_temperature_K: float | ArrayLike = 293.0,
    initial_alpha: float = 0.05,
    external_heat_source_W_m3: float | ArrayLike = 0.0,
    maximum_step_s: float = 10.0,
    coupling_tolerance_K: float = 1.0e-9,
    maximum_coupling_iterations: int = 8,
) -> SolverResult2D:
    """Integrate a layered 2-D thermochemical system in conservative form."""

    started = perf_counter()
    time = np.asarray(times_s, dtype=np.float64)
    air = np.asarray(air_temperature_K, dtype=np.float64)
    if time.ndim != 1 or air.shape != time.shape or time.size < 2:
        raise ValueError("times_s and air_temperature_K must be equal 1-D arrays.")
    if not np.all(np.diff(time) > 0.0):
        raise ValueError("times_s must be strictly increasing.")
    if not np.all(np.isfinite(time)) or not np.all(np.isfinite(air)):
        raise ValueError("times_s and air_temperature_K must be finite.")
    if np.any(air <= 0.0):
        raise ValueError("air_temperature_K must use positive absolute kelvin.")
    if not np.isfinite(maximum_step_s) or maximum_step_s <= 0.0:
        raise ValueError("maximum_step_s must be positive.")
    if not np.isfinite(coupling_tolerance_K) or coupling_tolerance_K <= 0.0:
        raise ValueError("coupling_tolerance_K must be finite and positive.")
    if maximum_coupling_iterations < 1:
        raise ValueError("maximum_coupling_iterations must be at least one.")
    if not np.isfinite(initial_alpha) or not 0.0 <= initial_alpha <= 1.0:
        raise ValueError("initial_alpha must be finite and within [0, 1].")

    active_grid = grid or rectangular_tool_composite_grid()
    _validate_grid_2d(active_grid)
    material = PublicCase1Material()
    active_boundaries = boundaries or RobinBoundaries2D(
        material.lower_h_W_m2_K, material.upper_h_W_m2_K
    )
    active_kinetics = kinetics or CureKinetics()
    external_source = np.asarray(external_heat_source_W_m3, dtype=np.float64)
    if external_source.ndim == 0:
        external_source = np.full(
            active_grid.shape, float(external_source), dtype=np.float64
        )
    if external_source.shape != active_grid.shape:
        raise ValueError("External heat source must be scalar or match the grid.")
    if not np.all(np.isfinite(external_source)):
        raise ValueError("External heat source must contain only finite values.")
    if np.ndim(initial_temperature_K) == 0:
        temperature = np.full(
            active_grid.shape, float(initial_temperature_K), dtype=np.float64
        )
    else:
        temperature = np.asarray(
            initial_temperature_K, dtype=np.float64
        ).copy()
        if temperature.shape != active_grid.shape:
            raise ValueError("Initial temperature must match the 2-D grid.")
    if not np.all(np.isfinite(temperature)) or np.any(temperature <= 0.0):
        raise ValueError(
            "Initial temperature must contain positive finite absolute kelvin."
        )
    alpha = np.zeros(active_grid.shape, dtype=np.float64)
    alpha[active_grid.composite_mask] = initial_alpha
    temperatures = np.empty(
        (time.size, *active_grid.shape), dtype=np.float64
    )
    alphas = np.empty_like(temperatures)
    temperatures[0] = temperature
    alphas[0] = alpha

    operator = _assemble_operator(active_grid, active_boundaries)
    factorization_cache: dict[float, tuple[FloatArray, SuperLU]] = {}
    maximum_local_residual = 0.0
    maximum_global_residual = 0.0
    maximum_interface_flux_imbalance = 0.0
    maximum_temperature_interface_jump = 0.0
    maximum_robin_flux_imbalance = 0.0
    total_substeps = 0
    used_coupling_iterations = 0
    maximum_converged_coupling_update = 0.0

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
                alpha_new = _rk4_alpha_2d(
                    alpha_old,
                    temperature_old,
                    temperature_guess,
                    dt_s,
                    active_grid.composite_mask,
                    active_kinetics,
                )
                temperature_new = _temperature_step_2d(
                    active_grid,
                    operator,
                    temperature_old,
                    alpha_old,
                    alpha_new,
                    air_new,
                    dt_s,
                    external_source,
                    factorization_cache,
                )
                coupling_update = float(
                    np.max(np.abs(temperature_new - temperature_guess))
                )
                if coupling_update <= coupling_tolerance_K:
                    maximum_converged_coupling_update = max(
                        maximum_converged_coupling_update, coupling_update
                    )
                    break
                temperature_guess = temperature_new
            else:
                raise RuntimeError(
                    "2-D thermochemical Picard coupling failed to converge at "
                    f"output_index={output_index}, substep={substep}, "
                    f"dt_s={dt_s:.12g}; final_update_K={coupling_update:.12g}, "
                    f"tolerance_K={coupling_tolerance_K:.12g}."
                )
            temperature = temperature_new
            alpha = alpha_new
            used_coupling_iterations = max(
                used_coupling_iterations, coupling_index
            )
            local_residual, global_residual = _energy_residual_2d(
                active_grid,
                operator,
                temperature_old,
                temperature,
                alpha_old,
                alpha,
                air_new,
                dt_s,
                external_source,
            )
            maximum_local_residual = max(
                maximum_local_residual, local_residual
            )
            maximum_global_residual = max(
                maximum_global_residual, global_residual
            )
            interface_flux, interface_jump = _interface_diagnostics_2d(
                active_grid, operator, temperature
            )
            maximum_interface_flux_imbalance = max(
                maximum_interface_flux_imbalance, interface_flux
            )
            maximum_temperature_interface_jump = max(
                maximum_temperature_interface_jump, interface_jump
            )
            maximum_robin_flux_imbalance = max(
                maximum_robin_flux_imbalance,
                _robin_flux_imbalance_2d(
                    active_grid,
                    operator,
                    active_boundaries,
                    temperature,
                    air_new,
                ),
            )
            total_substeps += 1
        temperatures[output_index] = temperature
        alphas[output_index] = alpha

    return SolverResult2D(
        times_s=time,
        x_m=active_grid.x_m,
        z_m=active_grid.z_m,
        temperature_K=temperatures,
        alpha=alphas,
        diagnostics=SolverDiagnostics2D(
            maximum_abs_energy_residual_W_m3=maximum_local_residual,
            maximum_relative_global_energy_residual=maximum_global_residual,
            maximum_interface_flux_imbalance_W=(
                maximum_interface_flux_imbalance
            ),
            maximum_temperature_interface_jump_K=(
                maximum_temperature_interface_jump
            ),
            maximum_robin_flux_imbalance_W=maximum_robin_flux_imbalance,
            all_coupling_steps_converged=True,
            maximum_converged_coupling_update_K=(
                maximum_converged_coupling_update
            ),
            substeps=total_substeps,
            maximum_coupling_iterations=used_coupling_iterations,
            wall_seconds=perf_counter() - started,
        ),
    )
