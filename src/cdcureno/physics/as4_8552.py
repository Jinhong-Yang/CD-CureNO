"""Sourced AS4/8552 thermochemical relations used by the conservative solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class CureKinetics:
    """Hubert--Johnston modified autocatalytic cure parameters in SI units."""

    A_per_s: float = 1.528e5
    delta_E_J_per_mol: float = 6.650e4
    M: float = 0.8129
    N: float = 2.7360
    C: float = 43.09
    C_0: float = -1.6840
    C_T_per_K: float = 5.475e-3
    R_J_per_mol_K: float = 8.314
    denominator_offset: float = 1.0


@dataclass(frozen=True)
class PublicCase1Material:
    """Material and boundary constants reported for the public ResFNO Case1."""

    fibre_volume_fraction: float = 0.574
    resin_volume_fraction: float = 0.426
    fibre_density_kg_m3: float = 1790.0
    resin_density_kg_m3: float = 1300.0
    tool_density_kg_m3: float = 8150.0
    fibre_cp_J_kg_K: float = 914.0
    resin_cp_J_kg_K: float = 1304.2
    tool_cp_J_kg_K: float = 510.0
    fibre_transverse_k_W_m_K: float = 3.960
    resin_k_W_m_K: float = 0.212
    tool_k_W_m_K: float = 13.0
    longitudinal_composite_k_intercept_W_m_K: float = 4.49
    longitudinal_composite_k_slope_W_m_K_per_C: float = 9.12e-3
    conductivity_reference_temperature_K: float = 293.15
    heat_of_reaction_J_kg_resin: float = 5.40e5
    tool_thickness_m: float = 0.020
    composite_thickness_m: float = 0.030
    lower_h_W_m2_K: float = 70.0
    upper_h_W_m2_K: float = 120.0
    initial_temperature_K: float = 293.0
    initial_alpha: float = 0.05

    @property
    def composite_density_kg_m3(self) -> float:
        return (
            self.resin_density_kg_m3 * self.resin_volume_fraction
            + self.fibre_density_kg_m3 * self.fibre_volume_fraction
        )

    @property
    def composite_cp_J_kg_K(self) -> float:
        # This is the mixture relation printed in Niaki et al. Eq. (32).
        return (
            self.resin_cp_J_kg_K * self.resin_volume_fraction
            + self.fibre_cp_J_kg_K * self.fibre_volume_fraction
        )

    @property
    def composite_k_W_m_K(self) -> float:
        """Springer--Tsai transverse conductivity relation."""

        gamma = 2.0 * (
            self.resin_k_W_m_K / self.fibre_transverse_k_W_m_K - 1.0
        )
        omega = np.sqrt(self.fibre_volume_fraction / np.pi)
        upsilon = np.sqrt(1.0 - gamma**2 * omega**2)
        bracket = np.pi - (4.0 / upsilon) * np.arctan(
            upsilon / (1.0 + omega * gamma)
        )
        return float(
            self.resin_k_W_m_K
            * ((1.0 - 2.0 * omega) + bracket / gamma)
        )

    @property
    def composite_longitudinal_k_W_m_K(self) -> float:
        """Measured cured-laminate longitudinal conductivity at 20 deg C.

        Johnston (1997), Fig. 6.5, reports
        ``k_c11 = 4.49 + 9.12e-3 * T_degC`` for AS4/8552. The first P4
        benchmark freezes this measured relation at the declared reference
        temperature because the linear solver currently assumes constant
        material coefficients.
        """

        temperature_C = self.conductivity_reference_temperature_K - 273.15
        return float(
            self.longitudinal_composite_k_intercept_W_m_K
            + self.longitudinal_composite_k_slope_W_m_K_per_C * temperature_C
        )

    @property
    def cure_source_J_m3_per_alpha(self) -> float:
        return (
            self.resin_density_kg_m3
            * self.resin_volume_fraction
            * self.heat_of_reaction_J_kg_resin
        )


def cure_rate_per_s(
    temperature_K: ArrayLike,
    alpha: ArrayLike,
    parameters: CureKinetics | None = None,
) -> NDArray[np.float64]:
    """Evaluate the sourced diffusion-limited cure law.

    Inputs may be scalars or broadcast-compatible arrays. Temperatures must be
    absolute kelvin and alpha is clipped only for evaluating powers; the solver
    remains responsible for enforcing state bounds.
    """

    p = parameters or CureKinetics()
    temperature = np.asarray(temperature_K, dtype=np.float64)
    degree = np.asarray(alpha, dtype=np.float64)
    if np.any(temperature <= 0.0):
        raise ValueError("Cure kinetics require absolute temperature in kelvin.")
    bounded = np.clip(degree, 0.0, 1.0)
    exponent = p.C * (bounded - p.C_T_per_K * temperature - p.C_0)
    denominator = p.denominator_offset + np.exp(np.clip(exponent, -80.0, 80.0))
    rate = (
        p.A_per_s
        * np.exp(-p.delta_E_J_per_mol / (p.R_J_per_mol_K * temperature))
        * np.power(bounded, p.M)
        * np.power(1.0 - bounded, p.N)
        / denominator
    )
    return np.maximum(rate, 0.0)
