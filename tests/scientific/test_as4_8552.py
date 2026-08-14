import numpy as np

from cdcureno.physics import CureKinetics, PublicCase1Material, cure_rate_per_s


def test_sourced_public_material_mixtures_are_positive() -> None:
    material = PublicCase1Material()
    assert np.isclose(material.composite_density_kg_m3, 1581.26)
    assert np.isclose(material.composite_cp_J_kg_K, 1080.2252)
    assert np.isclose(material.composite_k_W_m_K, 0.6369802947587432)
    assert material.cure_source_J_m3_per_alpha > 0.0


def test_cure_rate_is_nonnegative_and_diffusion_limited() -> None:
    temperature = np.full(200, 453.0)
    alpha = np.linspace(0.05, 0.999, temperature.size)
    rate = cure_rate_per_s(temperature, alpha)
    assert np.all(rate >= 0.0)
    assert rate[-1] < rate[50]


def test_printed_no_offset_variant_is_explicit_not_default() -> None:
    alpha = np.array([0.2, 0.7])
    temperature = np.array([453.0, 453.0])
    sourced = cure_rate_per_s(temperature, alpha)
    printed_variant = cure_rate_per_s(
        temperature, alpha, CureKinetics(denominator_offset=0.0)
    )
    assert np.any(np.abs(sourced - printed_variant) > 0.0)
