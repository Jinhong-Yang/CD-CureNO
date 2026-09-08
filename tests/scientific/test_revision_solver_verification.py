"""Small mathematical checks guarding the public revision-verification cases."""
import importlib.util
from pathlib import Path

import numpy as np

_path = Path(__file__).resolve().parents[2] / "scripts" / "verify_revision_solver.py"
_spec = importlib.util.spec_from_file_location("revision_solver_verification", _path)
verification = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verification)


def test_manufactured_field_satisfies_continuous_pde_away_from_interface():
    """Differentiate reference temperatures numerically, independently of q*."""
    for interface in (False, True):
        x = np.array([0.03, 0.09, 0.16])
        z = np.array([0.015, 0.035, 0.065, 0.085])
        step = 1e-4
        temp, source, _ = verification.exact_steady(x, z, interface)
        xp = verification.exact_steady(x + step, z, interface)[0]
        xm = verification.exact_steady(x - step, z, interface)[0]
        zp = verification.exact_steady(x, z + step, interface)[0]
        zm = verification.exact_steady(x, z - step, interface)[0]
        kz = np.where(z < 0.05, 1.0, 0.25) if interface else np.full(z.shape, 0.5)
        residual = 2.0 * (xp - 2 * temp + xm) / step**2 + kz[:, None] * (zp - 2 * temp + zm) / step**2 + source
        assert np.max(np.abs(residual)) < 5e-5


def test_manufactured_interface_temperature_and_flux_are_continuous():
    x = np.array([0.03, 0.11, 0.17])
    step = 1e-5
    interface = 0.05
    t = verification.exact_steady(x, np.array([interface]), True)[0][0]
    left = verification.exact_steady(x, np.array([interface-step, interface-2*step]), True)[0]
    right = verification.exact_steady(x, np.array([interface+step, interface+2*step]), True)[0]
    derivative_left = (3*t - 4*left[0] + left[1]) / (2*step)
    derivative_right = (-3*t + 4*right[0] - right[1]) / (2*step)
    assert np.max(np.abs(derivative_left - 0.25*derivative_right)) < 2e-8
    assert np.max(np.abs(derivative_left)) > 1.0


def test_manufactured_fields_satisfy_all_four_continuous_robin_conditions():
    step = 1e-5
    for interface in (False, True):
        x = np.array([0.03, 0.09, 0.16])
        z = np.array([0.015, 0.035, 0.065, 0.085])
        _, _, boundary = verification.exact_steady(x, z, interface)
        for upper in (False, True):
            sign = -1.0 if upper else 1.0
            edge_x = 0.2 if upper else 0.0
            tx = verification.exact_steady(edge_x + sign*step*np.arange(3), z, interface)[0]
            inward_dx = (-3*tx[:, 0] + 4*tx[:, 1] - tx[:, 2]) / (2*step)
            assert np.max(np.abs(2.0*inward_dx - boundary.left_h_W_m2_K*(tx[:, 0]-300.0))) < 2e-8
            edge_z = 0.1 if upper else 0.0
            tz = verification.exact_steady(x, edge_z + sign*step*np.arange(3), interface)[0]
            inward_dz = (-3*tz[0] + 4*tz[1] - tz[2]) / (2*step)
            kz = (0.25 if upper else 1.0) if interface else 0.5
            assert np.max(np.abs(kz*inward_dz - boundary.lower_h_W_m2_K*(tz[0]-300.0))) < 2e-8


def test_true_two_dimensional_coupled_case_uses_production_solver():
    coarse = verification.run_case("C", 12, 8, 20.0, np.array([0.0, 100.0]), "coarse", "test")
    fine = verification.run_case("C", 12, 8, 10.0, np.array([0.0, 100.0]), "fine", "test")
    ratio = coarse["errors"][-1]["alpha_linf"] / fine["errors"][-1]["alpha_linf"]
    assert 14.0 < ratio < 22.0
    assert fine["mean_reaction_heat_identity_error_K"] < 1e-9
    assert fine["x_variation_K"] > 1.0 and fine["z_variation_K"] > 1.0
