from __future__ import annotations

from backend.physics.optics import fresnel
from backend.physics.polarization import classify_phase
from backend.physics.transmission import envelope_values
from backend.models import OpticsInput


def test_fresnel_conserves_power_for_lossless_case() -> None:
    result = fresnel(OpticsInput(n1=1.0, n2=1.5, theta_deg=35.0, polarization="natural"))
    assert not result.is_tir
    assert abs((result.R_s + result.T_s) - 1.0) < 1e-9
    assert abs((result.R_p + result.T_p) - 1.0) < 1e-9


def test_tir_returns_full_reflection() -> None:
    result = fresnel(OpticsInput(n1=1.5, n2=1.0, theta_deg=60.0, polarization="p"))
    assert result.is_tir
    assert result.active_R == 1.0
    assert result.active_T == 0.0
    assert result.theta_t_deg is None


def test_phase_classification_detects_circular_polarization() -> None:
    status, color = classify_phase(1.0, 1.0, 90.0)
    assert "圆极化" in status
    assert color == "purple"


def test_envelope_values_use_reflection_sign_consistently() -> None:
    env_e, env_h = envelope_values(-0.6)
    assert env_e.min() >= 0.0
    assert env_h.min() >= 0.0
