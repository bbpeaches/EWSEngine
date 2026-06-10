from __future__ import annotations

import math

import numpy as np

from backend.models import INTRINSIC_IMPEDANCE, OpticsInput, PolarizationInput, SpeedInput, TemInput, TransmissionInput, WaveInput
from backend.physics.optics import OpticsEngine, fresnel
from backend.physics.polarization import PolarizationEngine, classify_phase
from backend.physics.speed import SpeedEngine
from backend.physics.tem import TemEngine, compute_wave
from backend.physics.transmission import TransmissionEngine, envelope_values
from backend.physics.wave import render_lossy, render_material


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


def test_critical_angle_is_treated_as_tir_boundary() -> None:
    critical = math.degrees(math.asin(1.0 / 1.5))
    result = fresnel(OpticsInput(n1=1.5, n2=1.0, theta_deg=critical, polarization="natural"))
    assert result.is_tir
    assert result.theta_t_deg is None


def test_apparent_speed_near_ninety_degrees_stays_finite() -> None:
    frame = SpeedEngine().simulate(SpeedInput(mode="apparent", theta_deg=85.0))
    assert frame.apparent_marker.point is not None
    assert all(np.isfinite(frame.apparent_marker.point))
    assert all("inf" not in line.lower() for line in frame.panel.status_lines + frame.panel.metrics_lines)


def test_phase_classification_detects_circular_polarization() -> None:
    status, color = classify_phase(1.0, 1.0, 90.0)
    assert "圆极化" in status
    assert color == "purple"


def test_envelope_values_use_reflection_sign_consistently() -> None:
    env_e, env_h = envelope_values(-0.6)
    assert env_e.min() >= 0.0
    assert env_h.min() >= 0.0


def test_lossy_tem_wave_uses_exponential_envelope() -> None:
    samples = np.asarray([0.0, 5.0, 10.0], dtype=float)
    values, _, _, envelope = compute_wave(
        TemInput(mode="lossy", alpha=0.2, beta=5.0, speed=2.0, time=0.0),
        samples,
    )
    assert envelope[0] > envelope[1] > envelope[2]
    assert abs(values[0]) > abs(values[-1])


def test_wave_material_and_lossy_frames_include_377h_line() -> None:
    material_frame = render_material(WaveInput(mode="material", h_display="377H"))
    lossy_frame = render_lossy(WaveInput(mode="lossy", alpha=0.4, beta=5.0, h_display="377H"))
    assert len(material_frame.magnetic_line.x) > 0
    assert len(lossy_frame.magnetic_line.x) > 0
    assert np.isclose(np.max(np.abs(material_frame.magnetic_line.y)) * INTRINSIC_IMPEDANCE, 1.0)
    assert np.max(np.abs(lossy_frame.magnetic_line.y)) * INTRINSIC_IMPEDANCE > 0.0


def test_propagation_coordinate_ranges_are_doubled() -> None:
    assert np.isclose(max(TemEngine().simulate(TemInput(direction="x")).propagation_axis.x), 30.0)
    assert np.isclose(max(SpeedEngine().simulate(SpeedInput(direction="z")).propagation_axis.z), 36.0)
    assert np.isclose(max(PolarizationEngine().simulate(PolarizationInput()).wave_line.z), 24.0)
    assert np.isclose(min(TransmissionEngine().simulate(TransmissionInput()).axis_line.x), -8.0)
    optics = OpticsEngine().simulate(OpticsInput(theta_deg=0.0))
    assert np.isclose(max(optics.incident_line.z), 5.2)


def test_polarization_frame_includes_physical_h_field() -> None:
    frame = PolarizationEngine().simulate(PolarizationInput(p1=1.0, p2=0.0, p3=0.0, h_display="H"))
    assert frame.h_display == "H"
    assert np.allclose(frame.magnetic_line.x, 0.0)
    assert np.allclose(frame.magnetic_line.y, frame.wave_line.x / INTRINSIC_IMPEDANCE)
    dot = frame.wave_field.u * frame.magnetic_field.u + frame.wave_field.v * frame.magnetic_field.v
    assert np.allclose(dot, 0.0)
