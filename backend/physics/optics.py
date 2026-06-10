from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.interfaces import SimulationEngine
from backend.models import (
    AxisLimits,
    FresnelResult,
    LineData,
    MarkerData,
    OpticsFrame,
    OpticsInput,
    PanelText,
    PlaneData,
    VectorData,
    empty_line,
)
from backend.physics.geometry import (
    build_lower_arc,
    build_upper_arc,
    incident_basis,
    limits,
    line_from_points,
    marker_from_point,
    plane,
    point,
    segment,
    vector,
)
from core.exceptions import ValidationError

INTERFACE_HALF = 2.8
RAY_LENGTH = 5.2
ARC_RADIUS = 0.60
DIPOLE_SCALE = 0.72
MIN_ZOOM = 0.65
MAX_ZOOM = 4.0
TOLERANCE = 1e-10
CRITICAL_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class OpticsEngine(SimulationEngine[OpticsInput, OpticsFrame]):
    """Interface optics and coplanarity simulation."""

    def simulate(self, input_data: OpticsInput) -> OpticsFrame:
        validate(input_data)
        result = fresnel(input_data)
        tangential_hat, side_hat, normal_hat = incident_basis(input_data.phi_deg)
        theta_i = result.theta_i_rad

        source = point((-np.sin(theta_i) * tangential_hat + np.cos(theta_i) * normal_hat) * RAY_LENGTH)
        reflected = point((np.sin(theta_i) * tangential_hat + np.cos(theta_i) * normal_hat) * RAY_LENGTH)

        planes = build_planes(tangential_hat, normal_hat)
        vectors: list[VectorData] = [vector(source, (0.0, 0.0, 0.0), "royalblue", 2.8)]

        incident_line = segment(source, (0.0, 0.0, 0.0))
        reflected_line = segment((0.0, 0.0, 0.0), reflected)
        refracted_line = empty_line()
        dipole_curve = empty_line()
        refracted_marker = marker_from_point(None)
        refract_alpha = 0.0
        refract_width = 0.0

        reflect_alpha = float(np.clip(0.18 + 0.82 * result.active_R, 0.18, 1.0))
        reflect_width = float(1.2 + 3.0 * np.sqrt(result.active_R))

        if result.is_brewster and input_data.polarization == "p":
            reflected_line = empty_line()
            reflected_marker = marker_from_point(None)
        else:
            vectors.append(vector((0.0, 0.0, 0.0), reflected, "firebrick", reflect_width, reflect_alpha))
            reflected_marker = marker_from_point(interpolate_point((0.0, 0.0, 0.0), reflected, pulse_progress(input_data.time)))

        if not result.is_tir and result.theta_t_rad is not None:
            theta_t = result.theta_t_rad
            refracted = point((np.sin(theta_t) * tangential_hat - np.cos(theta_t) * normal_hat) * RAY_LENGTH)
            refracted_line = segment((0.0, 0.0, 0.0), refracted)
            refract_alpha = float(np.clip(0.18 + 0.82 * result.active_T, 0.18, 1.0))
            refract_width = float(1.4 + 2.6 * np.sqrt(result.active_T))
            vectors.append(vector((0.0, 0.0, 0.0), refracted, "forestgreen", refract_width, refract_alpha))
            refracted_marker = marker_from_point(interpolate_point((0.0, 0.0, 0.0), refracted, pulse_progress(input_data.time)))
            dipole_curve = build_dipole(theta_t, tangential_hat, normal_hat)

        incident_marker = marker_from_point(interpolate_point(source, (0.0, 0.0, 0.0), pulse_progress(input_data.time)))
        axis_limits = build_limits(input_data.zoom)

        normal_line = segment((0.0, 0.0, -2.2), (0.0, 0.0, 2.2))
        side_trace = np.array([0.9 * side_hat, -0.9 * side_hat])

        return OpticsFrame(
            title="界面光学 3D 综合仿真",
            panel=panel_text(input_data, result),
            axis_limits=axis_limits,
            result=result,
            incident_line=incident_line,
            reflected_line=reflected_line,
            refracted_line=refracted_line,
            normal_line=normal_line,
            interface_x_line=segment((-INTERFACE_HALF, 0.0, 0.0), (INTERFACE_HALF, 0.0, 0.0)),
            interface_y_line=segment((0.0, -INTERFACE_HALF, 0.0), (0.0, INTERFACE_HALF, 0.0)),
            side_line=line_from_points(side_trace),
            incident_arc=build_upper_arc(theta_i, tangential_hat, normal_hat, -1.0, ARC_RADIUS),
            reflected_arc=build_upper_arc(theta_i, tangential_hat, normal_hat, 1.0, ARC_RADIUS),
            refracted_arc=build_lower_arc(result.theta_t_rad, tangential_hat, normal_hat, ARC_RADIUS),
            dipole_curve=dipole_curve,
            planes=planes,
            vectors=tuple(vectors),
            incident_marker=incident_marker,
            reflected_marker=reflected_marker,
            refracted_marker=refracted_marker,
            reflect_width=reflect_width,
            reflect_alpha=reflect_alpha,
            refract_width=refract_width,
            refract_alpha=refract_alpha,
            dipole_alpha=0.35 + 0.5 * result.active_T if not result.is_tir else 0.0,
        )


def validate(input_data: OpticsInput) -> None:
    if input_data.n1 <= 0.0 or input_data.n2 <= 0.0:
        raise ValidationError("Refractive indices must be positive.")
    if not 0.0 <= input_data.theta_deg < 90.0:
        raise ValidationError("Incident angle must be in [0, 90).")
    if input_data.polarization not in {"natural", "s", "p"}:
        raise ValidationError("Unsupported polarization mode.")
    if input_data.zoom <= 0.0:
        raise ValidationError("Zoom must be positive.")


def fresnel(input_data: OpticsInput) -> FresnelResult:
    theta_i = np.radians(input_data.theta_deg)
    sin_theta_t = (input_data.n1 / input_data.n2) * np.sin(theta_i)
    theta_b = float(np.degrees(np.arctan2(input_data.n2, input_data.n1)))
    theta_c = float(np.degrees(np.arcsin(input_data.n2 / input_data.n1))) if input_data.n1 > input_data.n2 else None

    if sin_theta_t >= 1.0 - CRITICAL_EPS:
        return tir_result(input_data, theta_i, theta_b, theta_c)

    sin_theta_t = float(np.clip(sin_theta_t, -1.0, 1.0))
    theta_t = float(np.arcsin(sin_theta_t))
    cos_i = float(np.cos(theta_i))
    cos_t = float(np.cos(theta_t))
    n1 = input_data.n1
    n2 = input_data.n2

    r_s = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
    r_p = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)
    t_s = (2.0 * n1 * cos_i) / (n1 * cos_i + n2 * cos_t)
    t_p = (2.0 * n1 * cos_i) / (n2 * cos_i + n1 * cos_t)

    power_scale = (n2 * cos_t) / (n1 * cos_i)
    R_s = float(r_s**2)
    R_p = float(r_p**2)
    T_s = float(power_scale * t_s**2)
    T_p = float(power_scale * t_p**2)
    is_brewster = abs(r_p) < 1e-3
    active_label, active_R, active_T = active_power(input_data.polarization, R_s, R_p, T_s, T_p)

    return FresnelResult(
        is_tir=False,
        is_brewster=is_brewster,
        theta_i_rad=float(theta_i),
        theta_t_rad=theta_t,
        theta_i_deg=input_data.theta_deg,
        theta_t_deg=float(np.degrees(theta_t)),
        theta_r_deg=input_data.theta_deg,
        theta_c_deg=theta_c,
        theta_b_deg=theta_b,
        r_s_amp=float(r_s),
        r_p_amp=float(r_p),
        t_s_amp=float(t_s),
        t_p_amp=float(t_p),
        R_s=R_s,
        R_p=R_p,
        T_s=T_s,
        T_p=T_p,
        active_label=active_label,
        active_R=active_R,
        active_T=active_T,
    )


def tir_result(input_data: OpticsInput, theta_i: float, theta_b: float, theta_c: float | None) -> FresnelResult:
    active_label, active_R, active_T = active_power(input_data.polarization, 1.0, 1.0, 0.0, 0.0)
    return FresnelResult(
        is_tir=True,
        is_brewster=False,
        theta_i_rad=float(theta_i),
        theta_t_rad=None,
        theta_i_deg=input_data.theta_deg,
        theta_t_deg=None,
        theta_r_deg=input_data.theta_deg,
        theta_c_deg=theta_c,
        theta_b_deg=theta_b,
        r_s_amp=1.0,
        r_p_amp=1.0,
        t_s_amp=None,
        t_p_amp=None,
        R_s=1.0,
        R_p=1.0,
        T_s=0.0,
        T_p=0.0,
        active_label=active_label,
        active_R=active_R,
        active_T=active_T,
        phase_shift_s=0.0,
        phase_shift_p=0.0,
    )


def active_power(mode: str, R_s: float, R_p: float, T_s: float, T_p: float) -> tuple[str, float, float]:
    if mode == "s":
        return "S 偏振", R_s, T_s
    if mode == "p":
        return "P 偏振", R_p, T_p
    return "自然光平均", 0.5 * (R_s + R_p), 0.5 * (T_s + T_p)


def build_planes(tangential_hat: np.ndarray, normal_hat: np.ndarray) -> tuple[PlaneData, ...]:
    return (
        plane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), INTERFACE_HALF, INTERFACE_HALF, "#c7d8ff", 0.14),
        plane((0.0, 0.0, 1.35), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), INTERFACE_HALF, INTERFACE_HALF, "aliceblue", 0.05),
        plane((0.0, 0.0, -1.35), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), INTERFACE_HALF, INTERFACE_HALF, "lavender", 0.05),
        plane((0.0, 0.0, 0.0), point(tangential_hat), point(normal_hat), INTERFACE_HALF, 2.2, "orange", 0.08),
    )


def build_dipole(theta_t: float, tangential_hat: np.ndarray, normal_hat: np.ndarray) -> LineData:
    psi = np.linspace(0.0, 2.0 * np.pi, 220)
    dipole_axis_angle = theta_t - np.pi / 2.0
    radius = DIPOLE_SCALE * np.abs(np.sin(psi - dipole_axis_angle))
    points = radius[:, None] * np.cos(psi)[:, None] * tangential_hat[None, :] + radius[:, None] * np.sin(psi)[:, None] * (-normal_hat)[None, :]
    return line_from_points(points)


def pulse_progress(time: float) -> float:
    return float((time * 0.45) % 1.0)


def interpolate_point(start: tuple[float, float, float], end: tuple[float, float, float], progress: float) -> tuple[float, float, float]:
    start_arr = np.asarray(start, dtype=float)
    end_arr = np.asarray(end, dtype=float)
    return point(start_arr + (end_arr - start_arr) * progress)


def build_limits(zoom: float) -> AxisLimits:
    zoom = float(np.clip(zoom, MIN_ZOOM, MAX_ZOOM))
    xy_half = 6.4 / zoom
    z_half = 5.4 / zoom
    return limits((-xy_half, xy_half), (-xy_half, xy_half), (-z_half, z_half))


def panel_text(input_data: OpticsInput, result: FresnelResult) -> PanelText:
    if result.is_tir:
        headline = "全反射：能量回到媒质 1，媒质 2 中无传播折射波。"
        status_color = "firebrick"
    elif result.is_brewster and input_data.polarization == "p":
        headline = "布儒斯特角：P 偏振反射系数接近零。"
        status_color = "forestgreen"
    else:
        headline = "一般界面响应：反射、折射和偏振相关的 Fresnel 系数同时存在。"
        status_color = "black"

    theta_c = "无临界角" if result.theta_c_deg is None else f"{result.theta_c_deg:.1f} deg"
    theta_t = "无折射角" if result.theta_t_deg is None else f"{result.theta_t_deg:.1f} deg"
    return PanelText(
        hint="实验说明\n滑动入射角与方位角，可观察入射面绕法线旋转。\n切换偏振可查看 Fresnel 系数对能量分配的影响。",
        status_lines=(
            f"当前状态: {headline}",
            f"偏振模式: {result.active_label}",
            f"n1 / n2: {input_data.n1:.2f} / {input_data.n2:.2f}",
            f"入射角 theta_i: {input_data.theta_deg:.1f} deg",
            f"入射面方位角 phi: {input_data.phi_deg:.1f} deg",
            f"缩放: {input_data.zoom:.1f}x",
        ),
        metrics_lines=(
            "关键量",
            f"反射角 theta_r: {result.theta_r_deg:.1f} deg",
            f"折射角 theta_t: {theta_t}",
            f"临界角 theta_c: {theta_c}",
            f"布儒斯特角 theta_B: {result.theta_b_deg:.1f} deg",
            f"Rs / Rp: {result.R_s:.3f} / {result.R_p:.3f}",
            f"Ts / Tp: {result.T_s:.3f} / {result.T_p:.3f}",
            f"当前显示 R / T: {result.active_R:.3f} / {result.active_T:.3f}",
        ),
        status_color=status_color,
    )


class ElectromagneticPhysics:
    @staticmethod
    def calculate_fresnel(n1: float, n2: float, theta_i_deg: float) -> FresnelResult:
        return fresnel(OpticsInput(n1=n1, n2=n2, theta_deg=theta_i_deg))
