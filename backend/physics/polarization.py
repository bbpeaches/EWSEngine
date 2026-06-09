from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.interfaces import SimulationEngine
from backend.models import (
    INTRINSIC_IMPEDANCE,
    AxisLimits,
    LineData,
    PanelText,
    PolarizationFrame,
    PolarizationInput,
    Series2D,
    VectorFieldData,
    empty_line,
)
from backend.physics.geometry import limits, line_from_components, segment
from core.exceptions import ValidationError

Z_LENGTH = 24.0
WAVE_LENGTH = 4.5
K = 2.0 * np.pi / WAVE_LENGTH
NUM_Z_POINTS = 320
NUM_ARROWS = 24
MIN_ZOOM = 0.65
MAX_ZOOM = 4.0

Z_POINTS = np.linspace(0.0, Z_LENGTH, NUM_Z_POINTS)
ARROW_Z = np.linspace(0.0, Z_LENGTH, NUM_ARROWS)

MODE_LABELS = {
    "phase": "Ex/Ey 相位合成",
    "circular": "LHCP/RHCP 基底合成",
    "match": "极化匹配/接收天线",
}


@dataclass(frozen=True, slots=True)
class PolarizationEngine(SimulationEngine[PolarizationInput, PolarizationFrame]):
    """Polarization synthesis and antenna-match simulation."""

    def simulate(self, input_data: PolarizationInput) -> PolarizationFrame:
        validate(input_data)
        mode_data = build_mode_data(input_data, Z_POINTS)
        arrow_data = build_mode_data(input_data, ARROW_Z)
        ex = mode_data.ex
        ey = mode_data.ey
        hx = -ey / INTRINSIC_IMPEDANCE
        hy = ex / INTRINSIC_IMPEDANCE
        arrow_hx = -arrow_data.ey / INTRINSIC_IMPEDANCE
        arrow_hy = arrow_data.ex / INTRINSIC_IMPEDANCE
        ex_tip = float(ex[0])
        ey_tip = float(ey[0])
        axis_limits = build_limits(mode_data.field_extent, input_data.zoom)
        projection_line = empty_line()
        antenna_line = empty_line()

        if mode_data.projection_tip is not None:
            projection_line = segment((0.0, 0.0, 0.0), mode_data.projection_tip)
        if mode_data.antenna_segment is not None:
            antenna_line = segment(mode_data.antenna_segment[0], mode_data.antenna_segment[1])

        return PolarizationFrame(
            title=f"{MODE_LABELS[input_data.mode]} · 沿 z 轴传播的 3D 极化波",
            panel=panel_text(input_data, mode_data),
            axis_limits=axis_limits,
            h_display=input_data.h_display,
            wave_line=line_from_components(ex, ey, Z_POINTS),
            trace_point=(ex_tip, ey_tip, 0.0),
            component_x_line=segment((0.0, 0.0, 0.0), (ex_tip, 0.0, 0.0)),
            component_y_line=segment((0.0, 0.0, 0.0), (0.0, ey_tip, 0.0)),
            total_vector_line=segment((0.0, 0.0, 0.0), (ex_tip, ey_tip, 0.0)),
            projection_line=projection_line,
            antenna_line=antenna_line,
            wave_field=VectorFieldData(
                np.zeros_like(ARROW_Z),
                np.zeros_like(ARROW_Z),
                ARROW_Z,
                arrow_data.ex,
                arrow_data.ey,
                np.zeros_like(ARROW_Z),
                mode_data.color,
                0.45,
                1.1,
            ),
            magnetic_line=line_from_components(hx, hy, Z_POINTS),
            magnetic_field=VectorFieldData(
                np.zeros_like(ARROW_Z),
                np.zeros_like(ARROW_Z),
                ARROW_Z,
                arrow_hx,
                arrow_hy,
                np.zeros_like(ARROW_Z),
                "royalblue",
                0.42,
                1.1,
            ),
            field_extent=mode_data.field_extent,
            status=mode_data.status,
            color=mode_data.color,
        )


@dataclass(frozen=True, slots=True)
class ModeData:
    ex: np.ndarray
    ey: np.ndarray
    status: str
    color: str
    field_extent: float
    details: tuple[str, ...]
    hint: str
    projection_tip: tuple[float, float, float] | None = None
    antenna_segment: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None


def validate(input_data: PolarizationInput) -> None:
    if input_data.mode not in MODE_LABELS:
        raise ValidationError("Unsupported polarization mode.")
    if input_data.h_display not in {"隐藏", "H", "377H"}:
        raise ValidationError("Unsupported H display mode.")
    if input_data.zoom <= 0.0:
        raise ValidationError("Zoom must be positive.")
    if input_data.mode in {"phase", "circular"} and (input_data.p1 < 0.0 or input_data.p2 < 0.0):
        raise ValidationError("Polarization amplitudes cannot be negative.")
    if input_data.mode == "match" and input_data.p3 <= 0.0:
        raise ValidationError("Antenna-match amplitude must be positive.")


def wrap_phase_deg(angle_deg: float) -> float:
    return float(((angle_deg + 180.0) % 360.0) - 180.0)


def classify_phase(ex_amp: float, ey_amp: float, phase_deg: float) -> tuple[str, str]:
    tolerance = 1e-3
    phase = wrap_phase_deg(phase_deg)
    if ex_amp < tolerance and ey_amp < tolerance:
        return "无场", "gray"
    if ex_amp < tolerance:
        return "垂直线极化 (Ey)", "black"
    if ey_amp < tolerance:
        return "水平线极化 (Ex)", "black"
    if abs(phase) < 1.0 or abs(abs(phase) - 180.0) < 1.0:
        return "线极化", "black"
    if abs(abs(phase) - 90.0) < 1.0 and abs(ex_amp - ey_amp) < 1e-2:
        return ("左旋圆极化 (LCP)", "purple") if phase > 0.0 else ("右旋圆极化 (RCP)", "purple")
    return ("左旋椭圆极化", "firebrick") if phase > 0.0 else ("右旋椭圆极化", "firebrick")


def classify_circular(lhcp_amp: float, rhcp_amp: float) -> tuple[str, str]:
    tolerance = 1e-2
    if lhcp_amp < tolerance and rhcp_amp < tolerance:
        return "无场", "gray"
    if abs(lhcp_amp - rhcp_amp) < tolerance:
        return "线极化", "black"
    if lhcp_amp > rhcp_amp:
        return ("纯左旋圆极化 (LHCP)", "royalblue") if rhcp_amp < tolerance else ("左旋椭圆极化", "royalblue")
    return ("纯右旋圆极化 (RHCP)", "forestgreen") if lhcp_amp < tolerance else ("右旋椭圆极化", "forestgreen")


def classify_match(incident_angle: float, antenna_angle: float) -> tuple[str, str, float]:
    mismatch = abs(((incident_angle - antenna_angle + 90.0) % 180.0) - 90.0)
    if mismatch < 1.0:
        return "极化完全匹配", "green", mismatch
    if abs(mismatch - 90.0) < 1.0:
        return "极化完全正交", "red", mismatch
    return "极化失配", "darkorange", mismatch


def current_phase(time: float, samples: np.ndarray) -> np.ndarray:
    return time - K * samples


def build_mode_data(input_data: PolarizationInput, samples: np.ndarray) -> ModeData:
    phase = current_phase(input_data.time, samples)
    if input_data.mode == "phase":
        ex_amp = input_data.p1
        ey_amp = input_data.p2
        delta_deg = input_data.p3
        ex = ex_amp * np.cos(phase)
        ey = ey_amp * np.cos(phase + np.radians(delta_deg))
        status, color = classify_phase(ex_amp, ey_amp, delta_deg)
        return ModeData(
            ex,
            ey,
            status,
            color,
            float(max(np.max(np.abs(ex)), np.max(np.abs(ey)), 1.0)),
            (f"Ex 幅度: {ex_amp:.2f}", f"Ey 幅度: {ey_amp:.2f}", f"相位差: {delta_deg:.0f} deg"),
            "直接调 Ex/Ey 幅度与相位差，观察线极化、圆极化和椭圆极化。",
        )
    if input_data.mode == "circular":
        lhcp_amp = input_data.p1
        rhcp_amp = input_data.p2
        ex = (lhcp_amp + rhcp_amp) * np.cos(phase)
        ey = (lhcp_amp - rhcp_amp) * np.sin(phase)
        status, color = classify_circular(lhcp_amp, rhcp_amp)
        return ModeData(
            ex,
            ey,
            status,
            color,
            float(max(np.max(np.abs(ex)), np.max(np.abs(ey)), 1.0)),
            (f"LHCP 幅度: {lhcp_amp:.2f}", f"RHCP 幅度: {rhcp_amp:.2f}"),
            "用左旋/右旋圆极化基底叠加，观察极化如何从圆变线再变椭圆。",
        )

    incident_angle = input_data.p1
    antenna_angle = input_data.p2
    amplitude = input_data.p3
    incident_rad = np.radians(incident_angle)
    antenna_rad = np.radians(antenna_angle)
    ex = amplitude * np.sin(incident_rad) * np.cos(phase)
    ey = amplitude * np.cos(incident_rad) * np.cos(phase)
    status, color, mismatch = classify_match(incident_angle, antenna_angle)
    antenna_dir = np.array([np.sin(antenna_rad), np.cos(antenna_rad), 0.0])
    total_tip = np.array([ex[0], ey[0], 0.0])
    projection_length = float(total_tip[0] * antenna_dir[0] + total_tip[1] * antenna_dir[1])
    projection_tip_arr = projection_length * antenna_dir
    antenna_half_length = max(amplitude, 1.0) * 0.95
    receive_power = (np.cos(np.radians(incident_angle - antenna_angle)) ** 2) * 100.0
    return ModeData(
        ex,
        ey,
        status,
        color,
        float(max(np.max(np.abs(ex)), np.max(np.abs(ey)), amplitude, 1.0)),
        (
            f"入射角: {incident_angle:.0f} deg",
            f"天线角度: {antenna_angle:.0f} deg",
            f"接收功率: {receive_power:.1f}%",
            f"失配角: {mismatch:.0f} deg",
        ),
        "线极化波沿 z 轴传播，接收天线按投影获得有效电场与接收功率。",
        tuple(float(v) for v in projection_tip_arr),
        (
            tuple(float(v) for v in -antenna_half_length * antenna_dir),
            tuple(float(v) for v in antenna_half_length * antenna_dir),
        ),
    )


def build_limits(field_extent: float, zoom: float) -> AxisLimits:
    zoom = float(np.clip(zoom, MIN_ZOOM, MAX_ZOOM))
    xy_half = max(field_extent * 1.35, 1.2) / zoom
    z_half = (Z_LENGTH * 0.55) / zoom
    z_center = Z_LENGTH / 2.0
    return limits((-xy_half, xy_half), (-xy_half, xy_half), (z_center - z_half, z_center + z_half))


def panel_text(input_data: PolarizationInput, mode_data: ModeData) -> PanelText:
    return PanelText(
        hint=f"{MODE_LABELS[input_data.mode]}\n{mode_data.hint}",
        status_lines=("当前模式", MODE_LABELS[input_data.mode], "", "当前极化", mode_data.status, "", *mode_data.details),
        metrics_lines=("传播轴: z", f"缩放: {input_data.zoom:.1f}x"),
        status_color=mode_data.color,
    )


class ElectromagneticPhysics:
    @staticmethod
    def calculate_polarization(ex_amp: float, ey_amp: float, phase_diff_deg: float) -> tuple[str, str]:
        return classify_phase(ex_amp, ey_amp, phase_diff_deg)
