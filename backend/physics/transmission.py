from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.interfaces import SimulationEngine
from backend.models import MarkerData, PanelText, TransmissionFrame, TransmissionInput, empty_line
from backend.physics.geometry import limits, line_from_components, marker_from_point
from core.exceptions import ValidationError

DISTANCE = np.linspace(-4.0, 0.0, 460)
BETA = 2.0 * np.pi
MIN_ZOOM = 0.65
MAX_ZOOM = 4.0


@dataclass(frozen=True, slots=True)
class TransmissionEngine(SimulationEngine[TransmissionInput, TransmissionFrame]):
    def simulate(self, input_data: TransmissionInput) -> TransmissionFrame:
        validate(input_data)
        reflection = input_data.reflection_coefficient
        env_e, env_h = envelope_values(reflection)
        scale = float(np.clip(input_data.zoom, MIN_ZOOM, MAX_ZOOM))
        field_extent = max(float(np.max(env_e)), float(np.max(env_h)), 2.2)
        axis_limits = limits((float(np.mean(DISTANCE) - 2.35 / scale), float(np.mean(DISTANCE) + 2.35 / scale)), (-1.75 / scale, 1.75 / scale), (0.0, max(field_extent, 2.35) / scale))
        axis_line = line_from_components((DISTANCE[0], 0.0), (0.0, 0.0), (0.0, 0.0))
        boundary_line = line_from_components((0.0, 0.0), (-1.45, 1.45), (0.0, 0.0))

        if input_data.mode == "vswr":
            electric = line_from_components(DISTANCE, np.full_like(DISTANCE, -0.72), env_e)
            magnetic = line_from_components(DISTANCE, np.full_like(DISTANCE, 0.72), env_h)
            probe_x = probe_position(input_data.time)
            electric_marker = marker_from_point((probe_x, -0.72, float(np.interp(probe_x, DISTANCE, env_e))))
            magnetic_marker = marker_from_point((probe_x, 0.72, float(np.interp(probe_x, DISTANCE, env_h))))
            return TransmissionFrame(
                "驻波包络 3D 监视器",
                panel_text(input_data, env_e, env_h),
                axis_limits,
                electric,
                magnetic,
                empty_line(),
                empty_line(),
                empty_line(),
                axis_line,
                boundary_line,
                electric_marker,
                magnetic_marker,
                MarkerData(None),
            )

        abs_r = abs(reflection)
        standing = 2.0 * abs_r * np.abs(np.sin(BETA * DISTANCE))
        traveling = np.full_like(DISTANCE, 1.0 - abs_r)
        probe_x = probe_position(input_data.time)
        return TransmissionFrame(
            "行驻波分解 3D 实验",
            panel_text(input_data, env_e, env_h),
            axis_limits,
            empty_line(),
            empty_line(),
            line_from_components(DISTANCE, np.zeros_like(DISTANCE), env_e),
            line_from_components(DISTANCE, np.full_like(DISTANCE, -0.86), standing),
            line_from_components(DISTANCE, np.full_like(DISTANCE, 0.86), traveling),
            axis_line,
            boundary_line,
            MarkerData(None),
            MarkerData(None),
            marker_from_point((probe_x, 0.0, float(np.interp(probe_x, DISTANCE, env_e)))),
        )


def validate(input_data: TransmissionInput) -> None:
    if input_data.mode not in {"vswr", "standing"}:
        raise ValidationError("Unsupported transmission-line mode.")
    if not -0.999 < input_data.reflection_coefficient < 0.999:
        raise ValidationError("Reflection coefficient must be in (-0.999, 0.999).")
    if input_data.zoom <= 0.0:
        raise ValidationError("Zoom must be positive.")


def envelope_values(reflection: float) -> tuple[np.ndarray, np.ndarray]:
    phase = 2.0 * BETA * DISTANCE
    env_e = np.sqrt(np.maximum(0.0, 1.0 + reflection**2 + 2.0 * reflection * np.cos(phase)))
    env_h = np.sqrt(np.maximum(0.0, 1.0 + reflection**2 - 2.0 * reflection * np.cos(phase)))
    return env_e, env_h


def probe_position(time: float) -> float:
    probe_fraction = (time * 0.28) % 1.0
    return float(DISTANCE[0] + (DISTANCE[-1] - DISTANCE[0]) * probe_fraction)


def panel_text(input_data: TransmissionInput, env_e: np.ndarray, env_h: np.ndarray) -> PanelText:
    reflection = input_data.reflection_coefficient
    abs_r = abs(reflection)
    if abs_r <= 1e-3:
        state_msg = "纯行波：匹配良好，没有可见驻波。"
        color = "forestgreen"
        vswr = "1.00"
    elif abs_r >= 0.98:
        state_msg = "强驻波：几乎所有能量都被反射。"
        color = "firebrick"
        vswr = "inf"
    else:
        state_msg = "行波与驻波叠加：存在部分传输和反射。"
        color = "royalblue"
        vswr = f"{(1.0 + abs_r) / (1.0 - abs_r):.2f}"
    return PanelText(
        hint="模式说明\n驻波包络显示 E/H 能量分布；分解模式显示合成包络、驻波分量和行波基线。",
        status_lines=(
            f"当前模式: {'驻波包络' if input_data.mode == 'vswr' else '行驻波分解'}",
            f"反射系数 R: {reflection:.2f}",
            f"驻波比 VSWR: {vswr}",
            f"状态: {state_msg}",
            f"缩放: {input_data.zoom:.1f}x",
        ),
        metrics_lines=(
            "关键关系",
            "|E| = sqrt(1 + R^2 + 2R cos(2 beta x))",
            "|H| = sqrt(1 + R^2 - 2R cos(2 beta x))",
            f"E max / min = {np.max(env_e):.2f} / {np.min(env_e):.2f}",
            f"H max / min = {np.max(env_h):.2f} / {np.min(env_h):.2f}",
        ),
        status_color=color,
    )
