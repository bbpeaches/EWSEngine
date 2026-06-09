from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.interfaces import SimulationEngine
from backend.models import INTRINSIC_IMPEDANCE, PanelText, Series2D, TemFrame, TemInput, VectorData, VectorFieldData
from backend.physics.geometry import limits, line_from_components, marker_from_point, vector
from core.exceptions import ValidationError

AXIS_LENGTH = 30.0
NUM_POINTS = 360
NUM_ARROWS = 24
MIN_ZOOM = 0.6
MAX_ZOOM = 4.0
AXIS_NAMES = ("X", "Y", "Z")
PROP_POINTS = np.linspace(0.0, AXIS_LENGTH, NUM_POINTS)
ARROW_POINTS = np.linspace(0.0, AXIS_LENGTH, NUM_ARROWS)

DIRECTION_CONFIG = {
    "x": {"label": "X轴传播", "english": "X-axis", "prop": 0, "e": 1, "h": 2, "color": "goldenrod"},
    "y": {"label": "Y轴传播", "english": "Y-axis", "prop": 1, "e": 2, "h": 0, "color": "seagreen"},
    "z": {"label": "Z轴传播", "english": "Z-axis", "prop": 2, "e": 0, "h": 1, "color": "mediumpurple"},
}


@dataclass(frozen=True, slots=True)
class TemEngine(SimulationEngine[TemInput, TemFrame]):
    def simulate(self, input_data: TemInput) -> TemFrame:
        validate(input_data)
        wave_data, k, omega, envelope = compute_wave(input_data, PROP_POINTS)
        arrow_data = compute_wave(input_data, ARROW_POINTS)[0]
        config = DIRECTION_CONFIG[input_data.direction]
        h_data = input_data.polarity * wave_data / INTRINSIC_IMPEDANCE
        h_arrow = input_data.polarity * arrow_data / INTRINSIC_IMPEDANCE
        peak_position = float((input_data.polarity * input_data.speed * input_data.time) % AXIS_LENGTH)
        peak_value = float(compute_wave(input_data, np.array([peak_position]))[0][0])
        peak = build_point(input_data, peak_position, peak_value, int(config["e"]))
        axis_coords = [np.zeros(2), np.zeros(2), np.zeros(2)]
        axis_coords[int(config["prop"])] = np.array([0.0, AXIS_LENGTH])
        e_field = build_field(input_data, ARROW_POINTS, arrow_data, int(config["e"]), "firebrick")
        h_field = build_field(input_data, ARROW_POINTS, h_arrow, int(config["h"]), "royalblue")
        axis_limits = tem_limits(input_data)
        return TemFrame(
            f"TEM {'有损耗' if input_data.mode == 'lossy' else '均匀'}平面波 · {config['english']}",
            panel_text(input_data, k, omega, envelope),
            axis_limits,
            input_data.h_display,
            input_data.direction,
            Series2D(PROP_POINTS, wave_data),
            Series2D(PROP_POINTS, reference_wave(input_data, PROP_POINTS, k)),
            (peak_position, input_data.amplitude),
            wave_formula_lines(input_data, k, omega),
            (f"当前传播: {config['label']}", f"相位追踪点: s = {peak_position:.2f}"),
            build_line(input_data, PROP_POINTS, wave_data, int(config["e"])),
            build_line(input_data, PROP_POINTS, h_data, int(config["h"])),
            line_from_components(axis_coords[0], axis_coords[1], axis_coords[2]),
            line_from_components((axis_limits.x[0], axis_limits.x[1]), (0.0, 0.0), (0.0, 0.0)),
            line_from_components((0.0, 0.0), (axis_limits.y[0], axis_limits.y[1]), (0.0, 0.0)),
            line_from_components((0.0, 0.0), (0.0, 0.0), (axis_limits.z[0], axis_limits.z[1])),
            marker_from_point(tuple(float(v) for v in peak)),
            e_field,
            h_field,
            local_vectors(input_data, omega),
        )


def validate(input_data: TemInput) -> None:
    if input_data.mode not in {"lossless", "lossy"}:
        raise ValidationError("Unsupported TEM mode.")
    if input_data.h_display not in {"隐藏", "H", "377H"}:
        raise ValidationError("Unsupported H display mode.")
    if input_data.direction not in DIRECTION_CONFIG:
        raise ValidationError("Unsupported TEM direction.")
    if input_data.polarity not in {-1.0, 1.0}:
        raise ValidationError("TEM polarity must be 1 or -1.")
    if input_data.amplitude <= 0.0 or input_data.wavelength <= 0.0 or input_data.speed <= 0.0:
        raise ValidationError("TEM amplitude, wavelength and speed must be positive.")
    if input_data.alpha < 0.0 or input_data.beta <= 0.0:
        raise ValidationError("TEM lossy alpha must be non-negative and beta must be positive.")


def wave_params(input_data: TemInput) -> tuple[float, float]:
    if input_data.mode == "lossy":
        beta = input_data.beta
        return beta, beta * input_data.speed
    k = 2.0 * np.pi / input_data.wavelength
    omega = k * input_data.speed
    return k, omega


def compute_wave(input_data: TemInput, samples: np.ndarray) -> tuple[np.ndarray, float, float, np.ndarray]:
    k, omega = wave_params(input_data)
    envelope = np.exp(-input_data.alpha * samples) if input_data.mode == "lossy" else np.ones_like(samples)
    phase = omega * input_data.time - input_data.polarity * k * samples
    return input_data.amplitude * envelope * np.cos(phase), k, omega, envelope


def reference_wave(input_data: TemInput, samples: np.ndarray, k: float) -> np.ndarray:
    envelope = np.exp(-input_data.alpha * samples) if input_data.mode == "lossy" else np.ones_like(samples)
    return input_data.amplitude * envelope * np.cos(-input_data.polarity * k * samples)


def wave_formula_lines(input_data: TemInput, k: float, omega: float) -> tuple[str, ...]:
    sign = "-" if input_data.polarity > 0 else "+"
    if input_data.mode == "lossy":
        return (
            f"E = A e^(-alpha s) cos(omega t {sign} beta s)",
            f"alpha = {input_data.alpha:.2f}, beta = {k:.2f}, omega = {omega:.2f}",
            "图中磁场可按 H 或 377H 显示",
        )
    return (
        f"相位项: cos(omega t {sign} ks)",
        f"k = {k:.2f}, omega = {omega:.2f}",
        "图中磁场按 377H 等效显示",
    )


def build_line(input_data: TemInput, samples: np.ndarray, field_values: np.ndarray, field_axis: int):
    config = DIRECTION_CONFIG[input_data.direction]
    coords = [np.zeros_like(samples), np.zeros_like(samples), np.zeros_like(samples)]
    coords[int(config["prop"])] = samples
    coords[field_axis] = field_values
    return line_from_components(coords[0], coords[1], coords[2])


def build_point(input_data: TemInput, prop_value: float, field_value: float, field_axis: int) -> np.ndarray:
    config = DIRECTION_CONFIG[input_data.direction]
    out = np.zeros(3)
    out[int(config["prop"])] = prop_value
    out[field_axis] = field_value
    return out


def build_field(input_data: TemInput, samples: np.ndarray, field_values: np.ndarray, field_axis: int, color: str) -> VectorFieldData:
    config = DIRECTION_CONFIG[input_data.direction]
    origins = [np.zeros_like(samples), np.zeros_like(samples), np.zeros_like(samples)]
    vectors = [np.zeros_like(samples), np.zeros_like(samples), np.zeros_like(samples)]
    origins[int(config["prop"])] = samples
    vectors[field_axis] = field_values
    return VectorFieldData(origins[0], origins[1], origins[2], vectors[0], vectors[1], vectors[2], color, 0.52, 1.15)


def axis_vector(axis_index: int, magnitude: float) -> np.ndarray:
    out = np.zeros(3)
    out[axis_index] = magnitude
    return out


def local_vectors(input_data: TemInput, omega: float) -> tuple[VectorData, ...]:
    config = DIRECTION_CONFIG[input_data.direction]
    local = input_data.amplitude * np.cos(omega * input_data.time)
    if input_data.mode == "lossy":
        local *= np.exp(-input_data.alpha * 0.0)
    e_local = axis_vector(int(config["e"]), local)
    h_local = axis_vector(int(config["h"]), input_data.polarity * local / INTRINSIC_IMPEDANCE)
    k_start = np.zeros(3)
    k_start[int(config["prop"])] = 0.6 if input_data.polarity > 0 else AXIS_LENGTH - 0.6
    k_local = axis_vector(int(config["prop"]), input_data.polarity * 2.3)
    return (
        vector((0.0, 0.0, 0.0), tuple(float(v) for v in e_local), "firebrick", 2.9, 0.95),
        vector((0.0, 0.0, 0.0), tuple(float(v) for v in h_local), "royalblue", 2.9, 0.95),
        vector(tuple(float(v) for v in k_start), tuple(float(v) for v in k_start + k_local), "forestgreen", 2.6, 0.95),
    )


def tem_limits(input_data: TemInput):
    config = DIRECTION_CONFIG[input_data.direction]
    zoom = float(np.clip(input_data.zoom, MIN_ZOOM, MAX_ZOOM))
    field_extent = max(input_data.amplitude * 1.9, 1.8)
    spans = [field_extent * 2.0, field_extent * 2.0, field_extent * 2.0]
    centers = [0.0, 0.0, 0.0]
    spans[int(config["prop"])] = AXIS_LENGTH
    centers[int(config["prop"])] = AXIS_LENGTH / 2.0
    scaled = [span / zoom for span in spans]
    halves = [span / 2.0 for span in scaled]
    return limits((centers[0] - halves[0], centers[0] + halves[0]), (centers[1] - halves[1], centers[1] + halves[1]), (centers[2] - halves[2], centers[2] + halves[2]))


def panel_text(input_data: TemInput, k: float, omega: float, envelope: np.ndarray) -> PanelText:
    config = DIRECTION_CONFIG[input_data.direction]
    polarity_label = "正向传播" if input_data.polarity > 0 else "反向传播"
    h_sign = "+" if input_data.polarity > 0 else "-"
    if input_data.mode == "lossy":
        headline = "有损耗 TEM：电场与磁场随传播距离指数衰减。"
        status_lines = (
            f"TEM 模式: 有损耗",
            f"传播轴: {config['label']}",
            f"传播方向: {polarity_label}",
            f"振幅: {input_data.amplitude:.1f}",
            f"alpha: {input_data.alpha:.2f}",
            f"beta: {k:.2f}",
            f"包络末端: {envelope[-1]:.3f}",
        )
    else:
        headline = "无损耗 TEM：E、H 与 k 两两正交并同相传播。"
        status_lines = (
            f"TEM 模式: 无损耗",
            f"传播轴: {config['label']}",
            f"传播方向: {polarity_label}",
            f"振幅: {input_data.amplitude:.1f}",
            f"波长: {input_data.wavelength:.1f}",
            f"相速度: {input_data.speed:.1f}",
        )
    return PanelText(
        f"模式说明\n{headline}\n蓝色磁场可切换为真实 H 或 377H 共尺度显示。",
        status_lines,
        ("核心关系", f"k/beta = {k:.2f}", f"omega = {omega:.2f}", "E perpendicular H perpendicular k", f"H 当前指向: {h_sign}{AXIS_NAMES[int(config['h'])]} 轴"),
    )
