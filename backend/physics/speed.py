from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.interfaces import SimulationEngine
from backend.models import PanelText, PlaneData, SpeedFrame, SpeedInput, VectorData, empty_line
from backend.physics.geometry import limits, line_from_components, marker_from_point, normalize, plane, point, vector
from core.exceptions import ValidationError

AXIS_LENGTH = 36.0
NUM_POINTS = 420
VP_REAL = 1.0
MIN_ZOOM = 0.6
MAX_ZOOM = 4.0
WAVEFRONT_COUNT = 5
COS_EPS = 0.08
PROP_POINTS = np.linspace(0.0, AXIS_LENGTH, NUM_POINTS)

DIRECTION_CONFIG = {
    "x": {"label": "X轴传播", "english": "X-axis", "prop": 0, "field": 1, "side": 2, "color": "goldenrod"},
    "y": {"label": "Y轴传播", "english": "Y-axis", "prop": 1, "field": 2, "side": 0, "color": "seagreen"},
    "z": {"label": "Z轴传播", "english": "Z-axis", "prop": 2, "field": 0, "side": 1, "color": "mediumpurple"},
}


@dataclass(frozen=True, slots=True)
class SpeedEngine(SimulationEngine[SpeedInput, SpeedFrame]):
    def simulate(self, input_data: SpeedInput) -> SpeedFrame:
        validate(input_data)
        if input_data.mode == "dispersion":
            return render_dispersion(input_data)
        return render_apparent(input_data)


def validate(input_data: SpeedInput) -> None:
    if input_data.mode not in {"dispersion", "apparent"}:
        raise ValidationError("Unsupported speed mode.")
    if input_data.direction not in DIRECTION_CONFIG:
        raise ValidationError("Unsupported speed direction.")
    if input_data.vg <= 0.0 or input_data.amplitude <= 0.0 or input_data.carrier_lambda <= 0.0:
        raise ValidationError("Speed simulation values must be positive.")
    if not 0.0 <= input_data.theta_deg <= 85.0:
        raise ValidationError("Apparent speed angle must be in [0, 85].")


def basis(input_data: SpeedInput) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    config = DIRECTION_CONFIG[input_data.direction]
    vectors = []
    for axis_key in ("prop", "field", "side"):
        out = np.zeros(3)
        out[int(config[axis_key])] = 1.0
        vectors.append(out)
    return tuple(vectors)  # type: ignore[return-value]


def compose_curve(input_data: SpeedInput, prop_values, field_values=None, side_values=None):
    config = DIRECTION_CONFIG[input_data.direction]
    prop_arr = np.asarray(prop_values, dtype=float)
    field_arr = np.full_like(prop_arr, field_values, dtype=float) if np.isscalar(field_values) or field_values is None else np.asarray(field_values, dtype=float)
    side_arr = np.full_like(prop_arr, side_values, dtype=float) if np.isscalar(side_values) or side_values is None else np.asarray(side_values, dtype=float)
    coords = [np.zeros_like(prop_arr), np.zeros_like(prop_arr), np.zeros_like(prop_arr)]
    coords[int(config["prop"])] = prop_arr
    coords[int(config["field"])] = field_arr
    coords[int(config["side"])] = side_arr
    return line_from_components(coords[0], coords[1], coords[2])


def compose_point(input_data: SpeedInput, prop_value: float, field_value: float = 0.0, side_value: float = 0.0) -> tuple[float, float, float]:
    config = DIRECTION_CONFIG[input_data.direction]
    out = np.zeros(3)
    out[int(config["prop"])] = prop_value
    out[int(config["field"])] = field_value
    out[int(config["side"])] = side_value
    return tuple(float(v) for v in out)


def render_dispersion(input_data: SpeedInput) -> SpeedFrame:
    config = DIRECTION_CONFIG[input_data.direction]
    e_prop, _, _ = basis(input_data)
    envelope_lambda = max(input_data.carrier_lambda * 5.0, 6.0)
    k_c = 2.0 * np.pi / input_data.carrier_lambda
    omega_c = k_c * VP_REAL
    dk = 2.0 * np.pi / envelope_lambda
    domega = dk * input_data.vg
    envelope = 2.0 * np.cos(domega * input_data.time - dk * PROP_POINTS)
    carrier = np.cos(omega_c * input_data.time - k_c * PROP_POINTS)
    wave_total = input_data.amplitude * envelope * carrier
    envelope_track = input_data.amplitude * np.abs(envelope)
    vg_pos = float((input_data.vg * input_data.time) % AXIS_LENGTH)
    vp_pos = float((VP_REAL * input_data.time) % AXIS_LENGTH)
    extent = float(max(np.max(np.abs(wave_total)), np.max(envelope_track), 2.5))
    vectors: tuple[VectorData, ...] = (
        vector(compose_point(input_data, 0.2, -input_data.amplitude * 2.4, -1.8), tuple(float(v) for v in compose_point(input_data, 0.2, -input_data.amplitude * 2.4, -1.8) + e_prop * 0), "goldenrod"),
    )
    return SpeedFrame(
        f"3D 波包色散实验 · {config['english']}",
        PanelText(
            "模式说明\n蓝线是合成载波，红色虚线是能量包络。",
            (f"当前模式: 波包色散", f"传播方向: {config['label']}", f"相速 Vp: {VP_REAL:.2f}", f"群速 Vg: {input_data.vg:.2f}", f"载波波长: {input_data.carrier_lambda:.2f}"),
            ("核心公式", "E = A * 2cos(delta omega t - delta k s) * cos(omega t - k s)", "群速位置: s = Vg * t", "相速位置: s = Vp * t"),
        ),
        speed_limits(input_data, extent),
        compose_curve(input_data, PROP_POINTS, wave_total, 0.0),
        compose_curve(input_data, PROP_POINTS, envelope_track, 0.0),
        compose_curve(input_data, PROP_POINTS, -envelope_track, 0.0),
        empty_line(),
        compose_curve(input_data, np.array([0.0, AXIS_LENGTH]), 0.0, 0.0),
        marker_from_point(compose_point(input_data, vp_pos, float(input_data.amplitude), 0.0)),
        marker_from_point(compose_point(input_data, vg_pos, float(input_data.amplitude * 2.0), 0.0)),
        marker_from_point(None),
        tuple(),
        tuple(),
    )


def render_apparent(input_data: SpeedInput) -> SpeedFrame:
    config = DIRECTION_CONFIG[input_data.direction]
    e_prop, e_field, e_side = basis(input_data)
    theta = np.radians(input_data.theta_deg)
    cos_theta = float(np.cos(theta))
    safe_cos = np.sign(cos_theta) * max(abs(cos_theta), COS_EPS)
    spacing = max(input_data.carrier_lambda * 1.5, 3.0)
    half_span = 2.3 + 0.45 * input_data.amplitude
    k_vec = normalize(np.cos(theta) * e_prop + np.sin(theta) * e_field)
    plane_vec = normalize(np.cross(k_vec, e_side))
    base_distance = AXIS_LENGTH * 0.12 + (VP_REAL * input_data.time) % spacing
    planes: list[PlaneData] = []
    for index in range(WAVEFRONT_COUNT):
        offset = index - WAVEFRONT_COUNT // 2
        center = point(k_vec * (base_distance + offset * spacing))
        planes.append(plane(center, point(e_side), point(plane_vec), half_span, half_span, "royalblue", 0.18))
    x_intersect = float((base_distance / safe_cos) % AXIS_LENGTH)
    vpr = VP_REAL / safe_cos
    return SpeedFrame(
        f"3D 视在相速实验 · {config['english']}",
        PanelText(
            "模式说明\n倾斜波阵面切过观察线时，交点会表现出更快的视在速度。",
            (f"当前模式: 视在相速", f"传播方向: {config['label']}", f"真实相速 Vp: {VP_REAL:.2f}", f"观察夹角 theta: {input_data.theta_deg:.1f} deg", f"视在相速 Vpr: {vpr:.2f}"),
            ("几何关系", "k dot r = 常数形成波阵面", "观察线交点速度: Vpr = Vp / cos(theta)", f"安全分母 cos(theta) = {safe_cos:.3f}"),
        ),
        speed_limits(input_data, max(half_span + 1.5, 4.0)),
        empty_line(),
        empty_line(),
        empty_line(),
        compose_curve(input_data, np.array([0.0, AXIS_LENGTH]), 0.0, 0.0),
        compose_curve(input_data, np.array([0.0, AXIS_LENGTH]), 0.0, 0.0),
        marker_from_point(None),
        marker_from_point(None),
        marker_from_point(compose_point(input_data, x_intersect, 0.0, 0.0)),
        tuple(planes),
        (vector(compose_point(input_data, AXIS_LENGTH * 0.18, 0.0, -1.4), tuple(float(v) for v in np.asarray(compose_point(input_data, AXIS_LENGTH * 0.18, 0.0, -1.4)) + k_vec * 3.2), "forestgreen", 2.6),),
    )


def speed_limits(input_data: SpeedInput, field_extent: float):
    config = DIRECTION_CONFIG[input_data.direction]
    zoom = float(np.clip(input_data.zoom, MIN_ZOOM, MAX_ZOOM))
    base_field_span = max(field_extent * 2.2, 4.0)
    base_side_span = max(field_extent * 1.8, 4.0)
    if input_data.mode == "apparent":
        base_field_span = max(base_field_span, 7.5)
        base_side_span = max(base_side_span, 6.0)
    spans = [base_field_span, base_field_span, base_field_span]
    centers = [0.0, 0.0, 0.0]
    spans[int(config["prop"])] = AXIS_LENGTH + 2.0
    spans[int(config["field"])] = base_field_span
    spans[int(config["side"])] = base_side_span
    centers[int(config["prop"])] = AXIS_LENGTH / 2.0
    scaled = [span / zoom for span in spans]
    halves = [span / 2.0 for span in scaled]
    return limits((centers[0] - halves[0], centers[0] + halves[0]), (centers[1] - halves[1], centers[1] + halves[1]), (centers[2] - halves[2], centers[2] + halves[2]))
