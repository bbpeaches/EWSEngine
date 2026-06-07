from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.interfaces import SimulationEngine
from backend.models import PanelText, PlaneData, WaveFrame, WaveInput, empty_line
from backend.physics.geometry import limits, line_from_components, marker_from_point, normalize, plane, point
from core.exceptions import ValidationError

LIGHT_SPEED = 299_792_458.0
Z_LINE = np.linspace(0.0, 1.6, 460)
PLANE_RANGE = 4.5
MIN_ZOOM = 0.65
MAX_ZOOM = 4.0
MATERIAL_TIME_UNIT = 1e-9
LOSSY_TIME_UNIT = 0.8

MATERIALS = {
    "真空/空气 (εr=1.0)": 1.0,
    "聚四氟乙烯 (εr=2.1)": 2.1,
    "FR4 玻纤板 (εr=4.4)": 4.4,
    "纯水 (εr=81.0)": 81.0,
}


@dataclass(frozen=True, slots=True)
class WaveEngine(SimulationEngine[WaveInput, WaveFrame]):
    def simulate(self, input_data: WaveInput) -> WaveFrame:
        validate(input_data)
        if input_data.mode == "material":
            return render_material(input_data)
        if input_data.mode == "lossy":
            return render_lossy(input_data)
        return render_planes(input_data)


def validate(input_data: WaveInput) -> None:
    if input_data.mode not in {"material", "lossy", "planes"}:
        raise ValidationError("Unsupported wave mode.")
    if input_data.material not in MATERIALS:
        raise ValidationError("Unknown material.")
    if input_data.freq_mhz <= 0.0:
        raise ValidationError("Frequency must be positive.")
    if input_data.alpha < 0.0 or input_data.beta <= 0.0 or input_data.spacing <= 0.0:
        raise ValidationError("Wave constants must be positive where required.")
    if input_data.zoom <= 0.0:
        raise ValidationError("Zoom must be positive.")


def material_params(input_data: WaveInput) -> tuple[float, float, float, float, float, float]:
    f_hz = input_data.freq_mhz * 1e6
    eps_r = MATERIALS[input_data.material]
    vp = LIGHT_SPEED / np.sqrt(eps_r)
    wavelength = vp / f_hz
    k = 2.0 * np.pi / wavelength
    omega = 2.0 * np.pi * f_hz
    return f_hz, eps_r, vp, wavelength, k, omega


def render_material(input_data: WaveInput) -> WaveFrame:
    _, eps_r, vp, wavelength, k, omega = material_params(input_data)
    wave_time = input_data.time * MATERIAL_TIME_UNIT
    wave_y = np.cos(omega * wave_time - k * Z_LINE)
    track_z = float((vp * wave_time) % Z_LINE[-1])
    track_y = float(np.cos(omega * wave_time - k * track_z))
    return WaveFrame(
        "材料行波 3D 实验",
        PanelText(
            "模式说明\n不同介质改变相速度、波长和波数。",
            (f"媒质: {input_data.material}", f"频率: {input_data.freq_mhz:.0f} MHz", f"相速度 Vp: {vp / 1e6:.2f} x10^6 m/s"),
            ("关键量", f"eps_r = {eps_r:.2f}", f"波长 lambda = {wavelength:.3f} m", f"波数 k = {k:.2f} rad/m"),
        ),
        wave_limits(input_data.zoom, 1.6, False),
        line_from_components(Z_LINE, np.zeros_like(Z_LINE), wave_y),
        empty_line(),
        empty_line(),
        line_from_components((0.0, Z_LINE[-1]), (0.0, 0.0), (0.0, 0.0)),
        empty_line(),
        marker_from_point((track_z, 0.0, track_y)),
        tuple(),
    )


def render_lossy(input_data: WaveInput) -> WaveFrame:
    env_y = np.exp(-input_data.alpha * Z_LINE)
    wave_time = input_data.time * LOSSY_TIME_UNIT
    wave_y = env_y * np.cos(2.0 * np.pi * wave_time - input_data.beta * Z_LINE)
    track_x = float(0.25 + ((wave_time * 0.18) % 1.0) * (Z_LINE[-1] - 0.25))
    track_env = float(np.exp(-input_data.alpha * track_x))
    track_y = float(track_env * np.cos(2.0 * np.pi * wave_time - input_data.beta * track_x))
    return WaveFrame(
        "有损耗媒质 3D 实验",
        PanelText(
            "模式说明\n灰色虚线是指数衰减包络，蓝线是随时间推进的衰减波。",
            (f"衰减常数 alpha: {input_data.alpha:.2f} Np/m", f"相位常数 beta: {input_data.beta:.2f} rad/m", f"缩放: {input_data.zoom:.1f}x"),
            ("关键关系", "Envelope = e^(-alpha z)", "E(z,t) = e^(-alpha z) cos(omega t - beta z)", f"末端包络 = {env_y[-1]:.2f}"),
        ),
        wave_limits(input_data.zoom, 1.6, False),
        line_from_components(Z_LINE, np.zeros_like(Z_LINE), wave_y),
        line_from_components(Z_LINE, np.zeros_like(Z_LINE), env_y),
        line_from_components(Z_LINE, np.zeros_like(Z_LINE), -env_y),
        line_from_components((0.0, Z_LINE[-1]), (0.0, 0.0), (0.0, 0.0)),
        empty_line(),
        marker_from_point((track_x, 0.0, track_y)),
        tuple(),
    )


def render_planes(input_data: WaveInput) -> WaveFrame:
    theta = np.radians(input_data.theta_deg)
    phi = np.radians(input_data.phi_deg)
    k_vec = normalize((np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)))
    if abs(k_vec[2]) > 0.999:
        u = np.array([1.0, 0.0, 0.0])
    else:
        u = normalize(np.cross(k_vec, np.array([0.0, 0.0, 1.0])))
    v = normalize(np.cross(k_vec, u))
    planes: list[PlaneData] = []
    for offset in (-input_data.spacing, 0.0, input_data.spacing):
        center = point(offset * k_vec)
        planes.append(plane(center, point(u), point(v), 2.6, 2.6, "royalblue", 0.22))
    return WaveFrame(
        "波矢等相面 3D 实验",
        PanelText(
            "模式说明\n红色线为波矢 k，蓝色平面表示等相位面。",
            (f"极角 theta: {input_data.theta_deg:.1f} deg", f"方位角 phi: {input_data.phi_deg:.1f} deg", f"面间距 lambda: {input_data.spacing:.2f}"),
            ("关键量", f"k = ({k_vec[0]:.2f}, {k_vec[1]:.2f}, {k_vec[2]:.2f})", "等相位面: k dot r = 常数"),
        ),
        wave_limits(input_data.zoom, PLANE_RANGE, True),
        empty_line(),
        empty_line(),
        empty_line(),
        empty_line(),
        line_from_components((0.0, k_vec[0] * input_data.spacing * 1.5), (0.0, k_vec[1] * input_data.spacing * 1.5), (0.0, k_vec[2] * input_data.spacing * 1.5)),
        marker_from_point(None),
        tuple(planes),
    )


def wave_limits(zoom: float, extent: float, plane_mode: bool) -> object:
    zoom = float(np.clip(zoom, MIN_ZOOM, MAX_ZOOM))
    if plane_mode:
        half = PLANE_RANGE / zoom
        return limits((-half, half), (-half, half), (-half, half))
    x_half = 0.95 / zoom
    y_half = 1.45 / zoom
    z_half = max(extent, 1.8) / zoom
    return limits((0.0, x_half * 2.0), (-y_half, y_half), (-z_half, z_half))
