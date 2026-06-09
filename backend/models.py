from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from core.types import FloatArray

Point3 = tuple[float, float, float]
Bounds = tuple[float, float]
PolarizationMode = Literal["natural", "s", "p"]
PolarizationSceneMode = Literal["phase", "circular", "match"]
TransmissionMode = Literal["vswr", "standing"]
WaveMode = Literal["material", "lossy", "planes"]
TemMode = Literal["lossless", "lossy"]
SpeedMode = Literal["dispersion", "apparent"]
DirectionKey = Literal["x", "y", "z"]
HDisplayMode = Literal["隐藏", "H", "377H"]
INTRINSIC_IMPEDANCE = 377.0


@dataclass(frozen=True, slots=True)
class LineData:
    x: FloatArray
    y: FloatArray
    z: FloatArray


@dataclass(frozen=True, slots=True)
class Series2D:
    x: FloatArray
    y: FloatArray


@dataclass(frozen=True, slots=True)
class MarkerData:
    point: Point3 | None


@dataclass(frozen=True, slots=True)
class VectorData:
    start: Point3
    end: Point3
    color: str
    width: float = 2.4
    alpha: float = 0.95


@dataclass(frozen=True, slots=True)
class VectorFieldData:
    x: FloatArray
    y: FloatArray
    z: FloatArray
    u: FloatArray
    v: FloatArray
    w: FloatArray
    color: str
    alpha: float = 0.5
    linewidth: float = 1.1


@dataclass(frozen=True, slots=True)
class PlaneData:
    center: Point3
    basis_u: Point3
    basis_v: Point3
    half_u: float
    half_v: float
    color: str
    alpha: float
    edge_color: str | None = None


@dataclass(frozen=True, slots=True)
class AxisLimits:
    x: Bounds
    y: Bounds
    z: Bounds


@dataclass(frozen=True, slots=True)
class PanelText:
    hint: str
    status_lines: tuple[str, ...]
    metrics_lines: tuple[str, ...]
    status_color: str = "black"


@dataclass(frozen=True, slots=True)
class OpticsInput:
    n1: float = 1.0
    n2: float = 1.5
    theta_deg: float = 40.0
    phi_deg: float = 0.0
    polarization: PolarizationMode = "natural"
    time: float = 0.0
    zoom: float = 1.0


@dataclass(frozen=True, slots=True)
class FresnelResult:
    is_tir: bool
    is_brewster: bool
    theta_i_rad: float
    theta_t_rad: float | None
    theta_i_deg: float
    theta_t_deg: float | None
    theta_r_deg: float
    theta_c_deg: float | None
    theta_b_deg: float
    r_s_amp: float
    r_p_amp: float
    t_s_amp: float | None
    t_p_amp: float | None
    R_s: float
    R_p: float
    T_s: float
    T_p: float
    active_label: str
    active_R: float
    active_T: float
    phase_shift_s: float | None = None
    phase_shift_p: float | None = None


@dataclass(frozen=True, slots=True)
class OpticsFrame:
    title: str
    panel: PanelText
    axis_limits: AxisLimits
    result: FresnelResult
    incident_line: LineData
    reflected_line: LineData
    refracted_line: LineData
    normal_line: LineData
    interface_x_line: LineData
    interface_y_line: LineData
    side_line: LineData
    incident_arc: LineData
    reflected_arc: LineData
    refracted_arc: LineData
    dipole_curve: LineData
    planes: tuple[PlaneData, ...]
    vectors: tuple[VectorData, ...]
    incident_marker: MarkerData
    reflected_marker: MarkerData
    refracted_marker: MarkerData
    reflect_width: float
    reflect_alpha: float
    refract_width: float
    refract_alpha: float
    dipole_alpha: float


@dataclass(frozen=True, slots=True)
class PhasePolarizationInput:
    ex_amp: float = 1.0
    ey_amp: float = 1.0
    phase_deg: float = 90.0
    time: float = 0.0
    zoom: float = 1.0


@dataclass(frozen=True, slots=True)
class CircularPolarizationInput:
    lhcp_amp: float = 1.0
    rhcp_amp: float = 0.0
    time: float = 0.0
    zoom: float = 1.0


@dataclass(frozen=True, slots=True)
class PolarizationMatchInput:
    incident_angle_deg: float = 30.0
    antenna_angle_deg: float = 0.0
    amplitude: float = 1.0
    time: float = 0.0
    zoom: float = 1.0


@dataclass(frozen=True, slots=True)
class PolarizationInput:
    mode: PolarizationSceneMode = "phase"
    p1: float = 1.0
    p2: float = 1.0
    p3: float = 90.0
    h_display: HDisplayMode = "隐藏"
    time: float = 0.0
    zoom: float = 1.0


@dataclass(frozen=True, slots=True)
class PolarizationFrame:
    title: str
    panel: PanelText
    axis_limits: AxisLimits
    h_display: HDisplayMode
    wave_line: LineData
    trace_point: Point3
    component_x_line: LineData
    component_y_line: LineData
    total_vector_line: LineData
    projection_line: LineData
    antenna_line: LineData
    wave_field: VectorFieldData
    magnetic_line: LineData
    magnetic_field: VectorFieldData
    field_extent: float
    status: str
    color: str


@dataclass(frozen=True, slots=True)
class TransmissionInput:
    mode: TransmissionMode = "vswr"
    reflection_coefficient: float = 0.0
    h_display: HDisplayMode = "隐藏"
    time: float = 0.0
    zoom: float = 1.0


@dataclass(frozen=True, slots=True)
class TransmissionFrame:
    title: str
    panel: PanelText
    axis_limits: AxisLimits
    h_display: HDisplayMode
    electric_line: LineData
    magnetic_line: LineData
    envelope_line: LineData
    standing_line: LineData
    traveling_line: LineData
    axis_line: LineData
    boundary_line: LineData
    electric_marker: MarkerData
    magnetic_marker: MarkerData
    envelope_marker: MarkerData


@dataclass(frozen=True, slots=True)
class WaveInput:
    mode: WaveMode = "material"
    freq_mhz: float = 1000.0
    material: str = "真空/空气 (εr=1.0)"
    alpha: float = 0.3
    beta: float = 5.0
    theta_deg: float = 45.0
    phi_deg: float = 45.0
    spacing: float = 1.5
    h_display: HDisplayMode = "隐藏"
    time: float = 0.0
    zoom: float = 1.0


@dataclass(frozen=True, slots=True)
class WaveFrame:
    title: str
    panel: PanelText
    axis_limits: AxisLimits
    h_display: HDisplayMode
    wave_line: LineData
    magnetic_line: LineData
    envelope_up: LineData
    envelope_down: LineData
    axis_line: LineData
    wave_vector_line: LineData
    track_marker: MarkerData
    planes: tuple[PlaneData, ...]


@dataclass(frozen=True, slots=True)
class TemInput:
    mode: TemMode = "lossless"
    direction: DirectionKey = "x"
    polarity: float = 1.0
    amplitude: float = 3.0
    wavelength: float = 5.0
    speed: float = 2.0
    alpha: float = 0.0
    beta: float = 5.0
    h_display: HDisplayMode = "377H"
    time_scale: float = 1.0
    time: float = 0.0
    zoom: float = 1.0


@dataclass(frozen=True, slots=True)
class TemFrame:
    title: str
    panel: PanelText
    axis_limits: AxisLimits
    h_display: HDisplayMode
    direction: DirectionKey
    wave_series: Series2D
    reference_series: Series2D
    peak_point_2d: tuple[float, float]
    wave_formula_lines: tuple[str, ...]
    wave_info_lines: tuple[str, ...]
    electric_line: LineData
    magnetic_line: LineData
    propagation_axis: LineData
    reference_x: LineData
    reference_y: LineData
    reference_z: LineData
    peak_marker_3d: MarkerData
    electric_field: VectorFieldData
    magnetic_field: VectorFieldData
    local_vectors: tuple[VectorData, ...]


@dataclass(frozen=True, slots=True)
class SpeedInput:
    mode: SpeedMode = "dispersion"
    direction: DirectionKey = "z"
    vg: float = 0.5
    theta_deg: float = 45.0
    amplitude: float = 1.0
    carrier_lambda: float = 2.0
    time: float = 0.0
    zoom: float = 1.0


@dataclass(frozen=True, slots=True)
class SpeedFrame:
    title: str
    panel: PanelText
    axis_limits: AxisLimits
    wave_line: LineData
    envelope_up: LineData
    envelope_down: LineData
    observer_line: LineData
    propagation_axis: LineData
    vp_marker: MarkerData
    vg_marker: MarkerData
    apparent_marker: MarkerData
    planes: tuple[PlaneData, ...]
    vectors: tuple[VectorData, ...]


def empty_line() -> LineData:
    empty = np.empty(0, dtype=float)
    return LineData(empty, empty, empty)
