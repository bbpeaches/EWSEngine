from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.types import RadioSpec, SliderSpec
from frontend.plot_tools import apply_limits, set_line_3d, set_marker
from frontend.scenes.base import BaseSimulationScene

PRESETS: dict[str, dict[str, float]] = {
    "空气 -> 玻璃": {"n1": 1.0, "n2": 1.5, "theta_deg": 40.0, "phi_deg": 0.0},
    "玻璃 -> 空气": {"n1": 1.5, "n2": 1.0, "theta_deg": 45.0, "phi_deg": 0.0},
    "水 -> 空气": {"n1": 1.33, "n2": 1.0, "theta_deg": 50.0, "phi_deg": 35.0},
}


class OpticsScene(BaseSimulationScene):
    """Optics scene that owns artists only; frame data comes from SimulationClient."""

    module_key = "optics"
    title = "界面光学 3D 综合仿真"
    slider_specs = (
        SliderSpec("n1", "入射媒质 n1", 1.0, 4.0, 1.0, 0.01),
        SliderSpec("n2", "透射媒质 n2", 1.0, 4.0, 1.5, 0.01),
        SliderSpec("theta_deg", "入射角 theta_i", 0.0, 89.9, 40.0, 0.1, "royalblue"),
        SliderSpec("phi_deg", "入射面方位角 phi", 0.0, 360.0, 0.0, 1.0, "darkorange"),
    )
    radio_specs = (RadioSpec("polarization", "偏振模式", ("natural", "s", "p"), "natural"),)
    presets = PRESETS

    default_elev = 20.0
    default_azim = -56.0
    axis_labels = ("X", "Y", "Z")

    def init_artists(self) -> None:
        self.incident_line, = self.ax.plot([], [], [], color="royalblue", lw=2.8, label="入射光线")
        self.reflected_line, = self.ax.plot([], [], [], color="firebrick", lw=2.4, label="反射光线")
        self.refracted_line, = self.ax.plot([], [], [], color="forestgreen", lw=2.4, label="折射光线")
        self.normal_line, = self.ax.plot([], [], [], color="slategray", lw=1.6, linestyle="--", label="法线")
        self.interface_x, = self.ax.plot([], [], [], color="gray", lw=1.1, alpha=0.25)
        self.interface_y, = self.ax.plot([], [], [], color="gray", lw=1.1, alpha=0.25)
        self.side_line, = self.ax.plot([], [], [], color="slategray", lw=2.0, alpha=0.45)
        self.incident_arc, = self.ax.plot([], [], [], color="royalblue", lw=1.4, alpha=0.85)
        self.reflected_arc, = self.ax.plot([], [], [], color="firebrick", lw=1.4, alpha=0.85)
        self.refracted_arc, = self.ax.plot([], [], [], color="forestgreen", lw=1.4, alpha=0.85)
        self.dipole_curve, = self.ax.plot(
            [],
            [],
            [],
            color="darkseagreen",
            lw=1.8,
            alpha=0.85,
            label="偶极辐射图样",
        )
        self.incident_marker, = self.ax.plot([], [], [], "o", color="royalblue", markersize=7)
        self.reflected_marker, = self.ax.plot([], [], [], "o", color="firebrick", markersize=7)
        self.refracted_marker, = self.ax.plot([], [], [], "o", color="forestgreen", markersize=7)
        self.plane_artists = self.create_plane_artists(4)
        self.vector_lines = self.create_vector_line_artists(3)
        self.ax.legend(loc="upper left", fontsize=9)

    def render(self, payload: dict[str, Any]) -> Mapping[str, Any]:
        """Apply one normalized frame dictionary to persistent Matplotlib artists."""

        frame = payload
        set_line_3d(self.incident_line, frame["incident_line"])
        set_line_3d(self.reflected_line, frame["reflected_line"])
        set_line_3d(self.refracted_line, frame["refracted_line"])
        set_line_3d(self.normal_line, frame["normal_line"])
        set_line_3d(self.interface_x, frame["interface_x_line"])
        set_line_3d(self.interface_y, frame["interface_y_line"])
        set_line_3d(self.side_line, frame["side_line"])
        set_line_3d(self.incident_arc, frame["incident_arc"])
        set_line_3d(self.reflected_arc, frame["reflected_arc"])
        set_line_3d(self.refracted_arc, frame["refracted_arc"])
        set_line_3d(self.dipole_curve, frame["dipole_curve"])
        self.reflected_line.set_alpha(float(frame["reflect_alpha"]))
        self.reflected_line.set_linewidth(float(frame["reflect_width"]))
        self.refracted_line.set_alpha(float(frame["refract_alpha"]))
        self.refracted_line.set_linewidth(float(frame["refract_width"]))
        self.dipole_curve.set_alpha(float(frame["dipole_alpha"]))
        set_marker(self.incident_marker, frame["incident_marker"])
        set_marker(self.reflected_marker, frame["reflected_marker"])
        set_marker(self.refracted_marker, frame["refracted_marker"])
        apply_limits(self.ax, frame["axis_limits"])
        self._update_planes(frame.get("planes", []))
        self._update_vectors(frame.get("vectors", []))
        return _panel(frame)

    def _update_planes(self, planes: list[Mapping[str, Any]]) -> None:
        self.update_planes(self.plane_artists, planes)

    def _update_vectors(self, vectors: list[Mapping[str, Any]]) -> None:
        self.update_vectors(self.vector_lines, vectors)

    def on_control_changed(self, key: str, value: str, app: Any) -> None:
        return


def _panel(frame: Mapping[str, Any]) -> Mapping[str, Any]:
    panel = frame.get("panel", {})
    return panel if isinstance(panel, Mapping) else {}
