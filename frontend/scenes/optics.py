from __future__ import annotations

from backend.models import OpticsInput
from backend.physics.optics import OpticsEngine
from frontend.plot_tools import apply_limits, draw_planes, draw_vectors, set_line_3d, set_marker
from frontend.scenes.base import BaseSimulationScene

PRESETS = {
    "空气 -> 玻璃": {"n1": 1.0, "n2": 1.5, "theta_deg": 40.0, "phi_deg": 0.0},
    "玻璃 -> 空气": {"n1": 1.5, "n2": 1.0, "theta_deg": 45.0, "phi_deg": 0.0},
    "水 -> 空气": {"n1": 1.33, "n2": 1.0, "theta_deg": 50.0, "phi_deg": 35.0},
}


class OpticsScene(BaseSimulationScene):
    def __init__(self) -> None:
        self.engine = OpticsEngine()
        self.dynamic_artists: list[object] = []
        super().__init__("界面光学 3D 综合仿真")

    def build_controls(self) -> None:
        self.add_radio_group("preset", "常用预设", tuple(PRESETS.keys()), "空气 -> 玻璃", height=0.12)
        self.add_radio_group("polarization", "偏振模式", ("natural", "s", "p"), "natural", height=0.10)
        self.add_slider("n1", "入射媒质 n1", 1.0, 4.0, 1.0)
        self.add_slider("n2", "透射媒质 n2", 1.0, 4.0, 1.5)
        self.add_slider("theta_deg", "入射角 theta_i", 0.0, 89.9, 40.0, "royalblue")
        self.add_slider("phi_deg", "入射面方位角 phi", 0.0, 360.0, 0.0, "darkorange")
        self.add_standard_controls()

    def on_radio_change(self, key: str, value: str) -> None:
        if key != "preset":
            return
        for slider_key, slider_value in PRESETS[value].items():
            self.set_slider_value(slider_key, slider_value)

    def init_artists(self) -> None:
        self.set_default_view(20.0, -56.0)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
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
        self.dipole_curve, = self.ax.plot([], [], [], color="darkseagreen", lw=1.8, alpha=0.85, label="偶极辐射图样")
        self.incident_marker, = self.ax.plot([], [], [], "o", color="royalblue", markersize=7)
        self.reflected_marker, = self.ax.plot([], [], [], "o", color="firebrick", markersize=7)
        self.refracted_marker, = self.ax.plot([], [], [], "o", color="forestgreen", markersize=7)
        self.ax.legend(loc="upper left", fontsize=9)

    def render(self) -> None:
        for artist in self.dynamic_artists:
            artist.remove()
        self.dynamic_artists.clear()
        frame = self.engine.simulate(
            OpticsInput(
                n1=self.sliders["n1"].val,
                n2=self.sliders["n2"].val,
                theta_deg=self.sliders["theta_deg"].val,
                phi_deg=self.sliders["phi_deg"].val,
                polarization=self.radio_values["polarization"],
                time=self.time,
                zoom=self.zoom,
            )
        )
        set_line_3d(self.incident_line, frame.incident_line)
        set_line_3d(self.reflected_line, frame.reflected_line)
        set_line_3d(self.refracted_line, frame.refracted_line)
        set_line_3d(self.normal_line, frame.normal_line)
        set_line_3d(self.interface_x, frame.interface_x_line)
        set_line_3d(self.interface_y, frame.interface_y_line)
        set_line_3d(self.side_line, frame.side_line)
        set_line_3d(self.incident_arc, frame.incident_arc)
        set_line_3d(self.reflected_arc, frame.reflected_arc)
        set_line_3d(self.refracted_arc, frame.refracted_arc)
        set_line_3d(self.dipole_curve, frame.dipole_curve)
        self.reflected_line.set_alpha(frame.reflect_alpha)
        self.reflected_line.set_linewidth(frame.reflect_width)
        self.refracted_line.set_alpha(frame.refract_alpha)
        self.refracted_line.set_linewidth(frame.refract_width)
        self.dipole_curve.set_alpha(frame.dipole_alpha)
        set_marker(self.incident_marker, frame.incident_marker)
        set_marker(self.reflected_marker, frame.reflected_marker)
        set_marker(self.refracted_marker, frame.refracted_marker)
        apply_limits(self.ax, frame.axis_limits)
        self.dynamic_artists.extend(draw_planes(self.ax, frame.planes))
        self.dynamic_artists.extend(draw_vectors(self.ax, frame.vectors))
        self.hint_text.set_text(frame.panel.hint)
        self.status_text.set_text("\n".join(frame.panel.status_lines))
        self.status_text.set_color(frame.panel.status_color)
        self.metrics_text.set_text("\n".join(frame.panel.metrics_lines))
