from __future__ import annotations

from backend.models import TemInput
from backend.physics.tem import TemEngine
from frontend.plot_tools import apply_limits, draw_vector_field, draw_vectors, set_marker, set_line_3d
from frontend.scenes.base import BaseSimulationScene

PRESETS = {
    "X 正向": {"direction": "x", "polarity": "1", "amplitude": 3.0, "wavelength": 5.0, "speed": 2.0},
    "Y 正向": {"direction": "y", "polarity": "1", "amplitude": 3.0, "wavelength": 5.0, "speed": 2.0},
    "Z 正向": {"direction": "z", "polarity": "1", "amplitude": 3.0, "wavelength": 5.0, "speed": 2.0},
    "X 反向": {"direction": "x", "polarity": "-1", "amplitude": 3.0, "wavelength": 5.0, "speed": 2.0},
}


class TemScene(BaseSimulationScene):
    def __init__(self) -> None:
        self.engine = TemEngine()
        self.dynamic_artists: list[object] = []
        super().__init__("TEM 平面波实验台")

    def build_controls(self) -> None:
        self.add_radio_group("preset", "快速预设", tuple(PRESETS.keys()), "X 正向", height=0.12)
        self.add_radio_group("direction", "传播轴", ("x", "y", "z"), "x", height=0.10)
        self.add_radio_group("polarity", "相位方向", ("1", "-1"), "1", height=0.10)
        self.add_slider("amplitude", "振幅", 0.5, 6.0, 3.0)
        self.add_slider("wavelength", "波长", 1.0, 10.0, 5.0)
        self.add_slider("speed", "传播速度", 0.5, 5.0, 2.0)
        self.add_slider("time_scale", "动画速度", 0.2, 3.0, 1.0)
        self.add_standard_controls()

    def on_radio_change(self, key: str, value: str) -> None:
        if key != "preset":
            return
        preset = PRESETS[value]
        self.set_radio_value("direction", preset["direction"])
        self.set_radio_value("polarity", preset["polarity"])
        for slider_key in ("amplitude", "wavelength", "speed"):
            self.set_slider_value(slider_key, preset[slider_key])

    def on_controls_changed(self) -> None:
        self.time_scale = self.sliders["time_scale"].val

    def init_artists(self) -> None:
        self.set_default_view(22.0, -58.0)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.electric_line, = self.ax.plot([], [], [], color="firebrick", lw=2.2, label="电场 E")
        self.magnetic_line, = self.ax.plot([], [], [], color="royalblue", lw=2.2, label="磁场 H")
        self.axis_line, = self.ax.plot([], [], [], color="goldenrod", lw=2.5, alpha=0.9, label="传播轴")
        self.ref_x, = self.ax.plot([], [], [], color="gray", linestyle="--", lw=1.0, alpha=0.22)
        self.ref_y, = self.ax.plot([], [], [], color="gray", linestyle="--", lw=1.0, alpha=0.22)
        self.ref_z, = self.ax.plot([], [], [], color="gray", linestyle="--", lw=1.0, alpha=0.22)
        self.peak_marker, = self.ax.plot([], [], [], "o", color="black", markersize=7, label="相位追踪点")
        self.ax.legend(loc="upper left", fontsize=9)

    def render(self) -> None:
        for artist in self.dynamic_artists:
            artist.remove()
        self.dynamic_artists.clear()
        frame = self.engine.simulate(
            TemInput(
                direction=self.radio_values["direction"],
                polarity=float(self.radio_values["polarity"]),
                amplitude=self.sliders["amplitude"].val,
                wavelength=self.sliders["wavelength"].val,
                speed=self.sliders["speed"].val,
                time_scale=self.sliders["time_scale"].val,
                time=self.time,
                zoom=self.zoom,
            )
        )
        set_line_3d(self.electric_line, frame.electric_line)
        set_line_3d(self.magnetic_line, frame.magnetic_line)
        set_line_3d(self.axis_line, frame.propagation_axis)
        set_line_3d(self.ref_x, frame.reference_x)
        set_line_3d(self.ref_y, frame.reference_y)
        set_line_3d(self.ref_z, frame.reference_z)
        set_marker(self.peak_marker, frame.peak_marker_3d)
        self.dynamic_artists.append(draw_vector_field(self.ax, frame.electric_field))
        self.dynamic_artists.append(draw_vector_field(self.ax, frame.magnetic_field))
        self.dynamic_artists.extend(draw_vectors(self.ax, frame.local_vectors))
        apply_limits(self.ax, frame.axis_limits)
        self.hint_text.set_text(frame.panel.hint)
        self.status_text.set_text("\n".join(frame.panel.status_lines))
        self.status_text.set_color(frame.panel.status_color)
        self.metrics_text.set_text("\n".join(frame.panel.metrics_lines))
