from __future__ import annotations

from backend.models import TransmissionInput
from backend.physics.transmission import TransmissionEngine
from frontend.plot_tools import apply_limits, set_marker, set_line_3d
from frontend.scenes.base import BaseSimulationScene

PRESETS = {
    "匹配传输": {"mode": "vswr", "reflection": 0.0},
    "电压波节": {"mode": "vswr", "reflection": -0.65},
    "电压波腹": {"mode": "vswr", "reflection": 0.65},
    "行驻混合": {"mode": "standing", "reflection": -0.45},
    "强驻波": {"mode": "standing", "reflection": -0.90},
}


class TransmissionScene(BaseSimulationScene):
    def __init__(self) -> None:
        self.engine = TransmissionEngine()
        super().__init__("行驻波传播实验台")

    def build_controls(self) -> None:
        self.add_radio_group("preset", "快速预设", tuple(PRESETS.keys()), "匹配传输", height=0.16)
        self.add_radio_group("mode", "实验模式", ("vswr", "standing"), "vswr", height=0.10)
        self.add_slider("reflection", "反射系数 R", -0.99, 0.99, 0.0)
        self.add_standard_controls()

    def on_radio_change(self, key: str, value: str) -> None:
        if key != "preset":
            return
        preset = PRESETS[value]
        self.set_radio_value("mode", preset["mode"])
        self.set_slider_value("reflection", preset["reflection"])

    def init_artists(self) -> None:
        self.set_default_view(24.0, -58.0)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("显示通道")
        self.ax.set_zlabel("归一化幅度")
        self.electric_line, = self.ax.plot([], [], [], color="firebrick", lw=2.5, label="电场包络")
        self.magnetic_line, = self.ax.plot([], [], [], color="royalblue", lw=2.5, linestyle="--", label="磁场包络")
        self.envelope_line, = self.ax.plot([], [], [], color="royalblue", lw=2.8, label="合成包络")
        self.standing_line, = self.ax.plot([], [], [], color="goldenrod", lw=2.0, linestyle="--", label="驻波分量")
        self.traveling_line, = self.ax.plot([], [], [], color="forestgreen", lw=2.0, linestyle="--", label="行波基线")
        self.axis_line, = self.ax.plot([], [], [], color="slategray", lw=2.2, alpha=0.85, label="传输线")
        self.boundary_line, = self.ax.plot([], [], [], color="black", lw=2.2, alpha=0.85, label="负载面")
        self.electric_marker, = self.ax.plot([], [], [], "o", color="firebrick", markersize=7)
        self.magnetic_marker, = self.ax.plot([], [], [], "o", color="royalblue", markersize=7)
        self.envelope_marker, = self.ax.plot([], [], [], "o", color="navy", markersize=7)
        self.ax.legend(loc="upper left", fontsize=9)

    def render(self) -> None:
        frame = self.engine.simulate(
            TransmissionInput(
                mode=self.radio_values["mode"],
                reflection_coefficient=self.sliders["reflection"].val,
                time=self.time,
                zoom=self.zoom,
            )
        )
        set_line_3d(self.electric_line, frame.electric_line)
        set_line_3d(self.magnetic_line, frame.magnetic_line)
        set_line_3d(self.envelope_line, frame.envelope_line)
        set_line_3d(self.standing_line, frame.standing_line)
        set_line_3d(self.traveling_line, frame.traveling_line)
        set_line_3d(self.axis_line, frame.axis_line)
        set_line_3d(self.boundary_line, frame.boundary_line)
        set_marker(self.electric_marker, frame.electric_marker)
        set_marker(self.magnetic_marker, frame.magnetic_marker)
        set_marker(self.envelope_marker, frame.envelope_marker)
        apply_limits(self.ax, frame.axis_limits)
        self.hint_text.set_text(frame.panel.hint)
        self.status_text.set_text("\n".join(frame.panel.status_lines))
        self.status_text.set_color(frame.panel.status_color)
        self.metrics_text.set_text("\n".join(frame.panel.metrics_lines))
