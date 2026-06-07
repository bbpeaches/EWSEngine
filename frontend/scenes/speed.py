from __future__ import annotations

from backend.models import SpeedInput
from backend.physics.speed import SpeedEngine
from frontend.plot_tools import apply_limits, draw_planes, draw_vectors, set_marker, set_line_3d
from frontend.scenes.base import BaseSimulationScene

PRESETS = {
    "低群速": {"mode": "dispersion", "direction": "z", "vg": 0.5, "theta_deg": 45.0, "amplitude": 1.0, "carrier_lambda": 2.0},
    "高群速": {"mode": "dispersion", "direction": "z", "vg": 1.6, "theta_deg": 45.0, "amplitude": 1.0, "carrier_lambda": 2.0},
    "视在低角": {"mode": "apparent", "direction": "z", "vg": 0.5, "theta_deg": 25.0, "amplitude": 1.0, "carrier_lambda": 2.0},
    "视在高角": {"mode": "apparent", "direction": "z", "vg": 0.5, "theta_deg": 78.0, "amplitude": 1.0, "carrier_lambda": 2.0},
}


class SpeedScene(BaseSimulationScene):
    def __init__(self) -> None:
        self.engine = SpeedEngine()
        self.dynamic_artists: list[object] = []
        super().__init__("波速效应实验台")

    def build_controls(self) -> None:
        self.add_radio_group("preset", "快速预设", tuple(PRESETS.keys()), "低群速", height=0.12)
        self.add_radio_group("mode", "实验模式", ("dispersion", "apparent"), "dispersion", height=0.10)
        self.add_radio_group("direction", "传播方向", ("x", "y", "z"), "z", height=0.10)
        self.add_slider("vg", "群速 Vg", 0.1, 2.5, 0.5, "firebrick")
        self.add_slider("theta_deg", "观察夹角 theta", 0.0, 85.0, 45.0, "forestgreen")
        self.add_slider("amplitude", "振幅", 0.4, 2.4, 1.0, "royalblue")
        self.add_slider("carrier_lambda", "载波波长 lambda", 1.2, 5.0, 2.0, "darkorange")
        self.add_standard_controls()
        self._apply_mode_visibility("dispersion")

    def on_radio_change(self, key: str, value: str) -> None:
        if key == "preset":
            preset = PRESETS[value]
            self.set_radio_value("mode", preset["mode"])
            self.set_radio_value("direction", preset["direction"])
            for slider_key in ("vg", "theta_deg", "amplitude", "carrier_lambda"):
                self.set_slider_value(slider_key, preset[slider_key])
            return
        if key == "mode":
            self._apply_mode_visibility(value)

    def _apply_mode_visibility(self, mode: str) -> None:
        self.configure_slider("vg", label="群速 Vg", minimum=0.1, maximum=2.5, value=self.sliders["vg"].val, visible=mode == "dispersion")
        self.configure_slider("theta_deg", label="观察夹角 theta", minimum=0.0, maximum=85.0, value=self.sliders["theta_deg"].val, visible=mode == "apparent")

    def init_artists(self) -> None:
        self.set_default_view(23.0, -62.0)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.wave_line, = self.ax.plot([], [], [], color="royalblue", lw=2.4, label="合成波")
        self.env_up, = self.ax.plot([], [], [], color="firebrick", linestyle="--", lw=1.8, alpha=0.85, label="包络")
        self.env_down, = self.ax.plot([], [], [], color="firebrick", linestyle="--", lw=1.8, alpha=0.85)
        self.observer_line, = self.ax.plot([], [], [], color="black", linestyle="--", lw=2.4, alpha=0.8, label="观察线")
        self.axis_line, = self.ax.plot([], [], [], color="goldenrod", lw=2.6, alpha=0.95, label="传播轴")
        self.vp_marker, = self.ax.plot([], [], [], "o", color="navy", markersize=7, label="相速追踪点")
        self.vg_marker, = self.ax.plot([], [], [], "o", color="darkred", markersize=8, label="群速追踪点")
        self.apparent_marker, = self.ax.plot([], [], [], "o", color="crimson", markersize=8, label="视在交点")
        self.ax.legend(loc="upper left", fontsize=9)

    def render(self) -> None:
        for artist in self.dynamic_artists:
            artist.remove()
        self.dynamic_artists.clear()
        frame = self.engine.simulate(
            SpeedInput(
                mode=self.radio_values["mode"],
                direction=self.radio_values["direction"],
                vg=self.sliders["vg"].val,
                theta_deg=self.sliders["theta_deg"].val,
                amplitude=self.sliders["amplitude"].val,
                carrier_lambda=self.sliders["carrier_lambda"].val,
                time=self.time,
                zoom=self.zoom,
            )
        )
        set_line_3d(self.wave_line, frame.wave_line)
        set_line_3d(self.env_up, frame.envelope_up)
        set_line_3d(self.env_down, frame.envelope_down)
        set_line_3d(self.observer_line, frame.observer_line)
        set_line_3d(self.axis_line, frame.propagation_axis)
        set_marker(self.vp_marker, frame.vp_marker)
        set_marker(self.vg_marker, frame.vg_marker)
        set_marker(self.apparent_marker, frame.apparent_marker)
        self.dynamic_artists.extend(draw_planes(self.ax, frame.planes))
        self.dynamic_artists.extend(draw_vectors(self.ax, frame.vectors))
        apply_limits(self.ax, frame.axis_limits)
        self.hint_text.set_text(frame.panel.hint)
        self.status_text.set_text("\n".join(frame.panel.status_lines))
        self.status_text.set_color(frame.panel.status_color)
        self.metrics_text.set_text("\n".join(frame.panel.metrics_lines))
