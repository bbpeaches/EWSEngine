from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.types import RadioSpec, SliderSpec
from frontend.plot_tools import apply_limits, set_line_3d, set_marker
from frontend.scenes.base import BaseSimulationScene

PRESETS: dict[str, dict[str, float | str]] = {
    "低群速": {"mode": "dispersion", "direction": "z", "vg": 0.5, "theta_deg": 45.0, "amplitude": 1.0, "carrier_lambda": 2.0},
    "高群速": {"mode": "dispersion", "direction": "z", "vg": 1.6, "theta_deg": 45.0, "amplitude": 1.0, "carrier_lambda": 2.0},
    "视在低角": {"mode": "apparent", "direction": "z", "vg": 0.5, "theta_deg": 25.0, "amplitude": 1.0, "carrier_lambda": 2.0},
    "视在高角": {"mode": "apparent", "direction": "z", "vg": 0.5, "theta_deg": 78.0, "amplitude": 1.0, "carrier_lambda": 2.0},
}


class SpeedScene(BaseSimulationScene):
    module_key = "speed"
    title = "波速效应实验台"
    slider_specs = (
        SliderSpec("vg", "群速 Vg", 0.1, 2.5, 0.5, 0.01, "firebrick"),
        SliderSpec("theta_deg", "观察夹角 theta", 0.0, 85.0, 45.0, 1.0, "forestgreen"),
        SliderSpec("amplitude", "振幅", 0.4, 2.4, 1.0, 0.05, "royalblue"),
        SliderSpec("carrier_lambda", "载波波长 lambda", 1.2, 5.0, 2.0, 0.05, "darkorange"),
    )
    radio_specs = (
        RadioSpec("mode", "实验模式", ("dispersion", "apparent"), "dispersion"),
        RadioSpec("direction", "传播方向", ("x", "y", "z"), "z"),
    )
    presets = PRESETS
    default_elev = 23.0
    default_azim = -62.0
    axis_labels = ("X", "Y", "Z")

    def init_artists(self) -> None:
        self.wave_line, = self.ax.plot([], [], [], color="royalblue", lw=2.4, label="合成波")
        self.env_up, = self.ax.plot([], [], [], color="firebrick", linestyle="--", lw=1.8, alpha=0.85, label="包络")
        self.env_down, = self.ax.plot([], [], [], color="firebrick", linestyle="--", lw=1.8, alpha=0.85)
        self.observer_line, = self.ax.plot([], [], [], color="black", linestyle="--", lw=2.4, alpha=0.8, label="观察线")
        self.axis_line, = self.ax.plot([], [], [], color="goldenrod", lw=2.6, alpha=0.95, label="传播轴")
        self.vp_marker, = self.ax.plot([], [], [], "o", color="navy", markersize=7, label="相速追踪点")
        self.vg_marker, = self.ax.plot([], [], [], "o", color="darkred", markersize=8, label="群速追踪点")
        self.apparent_marker, = self.ax.plot([], [], [], "o", color="crimson", markersize=8, label="视在交点")
        self.plane_artists = self.create_plane_artists(5)
        self.vector_lines = self.create_vector_line_artists(1)
        self.ax.legend(loc="upper left", fontsize=9)

    def on_mount(self, app: Any) -> None:
        self._apply_mode_visibility(app.get_radio_value("mode"), app)

    def on_control_changed(self, key: str, value: str, app: Any) -> None:
        if key == "mode":
            self._apply_mode_visibility(value, app)

    def _apply_mode_visibility(self, mode: str, app: Any) -> None:
        app.configure_slider(
            "vg",
            label="群速 Vg",
            minimum=0.1,
            maximum=2.5,
            value=app.get_slider_value("vg"),
            visible=mode == "dispersion",
        )
        app.configure_slider(
            "theta_deg",
            label="观察夹角 theta",
            minimum=0.0,
            maximum=85.0,
            value=app.get_slider_value("theta_deg"),
            visible=mode == "apparent",
        )

    def render(self, payload: dict[str, Any]) -> Mapping[str, Any]:
        frame = payload
        set_line_3d(self.wave_line, frame["wave_line"])
        set_line_3d(self.env_up, frame["envelope_up"])
        set_line_3d(self.env_down, frame["envelope_down"])
        set_line_3d(self.observer_line, frame["observer_line"])
        set_line_3d(self.axis_line, frame["propagation_axis"])
        set_marker(self.vp_marker, frame["vp_marker"])
        set_marker(self.vg_marker, frame["vg_marker"])
        set_marker(self.apparent_marker, frame["apparent_marker"])
        self.update_planes(self.plane_artists, frame.get("planes", []))
        self.update_vectors(self.vector_lines, frame.get("vectors", []))
        apply_limits(self.ax, frame["axis_limits"])
        return _panel(frame)


def _panel(frame: Mapping[str, Any]) -> Mapping[str, Any]:
    panel = frame.get("panel", {})
    return panel if isinstance(panel, Mapping) else {}
