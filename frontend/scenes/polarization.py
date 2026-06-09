from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from backend.models import INTRINSIC_IMPEDANCE
from core.types import RadioSpec, SliderSpec
from frontend.plot_tools import apply_limits, clear_line, set_line_3d, set_scaled_line_3d, set_scaled_vector_field_lines
from frontend.scenes.base import BaseSimulationScene

MODE_CONFIG = {
    "phase": {
        "p1": ("Ex 幅度", 0.0, 1.5, 1.0),
        "p2": ("Ey 幅度", 0.0, 1.5, 1.0),
        "p3": ("相位差 phase", -180.0, 180.0, 90.0),
    },
    "circular": {
        "p1": ("LHCP 幅度", 0.0, 1.5, 1.0),
        "p2": ("RHCP 幅度", 0.0, 1.5, 0.0),
        "p3": ("占位参数", -180.0, 180.0, 0.0),
    },
    "match": {
        "p1": ("入射角", 0.0, 180.0, 30.0),
        "p2": ("天线角度", 0.0, 180.0, 0.0),
        "p3": ("场强幅度", 0.3, 1.5, 1.0),
    },
}

PRESETS: dict[str, dict[str, float | str]] = {
    "圆极化": {"mode": "phase", "p1": 1.0, "p2": 1.0, "p3": 90.0},
    "线极化": {"mode": "phase", "p1": 1.0, "p2": 1.0, "p3": 0.0},
    "左旋基底": {"mode": "circular", "p1": 1.0, "p2": 0.0, "p3": 0.0},
    "天线匹配": {"mode": "match", "p1": 30.0, "p2": 30.0, "p3": 1.0},
    "天线失配": {"mode": "match", "p1": 30.0, "p2": 80.0, "p3": 1.0},
}


class PolarizationScene(BaseSimulationScene):
    module_key = "polarization"
    title = "电磁波极化合成实验室"
    slider_specs = (
        SliderSpec("p1", "Ex 幅度", 0.0, 1.5, 1.0, 0.05, "royalblue"),
        SliderSpec("p2", "Ey 幅度", 0.0, 1.5, 1.0, 0.05, "forestgreen"),
        SliderSpec("p3", "相位差 phase", -180.0, 180.0, 90.0, 1.0, "darkorange"),
    )
    radio_specs = (
        RadioSpec("mode", "实验模式", ("phase", "circular", "match"), "phase"),
        RadioSpec("h_display", "磁场显示", ("隐藏", "H", "377H"), "隐藏"),
    )
    presets = PRESETS
    default_elev = 24.0
    default_azim = -58.0
    axis_labels = ("Ex", "Ey", "z")

    def init_artists(self) -> None:
        self.ax.set_box_aspect((1.0, 1.0, 1.8))
        self.wave_line, = self.ax.plot([], [], [], color="firebrick", lw=2.2, label="电场波形")
        self.magnetic_line, = self.ax.plot([], [], [], color="royalblue", lw=2.0, linestyle="--", label="磁场 H / 377H")
        self.component_x, = self.ax.plot([], [], [], color="royalblue", lw=2.0, label="Ex 分量")
        self.component_y, = self.ax.plot([], [], [], color="forestgreen", lw=2.0, label="Ey 分量")
        self.total_line, = self.ax.plot([], [], [], color="firebrick", lw=3.0, label="总电场")
        self.projection_line, = self.ax.plot([], [], [], color="darkorange", lw=3.0, label="接收投影")
        self.antenna_line, = self.ax.plot([], [], [], color="black", lw=2.6, label="接收天线")
        self.wave_field_lines = self.create_vector_field_artists(24)
        self.magnetic_field_lines = self.create_vector_field_artists(24)
        self.ax.legend(loc="upper left", fontsize=9)

    def on_mount(self, app: Any) -> None:
        self._apply_mode(app.get_radio_value("mode"), app, use_defaults=False)

    def on_control_changed(self, key: str, value: str, app: Any) -> None:
        if key == "mode":
            self._apply_mode(value, app, use_defaults=True)

    def _apply_mode(self, mode: str, app: Any, *, use_defaults: bool) -> None:
        for slider_key, (label, minimum, maximum, default) in MODE_CONFIG[mode].items():
            value = default if use_defaults else app.get_slider_value(slider_key)
            app.configure_slider(
                slider_key,
                label=label,
                minimum=minimum,
                maximum=maximum,
                value=value,
                visible=True,
            )
        if mode == "circular":
            app.configure_slider("p3", label="占位参数", minimum=-180.0, maximum=180.0, value=0.0, visible=False)

    def render(self, payload: dict[str, Any]) -> Mapping[str, Any]:
        frame = payload
        set_line_3d(self.wave_line, frame["wave_line"])
        set_line_3d(self.component_x, frame["component_x_line"])
        set_line_3d(self.component_y, frame["component_y_line"])
        set_line_3d(self.total_line, frame["total_vector_line"])
        set_line_3d(self.projection_line, frame["projection_line"])
        set_line_3d(self.antenna_line, frame["antenna_line"])
        self.wave_line.set_color(frame["color"])
        self.total_line.set_color(frame["color"])
        self.update_vector_field(self.wave_field_lines, frame["wave_field"])
        self._render_magnetic(frame)
        apply_limits(self.ax, frame["axis_limits"])
        return _panel(frame)

    def _render_magnetic(self, frame: dict[str, Any]) -> None:
        h_display = frame.get("h_display", "隐藏")
        if h_display == "隐藏":
            clear_line(self.magnetic_line)
            self.magnetic_line.set_visible(False)
            for line in self.magnetic_field_lines:
                clear_line(line)
                line.set_visible(False)
            return
        scale = INTRINSIC_IMPEDANCE if h_display == "377H" else 1.0
        set_scaled_line_3d(self.magnetic_line, frame["magnetic_line"], ("x", "y"), scale)
        set_scaled_vector_field_lines(self.magnetic_field_lines, frame["magnetic_field"], ("x", "y"), scale)


def _panel(frame: Mapping[str, Any]) -> Mapping[str, Any]:
    panel = frame.get("panel", {})
    return panel if isinstance(panel, Mapping) else {}
