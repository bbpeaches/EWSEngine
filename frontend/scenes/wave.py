from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from backend.physics.wave import MATERIALS
from core.types import RadioSpec, SliderSpec
from frontend.plot_tools import apply_limits, set_line_3d, set_marker
from frontend.scenes.base import BaseSimulationScene

MODE_FIELDS = {
    "material": {"freq_mhz": True, "alpha": False, "beta": False, "theta_deg": False, "phi_deg": False, "spacing": False},
    "lossy": {"freq_mhz": False, "alpha": True, "beta": True, "theta_deg": False, "phi_deg": False, "spacing": False},
    "planes": {"freq_mhz": False, "alpha": False, "beta": False, "theta_deg": True, "phi_deg": True, "spacing": True},
}

SLIDER_CONFIG = {
    "freq_mhz": ("频率 MHz", 100.0, 3000.0, 1000.0),
    "alpha": ("衰减常数 alpha", 0.0, 1.0, 0.3),
    "beta": ("相位常数 beta", 1.0, 10.0, 5.0),
    "theta_deg": ("极角 theta", 0.0, 180.0, 45.0),
    "phi_deg": ("方位角 phi", 0.0, 360.0, 45.0),
    "spacing": ("面间距 lambda", 0.5, 4.0, 1.5),
}

DEFAULT_MATERIAL = next(iter(MATERIALS.keys()))
PRESETS: dict[str, dict[str, float | str]] = {
    "空气行波": {"mode": "material", "material": DEFAULT_MATERIAL, "freq_mhz": 1000.0},
    "FR4 慢波": {"mode": "material", "material": "FR4 玻纤板 (εr=4.4)", "freq_mhz": 600.0},
    "轻度衰减": {"mode": "lossy", "material": DEFAULT_MATERIAL, "alpha": 0.2, "beta": 4.0},
    "强衰减": {"mode": "lossy", "material": DEFAULT_MATERIAL, "alpha": 0.8, "beta": 7.0},
    "倾斜波矢": {"mode": "planes", "material": DEFAULT_MATERIAL, "theta_deg": 55.0, "phi_deg": 40.0, "spacing": 1.6},
    "近竖直波矢": {"mode": "planes", "material": DEFAULT_MATERIAL, "theta_deg": 20.0, "phi_deg": 140.0, "spacing": 1.2},
}


class WaveScene(BaseSimulationScene):
    module_key = "wave"
    title = "基础波动实验台"
    slider_specs = (
        SliderSpec("freq_mhz", "频率 MHz", 100.0, 3000.0, 1000.0, 10.0, "royalblue"),
        SliderSpec("alpha", "衰减常数 alpha", 0.0, 1.0, 0.3, 0.01),
        SliderSpec("beta", "相位常数 beta", 1.0, 10.0, 5.0, 0.05, "forestgreen"),
        SliderSpec("theta_deg", "极角 theta", 0.0, 180.0, 45.0, 1.0, "royalblue"),
        SliderSpec("phi_deg", "方位角 phi", 0.0, 360.0, 45.0, 1.0, "darkorange"),
        SliderSpec("spacing", "面间距 lambda", 0.5, 4.0, 1.5, 0.05, "mediumpurple"),
    )
    radio_specs = (
        RadioSpec("mode", "实验模式", ("material", "lossy", "planes"), "material"),
        RadioSpec("material", "材料", tuple(MATERIALS.keys()), DEFAULT_MATERIAL),
    )
    presets = PRESETS
    default_elev = 24.0
    default_azim = -58.0
    axis_labels = ("空间 / 坐标", "显示通道 / 坐标", "幅度 / 相位面法向")

    def init_artists(self) -> None:
        self.wave_line, = self.ax.plot([], [], [], color="royalblue", lw=2.5, label="行波")
        self.envelope_up, = self.ax.plot([], [], [], color="gray", linestyle="--", lw=1.6, alpha=0.75, label="衰减包络")
        self.envelope_down, = self.ax.plot([], [], [], color="gray", linestyle="--", lw=1.6, alpha=0.75)
        self.axis_line, = self.ax.plot([], [], [], color="slategray", lw=2.2, alpha=0.9, label="传播轴")
        self.k_line, = self.ax.plot([], [], [], color="firebrick", lw=2.6, label="波矢 k")
        self.track_marker, = self.ax.plot([], [], [], "o", color="firebrick", markersize=8, label="相位追踪点")
        self.plane_artists = self.create_plane_artists(3)
        self.ax.legend(loc="upper left", fontsize=9)

    def on_mount(self, app: Any) -> None:
        self._apply_mode_visibility(app.get_radio_value("mode"), app)

    def on_control_changed(self, key: str, value: str, app: Any) -> None:
        if key == "mode":
            self._apply_mode_visibility(value, app)

    def _apply_mode_visibility(self, mode: str, app: Any) -> None:
        for slider_key, visible in MODE_FIELDS[mode].items():
            label, minimum, maximum, default = SLIDER_CONFIG[slider_key]
            value = app.get_slider_value(slider_key) if visible else default
            app.configure_slider(
                slider_key,
                label=label,
                minimum=minimum,
                maximum=maximum,
                value=value,
                visible=visible,
            )

    def render(self, payload: dict[str, Any]) -> Mapping[str, Any]:
        frame = payload
        set_line_3d(self.wave_line, frame["wave_line"])
        set_line_3d(self.envelope_up, frame["envelope_up"])
        set_line_3d(self.envelope_down, frame["envelope_down"])
        set_line_3d(self.axis_line, frame["axis_line"])
        set_line_3d(self.k_line, frame["wave_vector_line"])
        set_marker(self.track_marker, frame["track_marker"])
        self.update_planes(self.plane_artists, frame.get("planes", []))
        apply_limits(self.ax, frame["axis_limits"])
        return _panel(frame)


def _panel(frame: Mapping[str, Any]) -> Mapping[str, Any]:
    panel = frame.get("panel", {})
    return panel if isinstance(panel, Mapping) else {}
