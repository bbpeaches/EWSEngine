from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.types import RadioSpec, SliderSpec
from frontend.plot_tools import apply_limits, set_line_3d, set_marker
from frontend.scenes.base import BaseSimulationScene

PRESETS: dict[str, dict[str, float | str]] = {
    "匹配传输": {"mode": "vswr", "reflection_coefficient": 0.0},
    "电压波节": {"mode": "vswr", "reflection_coefficient": -0.65},
    "电压波腹": {"mode": "vswr", "reflection_coefficient": 0.65},
    "行驻混合": {"mode": "standing", "reflection_coefficient": -0.45},
    "强驻波": {"mode": "standing", "reflection_coefficient": -0.90},
}


class TransmissionScene(BaseSimulationScene):
    module_key = "transmission"
    title = "行驻波传播实验台"
    slider_specs = (SliderSpec("reflection_coefficient", "反射系数 R", -0.99, 0.99, 0.0, 0.01),)
    radio_specs = (RadioSpec("mode", "实验模式", ("vswr", "standing"), "vswr"),)
    presets = PRESETS
    default_elev = 24.0
    default_azim = -58.0
    axis_labels = ("x", "显示通道", "归一化幅度")

    def prepare_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload["reflection_coefficient"] = payload.pop("reflection_coefficient", 0.0)
        return payload

    def init_artists(self) -> None:
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

    def render(self, payload: dict[str, Any]) -> Mapping[str, Any]:
        frame = payload
        set_line_3d(self.electric_line, frame["electric_line"])
        set_line_3d(self.magnetic_line, frame["magnetic_line"])
        set_line_3d(self.envelope_line, frame["envelope_line"])
        set_line_3d(self.standing_line, frame["standing_line"])
        set_line_3d(self.traveling_line, frame["traveling_line"])
        set_line_3d(self.axis_line, frame["axis_line"])
        set_line_3d(self.boundary_line, frame["boundary_line"])
        set_marker(self.electric_marker, frame["electric_marker"])
        set_marker(self.magnetic_marker, frame["magnetic_marker"])
        set_marker(self.envelope_marker, frame["envelope_marker"])
        apply_limits(self.ax, frame["axis_limits"])
        return _panel(frame)


def _panel(frame: Mapping[str, Any]) -> Mapping[str, Any]:
    panel = frame.get("panel", {})
    return panel if isinstance(panel, Mapping) else {}
