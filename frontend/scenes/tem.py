from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.types import RadioSpec, SliderSpec
from frontend.plot_tools import apply_limits, set_line_3d, set_marker
from frontend.scenes.base import BaseSimulationScene

PRESETS: dict[str, dict[str, float | str]] = {
    "X 正向": {"direction": "x", "polarity": "1", "amplitude": 3.0, "wavelength": 5.0, "speed": 2.0},
    "Y 正向": {"direction": "y", "polarity": "1", "amplitude": 3.0, "wavelength": 5.0, "speed": 2.0},
    "Z 正向": {"direction": "z", "polarity": "1", "amplitude": 3.0, "wavelength": 5.0, "speed": 2.0},
    "X 反向": {"direction": "x", "polarity": "-1", "amplitude": 3.0, "wavelength": 5.0, "speed": 2.0},
}


class TemScene(BaseSimulationScene):
    module_key = "tem"
    title = "TEM 平面波实验台"
    slider_specs = (
        SliderSpec("amplitude", "振幅", 0.5, 6.0, 3.0, 0.1),
        SliderSpec("wavelength", "波长", 1.0, 10.0, 5.0, 0.1),
        SliderSpec("speed", "传播速度", 0.5, 5.0, 2.0, 0.1),
        SliderSpec("time_scale", "动画速度", 0.2, 3.0, 1.0, 0.1),
    )
    radio_specs = (
        RadioSpec("direction", "传播轴", ("x", "y", "z"), "x"),
        RadioSpec("polarity", "相位方向", ("1", "-1"), "1"),
    )
    presets = PRESETS
    default_elev = 22.0
    default_azim = -58.0
    axis_labels = ("X", "Y", "Z")

    def prepare_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload["polarity"] = float(payload["polarity"])
        return payload

    def init_artists(self) -> None:
        self.electric_line, = self.ax.plot([], [], [], color="firebrick", lw=2.2, label="电场 E")
        self.magnetic_line, = self.ax.plot([], [], [], color="royalblue", lw=2.2, label="磁场 H")
        self.axis_line, = self.ax.plot([], [], [], color="goldenrod", lw=2.5, alpha=0.9, label="传播轴")
        self.ref_x, = self.ax.plot([], [], [], color="gray", linestyle="--", lw=1.0, alpha=0.22)
        self.ref_y, = self.ax.plot([], [], [], color="gray", linestyle="--", lw=1.0, alpha=0.22)
        self.ref_z, = self.ax.plot([], [], [], color="gray", linestyle="--", lw=1.0, alpha=0.22)
        self.peak_marker, = self.ax.plot([], [], [], "o", color="black", markersize=7, label="相位追踪点")
        self.electric_field_lines = self.create_vector_field_artists(24)
        self.magnetic_field_lines = self.create_vector_field_artists(24)
        self.local_vector_lines = self.create_vector_line_artists(3)
        self.ax.legend(loc="upper left", fontsize=9)

    def render(self, payload: dict[str, Any]) -> Mapping[str, Any]:
        frame = payload
        set_line_3d(self.electric_line, frame["electric_line"])
        set_line_3d(self.magnetic_line, frame["magnetic_line"])
        set_line_3d(self.axis_line, frame["propagation_axis"])
        set_line_3d(self.ref_x, frame["reference_x"])
        set_line_3d(self.ref_y, frame["reference_y"])
        set_line_3d(self.ref_z, frame["reference_z"])
        set_marker(self.peak_marker, frame["peak_marker_3d"])
        self.update_vector_field(self.electric_field_lines, frame["electric_field"])
        self.update_vector_field(self.magnetic_field_lines, frame["magnetic_field"])
        self.update_vectors(self.local_vector_lines, frame.get("local_vectors", []))
        apply_limits(self.ax, frame["axis_limits"])
        return _panel(frame)


def _panel(frame: Mapping[str, Any]) -> Mapping[str, Any]:
    panel = frame.get("panel", {})
    return panel if isinstance(panel, Mapping) else {}
