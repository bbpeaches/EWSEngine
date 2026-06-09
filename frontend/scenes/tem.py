from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from backend.models import INTRINSIC_IMPEDANCE
from core.types import RadioSpec, SliderSpec
from frontend.plot_tools import apply_limits, clear_line, set_line_3d, set_marker, set_scaled_line_3d, set_scaled_vector_field_lines, set_scaled_vector_line
from frontend.scenes.base import BaseSimulationScene

PRESETS: dict[str, dict[str, float | str]] = {
    "X 正向": {"mode": "lossless", "direction": "x", "polarity": "1", "amplitude": 3.0, "wavelength": 5.0, "speed": 2.0},
    "Y 正向": {"mode": "lossless", "direction": "y", "polarity": "1", "amplitude": 3.0, "wavelength": 5.0, "speed": 2.0},
    "Z 正向": {"mode": "lossless", "direction": "z", "polarity": "1", "amplitude": 3.0, "wavelength": 5.0, "speed": 2.0},
    "X 反向": {"mode": "lossless", "direction": "x", "polarity": "-1", "amplitude": 3.0, "wavelength": 5.0, "speed": 2.0},
    "X 有损耗": {"mode": "lossy", "direction": "x", "polarity": "1", "amplitude": 3.0, "speed": 2.0, "alpha": 0.25, "beta": 5.0},
    "Z 有损耗": {"mode": "lossy", "direction": "z", "polarity": "1", "amplitude": 3.0, "speed": 2.0, "alpha": 0.35, "beta": 6.0},
    "强衰减 TEM": {"mode": "lossy", "direction": "z", "polarity": "-1", "amplitude": 3.0, "speed": 2.0, "alpha": 0.75, "beta": 7.0},
}


class TemScene(BaseSimulationScene):
    module_key = "tem"
    title = "TEM 平面波实验台"
    slider_specs = (
        SliderSpec("amplitude", "振幅", 0.5, 6.0, 3.0, 0.1),
        SliderSpec("wavelength", "波长", 1.0, 10.0, 5.0, 0.1),
        SliderSpec("speed", "传播速度", 0.5, 5.0, 2.0, 0.1),
        SliderSpec("alpha", "衰减常数 alpha", 0.0, 1.2, 0.2, 0.01),
        SliderSpec("beta", "相位常数 beta", 1.0, 10.0, 5.0, 0.05),
        SliderSpec("time_scale", "动画速度", 0.2, 3.0, 1.0, 0.1),
    )
    radio_specs = (
        RadioSpec("mode", "TEM 模式", ("lossless", "lossy"), "lossless"),
        RadioSpec("direction", "传播轴", ("x", "y", "z"), "x"),
        RadioSpec("polarity", "相位方向", ("1", "-1"), "1"),
        RadioSpec("h_display", "磁场显示", ("隐藏", "H", "377H"), "377H"),
    )
    presets = PRESETS
    default_elev = 22.0
    default_azim = -58.0
    axis_labels = ("X", "Y", "Z")

    def prepare_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload["polarity"] = float(payload["polarity"])
        return payload

    def on_mount(self, app: Any) -> None:
        self._apply_mode_visibility(app.get_radio_value("mode"), app)

    def on_control_changed(self, key: str, value: str, app: Any) -> None:
        if key == "mode":
            self._apply_mode_visibility(value, app)

    def _apply_mode_visibility(self, mode: str, app: Any) -> None:
        app.configure_slider(
            "wavelength",
            label="波长",
            minimum=1.0,
            maximum=10.0,
            value=app.get_slider_value("wavelength"),
            visible=mode == "lossless",
        )
        app.configure_slider(
            "speed",
            label="传播速度",
            minimum=0.5,
            maximum=5.0,
            value=app.get_slider_value("speed"),
            visible=True,
        )
        app.configure_slider(
            "alpha",
            label="衰减常数 alpha",
            minimum=0.0,
            maximum=1.2,
            value=app.get_slider_value("alpha"),
            visible=mode == "lossy",
        )
        app.configure_slider(
            "beta",
            label="相位常数 beta",
            minimum=1.0,
            maximum=10.0,
            value=app.get_slider_value("beta"),
            visible=mode == "lossy",
        )

    def init_artists(self) -> None:
        self.electric_line, = self.ax.plot([], [], [], color="firebrick", lw=2.2, label="电场 E")
        self.magnetic_line, = self.ax.plot([], [], [], color="royalblue", lw=2.2, label="磁场 H / 377H")
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
        self._render_magnetic(frame)
        set_line_3d(self.axis_line, frame["propagation_axis"])
        set_line_3d(self.ref_x, frame["reference_x"])
        set_line_3d(self.ref_y, frame["reference_y"])
        set_line_3d(self.ref_z, frame["reference_z"])
        set_marker(self.peak_marker, frame["peak_marker_3d"])
        self.update_vector_field(self.electric_field_lines, frame["electric_field"])
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
            vectors = frame.get("local_vectors", [])
            self.update_vectors(self.local_vector_lines[0:1], vectors[0:1])
            clear_line(self.local_vector_lines[1])
            self.local_vector_lines[1].set_visible(False)
            self.update_vectors(self.local_vector_lines[2:], vectors[2:])
            return
        scale = INTRINSIC_IMPEDANCE if h_display == "377H" else 1.0
        config = {"x": "z", "y": "x", "z": "y"}
        h_axis = config.get(str(frame.get("direction", "x")), "z")
        set_scaled_line_3d(self.magnetic_line, frame["magnetic_line"], (h_axis,), scale)
        set_scaled_vector_field_lines(self.magnetic_field_lines, frame["magnetic_field"], (h_axis,), scale)
        vectors = frame.get("local_vectors", [])
        if len(vectors) >= 3:
            self.update_vectors(self.local_vector_lines[0:1], vectors[0:1])
            set_scaled_vector_line(self.local_vector_lines[1], vectors[1], (h_axis,), scale)
            self.update_vectors(self.local_vector_lines[2:], vectors[2:])
        else:
            self.update_vectors(self.local_vector_lines, vectors)


def _panel(frame: Mapping[str, Any]) -> Mapping[str, Any]:
    panel = frame.get("panel", {})
    return panel if isinstance(panel, Mapping) else {}
