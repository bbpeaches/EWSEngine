from __future__ import annotations

from backend.models import PolarizationInput
from backend.physics.polarization import PolarizationEngine
from frontend.plot_tools import apply_limits, draw_vector_field, set_line_3d
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

PRESETS = {
    "圆极化": {"mode": "phase", "p1": 1.0, "p2": 1.0, "p3": 90.0},
    "线极化": {"mode": "phase", "p1": 1.0, "p2": 1.0, "p3": 0.0},
    "左旋基底": {"mode": "circular", "p1": 1.0, "p2": 0.0, "p3": 0.0},
    "天线匹配": {"mode": "match", "p1": 30.0, "p2": 30.0, "p3": 1.0},
    "天线失配": {"mode": "match", "p1": 30.0, "p2": 80.0, "p3": 1.0},
}


class PolarizationScene(BaseSimulationScene):
    def __init__(self) -> None:
        self.engine = PolarizationEngine()
        self.dynamic_artists: list[object] = []
        super().__init__("电磁波极化合成实验室")

    def build_controls(self) -> None:
        self.add_radio_group("preset", "快速预设", tuple(PRESETS.keys()), "圆极化", height=0.16)
        self.add_radio_group("mode", "实验模式", ("phase", "circular", "match"), "phase", height=0.12)
        self.add_slider("p1", "Ex 幅度", 0.0, 1.5, 1.0, "royalblue")
        self.add_slider("p2", "Ey 幅度", 0.0, 1.5, 1.0, "forestgreen")
        self.add_slider("p3", "相位差 phase", -180.0, 180.0, 90.0, "darkorange")
        self.add_standard_controls()

    def on_radio_change(self, key: str, value: str) -> None:
        if key == "preset":
            preset = PRESETS[value]
            self.set_radio_value("mode", preset["mode"])
            for slider_key in ("p1", "p2", "p3"):
                self.set_slider_value(slider_key, preset[slider_key])
            return
        if key == "mode":
            self._apply_mode(value)

    def _apply_mode(self, mode: str) -> None:
        for slider_key, (label, minimum, maximum, default) in MODE_CONFIG[mode].items():
            self.configure_slider(slider_key, label=label, minimum=minimum, maximum=maximum, value=default, visible=True)
        if mode == "circular":
            self.configure_slider("p3", label="占位参数", minimum=-180.0, maximum=180.0, value=0.0, visible=False)

    def init_artists(self) -> None:
        self.set_default_view(24.0, -58.0)
        self.ax.set_xlabel("Ex")
        self.ax.set_ylabel("Ey")
        self.ax.set_zlabel("z")
        self.ax.set_box_aspect((1.0, 1.0, 1.8))
        self.wave_line, = self.ax.plot([], [], [], color="firebrick", lw=2.2, label="电场波形")
        self.component_x, = self.ax.plot([], [], [], color="royalblue", lw=2.0, label="Ex 分量")
        self.component_y, = self.ax.plot([], [], [], color="forestgreen", lw=2.0, label="Ey 分量")
        self.total_line, = self.ax.plot([], [], [], color="firebrick", lw=3.0, label="总电场")
        self.projection_line, = self.ax.plot([], [], [], color="darkorange", lw=3.0, label="接收投影")
        self.antenna_line, = self.ax.plot([], [], [], color="black", lw=2.6, label="接收天线")
        self.ax.legend(loc="upper left", fontsize=9)

    def render(self) -> None:
        for artist in self.dynamic_artists:
            artist.remove()
        self.dynamic_artists.clear()
        mode = self.radio_values["mode"]
        frame = self.engine.simulate(
            PolarizationInput(
                mode=mode,
                p1=self.sliders["p1"].val,
                p2=self.sliders["p2"].val,
                p3=self.sliders["p3"].val,
                time=self.time,
                zoom=self.zoom,
            )
        )
        set_line_3d(self.wave_line, frame.wave_line)
        set_line_3d(self.component_x, frame.component_x_line)
        set_line_3d(self.component_y, frame.component_y_line)
        set_line_3d(self.total_line, frame.total_vector_line)
        set_line_3d(self.projection_line, frame.projection_line)
        set_line_3d(self.antenna_line, frame.antenna_line)
        self.wave_line.set_color(frame.color)
        self.total_line.set_color(frame.color)
        apply_limits(self.ax, frame.axis_limits)
        self.dynamic_artists.append(draw_vector_field(self.ax, frame.wave_field))
        self.hint_text.set_text(frame.panel.hint)
        self.status_text.set_text("\n".join(frame.panel.status_lines))
        self.status_text.set_color(frame.panel.status_color)
        self.metrics_text.set_text("\n".join(frame.panel.metrics_lines))
