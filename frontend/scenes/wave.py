from __future__ import annotations

from backend.models import WaveInput
from backend.physics.wave import MATERIALS, WaveEngine
from frontend.plot_tools import apply_limits, draw_planes, set_marker, set_line_3d
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
PRESETS = {
    "空气行波": {"mode": "material", "material": DEFAULT_MATERIAL, "freq_mhz": 1000.0},
    "FR4 慢波": {"mode": "material", "material": "FR4 玻纤板 (εr=4.4)", "freq_mhz": 600.0},
    "轻度衰减": {"mode": "lossy", "material": DEFAULT_MATERIAL, "alpha": 0.2, "beta": 4.0},
    "强衰减": {"mode": "lossy", "material": DEFAULT_MATERIAL, "alpha": 0.8, "beta": 7.0},
    "倾斜波矢": {"mode": "planes", "material": DEFAULT_MATERIAL, "theta_deg": 55.0, "phi_deg": 40.0, "spacing": 1.6},
    "近竖直波矢": {"mode": "planes", "material": DEFAULT_MATERIAL, "theta_deg": 20.0, "phi_deg": 140.0, "spacing": 1.2},
}


class WaveScene(BaseSimulationScene):
    def __init__(self) -> None:
        self.engine = WaveEngine()
        self.dynamic_artists: list[object] = []
        super().__init__("基础波动实验台")

    def build_controls(self) -> None:
        self.add_radio_group("preset", "快速预设", tuple(PRESETS.keys()), "空气行波", height=0.18)
        self.add_radio_group("mode", "实验模式", ("material", "lossy", "planes"), "material", height=0.10)
        self.add_radio_group("material", "材料", tuple(MATERIALS.keys()), DEFAULT_MATERIAL, height=0.18)
        self.add_slider("freq_mhz", "频率 MHz", 100.0, 3000.0, 1000.0, "royalblue")
        self.add_slider("alpha", "衰减常数 alpha", 0.0, 1.0, 0.3)
        self.add_slider("beta", "相位常数 beta", 1.0, 10.0, 5.0, "forestgreen")
        self.add_slider("theta_deg", "极角 theta", 0.0, 180.0, 45.0, "royalblue")
        self.add_slider("phi_deg", "方位角 phi", 0.0, 360.0, 45.0, "darkorange")
        self.add_slider("spacing", "面间距 lambda", 0.5, 4.0, 1.5, "mediumpurple")
        self.add_standard_controls()
        self._apply_mode_visibility("material")

    def on_radio_change(self, key: str, value: str) -> None:
        if key == "preset":
            preset = PRESETS[value]
            self.set_radio_value("mode", preset["mode"])
            self.set_radio_value("material", preset["material"])
            for slider_key, slider_value in preset.items():
                if slider_key in self.sliders:
                    self.set_slider_value(slider_key, float(slider_value))
            return
        if key == "mode":
            self._apply_mode_visibility(value)

    def _apply_mode_visibility(self, mode: str) -> None:
        for slider_key, visible in MODE_FIELDS[mode].items():
            label, minimum, maximum, default = SLIDER_CONFIG[slider_key]
            value = self.sliders[slider_key].val if visible else default
            self.configure_slider(
                slider_key,
                label=label,
                minimum=minimum,
                maximum=maximum,
                value=value,
                visible=visible,
            )

    def init_artists(self) -> None:
        self.set_default_view(24.0, -58.0)
        self.ax.set_xlabel("空间 / 坐标")
        self.ax.set_ylabel("显示通道 / 坐标")
        self.ax.set_zlabel("幅度 / 相位面法向")
        self.wave_line, = self.ax.plot([], [], [], color="royalblue", lw=2.5, label="行波")
        self.envelope_up, = self.ax.plot([], [], [], color="gray", linestyle="--", lw=1.6, alpha=0.75, label="衰减包络")
        self.envelope_down, = self.ax.plot([], [], [], color="gray", linestyle="--", lw=1.6, alpha=0.75)
        self.axis_line, = self.ax.plot([], [], [], color="slategray", lw=2.2, alpha=0.9, label="传播轴")
        self.k_line, = self.ax.plot([], [], [], color="firebrick", lw=2.6, label="波矢 k")
        self.track_marker, = self.ax.plot([], [], [], "o", color="firebrick", markersize=8, label="相位追踪点")
        self.ax.legend(loc="upper left", fontsize=9)

    def render(self) -> None:
        for artist in self.dynamic_artists:
            artist.remove()
        self.dynamic_artists.clear()
        frame = self.engine.simulate(
            WaveInput(
                mode=self.radio_values["mode"],
                freq_mhz=self.sliders["freq_mhz"].val,
                material=self.radio_values["material"],
                alpha=self.sliders["alpha"].val,
                beta=self.sliders["beta"].val,
                theta_deg=self.sliders["theta_deg"].val,
                phi_deg=self.sliders["phi_deg"].val,
                spacing=self.sliders["spacing"].val,
                time=self.time,
                zoom=self.zoom,
            )
        )
        set_line_3d(self.wave_line, frame.wave_line)
        set_line_3d(self.envelope_up, frame.envelope_up)
        set_line_3d(self.envelope_down, frame.envelope_down)
        set_line_3d(self.axis_line, frame.axis_line)
        set_line_3d(self.k_line, frame.wave_vector_line)
        set_marker(self.track_marker, frame.track_marker)
        self.dynamic_artists.extend(draw_planes(self.ax, frame.planes))
        apply_limits(self.ax, frame.axis_limits)
        self.hint_text.set_text(frame.panel.hint)
        self.status_text.set_text("\n".join(frame.panel.status_lines))
        self.status_text.set_color(frame.panel.status_color)
        self.metrics_text.set_text("\n".join(frame.panel.metrics_lines))
