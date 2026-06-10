from __future__ import annotations

import platform
import time
import tkinter as tk
import tomllib
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - required for projection="3d".
from tkinter import ttk

from core.exceptions import FrontendError
from core.types import RadioSpec, SliderSpec
from frontend.client import SimulationClient
from frontend.scenes.optics import OpticsScene

def _package_version() -> str:
    try:
        return version("EWSEngine")
    except PackageNotFoundError:
        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        with pyproject.open("rb") as file:
            return str(tomllib.load(file)["project"]["version"])


APP_VERSION = _package_version()
RENDER_DELAY_MS = 20
ANIMATION_INTERVAL_MS = 90
ANIMATION_TIME_FACTOR = 0.55


class SceneFactory(Protocol):
    def __call__(self, ax: Axes3D | None = None, figure: Figure | None = None) -> Any:
        ...


class ModernAppBase:
    """Shared Tk shell behavior for standalone and launcher-owned scene windows."""

    def _init_modern_app(
        self,
        scene_factory: SceneFactory | None = None,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
    ) -> None:
        self.client = SimulationClient(host=host, port=port)
        self.time = 0.0
        self.zoom = 1.0
        self.zoom_var = tk.StringVar(value=self._format_zoom())
        self.is_paused = False
        self._is_rendering = False
        self._closed = False
        self._render_job: str | None = None
        self._animation_job: str | None = None
        self._suspend_controls = False
        self._last_tick_time = time.monotonic()
        self.slider_vars: dict[str, tk.DoubleVar] = {}
        self.slider_label_vars: dict[str, tk.StringVar] = {}
        self.slider_value_vars: dict[str, tk.StringVar] = {}
        self.slider_widgets: dict[str, ttk.Scale] = {}
        self.slider_specs: dict[str, SliderSpec] = {}
        self.radio_vars: dict[str, tk.StringVar] = {}
        self.radio_specs: dict[str, RadioSpec] = {}
        self.preset_var: tk.StringVar | None = None
        self.view_var: tk.StringVar | None = None
        self.error_var: tk.StringVar

        self.title(f"EWSEngine {APP_VERSION}")
        self.geometry("1280x820")
        self.minsize(900, 520)
        self.configure(bg="#eef1f6")
        self.protocol("WM_DELETE_WINDOW", self.close)

        self._build_shell()
        self.fig = plt.figure(figsize=(9.8, 7.6), dpi=100)
        self.fig.set_facecolor("#f7f8fb")
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_facecolor("#f5f7fb")
        factory = scene_factory or OpticsScene
        self.scene = factory(ax=self.ax, figure=self.fig)
        self.title(f"EWSEngine {APP_VERSION} - {self.scene.title}")
        self._build_view_controls()
        self._build_canvas()
        self._build_controls()
        getattr(self.scene, "on_mount", lambda _app: None)(self)
        self._bind_canvas_events()
        self.render_now()
        self._schedule_animation()

    def _configure_matplotlib(self) -> None:
        if platform.system() == "Darwin":
            plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
        else:
            plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["toolbar"] = "None"

    def _build_shell(self) -> None:
        toolbar = ttk.Frame(self, padding=(12, 8))
        toolbar.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(toolbar, text="Local 桌面模式", foreground="#51606f").pack(side=tk.LEFT, padx=(0, 12))
        self.pause_button = ttk.Button(toolbar, text="暂停", command=self.toggle_pause, width=8)
        self.pause_button.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(toolbar, text="重置视角", command=self.reset_view, width=10).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(toolbar, text="放大", command=lambda: self.set_zoom(self.zoom * 1.18), width=7).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="缩小", command=lambda: self.set_zoom(self.zoom / 1.18), width=7).pack(side=tk.LEFT, padx=6)
        ttk.Button(toolbar, text="重置缩放", command=lambda: self.set_zoom(1.0), width=10).pack(side=tk.LEFT)
        ttk.Label(toolbar, textvariable=self.zoom_var, foreground="#51606f", width=10).pack(side=tk.LEFT, padx=(8, 0))

        body = ttk.Frame(self)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.control_panel = ttk.Frame(body, width=340, padding=(12, 10))
        self.control_panel.pack(side=tk.LEFT, fill=tk.Y)
        self.control_panel.pack_propagate(False)
        self.canvas_area = ttk.Frame(body, padding=(0, 0, 10, 10))
        self.canvas_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def _build_canvas(self) -> None:
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_area)
        widget = self.canvas.get_tk_widget()
        widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _build_view_controls(self) -> None:
        presets = tuple(getattr(self.scene, "view_presets", {}).keys())
        if not presets:
            return
        row = ttk.Frame(self.canvas_area, padding=(8, 6))
        row.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(row, text="投影视角").pack(side=tk.LEFT, padx=(0, 8))
        self.view_var = tk.StringVar(value="默认")
        combo = ttk.Combobox(row, textvariable=self.view_var, values=presets, state="readonly", width=12)
        combo.pack(side=tk.LEFT)
        combo.bind("<<ComboboxSelected>>", lambda _: self.apply_view_preset(self.view_var.get()))

    def _build_controls(self) -> None:
        self.scroll_canvas = tk.Canvas(self.control_panel, highlightthickness=0, bg="#eef1f6")
        scrollbar = ttk.Scrollbar(self.control_panel, orient=tk.VERTICAL, command=self.scroll_canvas.yview)
        self.scroll_frame = ttk.Frame(self.scroll_canvas)
        self.scroll_window = self.scroll_canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.scroll_canvas.configure(yscrollcommand=scrollbar.set)
        self.scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.scroll_frame.bind("<Configure>", self._update_scroll_region)
        self.scroll_canvas.bind("<Configure>", self._resize_scroll_window)
        self.scroll_canvas.bind("<Enter>", self._bind_panel_scroll)
        self.scroll_canvas.bind("<Leave>", self._unbind_panel_scroll)
        self.scroll_frame.bind("<Enter>", self._bind_panel_scroll)
        self.scroll_frame.bind("<Leave>", self._unbind_panel_scroll)
        self.scroll_canvas.bind("<ButtonPress-1>", self._start_panel_drag)
        self.scroll_canvas.bind("<B1-Motion>", self._drag_panel)
        self.scroll_frame.bind("<ButtonPress-1>", self._start_panel_drag)
        self.scroll_frame.bind("<B1-Motion>", self._drag_panel)

        ttk.Label(
            self.scroll_frame,
            text=self.scene.title,
            font=("Microsoft YaHei", 14, "bold"),
            foreground="#243447",
        ).pack(anchor=tk.W, pady=(0, 12))
        self._build_preset_controls()
        self._build_slider_controls(tuple(self.scene.slider_specs))
        self._build_radio_controls(tuple(self.scene.radio_specs))
        self._build_panel_text()
        self._refresh_control_labels()
        self._bind_control_panel_scroll(self.control_panel)

    def _build_preset_controls(self) -> None:
        presets = getattr(self.scene, "presets", {})
        if not presets:
            return
        frame = ttk.LabelFrame(self.scroll_frame, text="常用预设", padding=8)
        frame.pack(fill=tk.X, pady=(0, 12))
        self.preset_var = tk.StringVar(value=next(iter(presets)))
        combo = ttk.Combobox(
            frame,
            textvariable=self.preset_var,
            values=tuple(presets.keys()),
            state="readonly",
        )
        combo.pack(fill=tk.X)
        combo.bind("<<ComboboxSelected>>", lambda _: self.apply_preset(self.preset_var.get()))

    def _build_slider_controls(self, specs: tuple[SliderSpec, ...]) -> None:
        if not specs:
            return
        frame = ttk.LabelFrame(self.scroll_frame, text="参数", padding=8)
        frame.pack(fill=tk.X, pady=(0, 12))
        for spec in specs:
            self.slider_specs[spec.key] = spec
            row = ttk.Frame(frame)
            row.pack(fill=tk.X, pady=(0, 10))
            label_row = ttk.Frame(row)
            label_row.pack(fill=tk.X)
            label_var = tk.StringVar(value=spec.label)
            ttk.Label(label_row, textvariable=label_var).pack(side=tk.LEFT)
            value_var = tk.StringVar(value=self._format_slider_value(spec, spec.value))
            ttk.Label(label_row, textvariable=value_var, width=9, anchor=tk.E).pack(side=tk.RIGHT)
            var = tk.DoubleVar(value=spec.value)
            scale = ttk.Scale(
                row,
                from_=spec.minimum,
                to=spec.maximum,
                variable=var,
                command=lambda raw, key=spec.key: self._on_slider_changed(key, raw),
            )
            scale.pack(fill=tk.X, pady=(4, 0))
            self.slider_vars[spec.key] = var
            self.slider_label_vars[spec.key] = label_var
            self.slider_value_vars[spec.key] = value_var
            self.slider_widgets[spec.key] = scale

    def _build_radio_controls(self, specs: tuple[RadioSpec, ...]) -> None:
        for spec in specs:
            self.radio_specs[spec.key] = spec
            frame = ttk.LabelFrame(self.scroll_frame, text=spec.label, padding=8)
            frame.pack(fill=tk.X, pady=(0, 12))
            var = tk.StringVar(value=spec.value)
            self.radio_vars[spec.key] = var
            for option in spec.options:
                ttk.Radiobutton(
                    frame,
                    text=option,
                    value=option,
                    variable=var,
                    command=lambda key=spec.key: self._on_radio_changed(key),
                ).pack(anchor=tk.W, pady=2)

    def _build_panel_text(self) -> None:
        frame = ttk.LabelFrame(self.scroll_frame, text="状态", padding=8)
        frame.pack(fill=tk.BOTH, expand=True, pady=(0, 12))
        self.hint_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="")
        self.metrics_var = tk.StringVar(value="")
        self.error_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.hint_var, justify=tk.LEFT, wraplength=292).pack(fill=tk.X)
        ttk.Separator(frame).pack(fill=tk.X, pady=8)
        self.status_label = ttk.Label(frame, textvariable=self.status_var, justify=tk.LEFT, wraplength=292)
        self.status_label.pack(fill=tk.X)
        ttk.Separator(frame).pack(fill=tk.X, pady=8)
        ttk.Label(frame, textvariable=self.metrics_var, justify=tk.LEFT, wraplength=292).pack(fill=tk.X)
        self.error_label = ttk.Label(
            frame,
            textvariable=self.error_var,
            justify=tk.LEFT,
            wraplength=292,
            foreground="firebrick",
        )
        self.error_label.pack(fill=tk.X, pady=(8, 0))

    def _bind_canvas_events(self) -> None:
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _update_scroll_region(self, _: tk.Event[Any]) -> None:
        self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all"))

    def _resize_scroll_window(self, event: tk.Event[Any]) -> None:
        self.scroll_canvas.itemconfigure(self.scroll_window, width=event.width)

    def _bind_panel_scroll(self, _: tk.Event[Any]) -> None:
        self.bind_all("<MouseWheel>", self._on_panel_mousewheel)
        self.bind_all("<Button-4>", self._on_panel_mousewheel)
        self.bind_all("<Button-5>", self._on_panel_mousewheel)

    def _unbind_panel_scroll(self, _: tk.Event[Any]) -> None:
        self.unbind_all("<MouseWheel>")
        self.unbind_all("<Button-4>")
        self.unbind_all("<Button-5>")

    def _on_panel_mousewheel(self, event: tk.Event[Any]) -> str:
        if getattr(event, "num", None) == 4: 
            units = -3
        elif getattr(event, "num", None) == 5:
            units = 3
        else:
            if platform.system() == "Darwin":
                units = -event.delta
            else:
                units = int(-event.delta / 120)

        if units:
            self.scroll_canvas.yview_scroll(units, "units")
        return "break"

    def _start_panel_drag(self, event: tk.Event[Any]) -> None:
        x = event.x_root - self.scroll_canvas.winfo_rootx()
        y = event.y_root - self.scroll_canvas.winfo_rooty()
        self.scroll_canvas.scan_mark(x, y)

    def _drag_panel(self, event: tk.Event[Any]) -> None:
        x = event.x_root - self.scroll_canvas.winfo_rootx()
        y = event.y_root - self.scroll_canvas.winfo_rooty()
        self.scroll_canvas.scan_dragto(x, y, gain=1)

    def _bind_control_panel_scroll(self, widget: tk.Misc) -> None:
        widget.bind("<Enter>", self._bind_panel_scroll, add="+")
        widget.bind("<Leave>", self._unbind_panel_scroll, add="+")
        widget.bind("<MouseWheel>", self._on_panel_mousewheel, add="+")
        widget.bind("<Button-4>", self._on_panel_mousewheel, add="+")
        widget.bind("<Button-5>", self._on_panel_mousewheel, add="+")
        for child in widget.winfo_children():
            self._bind_control_panel_scroll(child)

    def _on_slider_changed(self, key: str, raw_value: str) -> None:
        spec = self.slider_specs[key]
        value = float(raw_value)
        self.slider_value_vars[key].set(self._format_slider_value(spec, value))
        if not self._suspend_controls:
            self._sync_preset_selection_from_controls()
            self._request_render()

    def _on_radio_changed(self, key: str) -> None:
        if self._suspend_controls:
            return
        self.scene.on_control_changed(key, self.radio_vars[key].get(), self)
        self._sync_preset_selection_from_controls()
        self._request_render()

    def apply_preset(self, name: str) -> None:
        presets: Mapping[str, Mapping[str, Any]] = self.scene.presets
        if name not in presets:
            return
        values = presets[name]
        self._suspend_controls = True
        try:
            for key, value in values.items():
                if key in self.radio_vars:
                    self.radio_vars[key].set(str(value))
                    self.scene.on_control_changed(key, str(value), self)
            for key, spec in self.radio_specs.items():
                if key not in values:
                    self.radio_vars[key].set(spec.value)
                    self.scene.on_control_changed(key, spec.value, self)
            for key, value in values.items():
                if key in self.slider_vars:
                    self.set_slider_value(key, float(value))
        finally:
            self._suspend_controls = False
        if self.preset_var is not None:
            self.preset_var.set(name)
        self._request_render()

    def get_slider_value(self, key: str) -> float:
        return float(self.slider_vars[key].get())

    def _sync_preset_selection_from_controls(self) -> None:
        if self.preset_var is None:
            return
        presets = getattr(self.scene, "presets", {})
        for name, values in presets.items():
            if self._controls_match_preset(values):
                self.preset_var.set(name)
                return
        self.preset_var.set("")

    def _controls_match_preset(self, values: Mapping[str, Any]) -> bool:
        for key, value in values.items():
            if key in self.slider_vars:
                if abs(self.get_slider_value(key) - float(value)) > 1e-6:
                    return False
            elif key in self.radio_vars:
                if self.get_radio_value(key) != str(value):
                    return False
        for key, var in self.radio_vars.items():
            if key in values:
                continue
            spec = self.radio_specs[key]
            if var.get() != spec.value:
                return False
        return True

    def set_slider_value(self, key: str, value: float) -> None:
        spec = self.slider_specs[key]
        clamped = float(max(spec.minimum, min(spec.maximum, value)))
        self.slider_vars[key].set(clamped)
        self.slider_value_vars[key].set(self._format_slider_value(spec, clamped))

    def get_radio_value(self, key: str) -> str:
        return self.radio_vars[key].get()

    def set_radio_value(self, key: str, value: str) -> None:
        self.radio_vars[key].set(value)

    def configure_slider(
        self,
        key: str,
        *,
        label: str | None = None,
        minimum: float | None = None,
        maximum: float | None = None,
        value: float | None = None,
        visible: bool = True,
    ) -> None:
        if key not in self.slider_widgets:
            return
        spec = self.slider_specs[key]
        next_spec = SliderSpec(
            key=spec.key,
            label=label or spec.label,
            minimum=spec.minimum if minimum is None else minimum,
            maximum=spec.maximum if maximum is None else maximum,
            value=spec.value if value is None else value,
            step=spec.step,
            color=spec.color,
        )
        self.slider_specs[key] = next_spec
        self.slider_label_vars[key].set(next_spec.label)
        widget = self.slider_widgets[key]
        widget.configure(from_=next_spec.minimum, to=next_spec.maximum)
        if value is not None:
            self.set_slider_value(key, float(value))
        if visible:
            widget.master.pack(fill=tk.X, pady=(0, 10))
        else:
            widget.master.pack_forget()

    def collect_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {key: var.get() for key, var in self.slider_vars.items()}
        payload.update({key: var.get() for key, var in self.radio_vars.items()})
        payload["time"] = self.time
        payload["zoom"] = self.zoom
        return payload

    def render_now(self) -> None:
        if self._closed or getattr(self, "_is_rendering", False):
            return

        self._is_rendering = True
        try:
            self._render_job = None
            payload = self.collect_payload()
            try:
                payload = self.scene.prepare_payload(payload)
                frame = self.client.fetch_frame(self.scene.module_key, payload, "local")
                panel = self.scene.render(frame)
            except FrontendError as exc:
                self.error_var.set(str(exc))
                return
            except Exception as exc:  # noqa: BLE001
                self.error_var.set(f"Render failed: {exc}")
                return

            self.error_var.set("")
            self._update_panel(panel)
            self.canvas.draw_idle()

        finally:
            self._is_rendering = False

    def _request_render(self) -> None:
        if self._closed:
            return
        if self._render_job is not None:
            self.after_cancel(self._render_job)
        self._render_job = self.after(RENDER_DELAY_MS, self.render_now)

    def _schedule_animation(self) -> None:
        if self._closed:
            return
        self._animation_job = self.after(ANIMATION_INTERVAL_MS, self._animation_tick)

    def _animation_tick(self) -> None:
        now = time.monotonic()
        dt = now - self._last_tick_time
        self._last_tick_time = now
        if self._closed:
            return
        if not self.is_paused and not self._is_rendering:
            self.time += dt * ANIMATION_TIME_FACTOR * self._time_scale()
            self._request_render()

        self._schedule_animation()

    def _time_scale(self) -> float:
        if "time_scale" not in self.slider_vars:
            return 1.0
        return float(self.slider_vars["time_scale"].get())

    def _update_panel(self, panel: Mapping[str, Any]) -> None:
        self.hint_var.set(str(panel.get("hint", "")))
        status_lines = panel.get("status_lines", [])
        metrics_lines = panel.get("metrics_lines", [])
        self.status_var.set("\n".join(str(item) for item in status_lines))
        self.metrics_var.set("\n".join(str(item) for item in metrics_lines))
        status_color = str(panel.get("status_color", "black"))
        self.status_label.configure(foreground=status_color)

    def _refresh_control_labels(self) -> None:
        for key, spec in self.slider_specs.items():
            if key in self.slider_label_vars:
                self.slider_label_vars[key].set(spec.label)
            if key in self.slider_vars:
                self.slider_value_vars[key].set(self._format_slider_value(spec, self.get_slider_value(key)))

    def toggle_pause(self) -> None:
        self.is_paused = not self.is_paused
        self.pause_button.configure(text="继续" if self.is_paused else "暂停")

    def reset_view(self) -> None:
        self.scene.reset_view()
        if self.view_var is not None:
            self.view_var.set("默认")
        self._request_render()

    def apply_view_preset(self, name: str) -> None:
        self.scene.apply_view_preset(name)
        self._request_render()

    def set_zoom(self, zoom: float) -> None:
        self.zoom = float(max(0.6, min(4.0, zoom)))
        self.zoom_var.set(self._format_zoom())
        self._request_render()

    def _on_scroll(self, event: Any) -> None:
        if event.inaxes is not self.ax:
            return
        factor = 1.08 if event.button == "up" else 1.0 / 1.08
        self.set_zoom(self.zoom * factor)

    def _on_key(self, event: Any) -> None:
        key = (event.key or "").lower()
        if key in (" ", "space"):
            self.toggle_pause()
        elif key == "r":
            self.reset_view()
        elif key in ("+", "="):
            self.set_zoom(self.zoom * 1.12)
        elif key in ("-", "_"):
            self.set_zoom(self.zoom / 1.12)

    def _format_zoom(self) -> str:
        return f"缩放 {self.zoom:.2f}x"

    def _format_slider_value(self, spec: SliderSpec, value: float) -> str:
        if spec.step is None or spec.step >= 1.0:
            return f"{value:.1f}"
        if spec.step >= 0.1:
            return f"{value:.1f}"
        return f"{value:.2f}"

    def close(self) -> None:
        self._closed = True
        if self._render_job is not None:
            self.after_cancel(self._render_job)
            self._render_job = None
        if self._animation_job is not None:
            self.after_cancel(self._animation_job)
            self._animation_job = None
        plt.close(self.fig)
        self.destroy()


class ModernAppWindow(ModernAppBase, tk.Tk):
    """Standalone Tk root for directly running a modern scene desktop."""

    def __init__(
        self,
        scene_factory: SceneFactory | None = None,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
    ) -> None:
        self._configure_matplotlib()
        tk.Tk.__init__(self)
        self._init_modern_app(scene_factory=scene_factory, host=host, port=port)


class ModernAppToplevel(ModernAppBase, tk.Toplevel):
    """Launcher-owned scene window that leaves the module menu open."""

    def __init__(
        self,
        master: tk.Misc,
        scene_factory: SceneFactory | None = None,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
    ) -> None:
        self._configure_matplotlib()
        tk.Toplevel.__init__(self, master=master)
        self._init_modern_app(scene_factory=scene_factory, host=host, port=port)


def run_modern_app(
    scene_factory: SceneFactory | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    app = ModernAppWindow(scene_factory=scene_factory, host=host, port=port)
    app.mainloop()


if __name__ == "__main__":
    run_modern_app()