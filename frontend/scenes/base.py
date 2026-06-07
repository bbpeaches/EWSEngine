from __future__ import annotations

import platform
from abc import ABC, abstractmethod
from typing import Any, Iterable

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, Slider

from core.exceptions import FrontendError


class BaseSimulationScene(ABC):
    """Shared Matplotlib scene lifecycle for all migrated simulations."""

    def __init__(self, title: str, *, is_3d: bool = True) -> None:
        self.title = title
        self.is_3d = is_3d
        self.time = 0.0
        self.zoom = 1.0
        self.time_scale = 1.0
        self.is_paused = False
        self._suspend_events = False
        self._default_elev = 24.0
        self._default_azim = -58.0
        self._buttons: list[Button] = []

        self.fig = plt.figure(figsize=(15, 9))
        self.fig.set_facecolor("#f7f8fb")
        self._configure_fonts()
        self.fig.suptitle(title, fontsize=18, fontweight="bold", y=0.965)
        plt.subplots_adjust(left=0.06, right=0.74, bottom=0.34, top=0.91)
        self.ax = self.fig.add_subplot(111, projection="3d") if is_3d else self.fig.add_subplot(111)
        self.ax.set_facecolor("#f5f7fb")
        if is_3d:
            self.ax.view_init(elev=self._default_elev, azim=self._default_azim)
        self.status_text = self.fig.text(0.77, 0.60, "", fontsize=10, va="top")
        self.metrics_text = self.fig.text(0.77, 0.35, "", fontsize=10, va="top")
        self.hint_text = self.fig.text(
            0.77,
            0.88,
            "",
            fontsize=10,
            va="top",
            bbox=dict(facecolor="whitesmoke", edgecolor="lightgray", boxstyle="round,pad=0.4"),
        )
        self._slider_offset = 0.29
        self._slider_step = 0.045
        self._radio_offset = 0.83
        self.sliders: dict[str, Slider] = {}
        self.slider_meta: dict[str, tuple[float, float]] = {}
        self.radios: dict[str, RadioButtons] = {}
        self.radio_values: dict[str, str] = {}
        self.build_controls()
        self.init_artists()
        self.render()
        self._bind_events()
        self._animation = animation.FuncAnimation(
            self.fig,
            self._tick,
            interval=40,
            blit=False,
            cache_frame_data=False,
        )

    def _configure_fonts(self) -> None:
        if platform.system() == "Darwin":
            plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
        else:
            plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["toolbar"] = "None"

    def _bind_events(self) -> None:
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def set_default_view(self, elev: float, azim: float) -> None:
        self._default_elev = elev
        self._default_azim = azim
        if self.is_3d:
            self.ax.view_init(elev=elev, azim=azim)

    def _tick(self, _: int) -> tuple[Any, ...]:
        if not self.is_paused:
            self.time += 0.05 * self.time_scale
        self.render()
        return tuple()

    def _on_scroll(self, event: Any) -> None:
        if event.inaxes is not self.ax:
            return
        factor = 1.08 if event.button == "up" else 1.0 / 1.08
        self.set_zoom(self.zoom * factor)

    def _on_key(self, event: Any) -> None:
        key = (event.key or "").lower()
        if key in (" ", "space"):
            self.toggle_pause()
            return
        if key == "r":
            self.reset_view()
            return
        if key in ("+", "="):
            self.set_zoom(self.zoom * 1.12)
            return
        if key in ("-", "_"):
            self.set_zoom(self.zoom / 1.12)

    def add_slider(
        self,
        key: str,
        label: str,
        minimum: float,
        maximum: float,
        value: float,
        color: str = "slategray",
    ) -> Slider:
        if self._slider_offset < 0.035:
            raise FrontendError("Too many sliders for the current layout.")
        axis = self.fig.add_axes([0.10, self._slider_offset, 0.56, 0.026], facecolor="#eef1f6")
        slider = Slider(axis, label, minimum, maximum, valinit=value, color=color)
        slider.on_changed(lambda _: self._on_slider())
        self.sliders[key] = slider
        self.slider_meta[key] = (minimum, maximum)
        self._slider_offset -= self._slider_step
        return slider

    def set_slider_value(self, key: str, value: float) -> None:
        minimum, maximum = self.slider_meta[key]
        self.configure_slider(
            key,
            label=getattr(self.sliders[key], "label").get_text(),
            minimum=minimum,
            maximum=maximum,
            value=value,
            visible=self.sliders[key].ax.get_visible(),
        )

    def configure_slider(
        self,
        key: str,
        *,
        label: str,
        minimum: float,
        maximum: float,
        value: float,
        visible: bool = True,
    ) -> None:
        slider = self.sliders[key]
        slider.ax.set_visible(visible)
        slider.valmin = minimum
        slider.valmax = maximum
        slider.ax.set_xlim(minimum, maximum)
        getattr(slider, "label").set_text(label)
        self.slider_meta[key] = (minimum, maximum)
        if not visible:
            return
        next_value = float(max(minimum, min(maximum, value)))
        if abs(float(slider.val) - next_value) < 1e-12:
            return
        previous = self._suspend_events
        self._suspend_events = True
        try:
            slider.set_val(next_value)
        finally:
            self._suspend_events = previous

    def add_radio_group(
        self,
        key: str,
        label: str,
        options: Iterable[str],
        value: str,
        *,
        height: float = 0.10,
    ) -> RadioButtons:
        option_list = tuple(options)
        axis = self.fig.add_axes([0.77, self._radio_offset - height, 0.18, height], facecolor="#eef1f6")
        radio = RadioButtons(axis, option_list, active=option_list.index(value))
        axis.set_title(label, fontsize=11)
        radio.on_clicked(lambda selected, radio_key=key: self._on_radio(radio_key, selected))
        self.radios[key] = radio
        self.radio_values[key] = value
        self._radio_offset -= height + 0.03
        return radio

    def set_radio_value(self, key: str, value: str) -> None:
        radio = self.radios[key]
        labels = tuple(label.get_text() for label in radio.labels)
        previous = self._suspend_events
        self._suspend_events = True
        try:
            self.radio_values[key] = value
            radio.set_active(labels.index(value))
        finally:
            self._suspend_events = previous
        self.on_radio_change(key, value)

    def _on_slider(self) -> None:
        if self._suspend_events:
            return
        self.on_controls_changed()
        self.render()
        self.fig.canvas.draw_idle()

    def _on_radio(self, key: str, value: str) -> None:
        if self._suspend_events:
            self.radio_values[key] = value
            return
        self.radio_values[key] = value
        self.on_radio_change(key, value)
        self.on_controls_changed()
        self.render()
        self.fig.canvas.draw_idle()

    def on_radio_change(self, key: str, value: str) -> None:
        return

    def on_controls_changed(self) -> None:
        return

    def add_action_button(self, label: str, bounds: tuple[float, float, float, float], callback: Any) -> Button:
        axis = self.fig.add_axes(bounds)
        button = Button(axis, label)
        button.on_clicked(callback)
        self._buttons.append(button)
        return button

    def add_standard_controls(self) -> None:
        self.pause_button = self.add_action_button("暂停", (0.77, 0.26, 0.08, 0.05), self._toggle_pause)
        self.add_action_button("重置视角", (0.87, 0.26, 0.08, 0.05), lambda _: self.reset_view())
        self.add_action_button("放大", (0.77, 0.19, 0.05, 0.05), lambda _: self.set_zoom(self.zoom * 1.18))
        self.add_action_button("缩小", (0.835, 0.19, 0.05, 0.05), lambda _: self.set_zoom(self.zoom / 1.18))
        self.add_action_button("重置", (0.90, 0.19, 0.05, 0.05), lambda _: self.set_zoom(1.0))

    def add_pause_button(self) -> None:
        self.pause_button = self.add_action_button("暂停", (0.77, 0.26, 0.08, 0.05), self._toggle_pause)

    def toggle_pause(self) -> None:
        self.is_paused = not self.is_paused
        getattr(self.pause_button, "label").set_text("继续" if self.is_paused else "暂停")
        self.render()
        self.fig.canvas.draw_idle()

    def _toggle_pause(self, _: Any) -> None:
        self.toggle_pause()

    def reset_view(self) -> None:
        if self.is_3d:
            self.ax.view_init(elev=self._default_elev, azim=self._default_azim)
        self.render()
        self.fig.canvas.draw_idle()

    def set_zoom(self, zoom: float) -> None:
        self.zoom = float(max(0.6, min(4.0, zoom)))
        self.render()
        self.fig.canvas.draw_idle()

    def show(self) -> None:
        plt.show()

    @abstractmethod
    def build_controls(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def init_artists(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def render(self) -> None:
        raise NotImplementedError
