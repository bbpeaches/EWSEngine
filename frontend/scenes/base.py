from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D

from core.types import RadioSpec, SliderSpec
from frontend.client import SimulationClient
from frontend.plot_tools import set_plane_collection, set_vector_field_lines, set_vector_line


class BaseSimulationScene(ABC):
    """Lightweight scene contract shared by Tk windows and headless smoke tests."""

    module_key = ""
    title = ""
    slider_specs: tuple[SliderSpec, ...] = tuple()
    radio_specs: tuple[RadioSpec, ...] = tuple()
    presets: Mapping[str, Mapping[str, Any]] = {}
    default_elev = 24.0
    default_azim = -58.0
    axis_labels = ("X", "Y", "Z")

    def __init__(self, ax: Axes3D | None = None, figure: Figure | None = None) -> None:
        self.fig = figure or plt.figure(figsize=(10.5, 7.8))
        self.ax: Axes3D = ax or self.fig.add_subplot(111, projection="3d")
        self._standalone_client = SimulationClient()
        self._configure_axes()
        self.init_artists()
        if ax is None:
            self.render(self._fetch_default_frame())

    def _configure_axes(self) -> None:
        self.fig.set_facecolor("#f7f8fb")
        self.fig.suptitle(self.title, fontsize=16, fontweight="bold", y=0.97)
        self.ax.set_facecolor("#f5f7fb")
        self.ax.set_xlabel(self.axis_labels[0])
        self.ax.set_ylabel(self.axis_labels[1])
        self.ax.set_zlabel(self.axis_labels[2])
        self.reset_view()

    def reset_view(self) -> None:
        self.ax.view_init(elev=self.default_elev, azim=self.default_azim)

    def show(self) -> None:
        plt.show()

    def on_mount(self, app: Any) -> None:
        return

    def on_control_changed(self, key: str, value: str, app: Any) -> None:
        return

    def prepare_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return payload

    def default_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {spec.key: spec.value for spec in self.slider_specs}
        payload.update({spec.key: spec.value for spec in self.radio_specs})
        payload["time"] = 0.0
        payload["zoom"] = 1.0
        return self.prepare_payload(payload)

    def _fetch_default_frame(self) -> dict[str, Any]:
        return self._standalone_client.fetch_frame(self.module_key, self.default_payload(), "local")

    def create_plane_artists(self, count: int) -> list[Poly3DCollection]:
        artists: list[Poly3DCollection] = []
        for _ in range(count):
            artist = Poly3DCollection([np.zeros((4, 3), dtype=float)], alpha=0.0, linewidths=0.35)
            artist.set_visible(False)
            self.ax.add_collection3d(artist)
            artists.append(artist)
        return artists

    def update_planes(self, artists: Sequence[Poly3DCollection], planes: Sequence[Mapping[str, Any]]) -> None:
        for index, artist in enumerate(artists):
            plane = planes[index] if index < len(planes) else None
            set_plane_collection(artist, plane)

    def create_vector_line_artists(self, count: int, *, marker_size: float = 4.0) -> list[object]:
        artists: list[object] = []
        for _ in range(count):
            line, = self.ax.plot(
                [],
                [],
                [],
                color="black",
                lw=2.4,
                alpha=0.95,
                marker="o",
                markevery=[-1],
                markersize=marker_size,
            )
            line.set_visible(False)
            artists.append(line)
        return artists

    def update_vectors(self, artists: Sequence[object], vectors: Sequence[Mapping[str, Any]]) -> None:
        for index, artist in enumerate(artists):
            vector = vectors[index] if index < len(vectors) else None
            set_vector_line(artist, vector)

    def create_vector_field_artists(self, count: int, *, marker_size: float = 2.5) -> list[object]:
        return self.create_vector_line_artists(count, marker_size=marker_size)

    def update_vector_field(self, artists: Sequence[object], field: Mapping[str, Any]) -> None:
        set_vector_field_lines(artists, field)

    @abstractmethod
    def init_artists(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def render(self, payload: dict[str, Any]) -> Mapping[str, Any]:
        raise NotImplementedError
