from __future__ import annotations

from typing import Iterable

import numpy as np

from backend.models import AxisLimits, LineData, MarkerData, PlaneData, VectorData, VectorFieldData


def set_line_3d(line: object, data: LineData) -> None:
    line.set_data_3d(data.x, data.y, data.z)


def clear_line(line: object) -> None:
    line.set_data_3d([], [], [])


def set_marker(line: object, marker: MarkerData) -> None:
    if marker.point is None:
        clear_line(line)
        return
    x, y, z = marker.point
    line.set_data_3d([x], [y], [z])


def apply_limits(ax: object, limits: AxisLimits) -> None:
    ax.set_xlim(*limits.x)
    ax.set_ylim(*limits.y)
    ax.set_zlim(*limits.z)


def draw_planes(ax: object, planes: Iterable[PlaneData]) -> list[object]:
    artists: list[object] = []
    for item in planes:
        basis_u = np.asarray(item.basis_u, dtype=float)
        basis_v = np.asarray(item.basis_v, dtype=float)
        center = np.asarray(item.center, dtype=float)
        s, t = np.meshgrid(np.linspace(-item.half_u, item.half_u, 2), np.linspace(-item.half_v, item.half_v, 2))
        x = center[0] + s * basis_u[0] + t * basis_v[0]
        y = center[1] + s * basis_u[1] + t * basis_v[1]
        z = center[2] + s * basis_u[2] + t * basis_v[2]
        artist = ax.plot_surface(x, y, z, color=item.color, alpha=item.alpha, edgecolor=item.edge_color or item.color, linewidth=0.35, shade=False)
        artists.append(artist)
    return artists


def draw_vectors(ax: object, vectors: Iterable[VectorData]) -> list[object]:
    artists: list[object] = []
    for item in vectors:
        start = np.asarray(item.start, dtype=float)
        end = np.asarray(item.end, dtype=float)
        delta = end - start
        artist = ax.quiver(start[0], start[1], start[2], delta[0], delta[1], delta[2], color=item.color, linewidth=item.width, alpha=item.alpha, arrow_length_ratio=0.14)
        artists.append(artist)
    return artists


def draw_vector_field(ax: object, field: VectorFieldData) -> object:
    return ax.quiver(field.x, field.y, field.z, field.u, field.v, field.w, color=field.color, alpha=field.alpha, arrow_length_ratio=0.16, linewidth=field.linewidth)
