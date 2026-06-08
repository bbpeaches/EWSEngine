from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

_MISSING = object()


def set_line_3d(line: object, data: Mapping[str, Any] | object) -> None:
    line.set_data_3d(
        _as_float_array(_get_value(data, "x")),
        _as_float_array(_get_value(data, "y")),
        _as_float_array(_get_value(data, "z")),
    )


def clear_line(line: object) -> None:
    empty = np.empty(0, dtype=float)
    line.set_data_3d(empty, empty, empty)


def set_marker(line: object, marker: Mapping[str, Any] | object) -> None:
    point = _get_value(marker, "point")
    if point is None:
        clear_line(line)
        return
    coords = _as_point(point)
    line.set_data_3d(coords[0:1], coords[1:2], coords[2:3])


def apply_limits(ax: object, limits: Mapping[str, Any] | object) -> None:
    x_bounds = _as_point_like(_get_value(limits, "x"))
    y_bounds = _as_point_like(_get_value(limits, "y"))
    z_bounds = _as_point_like(_get_value(limits, "z"))
    ax.set_xlim(float(x_bounds[0]), float(x_bounds[1]))
    ax.set_ylim(float(y_bounds[0]), float(y_bounds[1]))
    ax.set_zlim(float(z_bounds[0]), float(z_bounds[1]))


def draw_planes(ax: object, planes: Iterable[Mapping[str, Any] | object]) -> list[object]:
    artists: list[object] = []
    for plane in planes:
        artist = Poly3DCollection(
            [_plane_vertices(plane)],
            facecolors=_get_value(plane, "color"),
            edgecolors=_get_value(plane, "edge_color", _get_value(plane, "color"))
            or _get_value(plane, "color"),
            linewidths=0.35,
            alpha=float(_get_value(plane, "alpha")),
        )
        ax.add_collection3d(artist)
        artists.append(artist)
    return artists


def set_plane_collection(artist: Poly3DCollection, plane: Mapping[str, Any] | object | None) -> None:
    if plane is None:
        artist.set_visible(False)
        artist.set_alpha(0.0)
        return
    artist.set_verts([_plane_vertices(plane)])
    color = _get_value(plane, "color")
    edge_color = _get_value(plane, "edge_color", color) or color
    artist.set_facecolor(color)
    artist.set_edgecolor(edge_color)
    artist.set_alpha(float(_get_value(plane, "alpha")))
    artist.set_visible(True)


def draw_vectors(ax: object, vectors: Iterable[Mapping[str, Any] | object]) -> list[object]:
    artists: list[object] = []
    for vector in vectors:
        start = _as_point(_get_value(vector, "start"))
        end = _as_point(_get_value(vector, "end"))
        delta = end - start
        artist = ax.quiver(
            float(start[0]),
            float(start[1]),
            float(start[2]),
            float(delta[0]),
            float(delta[1]),
            float(delta[2]),
            color=_get_value(vector, "color"),
            linewidth=float(_get_value(vector, "width", 2.4)),
            alpha=float(_get_value(vector, "alpha", 0.95)),
            arrow_length_ratio=0.14,
        )
        artists.append(artist)
    return artists


def set_vector_line(line: object, vector: Mapping[str, Any] | object | None) -> None:
    if vector is None:
        clear_line(line)
        line.set_visible(False)
        return
    start = _as_point(_get_value(vector, "start"))
    end = _as_point(_get_value(vector, "end"))
    line.set_data_3d(
        np.asarray([start[0], end[0]], dtype=float),
        np.asarray([start[1], end[1]], dtype=float),
        np.asarray([start[2], end[2]], dtype=float),
    )
    line.set_color(_get_value(vector, "color"))
    line.set_linewidth(float(_get_value(vector, "width", 2.4)))
    line.set_alpha(float(_get_value(vector, "alpha", 0.95)))
    line.set_visible(True)


def draw_vector_field(ax: object, field: Mapping[str, Any] | object) -> object:
    return ax.quiver(
        _as_float_array(_get_value(field, "x")),
        _as_float_array(_get_value(field, "y")),
        _as_float_array(_get_value(field, "z")),
        _as_float_array(_get_value(field, "u")),
        _as_float_array(_get_value(field, "v")),
        _as_float_array(_get_value(field, "w")),
        color=_get_value(field, "color"),
        alpha=float(_get_value(field, "alpha", 0.5)),
        arrow_length_ratio=0.16,
        linewidth=float(_get_value(field, "linewidth", 1.1)),
    )


def set_vector_field_lines(lines: Sequence[object], field: Mapping[str, Any] | object) -> None:
    starts = np.stack(
        [
            _as_float_array(_get_value(field, "x")).reshape(-1),
            _as_float_array(_get_value(field, "y")).reshape(-1),
            _as_float_array(_get_value(field, "z")).reshape(-1),
        ],
        axis=1,
    )
    deltas = np.stack(
        [
            _as_float_array(_get_value(field, "u")).reshape(-1),
            _as_float_array(_get_value(field, "v")).reshape(-1),
            _as_float_array(_get_value(field, "w")).reshape(-1),
        ],
        axis=1,
    )
    color = _get_value(field, "color")
    alpha = float(_get_value(field, "alpha", 0.5))
    linewidth = float(_get_value(field, "linewidth", 1.1))
    count = min(len(lines), len(starts))
    for index, line in enumerate(lines):
        if index >= count:
            clear_line(line)
            line.set_visible(False)
            continue
        start = starts[index]
        end = start + deltas[index]
        line.set_data_3d(
            np.asarray([start[0], end[0]], dtype=float),
            np.asarray([start[1], end[1]], dtype=float),
            np.asarray([start[2], end[2]], dtype=float),
        )
        line.set_color(color)
        line.set_alpha(alpha)
        line.set_linewidth(linewidth)
        line.set_visible(True)


def _plane_vertices(plane: Mapping[str, Any] | object) -> np.ndarray:
    center = _as_point(_get_value(plane, "center"))
    basis_u = _as_point(_get_value(plane, "basis_u"))
    basis_v = _as_point(_get_value(plane, "basis_v"))
    half_u = float(_get_value(plane, "half_u"))
    half_v = float(_get_value(plane, "half_v"))
    s = np.asarray([-half_u, half_u, half_u, -half_u], dtype=float)
    t = np.asarray([-half_v, -half_v, half_v, half_v], dtype=float)
    return center[None, :] + s[:, None] * basis_u[None, :] + t[:, None] * basis_v[None, :]


def _get_value(data: Mapping[str, Any] | object, key: str, default: Any = _MISSING) -> Any:
    if isinstance(data, Mapping):
        if default is _MISSING:
            return data[key]
        return data.get(key, default)
    if default is _MISSING:
        return getattr(data, key)
    return getattr(data, key, default)


def _as_float_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float)


def _as_point(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(3)


def _as_point_like(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(2)
