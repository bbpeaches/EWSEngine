from __future__ import annotations

from typing import Iterable

import numpy as np

from backend.models import AxisLimits, LineData, MarkerData, PlaneData, Point3, VectorData, empty_line
from core.exceptions import ValidationError


def as_array(values: Iterable[float] | np.ndarray) -> np.ndarray:
    return np.asarray(tuple(values) if not isinstance(values, np.ndarray) else values, dtype=float)


def point(values: Iterable[float]) -> Point3:
    coords = tuple(float(value) for value in values)
    if len(coords) != 3:
        raise ValidationError(f"Expected 3 coordinates, got {len(coords)}.")
    return coords  # type: ignore[return-value]


def line_from_points(points: np.ndarray) -> LineData:
    if points.size == 0:
        return empty_line()
    points = np.asarray(points, dtype=float)
    return LineData(points[:, 0], points[:, 1], points[:, 2])


def line_from_components(x: Iterable[float], y: Iterable[float], z: Iterable[float]) -> LineData:
    return LineData(as_array(x), as_array(y), as_array(z))


def marker_from_point(values: Point3 | None) -> MarkerData:
    return MarkerData(values)


def segment(start: Point3, end: Point3) -> LineData:
    return line_from_components((start[0], end[0]), (start[1], end[1]), (start[2], end[2]))


def build_upper_arc(theta: float, tangential_hat: np.ndarray, normal_hat: np.ndarray, sign: float, radius: float) -> LineData:
    if theta <= 1e-6:
        return empty_line()
    angles = np.linspace(0.0, theta, 80)
    points = np.array([
        radius * (sign * np.sin(angle) * tangential_hat + np.cos(angle) * normal_hat)
        for angle in angles
    ])
    return line_from_points(points)


def build_lower_arc(theta: float | None, tangential_hat: np.ndarray, normal_hat: np.ndarray, radius: float) -> LineData:
    if theta is None or theta <= 1e-6:
        return empty_line()
    angles = np.linspace(0.0, theta, 80)
    points = np.array([
        radius * (np.sin(angle) * tangential_hat - np.cos(angle) * normal_hat)
        for angle in angles
    ])
    return line_from_points(points)


def normalize(vector: Iterable[float]) -> np.ndarray:
    arr = np.asarray(tuple(vector), dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        raise ValidationError("Zero-length vector cannot be normalized.")
    return arr / norm


def incident_basis(phi_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phi = np.radians(phi_deg)
    tangential_hat = np.array([np.cos(phi), np.sin(phi), 0.0], dtype=float)
    side_hat = np.array([-np.sin(phi), np.cos(phi), 0.0], dtype=float)
    normal_hat = np.array([0.0, 0.0, 1.0], dtype=float)
    return tangential_hat, side_hat, normal_hat


def plane(center: Point3, basis_u: Point3, basis_v: Point3, half_u: float, half_v: float, color: str, alpha: float, edge_color: str | None = None) -> PlaneData:
    return PlaneData(center, basis_u, basis_v, half_u, half_v, color, alpha, edge_color)


def vector(start: Point3, end: Point3, color: str, width: float = 2.4, alpha: float = 0.95) -> VectorData:
    return VectorData(start, end, color, width, alpha)


def limits(x: tuple[float, float], y: tuple[float, float], z: tuple[float, float]) -> AxisLimits:
    return AxisLimits(x=x, y=y, z=z)
