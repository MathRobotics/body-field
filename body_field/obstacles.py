from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .core import Vector3


class DistanceObstacle(Protocol):
    name: str

    def signed_distances(self, points: np.ndarray) -> np.ndarray:
        ...

    def closest_points(self, points: np.ndarray) -> np.ndarray:
        ...


@dataclass(frozen=True)
class SphereObstacle:
    name: str
    center: Vector3
    radius: float

    def __post_init__(self) -> None:
        if self.radius < 0.0:
            raise ValueError("SphereObstacle.radius must be non-negative")

    def signed_distances(self, points: np.ndarray) -> np.ndarray:
        center = np.asarray(self.center, dtype=np.float32)
        return np.linalg.norm(points - center, axis=1) - float(self.radius)

    def closest_points(self, points: np.ndarray) -> np.ndarray:
        center = np.asarray(self.center, dtype=np.float32)
        offsets = points - center
        norms = np.linalg.norm(offsets, axis=1)
        safe_norms = np.where(norms > 0.0, norms, 1.0)
        directions = offsets / safe_norms[:, None]
        if self.radius == 0.0:
            return np.repeat(center[None, :], points.shape[0], axis=0)

        fallback = np.zeros_like(directions)
        fallback[:, 0] = 1.0
        directions = np.where((norms > 0.0)[:, None], directions, fallback)
        return center + directions * float(self.radius)


@dataclass(frozen=True)
class AxisAlignedBoxObstacle:
    name: str
    min_corner: Vector3
    max_corner: Vector3

    def __post_init__(self) -> None:
        min_corner = np.asarray(self.min_corner, dtype=float)
        max_corner = np.asarray(self.max_corner, dtype=float)
        if np.any(min_corner > max_corner):
            raise ValueError("AxisAlignedBoxObstacle min_corner must be <= max_corner")

    def signed_distances(self, points: np.ndarray) -> np.ndarray:
        min_corner = np.asarray(self.min_corner, dtype=np.float32)
        max_corner = np.asarray(self.max_corner, dtype=np.float32)
        center = (min_corner + max_corner) * 0.5
        half_size = (max_corner - min_corner) * 0.5
        q = np.abs(points - center) - half_size
        outside = np.maximum(q, 0.0)
        outside_distance = np.linalg.norm(outside, axis=1)
        inside_distance = np.minimum(np.max(q, axis=1), 0.0)
        return outside_distance + inside_distance

    def closest_points(self, points: np.ndarray) -> np.ndarray:
        min_corner = np.asarray(self.min_corner, dtype=np.float32)
        max_corner = np.asarray(self.max_corner, dtype=np.float32)
        closest = np.minimum(np.maximum(points, min_corner), max_corner)

        inside = np.all((points >= min_corner) & (points <= max_corner), axis=1)
        if not np.any(inside):
            return closest

        inside_points = points[inside]
        distances_to_min = inside_points - min_corner
        distances_to_max = max_corner - inside_points
        face_distances = np.concatenate([distances_to_min, distances_to_max], axis=1)
        face_ids = np.argmin(face_distances, axis=1)
        closest_inside = inside_points.copy()
        for row, face_id in enumerate(face_ids):
            axis = int(face_id % 3)
            if face_id < 3:
                closest_inside[row, axis] = min_corner[axis]
            else:
                closest_inside[row, axis] = max_corner[axis]
        closest[inside] = closest_inside
        return closest
