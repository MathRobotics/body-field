from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SDFQueryResult:
    distance: np.ndarray
    normal: np.ndarray
    nearest: np.ndarray


class GaussianProcessDistanceField:
    def __init__(
        self,
        samples: np.ndarray,
        *,
        a: float = 400.0,
        noise: float = 1.0e-8,
        batch_size: int = 4096,
    ) -> None:
        if a <= 0.0:
            raise ValueError("a must be positive")
        if noise < 0.0:
            raise ValueError("noise must be non-negative")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        samples_array = _as_points(samples).astype(np.float64, copy=False)
        self.samples = samples_array
        self.a = float(a)
        self.noise = float(noise)
        self.batch_size = int(batch_size)

        kernel = _se_kernel(samples_array, samples_array, self.a)
        if self.noise > 0.0:
            kernel = kernel + self.noise * np.eye(samples_array.shape[0])
        y = np.ones(samples_array.shape[0], dtype=np.float64)
        self.alpha = np.linalg.solve(kernel, y)

    def query(self, points: np.ndarray) -> SDFQueryResult:
        query_points = _as_points(points).astype(np.float64, copy=False)
        point_count = query_points.shape[0]
        distances = np.empty(point_count, dtype=np.float32)
        normals = np.empty((point_count, 3), dtype=np.float32)
        nearest = np.empty((point_count, 3), dtype=np.float32)

        for start in range(0, point_count, self.batch_size):
            end = min(start + self.batch_size, point_count)
            batch_points = query_points[start:end]
            f_value, f_grad = self._predict_value_grad(batch_points)
            batch_distances, batch_normals = self._distance_and_normal(f_value, f_grad)
            distances[start:end] = batch_distances
            normals[start:end] = batch_normals
            nearest[start:end] = batch_points.astype(np.float32) - (
                batch_normals * batch_distances[:, None]
            )

        return SDFQueryResult(
            distance=distances,
            normal=normals,
            nearest=nearest,
        )

    def _predict_value_grad(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        diff = self.samples[None, :, :] - points[:, None, :]
        r2 = np.sum(diff * diff, axis=2)
        kernel = np.exp(-0.5 * self.a * r2)
        value = kernel @ self.alpha

        weighted = kernel * self.alpha[None, :]
        grad_value = self.a * np.einsum("mnd,mn->md", diff, weighted)
        return value, grad_value

    def _distance_and_normal(
        self,
        value: np.ndarray,
        grad_value: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        safe_value = np.clip(value, np.finfo(np.float64).tiny, 1.0)
        distances = np.sqrt(np.maximum(0.0, -2.0 * np.log(safe_value) / self.a))

        normals = -grad_value
        normal_lengths = np.linalg.norm(normals, axis=1)
        safe_lengths = np.where(normal_lengths > 1.0e-12, normal_lengths, 1.0)
        normals = normals / safe_lengths[:, None]
        normals[normal_lengths <= 1.0e-12] = 0.0
        return distances.astype(np.float32), normals.astype(np.float32)


def _as_points(points: np.ndarray) -> np.ndarray:
    points_array = np.asarray(points, dtype=np.float32)
    if points_array.ndim == 1:
        if points_array.shape[0] != 3:
            raise ValueError("points must have shape (N, 3) or (3,)")
        points_array = points_array[None, :]
    if points_array.ndim != 2 or points_array.shape[1] != 3:
        raise ValueError("points must have shape (N, 3) or (3,)")
    return points_array


def _se_kernel(X1: np.ndarray, X2: np.ndarray, a: float) -> np.ndarray:
    x1_sq = np.sum(X1 * X1, axis=1, keepdims=True)
    x2_sq = np.sum(X2 * X2, axis=1, keepdims=True).T
    r2 = np.maximum(0.0, x1_sq + x2_sq - 2.0 * (X1 @ X2.T))
    return np.exp(-0.5 * a * r2)
