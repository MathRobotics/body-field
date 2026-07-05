from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from body_field.core import (
    Backend,
    LinkStateProvider,
    ParallelProfile,
    PreparedBackend,
    QuantitySpec,
    QuantityValue,
    RobotState,
    RobotSurfaceModel,
    SurfacePoint,
)
from body_field.backends._surface_points import pack_surface_point_transforms
from body_field.obstacles import DistanceObstacle


SUPPORTED_OBSTACLE_DISTANCE_QUANTITIES = {
    "geometry.obstacle.distance",
    "geometry.obstacle.closest_point",
    "geometry.obstacle.normal",
    "geometry.obstacle.vector",
}


@dataclass(frozen=True)
class NumpyObstacleDistanceBackend(Backend):
    link_state_provider: LinkStateProvider
    obstacles: Sequence[DistanceObstacle]
    name: str = "obstacle_distance.numpy"

    def supported_quantities(self) -> set[str]:
        return set(SUPPORTED_OBSTACLE_DISTANCE_QUANTITIES)

    def supports(self, model: RobotSurfaceModel, quantity: QuantitySpec) -> bool:
        return quantity.name in self.supported_quantities()

    def prepare(self, model: RobotSurfaceModel) -> PreparedBackend:
        return PreparedNumpyObstacleDistanceBackend(
            model=model,
            link_state_provider=self.link_state_provider,
            obstacles=tuple(self.obstacles),
            backend_name=self.name,
        )

    def parallel_profile(self) -> ParallelProfile:
        return ParallelProfile(
            point_parallel=True,
            quantity_parallel=False,
            device="cpu",
            backend_kind="numpy-obstacle-distance",
            preferred_min_points=1,
        )


@dataclass(frozen=True)
class PreparedNumpyObstacleDistanceBackend:
    model: RobotSurfaceModel
    link_state_provider: LinkStateProvider
    obstacles: tuple[DistanceObstacle, ...]
    backend_name: str

    def evaluate(
        self,
        points: list[SurfacePoint],
        quantities: list[QuantitySpec],
        state: RobotState | None = None,
    ) -> list[QuantityValue]:
        _require_supported_quantities(quantities)
        link_states = self.link_state_provider.compute_link_states(self.model, state)
        positions = pack_surface_point_transforms(
            self.model, points, link_states
        ).world_position
        outputs = _evaluate_obstacle_distances(positions, self.obstacles)
        return _collect_values(points, quantities, outputs, self.backend_name)


@dataclass(frozen=True)
class _ObstacleDistanceOutputs:
    world_position: np.ndarray
    signed_distance: np.ndarray
    closest_point: np.ndarray
    normal: np.ndarray
    closest_obstacle_name: list[str | None]


def _evaluate_obstacle_distances(
    positions: np.ndarray,
    obstacles: tuple[DistanceObstacle, ...],
) -> _ObstacleDistanceOutputs:
    point_count = positions.shape[0]
    best_distance = np.full(point_count, np.inf, dtype=np.float32)
    best_closest_point = np.full((point_count, 3), np.nan, dtype=np.float32)
    best_normal = np.full((point_count, 3), np.nan, dtype=np.float32)
    best_obstacle_name: list[str | None] = [None] * point_count

    for obstacle in obstacles:
        distances, closest_points, normals = _query_obstacle(obstacle, positions)
        update = distances < best_distance
        if not np.any(update):
            continue
        best_distance[update] = distances[update]
        best_closest_point[update] = closest_points[update]
        best_normal[update] = normals[update]
        for index in np.nonzero(update)[0]:
            best_obstacle_name[int(index)] = obstacle.name

    return _ObstacleDistanceOutputs(
        world_position=positions,
        signed_distance=best_distance,
        closest_point=best_closest_point,
        normal=best_normal,
        closest_obstacle_name=best_obstacle_name,
    )


def _collect_values(
    points: list[SurfacePoint],
    quantities: list[QuantitySpec],
    outputs: _ObstacleDistanceOutputs,
    backend_name: str,
) -> list[QuantityValue]:
    values: list[QuantityValue] = []
    for quantity in quantities:
        for index, point in enumerate(points):
            if quantity.name == "geometry.obstacle.distance":
                value = float(outputs.signed_distance[index])
                unit = quantity.unit or "m"
            elif quantity.name == "geometry.obstacle.closest_point":
                value = _tuple3(outputs.closest_point[index])
                unit = quantity.unit or "m"
            elif quantity.name == "geometry.obstacle.normal":
                value = _tuple3(outputs.normal[index])
                unit = quantity.unit
            elif quantity.name == "geometry.obstacle.vector":
                value = _tuple3(outputs.world_position[index] - outputs.closest_point[index])
                unit = quantity.unit or "m"
            else:
                raise ValueError(f"{backend_name} does not support {quantity.name}")

            values.append(
                QuantityValue(
                    spec=quantity,
                    point=point,
                    value=value,
                    frame=quantity.frame or "world",
                    unit=unit,
                    metadata={
                        "backend": backend_name,
                        "world_position": _tuple3(outputs.world_position[index]),
                        "closest_obstacle": outputs.closest_obstacle_name[index],
                        "closest_point": _tuple3(outputs.closest_point[index]),
                        "normal": _tuple3(outputs.normal[index]),
                    },
                )
            )
    return values


def _require_supported_quantities(quantities: list[QuantitySpec]) -> None:
    for quantity in quantities:
        if quantity.frame not in {None, "world"}:
            raise ValueError(f"{quantity.name} supports frame='world' only")
        if quantity.name not in SUPPORTED_OBSTACLE_DISTANCE_QUANTITIES:
            raise ValueError(f"Unsupported obstacle-distance quantity: {quantity.name}")


def _tuple3(array: np.ndarray) -> tuple[float, float, float]:
    return (float(array[0]), float(array[1]), float(array[2]))


def _query_obstacle(
    obstacle: DistanceObstacle,
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query = getattr(obstacle, "query", None)
    if callable(query):
        result = query(positions)
        distances = np.asarray(result.distance, dtype=np.float32)
        closest_points = np.asarray(result.nearest, dtype=np.float32)
        normals = np.asarray(result.normal, dtype=np.float32)
        return distances, closest_points, normals

    distances = obstacle.signed_distances(positions).astype(np.float32, copy=False)
    closest_points = obstacle.closest_points(positions).astype(np.float32, copy=False)
    normals_method = getattr(obstacle, "normals", None)
    if callable(normals_method):
        normals = np.asarray(normals_method(positions), dtype=np.float32)
    else:
        normals = _derive_normals(positions, closest_points, distances)
    return distances, closest_points, normals


def _derive_normals(
    positions: np.ndarray,
    closest_points: np.ndarray,
    distances: np.ndarray,
) -> np.ndarray:
    offsets = positions - closest_points
    lengths = np.linalg.norm(offsets, axis=1)
    normals = np.zeros_like(positions, dtype=np.float32)
    nonzero = lengths > 1.0e-8
    normals[nonzero] = offsets[nonzero] / lengths[nonzero, None]
    signs = np.where(distances < 0.0, -1.0, 1.0).astype(np.float32)
    normals[nonzero] *= signs[nonzero, None]
    return normals
