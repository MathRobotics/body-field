from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from body_field.core import BodyField, QuantitySpec, RobotState, SurfacePoint, Vector3


@dataclass(frozen=True)
class SelfCollisionMinimum:
    signed_distance: float
    point_index: int
    closest_point_index: int
    point: SurfacePoint | None
    closest_point: SurfacePoint | None
    point_link_name: str | None
    closest_link_name: str | None
    world_position: Vector3 | None
    closest_world_position: Vector3 | None
    vector: Vector3 | None
    normal: Vector3 | None
    distance_gradient: tuple[float, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def evaluate_minimum_self_collision(
    field: BodyField,
    points: list[SurfacePoint],
    state: RobotState | None = None,
    *,
    self_collision_backend: str = "self_collision.numpy",
    jacobian_backend: str | None = None,
    point_radius: float | None = None,
) -> SelfCollisionMinimum:
    """Return the active self-collision pair for a robot state.

    The returned signed distance is the scalar objective to minimize when searching
    for colliding joint angles. If a Jacobian backend is provided, the result also
    includes d(distance) / dq for the active pair.
    """

    params: dict[str, Any] = {}
    if point_radius is not None:
        params["point_radius"] = point_radius

    distance_spec = QuantitySpec(
        "geometry.self_collision.distance",
        output_type="scalar",
        frame="world",
        unit="m",
        params=params,
    )
    distance_values = field.evaluate(
        points,
        [distance_spec],
        state,
        backend=self_collision_backend,
    )
    if not distance_values:
        return _empty_minimum()

    distances = np.asarray([float(value.value) for value in distance_values], dtype=float)
    finite = np.isfinite(distances)
    if not np.any(finite):
        return _empty_minimum(signed_distance=float(np.min(distances)))

    active_index = int(np.argmin(distances))
    active_value = distance_values[active_index]
    active_metadata = active_value.metadata
    closest_index = int(active_metadata["closest_point_index"])

    world_position = _metadata_vector(active_metadata, "world_position")
    closest_world_position = _metadata_vector(active_metadata, "closest_point")
    vector = _subtract(world_position, closest_world_position)
    normal = _normalize(vector)
    gradient = None
    if jacobian_backend is not None and normal is not None and closest_index >= 0:
        gradient = _active_distance_gradient(
            field=field,
            points=points,
            state=state,
            active_index=active_index,
            closest_index=closest_index,
            normal=normal,
            jacobian_backend=jacobian_backend,
        )

    return SelfCollisionMinimum(
        signed_distance=float(active_value.value),
        point_index=active_index,
        closest_point_index=closest_index,
        point=points[active_index],
        closest_point=points[closest_index] if closest_index >= 0 else None,
        point_link_name=points[active_index].link_name,
        closest_link_name=active_metadata["closest_link"],
        world_position=world_position,
        closest_world_position=closest_world_position,
        vector=vector,
        normal=normal,
        distance_gradient=gradient,
        metadata={
            "backend": active_metadata.get("backend"),
            "point_radius": point_radius,
        },
    )


def _active_distance_gradient(
    field: BodyField,
    points: list[SurfacePoint],
    state: RobotState | None,
    active_index: int,
    closest_index: int,
    normal: Vector3,
    jacobian_backend: str,
) -> tuple[float, ...]:
    jacobian_spec = QuantitySpec(
        "kinematics.jacobian",
        output_type="matrix",
        frame="world",
        unit="m/rad",
    )
    jacobian_values = field.evaluate(
        [points[active_index], points[closest_index]],
        [jacobian_spec],
        state,
        backend=jacobian_backend,
    )
    if len(jacobian_values) != 2:
        raise ValueError("Jacobian backend must return one Jacobian for each active point")

    active_jacobian = np.asarray(jacobian_values[0].value, dtype=float)
    closest_jacobian = np.asarray(jacobian_values[1].value, dtype=float)
    if active_jacobian.shape != closest_jacobian.shape:
        raise ValueError(
            "Active self-collision Jacobians must have the same shape, got "
            f"{active_jacobian.shape} and {closest_jacobian.shape}"
        )
    if active_jacobian.ndim != 2 or active_jacobian.shape[0] != 3:
        raise ValueError(f"Expected 3xN point Jacobians, got {active_jacobian.shape}")

    normal_array = np.asarray(normal, dtype=float)
    gradient = normal_array @ (active_jacobian - closest_jacobian)
    return tuple(float(value) for value in gradient)


def _empty_minimum(signed_distance: float = float("inf")) -> SelfCollisionMinimum:
    return SelfCollisionMinimum(
        signed_distance=signed_distance,
        point_index=-1,
        closest_point_index=-1,
        point=None,
        closest_point=None,
        point_link_name=None,
        closest_link_name=None,
        world_position=None,
        closest_world_position=None,
        vector=None,
        normal=None,
    )


def _metadata_vector(metadata: dict[str, Any], key: str) -> Vector3:
    value = metadata[key]
    return (float(value[0]), float(value[1]), float(value[2]))


def _subtract(first: Vector3, second: Vector3) -> Vector3:
    return (first[0] - second[0], first[1] - second[1], first[2] - second[2])


def _normalize(vector: Vector3) -> Vector3 | None:
    norm = float(np.linalg.norm(np.asarray(vector, dtype=float)))
    if norm <= 1e-12:
        return None
    return (vector[0] / norm, vector[1] / norm, vector[2] / norm)
