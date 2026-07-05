from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from body_field.core import LinkState, RobotSurfaceModel, SurfacePoint


@dataclass(frozen=True)
class SurfacePointTransformArrays:
    world_position: np.ndarray
    link_offset: np.ndarray
    point_link_ids: np.ndarray
    link_names: tuple[str, ...]


def pack_surface_point_transforms(
    model: RobotSurfaceModel,
    points: list[SurfacePoint],
    link_states: dict[str, LinkState],
) -> SurfacePointTransformArrays:
    """Pack point transforms into arrays, batching local-to-world work by link."""

    link_names = tuple(model.links)
    link_ids = {name: index for index, name in enumerate(link_names)}
    point_count = len(points)

    point_position = np.empty((point_count, 3), dtype=np.float32)
    point_is_world = np.empty(point_count, dtype=bool)
    point_link_ids = np.empty(point_count, dtype=np.int32)

    for index, point in enumerate(points):
        model.require_link(point.link_name)
        if point.link_name not in link_states:
            raise ValueError(f"Missing link state for: {point.link_name}")
        if point.frame not in {"world", point.link_name, "link", "local"}:
            raise ValueError(
                "Surface point transforms expect point.frame to be 'world', the link name, "
                f"'link', or 'local'; got {point.frame!r}"
            )
        point_position[index] = np.asarray(point.position, dtype=np.float32)
        point_is_world[index] = point.frame == "world"
        point_link_ids[index] = link_ids[point.link_name]

    world_position = np.empty((point_count, 3), dtype=np.float32)
    link_offset = np.empty((point_count, 3), dtype=np.float32)

    for link_name in _queried_link_names(points):
        link_id = link_ids[link_name]
        indices = np.nonzero(point_link_ids == link_id)[0]
        link_state = link_states[link_name]
        origin = np.asarray(link_state.position, dtype=np.float32)
        rotation = np.asarray(link_state.rotation, dtype=np.float32)

        world_indices = indices[point_is_world[indices]]
        if world_indices.size:
            world_position[world_indices] = point_position[world_indices]
            link_offset[world_indices] = point_position[world_indices] - origin

        local_indices = indices[~point_is_world[indices]]
        if local_indices.size:
            offsets = point_position[local_indices] @ rotation.T
            link_offset[local_indices] = offsets
            world_position[local_indices] = origin + offsets

    return SurfacePointTransformArrays(
        world_position=world_position,
        link_offset=link_offset,
        point_link_ids=point_link_ids,
        link_names=link_names,
    )


def _queried_link_names(points: list[SurfacePoint]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(point.link_name for point in points))
