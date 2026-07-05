from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from body_field.core import QuantityValue, RobotSurfaceModel, SurfacePoint, Vector3


@dataclass(frozen=True)
class LinkTransform:
    position: Vector3
    rotation: tuple[Vector3, Vector3, Vector3] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )


def make_transform(transform: LinkTransform) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = np.asarray(transform.rotation, dtype=float)
    matrix[:3, 3] = np.asarray(transform.position, dtype=float)
    return matrix


class MeshcatVisualizer:
    def __init__(
        self,
        zmq_url: str | None = None,
        server_args: list[str] | None = None,
    ) -> None:
        try:
            import meshcat
        except ImportError as exc:
            raise ImportError("MeshcatVisualizer requires meshcat. Install body-field[viz].") from exc

        self._meshcat = meshcat
        self.vis = meshcat.Visualizer(zmq_url=zmq_url, server_args=server_args or [])

    def url(self) -> str:
        return self.vis.url()

    def open(self) -> None:
        self.vis.open()

    def clear(self) -> None:
        self.vis.delete()

    def draw_surface_model(
        self,
        model: RobotSurfaceModel,
        *,
        link_transforms: dict[str, LinkTransform] | None = None,
        path: str = "robot",
        color: int = 0x8DA0CB,
        opacity: float = 0.65,
    ) -> None:
        import meshcat.geometry as g

        for link_name, link_surface in model.links.items():
            link_path = f"{path}/{link_name}"
            mesh = link_surface.mesh
            material = g.MeshLambertMaterial(color=color, transparent=True, opacity=opacity)

            if mesh.vertices and mesh.faces:
                geometry = g.TriangularMeshGeometry(
                    np.asarray(mesh.vertices, dtype=float),
                    np.asarray(mesh.faces, dtype=np.int32),
                )
                self.vis[link_path].set_object(geometry, material)
            else:
                self.vis[link_path].set_object(g.Sphere(0.035), material)

            if link_transforms and link_name in link_transforms:
                self.vis[link_path].set_transform(make_transform(link_transforms[link_name]))

    def draw_points(
        self,
        values: Iterable[QuantityValue],
        *,
        path: str = "field/points",
        color: int = 0x222222,
        size: float = 0.025,
    ) -> None:
        import meshcat.geometry as g

        positions = [_value_as_vector3(value) for value in values]
        if not positions:
            return
        self.vis[path].set_object(
            g.Points(
                g.PointsGeometry(np.asarray(positions, dtype=float).T),
                g.PointsMaterial(size=size, color=color),
            )
        )

    def draw_scalar_field(
        self,
        values: Iterable[QuantityValue],
        *,
        path: str = "field/scalars",
        size: float = 0.035,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> None:
        import meshcat.geometry as g

        items = list(values)
        if not items:
            return

        scalars = [float(item.value) for item in items]
        lo = min(scalars) if min_value is None else min_value
        hi = max(scalars) if max_value is None else max_value
        span = hi - lo if hi > lo else 1.0

        for index, item in enumerate(items):
            position = _point_world_position(item.point, item)
            t = (float(item.value) - lo) / span
            self.vis[f"{path}/{index}"].set_object(
                g.Sphere(size),
                g.MeshLambertMaterial(color=_blue_to_red(t)),
            )
            self.vis[f"{path}/{index}"].set_transform(_translation(position))

    def draw_scalar_points(
        self,
        values: Iterable[QuantityValue],
        *,
        path: str = "field/scalars",
        size: float = 0.006,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> None:
        import meshcat.geometry as g

        items = list(values)
        if not items:
            return

        scalars = [float(item.value) for item in items]
        lo = min(scalars) if min_value is None else min_value
        hi = max(scalars) if max_value is None else max_value
        span = hi - lo if hi > lo else 1.0

        positions = [_point_world_position(item.point, item) for item in items]
        colors = [_blue_to_red_rgb((float(item.value) - lo) / span) for item in items]
        self.vis[path].set_object(
            g.Points(
                g.PointsGeometry(
                    np.asarray(positions, dtype=float).T,
                    np.asarray(colors, dtype=float).T,
                ),
                g.PointsMaterial(size=size),
            )
        )

    def draw_vectors(
        self,
        origins: Iterable[QuantityValue],
        vectors: Iterable[QuantityValue],
        *,
        path: str = "field/vectors",
        color: int = 0xD95F02,
        scale: float = 1.0,
        linewidth: float = 5.0,
    ) -> None:
        import meshcat.geometry as g

        line_vertices: list[Vector3] = []
        for origin_value, vector_value in zip(origins, vectors, strict=True):
            origin = _value_as_vector3(origin_value)
            vector = _value_as_vector3(vector_value)
            end = (
                origin[0] + scale * vector[0],
                origin[1] + scale * vector[1],
                origin[2] + scale * vector[2],
            )
            line_vertices.extend([origin, end])

        if not line_vertices:
            return

        self.vis[path].set_object(
            g.LineSegments(
                g.PointsGeometry(np.asarray(line_vertices, dtype=float).T),
                g.LineBasicMaterial(color=color, linewidth=linewidth),
            )
        )

    def draw_vector_tubes(
        self,
        origins: Iterable[QuantityValue],
        vectors: Iterable[QuantityValue],
        *,
        path: str = "field/vector_tubes",
        color: int = 0xD95F02,
        scale: float = 1.0,
        radius: float = 0.006,
    ) -> None:
        import meshcat.geometry as g

        self.vis[path].delete()
        material = g.MeshLambertMaterial(color=color)
        for count, (origin_value, vector_value) in enumerate(zip(origins, vectors, strict=True)):
            origin = _value_as_vector3(origin_value)
            vector = _value_as_vector3(vector_value)
            end = (
                origin[0] + scale * vector[0],
                origin[1] + scale * vector[1],
                origin[2] + scale * vector[2],
            )
            length = _distance(origin, end)
            vector_path = f"{path}/{count}"
            if length <= 1e-12:
                continue
            self.vis[vector_path].set_object(g.Cylinder(1.0, 1.0), material)
            self.vis[vector_path].set_transform(_scaled_cylinder_transform(origin, end, radius))

    def draw_tensor_axes(
        self,
        origins: Iterable[QuantityValue],
        tensors: Iterable[QuantityValue],
        *,
        path: str = "field/tensor_axes",
        scale: float = 1.0,
        colors: tuple[int, int, int] = (0xD73027, 0x1A9850, 0x4575B4),
    ) -> None:
        import meshcat.geometry as g

        axis_vertices: list[list[Vector3]] = [[], [], []]
        for origin_value, tensor_value in zip(origins, tensors, strict=True):
            origin = _value_as_vector3(origin_value)
            for axis_index, axis in enumerate(_value_as_tensor_axes(tensor_value)):
                length = _distance((0.0, 0.0, 0.0), axis)
                if length <= 1e-12:
                    continue
                scaled_axis = (scale * axis[0], scale * axis[1], scale * axis[2])
                start = (
                    origin[0] - scaled_axis[0],
                    origin[1] - scaled_axis[1],
                    origin[2] - scaled_axis[2],
                )
                end = (
                    origin[0] + scaled_axis[0],
                    origin[1] + scaled_axis[1],
                    origin[2] + scaled_axis[2],
                )
                axis_vertices[axis_index].extend([start, end])

        for axis_index, vertices in enumerate(axis_vertices):
            axis_path = f"{path}/{axis_index}"
            if not vertices:
                self.vis[axis_path].delete()
                continue
            self.vis[axis_path].set_object(
                g.LineSegments(
                    g.PointsGeometry(np.asarray(vertices, dtype=float).T),
                    g.LineBasicMaterial(color=colors[axis_index], linewidth=2.0),
                )
            )

    def draw_tensor_ellipsoids(
        self,
        origins: Iterable[QuantityValue],
        tensors: Iterable[QuantityValue],
        *,
        path: str = "field/tensor_ellipsoids",
        scale: float = 1.0,
        normalize: bool = False,
        color: int = 0xFFF2A8,
        opacity: float = 0.32,
    ) -> None:
        import meshcat.geometry as g

        self.vis[path].delete()
        material = g.MeshLambertMaterial(color=color, transparent=True, opacity=opacity)
        for count, (origin_value, tensor_value) in enumerate(zip(origins, tensors, strict=True)):
            axes = _value_as_tensor_axes(tensor_value)
            radii, transform = _ellipsoid_radii_and_transform(
                _value_as_vector3(origin_value),
                axes,
                scale=scale,
                normalize=normalize,
            )
            ellipsoid_path = f"{path}/{count}"
            self.vis[ellipsoid_path].set_object(g.Ellipsoid(np.asarray(radii, dtype=float)), material)
            self.vis[ellipsoid_path].set_transform(transform)

    def draw_line_segments(
        self,
        segments: Iterable[tuple[Vector3, Vector3]],
        *,
        path: str = "robot/lines",
        color: int = 0x4C566A,
        linewidth: float = 5.0,
    ) -> None:
        import meshcat.geometry as g

        vertices: list[Vector3] = []
        for start, end in segments:
            vertices.extend([start, end])

        if not vertices:
            return

        self.vis[path].set_object(
            g.LineSegments(
                g.PointsGeometry(np.asarray(vertices, dtype=float).T),
                g.LineBasicMaterial(color=color, linewidth=linewidth),
            )
        )

    def draw_cylinder_links(
        self,
        segments: Iterable[tuple[Vector3, Vector3]],
        *,
        path: str = "robot/cylinders",
        radius: float = 0.025,
        color: int = 0x8DA0CB,
        opacity: float = 0.75,
    ) -> None:
        import meshcat.geometry as g

        for index, (start, end) in enumerate(segments):
            length = _distance(start, end)
            if length <= 0.0:
                continue
            self.vis[f"{path}/{index}"].set_object(
                g.Cylinder(length, radius),
                g.MeshLambertMaterial(color=color, transparent=True, opacity=opacity),
            )
            self.vis[f"{path}/{index}"].set_transform(_cylinder_transform(start, end))

    def update_cylinder_link_transforms(
        self,
        segments: Iterable[tuple[Vector3, Vector3]],
        *,
        path: str = "robot/cylinders",
    ) -> None:
        for index, (start, end) in enumerate(segments):
            if _distance(start, end) <= 0.0:
                continue
            self.vis[f"{path}/{index}"].set_transform(_cylinder_transform(start, end))

    def draw_joint_spheres(
        self,
        positions: Iterable[Vector3],
        *,
        path: str = "robot/joints",
        radius: float = 0.045,
        color: int = 0x3B4252,
    ) -> None:
        import meshcat.geometry as g

        material = g.MeshLambertMaterial(color=color)
        for index, position in enumerate(positions):
            self.vis[f"{path}/{index}"].set_object(g.Sphere(radius), material)
            self.vis[f"{path}/{index}"].set_transform(_translation(position))

    def update_joint_sphere_transforms(
        self,
        positions: Iterable[Vector3],
        *,
        path: str = "robot/joints",
    ) -> None:
        for index, position in enumerate(positions):
            self.vis[f"{path}/{index}"].set_transform(_translation(position))


def _value_as_vector3(value: QuantityValue) -> Vector3:
    return (
        float(value.value[0]),
        float(value.value[1]),
        float(value.value[2]),
    )


def _value_as_tensor_axes(value: QuantityValue) -> tuple[Vector3, Vector3, Vector3]:
    axes = value.value
    if len(axes) != 3:
        raise ValueError("Tensor axis values must contain exactly three axes")
    return (
        _coerce_vector3(axes[0]),
        _coerce_vector3(axes[1]),
        _coerce_vector3(axes[2]),
    )


def _coerce_vector3(value) -> Vector3:
    return (float(value[0]), float(value[1]), float(value[2]))


def _ellipsoid_radii_and_transform(
    origin: Vector3,
    axes: tuple[Vector3, Vector3, Vector3],
    *,
    scale: float,
    normalize: bool,
) -> tuple[Vector3, np.ndarray]:
    axis_vectors = [np.asarray(axis, dtype=float) for axis in axes]
    lengths = [float(np.linalg.norm(axis)) for axis in axis_vectors]
    max_length = max(lengths)
    if normalize and max_length > 1e-12:
        radii = [length / max_length * scale for length in lengths]
    else:
        radii = [length * scale for length in lengths]
    safe_radii = [max(radius, 1e-9) for radius in radii]

    directions = [
        axis_vectors[0] / max(lengths[0], 1e-12),
        axis_vectors[1] / max(lengths[1], 1e-12),
        axis_vectors[2] / max(lengths[2], 1e-12),
    ]
    directions = _orthonormalize_axes(directions)

    transform = np.eye(4)
    transform[:3, :3] = np.column_stack(directions)
    transform[:3, 3] = np.asarray(origin, dtype=float)
    return (
        (safe_radii[0], safe_radii[1], safe_radii[2]),
        transform,
    )


def _orthonormalize_axes(axes: list[np.ndarray]) -> list[np.ndarray]:
    first = _normal_or_fallback(axes[0], np.array([1.0, 0.0, 0.0], dtype=float))
    second = axes[1] - first * float(np.dot(first, axes[1]))
    second = _normal_or_fallback(second, _perpendicular_axis(first))
    third = _normal_axis(first, second)
    second = _normal_axis(third, first)
    return [first, second, third]


def _normal_or_fallback(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return fallback / np.linalg.norm(fallback)
    return vector / norm


def _perpendicular_axis(axis: np.ndarray) -> np.ndarray:
    candidate = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(axis, candidate))) > 0.9:
        candidate = np.array([0.0, 1.0, 0.0], dtype=float)
    perpendicular = candidate - axis * float(np.dot(axis, candidate))
    return perpendicular / np.linalg.norm(perpendicular)


def _normal_axis(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    normal = np.cross(first, second)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return normal / norm


def _point_world_position(point: SurfacePoint, fallback: QuantityValue) -> Vector3:
    if point.frame == "world":
        return point.position
    if fallback.spec.name == "geometry.position":
        return _value_as_vector3(fallback)
    metadata_position = fallback.metadata.get("world_position")
    if metadata_position is not None:
        return (
            float(metadata_position[0]),
            float(metadata_position[1]),
            float(metadata_position[2]),
        )
    return point.position


def _translation(position: Vector3) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, 3] = np.asarray(position, dtype=float)
    return matrix


def _distance(start: Vector3, end: Vector3) -> float:
    return float(np.linalg.norm(np.asarray(end, dtype=float) - np.asarray(start, dtype=float)))


def _cylinder_transform(start: Vector3, end: Vector3) -> np.ndarray:
    start_vec = np.asarray(start, dtype=float)
    end_vec = np.asarray(end, dtype=float)
    direction = end_vec - start_vec
    length = np.linalg.norm(direction)
    unit = direction / length
    midpoint = (start_vec + end_vec) / 2.0

    # Meshcat CylinderGeometry follows Three.js and is aligned with the local y axis.
    y_axis = np.array([0.0, 1.0, 0.0])
    rotation = _rotation_between(y_axis, unit)

    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = midpoint
    return matrix


def _scaled_cylinder_transform(start: Vector3, end: Vector3, radius: float) -> np.ndarray:
    base = _cylinder_transform(start, end)
    length = _distance(start, end)
    scale = np.diag([radius, length, radius])
    base[:3, :3] = base[:3, :3] @ scale
    return base


def _rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    cross = np.cross(source, target)
    dot = float(np.dot(source, target))

    if dot > 1.0 - 1e-12:
        return np.eye(3)
    if dot < -1.0 + 1e-12:
        axis = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(source, axis))) > 0.9:
            axis = np.array([0.0, 0.0, 1.0])
        axis = axis - source * np.dot(source, axis)
        axis = axis / np.linalg.norm(axis)
        return -np.eye(3) + 2.0 * np.outer(axis, axis)

    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    return np.eye(3) + skew + skew @ skew * (1.0 / (1.0 + dot))


def _blue_to_red(t: float) -> int:
    clipped = max(0.0, min(1.0, t))
    red = int(255 * clipped)
    green = int(120 * (1.0 - abs(clipped - 0.5) * 2.0))
    blue = int(255 * (1.0 - clipped))
    return (red << 16) + (green << 8) + blue


def _blue_to_red_rgb(t: float) -> tuple[float, float, float]:
    color = _blue_to_red(t)
    return (
        ((color >> 16) & 0xFF) / 255.0,
        ((color >> 8) & 0xFF) / 255.0,
        (color & 0xFF) / 255.0,
    )
