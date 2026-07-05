from __future__ import annotations

import math

from body_field import SurfaceMesh, SurfacePoint, Vector3


def sample_line_link_points(
    link_name: str,
    length: float,
    count: int,
) -> list[SurfacePoint]:
    return [
        SurfacePoint(link_name, (x, 0.0, 0.0), link_name)
        for x in linspace_midpoints(0.0, length, count)
    ]


def sample_box_surface_points(
    link_name: str,
    length: float,
    width: float,
    height: float,
    *,
    x_count: int,
    y_count: int,
    z_count: int,
) -> list[SurfacePoint]:
    points: list[SurfacePoint] = []
    x_values = linspace_midpoints(0.0, length, x_count)
    y_values = linspace_midpoints(-width / 2.0, width / 2.0, y_count)
    z_values = linspace_midpoints(-height / 2.0, height / 2.0, z_count)

    for x in x_values:
        for y in y_values:
            points.append(SurfacePoint(link_name, (x, y, -height / 2.0), link_name))
            points.append(SurfacePoint(link_name, (x, y, height / 2.0), link_name))

    for x in x_values:
        for z in z_values:
            points.append(SurfacePoint(link_name, (x, -width / 2.0, z), link_name))
            points.append(SurfacePoint(link_name, (x, width / 2.0, z), link_name))

    for y in y_values:
        for z in z_values:
            points.append(SurfacePoint(link_name, (0.0, y, z), link_name))
            points.append(SurfacePoint(link_name, (length, y, z), link_name))

    return points


def box_link_mesh(length: float, width: float, height: float) -> SurfaceMesh:
    y = width / 2.0
    z = height / 2.0
    vertices = [
        (0.0, -y, -z),
        (length, -y, -z),
        (length, y, -z),
        (0.0, y, -z),
        (0.0, -y, z),
        (length, -y, z),
        (length, y, z),
        (0.0, y, z),
    ]
    faces = [
        (0, 1, 2),
        (0, 2, 3),
        (4, 6, 5),
        (4, 7, 6),
        (0, 4, 5),
        (0, 5, 1),
        (1, 5, 6),
        (1, 6, 2),
        (2, 6, 7),
        (2, 7, 3),
        (3, 7, 4),
        (3, 4, 0),
    ]
    return SurfaceMesh(vertices=vertices, faces=faces)


def joint_positions_world(
    segments: list[tuple[Vector3, Vector3]],
    *,
    tolerance: float = 1e-9,
) -> list[Vector3]:
    positions: list[Vector3] = []
    for start, end in segments:
        for position in [start, end]:
            if not any(norm(sub(position, known)) < tolerance for known in positions):
                positions.append(position)
    return positions


def linspace_midpoints(start: float, stop: float, count: int) -> list[float]:
    step = (stop - start) / count
    return [start + step * (index + 0.5) for index in range(count)]


def rotz(theta: float) -> tuple[Vector3, Vector3, Vector3]:
    c = math.cos(theta)
    s = math.sin(theta)
    return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))


def matvec(matrix: tuple[Vector3, Vector3, Vector3], vector: Vector3) -> Vector3:
    return (
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    )


def add(left: Vector3, right: Vector3) -> Vector3:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def sub(left: Vector3, right: Vector3) -> Vector3:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def cross(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def norm(vector: Vector3) -> float:
    return math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
