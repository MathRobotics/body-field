"""Shared helpers for the realtime GP arm SDF example."""

from __future__ import annotations

import math
import os
import socket
from dataclasses import dataclass

import numpy as np

from body_field import (
    GaussianProcessDistanceField,
    LinkState,
    LinkSurface,
    RobotState,
    RobotSurfaceModel,
    SDFQueryResult,
    SurfaceMesh,
    SurfacePoint,
    Vector3,
)


DURATION_SECONDS = 8.0
FRAME_COUNT = 96
COARSE_GRID_SPACING = 0.225
COARSE_GRID_MARGIN = 0.25
GP_A = 260.0
GP_NOISE = 1.0e-8


@dataclass(frozen=True)
class ArmLink:
    link_name: str
    length: float
    radius: float
    joint_axis: Vector3
    base_angle: float
    angle_amplitude: float
    angle_frequency: float
    angle_phase: float


@dataclass(frozen=True)
class LinkTransform:
    position: Vector3
    rotation: tuple[Vector3, Vector3, Vector3]


class SerialCapsuleArmStateProvider:
    name = "demo.serial_capsule_arm_state"

    def __init__(self, links: list[ArmLink]) -> None:
        self.links = links

    def compute_link_states(
        self,
        model: RobotSurfaceModel,
        state: RobotState | None = None,
    ) -> dict[str, LinkState]:
        transforms = self.link_transforms(model, state)
        return {
            link_name: LinkState(
                link_name=link_name,
                position=transform.position,
                rotation=transform.rotation,
                angular_velocity=(0.0, 0.0, 0.0),
                linear_velocity=(0.0, 0.0, 0.0),
            )
            for link_name, transform in transforms.items()
        }

    def link_transforms(
        self,
        model: RobotSurfaceModel,
        state: RobotState | None = None,
    ) -> dict[str, LinkTransform]:
        t = 0.0 if state is None or state.time is None else float(state.time)
        position = np.asarray((0.0, 0.0, 0.20), dtype=np.float32)
        rotation = _rpy_matrix(0.0, 0.0, 0.15)

        transforms: dict[str, LinkTransform] = {}
        for link in self.links:
            model.require_link(link.link_name)
            angle = _joint_angle(link, t)
            rotation = rotation @ _axis_angle_matrix(link.joint_axis, angle)
            transforms[link.link_name] = LinkTransform(
                position=tuple(float(v) for v in position),
                rotation=tuple(tuple(float(v) for v in row) for row in rotation),
            )
            position = position + rotation @ np.asarray((link.length, 0.0, 0.0), dtype=np.float32)
        return transforms


def arm_links() -> list[ArmLink]:
    return [
        ArmLink("base_link", 0.42, 0.055, (0.0, 0.0, 1.0), 0.15, 0.85, 0.75, 0.0),
        ArmLink("shoulder_link", 0.38, 0.050, (0.0, 1.0, 0.0), 0.70, 1.10, 0.95, 1.2),
        ArmLink("elbow_link", 0.34, 0.047, (0.0, 1.0, 0.0), -1.15, 1.55, 1.20, 2.4),
        ArmLink("wrist_link", 0.28, 0.042, (1.0, 0.0, 0.0), 0.20, 1.10, 1.35, 0.7),
        ArmLink("tool_link", 0.22, 0.038, (0.0, 1.0, 0.0), 1.05, 1.45, 1.05, 3.0),
    ]


def build_arm_robot(links: list[ArmLink]) -> RobotSurfaceModel:
    return RobotSurfaceModel(
        name="serial_3d_capsule_arm_demo_robot",
        links={
            link.link_name: LinkSurface(
                link.link_name,
                capsule_mesh(link.length, link.radius),
            )
            for link in links
        },
    )


def build_link_distance_fields(
    links: list[ArmLink],
) -> dict[str, GaussianProcessDistanceField]:
    fields: dict[str, GaussianProcessDistanceField] = {}
    for link in links:
        samples = [
            point.position
            for point in sample_capsule_surface_points(
                link.link_name,
                link.length,
                link.radius,
                x_count=8,
                theta_count=14,
                cap_rings=4,
            )
        ]
        fields[link.link_name] = GaussianProcessDistanceField(
            np.asarray(samples, dtype=np.float32),
            a=GP_A,
            noise=GP_NOISE,
        )
    return fields


def build_coarse_grid_points(
    provider: SerialCapsuleArmStateProvider,
    model: RobotSurfaceModel,
) -> np.ndarray:
    surface_points = arm_surface_query_points(provider.links)
    times = np.linspace(0.0, DURATION_SECONDS, FRAME_COUNT, endpoint=False)
    surface_frames = []
    for t in times:
        link_transforms = provider.link_transforms(model, RobotState(time=float(t)))
        surface_frames.append(surface_points_world(surface_points, link_transforms))
    return coarse_space_grid_points(
        np.stack(surface_frames),
        COARSE_GRID_SPACING,
        COARSE_GRID_MARGIN,
    )


def query_robot_gp_distance(
    points: np.ndarray,
    link_fields: dict[str, GaussianProcessDistanceField],
    link_transforms: dict[str, LinkTransform],
) -> SDFQueryResult:
    best_distance = np.full(points.shape[0], np.inf, dtype=np.float32)
    best_nearest = np.full((points.shape[0], 3), np.nan, dtype=np.float32)
    best_normal = np.full((points.shape[0], 3), np.nan, dtype=np.float32)

    for link_name, distance_field in link_fields.items():
        transform = link_transforms[link_name]
        rotation = np.asarray(transform.rotation, dtype=np.float32)
        position = np.asarray(transform.position, dtype=np.float32)
        query_points_local = (points - position) @ rotation
        result_local = distance_field.query(query_points_local)
        update = result_local.distance < best_distance
        if not np.any(update):
            continue
        best_distance[update] = result_local.distance[update]
        best_nearest[update] = result_local.nearest[update] @ rotation.T + position
        best_normal[update] = result_local.normal[update] @ rotation.T
    return SDFQueryResult(best_distance, best_normal, best_nearest)


def query_coarse_distances(
    coarse_grid_points: np.ndarray,
    link_fields: dict[str, GaussianProcessDistanceField],
    provider: SerialCapsuleArmStateProvider,
    model: RobotSurfaceModel,
    t: float,
) -> np.ndarray:
    link_transforms = provider.link_transforms(model, RobotState(time=float(t)))
    result = query_robot_gp_distance(coarse_grid_points, link_fields, link_transforms)
    return result.distance.astype(np.float32)


def arm_surface_query_points(links: list[ArmLink]) -> list[SurfacePoint]:
    points: list[SurfacePoint] = []
    for link in links:
        points.extend(
            sample_capsule_surface_points(
                link.link_name,
                link.length,
                link.radius,
                x_count=8,
                theta_count=14,
                cap_rings=4,
            )
        )
    return points


def sample_capsule_surface_points(
    link_name: str,
    length: float,
    radius: float,
    *,
    x_count: int,
    theta_count: int,
    cap_rings: int,
) -> list[SurfacePoint]:
    points: list[SurfacePoint] = []
    theta_values = np.linspace(0.0, 2.0 * np.pi, theta_count, endpoint=False)
    for x in np.linspace(0.0, length, x_count):
        for theta in theta_values:
            points.append(
                SurfacePoint(
                    link_name,
                    (float(x), float(radius * np.cos(theta)), float(radius * np.sin(theta))),
                    link_name,
                )
            )

    phi_values = np.linspace(0.0, 0.5 * np.pi, cap_rings + 2)[1:-1]
    for phi in phi_values:
        ring_radius = radius * np.sin(phi)
        x_offset = radius * np.cos(phi)
        for theta in theta_values:
            yz = (float(ring_radius * np.cos(theta)), float(ring_radius * np.sin(theta)))
            points.append(SurfacePoint(link_name, (float(-x_offset), yz[0], yz[1]), link_name))
            points.append(
                SurfacePoint(link_name, (float(length + x_offset), yz[0], yz[1]), link_name)
            )
    points.append(SurfacePoint(link_name, (-radius, 0.0, 0.0), link_name))
    points.append(SurfacePoint(link_name, (length + radius, 0.0, 0.0), link_name))
    return points


def capsule_mesh(
    length: float,
    radius: float,
    *,
    theta_count: int = 20,
    cap_rings: int = 6,
    cylinder_segments: int = 4,
) -> SurfaceMesh:
    ring_specs: list[tuple[float, float]] = []
    for phi in np.linspace(0.0, 0.5 * np.pi, cap_rings + 1):
        ring_specs.append((float(-radius * np.cos(phi)), float(radius * np.sin(phi))))
    for x in np.linspace(0.0, length, cylinder_segments + 1)[1:-1]:
        ring_specs.append((float(x), radius))
    for phi in np.linspace(0.5 * np.pi, 0.0, cap_rings + 1):
        ring_specs.append((float(length + radius * np.cos(phi)), float(radius * np.sin(phi))))

    vertices: list[Vector3] = []
    for x, ring_radius in ring_specs:
        for index in range(theta_count):
            theta = 2.0 * np.pi * index / theta_count
            vertices.append(
                (x, float(ring_radius * np.cos(theta)), float(ring_radius * np.sin(theta)))
            )

    faces: list[tuple[int, int, int]] = []
    for ring in range(len(ring_specs) - 1):
        start = ring * theta_count
        next_start = (ring + 1) * theta_count
        for index in range(theta_count):
            a = start + index
            b = start + (index + 1) % theta_count
            c = next_start + index
            d = next_start + (index + 1) % theta_count
            faces.append((a, c, b))
            faces.append((b, c, d))
    return SurfaceMesh(vertices=vertices, faces=faces)


def surface_points_world(
    points: list[SurfacePoint],
    link_transforms: dict[str, LinkTransform],
) -> np.ndarray:
    out = np.empty((len(points), 3), dtype=np.float32)
    for index, point in enumerate(points):
        transform = link_transforms[point.link_name]
        rotation = np.asarray(transform.rotation, dtype=np.float32)
        position = np.asarray(transform.position, dtype=np.float32)
        out[index] = position + rotation @ np.asarray(point.position, dtype=np.float32)
    return out


def coarse_space_grid_points(
    surface_frames: np.ndarray,
    spacing: float,
    margin: float,
) -> np.ndarray:
    all_surface_points = surface_frames.reshape(-1, 3)
    lo = np.min(all_surface_points, axis=0) - float(margin)
    hi = np.max(all_surface_points, axis=0) + float(margin)
    xs = np.arange(lo[0], hi[0] + 0.5 * spacing, spacing)
    ys = np.arange(lo[1], hi[1] + 0.5 * spacing, spacing)
    zs = np.arange(lo[2], hi[2] + 0.5 * spacing, spacing)
    return np.asarray([(x, y, z) for x in xs for y in ys for z in zs], dtype=np.float32)


def add_arm_meshes(server, links: list[ArmLink]):
    handles = {}
    for link in links:
        mesh = capsule_mesh(link.length, link.radius)
        handles[link.link_name] = server.scene.add_mesh_simple(
            f"/capsule_arm/{link.link_name}",
            vertices=np.asarray(mesh.vertices, dtype=np.float32),
            faces=np.asarray(mesh.faces, dtype=np.int32),
            color=(31, 41, 55),
            opacity=0.32,
            flat_shading=True,
            side="double",
        )
    return handles


def update_arm_meshes(handles, provider, model, t: float) -> None:
    link_transforms = provider.link_transforms(model, RobotState(time=t))
    for link_name, transform in link_transforms.items():
        rotation = np.asarray(transform.rotation, dtype=np.float32)
        handles[link_name].position = np.asarray(transform.position, dtype=np.float32)
        handles[link_name].wxyz = matrix_to_wxyz(rotation)


def unit_cube_mesh() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ],
        dtype=np.int32,
    )
    return vertices, faces


def distance_colors(distance: np.ndarray) -> np.ndarray:
    colors = np.empty((len(distance), 3), dtype=np.uint8)
    d = np.clip((distance - 0.035) / 0.55, 0.0, 1.0)
    near = d < 0.35
    far = ~near
    near_indices = np.nonzero(near)[0]
    far_indices = np.nonzero(far)[0]

    t_near = d[near] / 0.35
    colors[near_indices, 0] = np.round(248.0 - 60.0 * t_near).astype(np.uint8)
    colors[near_indices, 1] = np.round(186.0 - 6.0 * t_near).astype(np.uint8)
    colors[near_indices, 2] = np.round(28.0 + 150.0 * t_near).astype(np.uint8)

    t_far = (d[far] - 0.35) / 0.65
    colors[far_indices, 0] = np.round(188.0 - 157.0 * t_far).astype(np.uint8)
    colors[far_indices, 1] = np.round(180.0 - 85.0 * t_far).astype(np.uint8)
    colors[far_indices, 2] = np.round(178.0 + 13.0 * t_far).astype(np.uint8)
    return colors


def coarse_distance_opacities(distance: np.ndarray) -> np.ndarray:
    d = np.clip((distance - 0.04) / 0.85, 0.0, 1.0)
    return np.where(d < 0.35, 0.34, 0.26 - 0.12 * ((d - 0.35) / 0.65)).astype(np.float32)


def viser_port() -> int:
    port = os.environ.get("BODY_FIELD_VISER_PORT")
    if port is not None:
        return int(port)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def matrix_to_wxyz(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            w = (rotation[2, 1] - rotation[1, 2]) / s
            x = 0.25 * s
            y = (rotation[0, 1] + rotation[1, 0]) / s
            z = (rotation[0, 2] + rotation[2, 0]) / s
        elif axis == 1:
            s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            w = (rotation[0, 2] - rotation[2, 0]) / s
            x = (rotation[0, 1] + rotation[1, 0]) / s
            y = 0.25 * s
            z = (rotation[1, 2] + rotation[2, 1]) / s
        else:
            s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            w = (rotation[1, 0] - rotation[0, 1]) / s
            x = (rotation[0, 2] + rotation[2, 0]) / s
            y = (rotation[1, 2] + rotation[2, 1]) / s
            z = 0.25 * s
    quat = np.asarray([w, x, y, z], dtype=np.float32)
    return quat / np.linalg.norm(quat)


def _joint_angle(link: ArmLink, t: float) -> float:
    return float(
        link.base_angle
        + link.angle_amplitude * math.sin(link.angle_frequency * t + link.angle_phase)
    )


def _axis_angle_matrix(axis: Vector3, angle: float) -> np.ndarray:
    axis_array = np.asarray(axis, dtype=np.float32)
    axis_array = axis_array / np.linalg.norm(axis_array)
    x, y, z = axis_array
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c
    return np.asarray(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float32,
    )


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float32,
    )
