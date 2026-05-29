"""Visualize robot surface points and quantities with Meshcat.

Run:
    python examples/meshcat_robot_quantities.py

The example uses a small built-in two-link robot so it works without RoboKots.
To connect this to RoboKots, build a BodyField with KotsBackend and pass the
resulting QuantityValue objects to MeshcatVisualizer in the same way.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from body_field import (
    BodyField,
    LinkSurface,
    LinkState,
    ParallelProfile,
    QuantitySpec,
    QuantityValue,
    RobotState,
    RobotSurfaceModel,
    SurfaceMesh,
    SurfacePoint,
    Vector3,
)
from body_field.backends import NumpyPointFieldBackend, TaichiPointFieldBackend
from body_field.visualization import LinkTransform, MeshcatVisualizer


@dataclass(frozen=True)
class LinkPose:
    transform: LinkTransform
    angular_velocity_z: float
    linear_velocity: Vector3
    length: float
    width: float
    height: float


class DemoPlanarLinkStateProvider:
    name = "demo.planar_link_state"

    def __init__(self, poses: dict[str, LinkPose]) -> None:
        self.poses = poses

    def parallel_profile(self) -> ParallelProfile:
        return ParallelProfile(
            point_parallel=False,
            quantity_parallel=False,
            device="cpu",
            backend_kind="demo-link-state",
        )

    def link_transforms(self) -> dict[str, LinkTransform]:
        return {name: pose.transform for name, pose in self.poses.items()}

    def compute_link_states(
        self,
        model: RobotSurfaceModel,
        state: RobotState | None = None,
    ) -> dict[str, LinkState]:
        return {
            name: LinkState(
                link_name=name,
                position=pose.transform.position,
                rotation=pose.transform.rotation,
                angular_velocity=(0.0, 0.0, pose.angular_velocity_z),
                linear_velocity=pose.linear_velocity,
            )
            for name, pose in self.poses.items()
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-open", action="store_true", help="Do not open the Meshcat browser tab.")
    parser.add_argument("--no-block", action="store_true", help="Exit after publishing the scene.")
    parser.add_argument(
        "--zmq-url",
        default=None,
        help="Connect to an existing Meshcat server, e.g. tcp://127.0.0.1:6000.",
    )
    parser.add_argument(
        "--server-zmq-url",
        default=None,
        help="Start a new Meshcat server with this ZMQ URL, e.g. tcp://127.0.0.1:7500.",
    )
    parser.add_argument("--point-size", type=float, default=0.0035)
    parser.add_argument("--scalar-size", type=float, default=0.005)
    parser.add_argument("--vector-stride", type=int, default=8)
    parser.add_argument("--show-black-points", action="store_true")
    parser.add_argument("--point-backend", choices=["numpy", "taichi"], default="numpy")
    parser.add_argument(
        "--geometry",
        choices=["line", "box"],
        default="line",
        help="Use line links for the first calculation, or box links for surface coverage.",
    )
    args = parser.parse_args()

    model, link_state_provider = build_demo_robot()
    field = BodyField(model)
    point_backend_name = f"point_field.{args.point_backend}"
    if args.point_backend == "numpy":
        field.register_backend(NumpyPointFieldBackend(link_state_provider))
    else:
        field.register_backend(TaichiPointFieldBackend(link_state_provider))

    points = sample_points(args.geometry)
    quantities = [
        QuantitySpec("geometry.position", output_type="vector3", frame="world", unit="m"),
        QuantitySpec("kinematics.velocity", output_type="vector3", frame="world", unit="m/s"),
        QuantitySpec("kinematics.speed", output_type="scalar", frame="world", unit="m/s"),
    ]
    values = field.evaluate(points, quantities, backend=point_backend_name)
    positions = _filter(values, "geometry.position")
    velocities = _filter(values, "kinematics.velocity")
    speeds = _filter(values, "kinematics.speed")

    server_args = [f"--zmq-url={args.server_zmq_url}"] if args.server_zmq_url else None
    meshcat = MeshcatVisualizer(zmq_url=args.zmq_url, server_args=server_args)
    meshcat.clear()
    if args.geometry == "line":
        line_segments = _line_segments_world(link_state_provider)
        meshcat.draw_cylinder_links(line_segments, radius=0.018)
        meshcat.draw_joint_spheres(_joint_positions_world(line_segments), radius=0.035)
    else:
        meshcat.draw_surface_model(model, link_transforms=link_state_provider.link_transforms())
    if args.show_black_points:
        meshcat.draw_points(positions, size=args.point_size)
    meshcat.draw_vectors(
        positions[:: args.vector_stride],
        velocities[:: args.vector_stride],
        scale=0.16,
    )
    meshcat.draw_scalar_field(speeds, size=args.scalar_size)

    print(f"Meshcat URL: {meshcat.url()}")
    print("Scene quantities are evaluated in the world frame.")
    print(f"Point-field backend: {point_backend_name}")
    print("Robot links, speed-colored link points, orange world-frame velocity vectors.")
    if not args.no_open:
        meshcat.open()
    if not args.no_block:
        input("Press Enter to exit...")


def build_demo_robot() -> tuple[RobotSurfaceModel, DemoPlanarLinkStateProvider]:
    link_specs = [
        ("link1", 1.0, 0.55, 0.8),
        ("link2", 0.75, -0.75, 1.2),
        ("link3", 0.58, 0.65, -0.9),
    ]
    width = 0.12
    height = 0.08

    poses: dict[str, LinkPose] = {}
    links: dict[str, LinkSurface] = {}
    origin = (0.0, 0.0, 0.0)
    origin_velocity = (0.0, 0.0, 0.0)
    cumulative_theta = 0.0
    cumulative_omega = 0.0

    for link_name, length, theta, joint_omega in link_specs:
        cumulative_theta += theta
        cumulative_omega += joint_omega
        rotation = _rotz(cumulative_theta)
        poses[link_name] = LinkPose(
            LinkTransform(origin, rotation),
            cumulative_omega,
            origin_velocity,
            length,
            width,
            height,
        )
        links[link_name] = LinkSurface(link_name, _box_link_mesh(length, width, height))

        next_origin = _add(origin, _matvec(rotation, (length, 0.0, 0.0)))
        origin_velocity = _add(
            origin_velocity,
            _cross((0.0, 0.0, cumulative_omega), _sub(next_origin, origin)),
        )
        origin = next_origin

    model = RobotSurfaceModel(
        name="meshcat_demo_robot",
        links=links,
    )
    return model, DemoPlanarLinkStateProvider(poses)


def sample_points(geometry: str = "line") -> list[SurfacePoint]:
    points: list[SurfacePoint] = []
    for link_name, length, width, height in [
        ("link1", 1.0, 0.12, 0.08),
        ("link2", 0.75, 0.12, 0.08),
        ("link3", 0.58, 0.12, 0.08),
    ]:
        if geometry == "line":
            points.extend(_sample_line_link_points(link_name, length))
        elif geometry == "box":
            points.extend(_sample_box_surface_points(link_name, length, width, height))
        else:
            raise ValueError(f"Unknown geometry: {geometry}")
    return points


def _sample_line_link_points(link_name: str, length: float) -> list[SurfacePoint]:
    return [
        SurfacePoint(link_name, (x, 0.0, 0.0), link_name)
        for x in _linspace_midpoints(0.0, length, 180)
    ]


def _sample_box_surface_points(
    link_name: str,
    length: float,
    width: float,
    height: float,
) -> list[SurfacePoint]:
    points: list[SurfacePoint] = []
    x_values = _linspace_midpoints(0.0, length, 42)
    y_values = _linspace_midpoints(-width / 2.0, width / 2.0, 9)
    z_values = _linspace_midpoints(-height / 2.0, height / 2.0, 7)

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


def _linspace_midpoints(start: float, stop: float, count: int) -> list[float]:
    step = (stop - start) / count
    return [start + step * (index + 0.5) for index in range(count)]


def _filter(values: list[QuantityValue], name: str) -> list[QuantityValue]:
    return [value for value in values if value.spec.name == name]


def _line_segments_world(provider: DemoPlanarLinkStateProvider) -> list[tuple[Vector3, Vector3]]:
    segments: list[tuple[Vector3, Vector3]] = []
    for pose in provider.poses.values():
        start = pose.transform.position
        end = _add(start, _matvec(pose.transform.rotation, (pose.length, 0.0, 0.0)))
        segments.append((start, end))
    return segments


def _joint_positions_world(segments: list[tuple[Vector3, Vector3]]) -> list[Vector3]:
    positions: list[Vector3] = []
    for start, end in segments:
        for position in [start, end]:
            if not any(_norm(_sub(position, known)) < 1e-9 for known in positions):
                positions.append(position)
    return positions


def _box_link_mesh(length: float, width: float, height: float) -> SurfaceMesh:
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


def _rotz(theta: float) -> tuple[Vector3, Vector3, Vector3]:
    c = math.cos(theta)
    s = math.sin(theta)
    return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))


def _matvec(matrix: tuple[Vector3, Vector3, Vector3], vector: Vector3) -> Vector3:
    return (
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    )


def _add(left: Vector3, right: Vector3) -> Vector3:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def _sub(left: Vector3, right: Vector3) -> Vector3:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def _cross(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _norm(vector: Vector3) -> float:
    return math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)


if __name__ == "__main__":
    main()
