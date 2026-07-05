"""Visualize robot surface points and quantities with Meshcat.

Run:
    python examples/meshcat_robot_quantities.py

The example uses a small built-in two-link robot.
"""

from __future__ import annotations

import argparse
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
    SurfacePoint,
    Vector3,
)
from body_field.backends import NumpyPointFieldBackend, TaichiPointFieldBackend
from body_field.visualization import LinkTransform, MeshcatVisualizer
from examples._demo_geometry import (
    add,
    box_link_mesh,
    cross,
    joint_positions_world,
    matvec,
    rotz,
    sample_box_surface_points,
    sample_line_link_points,
    sub,
)


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
        meshcat.draw_joint_spheres(joint_positions_world(line_segments), radius=0.035)
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
        rotation = rotz(cumulative_theta)
        poses[link_name] = LinkPose(
            LinkTransform(origin, rotation),
            cumulative_omega,
            origin_velocity,
            length,
            width,
            height,
        )
        links[link_name] = LinkSurface(link_name, box_link_mesh(length, width, height))

        next_origin = add(origin, matvec(rotation, (length, 0.0, 0.0)))
        origin_velocity = add(
            origin_velocity,
            cross((0.0, 0.0, cumulative_omega), sub(next_origin, origin)),
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
            points.extend(sample_line_link_points(link_name, length, 180))
        elif geometry == "box":
            points.extend(
                sample_box_surface_points(
                    link_name,
                    length,
                    width,
                    height,
                    x_count=42,
                    y_count=9,
                    z_count=7,
                )
            )
        else:
            raise ValueError(f"Unknown geometry: {geometry}")
    return points


def _filter(values: list[QuantityValue], name: str) -> list[QuantityValue]:
    return [value for value in values if value.spec.name == name]


def _line_segments_world(provider: DemoPlanarLinkStateProvider) -> list[tuple[Vector3, Vector3]]:
    segments: list[tuple[Vector3, Vector3]] = []
    for pose in provider.poses.values():
        start = pose.transform.position
        end = add(start, matvec(pose.transform.rotation, (pose.length, 0.0, 0.0)))
        segments.append((start, end))
    return segments


if __name__ == "__main__":
    main()
