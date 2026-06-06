"""Animate a robot and update point-field quantities in Meshcat.

Run:
    uv run python examples/realtime_point_field_meshcat.py

The sample drives a small planar robot from RobotState.time. Each frame it
recomputes world positions, velocities, and speeds on body points, then updates
the Meshcat scene.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from body_field import (
    BodyField,
    LinkSurface,
    LinkState,
    QuantitySpec,
    QuantityValue,
    RobotState,
    RobotSurfaceModel,
    SurfaceMesh,
    SurfacePoint,
    Vector3,
)
from body_field.backends import (
    NumpyPointFieldBackend,
    NumpyPointJacobianBackend,
    TaichiPointFieldBackend,
)
from body_field.visualization import LinkTransform, MeshcatVisualizer, make_transform


@dataclass(frozen=True)
class LinkMotion:
    link_name: str
    length: float
    base_angle: float
    amplitude: float
    frequency: float
    phase: float


class MovingPlanarLinkStateProvider:
    name = "demo.moving_planar_link_state"

    def __init__(self, motions: list[LinkMotion]) -> None:
        self.motions = motions

    def link_transforms(
        self,
        model: RobotSurfaceModel,
        state: RobotState | None = None,
    ) -> dict[str, LinkTransform]:
        link_states = self.compute_link_states(model, state)
        return {
            name: LinkTransform(link_state.position, link_state.rotation)
            for name, link_state in link_states.items()
        }

    def compute_link_states(
        self,
        model: RobotSurfaceModel,
        state: RobotState | None = None,
    ) -> dict[str, LinkState]:
        t = 0.0 if state is None or state.time is None else float(state.time)
        origin = (0.0, 0.0, 0.0)
        origin_velocity = (0.0, 0.0, 0.0)
        origin_acceleration = (0.0, 0.0, 0.0)
        cumulative_theta = 0.0
        cumulative_omega = 0.0
        cumulative_alpha = 0.0
        link_states: dict[str, LinkState] = {}

        for motion in self.motions:
            model.require_link(motion.link_name)
            angle, angular_velocity, angular_acceleration = _joint_motion(motion, t)
            cumulative_theta += angle
            cumulative_omega += angular_velocity
            cumulative_alpha += angular_acceleration

            rotation = _rotz(cumulative_theta)
            link_states[motion.link_name] = LinkState(
                link_name=motion.link_name,
                position=origin,
                rotation=rotation,
                angular_velocity=(0.0, 0.0, cumulative_omega),
                linear_velocity=origin_velocity,
                angular_acceleration=(0.0, 0.0, cumulative_alpha),
                linear_acceleration=origin_acceleration,
            )

            link_offset = _matvec(rotation, (motion.length, 0.0, 0.0))
            omega = (0.0, 0.0, cumulative_omega)
            alpha = (0.0, 0.0, cumulative_alpha)
            next_origin = _add(origin, link_offset)
            next_velocity = _add(origin_velocity, _cross(omega, link_offset))
            next_acceleration = _add(
                origin_acceleration,
                _add(_cross(alpha, link_offset), _cross(omega, _cross(omega, link_offset))),
            )

            origin = next_origin
            origin_velocity = next_velocity
            origin_acceleration = next_acceleration

        return link_states

    def compute_point_jacobians(
        self,
        model: RobotSurfaceModel,
        points: list[SurfacePoint],
        state: RobotState | None = None,
    ) -> list[np.ndarray]:
        link_states = self.compute_link_states(model, state)
        link_order = {motion.link_name: index for index, motion in enumerate(self.motions)}
        return [_point_jacobian(model, point, link_states, link_order) for point in points]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-open", action="store_true", help="Do not open the Meshcat browser tab.")
    parser.add_argument("--duration", type=float, default=None, help="Seconds to run. Default: run until Ctrl-C.")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--point-backend", choices=["numpy", "taichi"], default="numpy")
    parser.add_argument("--geometry", choices=["line", "box"], default="line")
    parser.add_argument("--point-size", type=float, default=0.0035)
    parser.add_argument("--scalar-size", type=float, default=0.006)
    parser.add_argument("--scalar-stride", type=int, default=3)
    parser.add_argument("--vector-stride", type=int, default=12)
    parser.add_argument("--vector-scale", type=float, default=0.11)
    parser.add_argument("--vector-radius", type=float, default=0.007)
    parser.add_argument("--ellipsoid-stride", type=int, default=48)
    parser.add_argument("--ellipsoid-scale", type=float, default=0.12)
    parser.add_argument("--raw-ellipsoids", action="store_true")
    parser.add_argument("--hide-ellipsoids", action="store_true")
    parser.add_argument("--show-black-points", action="store_true")
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
    args = parser.parse_args()
    if args.fps <= 0.0:
        raise SystemExit("--fps must be positive.")
    if args.scalar_stride <= 0:
        raise SystemExit("--scalar-stride must be positive.")
    if args.vector_stride <= 0:
        raise SystemExit("--vector-stride must be positive.")
    if args.vector_radius <= 0.0:
        raise SystemExit("--vector-radius must be positive.")
    if args.ellipsoid_stride <= 0:
        raise SystemExit("--ellipsoid-stride must be positive.")

    provider = MovingPlanarLinkStateProvider(_demo_link_motions())
    model = _build_demo_robot(provider.motions)
    field = BodyField(model)
    point_backend_name = f"point_field.{args.point_backend}"
    if args.point_backend == "numpy":
        field.register_backend(NumpyPointFieldBackend(provider))
    else:
        field.register_backend(TaichiPointFieldBackend(provider))
    field.register_backend(NumpyPointJacobianBackend(provider))

    points = _sample_points(provider.motions, args.geometry)
    quantities = [
        QuantitySpec("geometry.position", output_type="vector3", frame="world", unit="m"),
        QuantitySpec("kinematics.velocity", output_type="vector3", frame="world", unit="m/s"),
        QuantitySpec("kinematics.speed", output_type="scalar", frame="world", unit="m/s"),
    ]
    manipulability_quantity = QuantitySpec(
        "kinematics.manipulability.axes",
        output_type="tensor3",
        frame="world",
        unit="m/rad",
    )

    server_args = [f"--zmq-url={args.server_zmq_url}"] if args.server_zmq_url else None
    meshcat = MeshcatVisualizer(zmq_url=args.zmq_url, server_args=server_args)
    meshcat.clear()
    _draw_robot_objects(meshcat, model, provider, RobotState(time=0.0), args.geometry)

    print(f"Meshcat URL: {meshcat.url()}")
    print(f"Point-field backend: {point_backend_name}")
    ellipsoid_mode = "raw" if args.raw_ellipsoids else "normalized"
    print(f"Manipulability ellipsoids: {ellipsoid_mode} translucent surfaces from eig(J J^T).")
    print("Updating robot motion and world-frame field values. Press Ctrl-C to stop.")
    if not args.no_open:
        meshcat.open()

    frame_period = 1.0 / args.fps
    start = time.perf_counter()
    frame = 0
    try:
        while True:
            elapsed = time.perf_counter() - start
            if args.duration is not None and elapsed > args.duration:
                break

            state = RobotState(time=elapsed)
            values = field.evaluate(points, quantities, state=state, backend=point_backend_name)
            positions = _filter(values, "geometry.position")
            velocities = _filter(values, "kinematics.velocity")
            speeds = _filter(values, "kinematics.speed")
            manipulability_axes = field.evaluate(
                points[:: args.ellipsoid_stride],
                [manipulability_quantity],
                state=state,
                backend="jacobian.numpy",
            )

            _update_robot(meshcat, model, provider, state, args.geometry)
            if args.show_black_points:
                meshcat.draw_points(positions, size=args.point_size)
            meshcat.draw_vector_tubes(
                positions[:: args.vector_stride],
                velocities[:: args.vector_stride],
                scale=args.vector_scale,
                radius=args.vector_radius,
            )
            meshcat.draw_scalar_points(
                speeds[:: args.scalar_stride],
                size=args.scalar_size,
                min_value=0.0,
                max_value=3.5,
            )
            if not args.hide_ellipsoids:
                meshcat.draw_tensor_ellipsoids(
                    positions[:: args.ellipsoid_stride],
                    manipulability_axes,
                    scale=args.ellipsoid_scale,
                    normalize=not args.raw_ellipsoids,
                )

            frame += 1
            sleep_s = start + frame * frame_period - time.perf_counter()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
    except KeyboardInterrupt:
        pass


def _surface_point_world_position(point: SurfacePoint, link_state: LinkState) -> Vector3:
    if point.frame == "world":
        return point.position
    if point.frame not in {point.link_name, "link", "local"}:
        raise ValueError(
            "Manipulability demo expects point.frame to be 'world', the link name, "
            f"'link', or 'local'; got {point.frame!r}"
        )
    return _add(link_state.position, _matvec(link_state.rotation, point.position))


def _point_jacobian(
    model: RobotSurfaceModel,
    point: SurfacePoint,
    link_states: dict[str, LinkState],
    link_order: dict[str, int],
) -> np.ndarray:
    model.require_link(point.link_name)
    point_world = _surface_point_world_position(point, link_states[point.link_name])
    point_link_index = link_order[point.link_name]
    jacobian = np.zeros((3, len(link_order)), dtype=float)
    point_array = np.asarray(point_world, dtype=float)
    z_axis = np.array([0.0, 0.0, 1.0], dtype=float)

    for link_name, joint_index in sorted(link_order.items(), key=lambda item: item[1]):
        if joint_index > point_link_index:
            continue
        joint_origin = np.asarray(link_states[link_name].position, dtype=float)
        jacobian[:, joint_index] = np.cross(z_axis, point_array - joint_origin)
    return jacobian


def _demo_link_motions() -> list[LinkMotion]:
    return [
        LinkMotion("link1", 0.72, 0.22, 0.48, 0.70, 0.0),
        LinkMotion("link2", 0.62, -0.38, 0.58, 1.05, 0.7),
        LinkMotion("link3", 0.54, 0.34, 0.50, 1.30, 1.4),
        LinkMotion("link4", 0.46, -0.26, 0.44, 1.65, 2.0),
        LinkMotion("link5", 0.38, 0.18, 0.36, 2.00, 2.6),
        LinkMotion("link6", 0.30, -0.12, 0.30, 2.35, 3.1),
    ]


def _build_demo_robot(motions: list[LinkMotion]) -> RobotSurfaceModel:
    width = 0.10
    height = 0.07
    return RobotSurfaceModel(
        name="realtime_6link_demo_robot",
        links={
            motion.link_name: LinkSurface(
                motion.link_name,
                _box_link_mesh(motion.length, width, height),
            )
            for motion in motions
        },
    )


def _sample_points(motions: list[LinkMotion], geometry: str) -> list[SurfacePoint]:
    points: list[SurfacePoint] = []
    width = 0.10
    height = 0.07
    for motion in motions:
        if geometry == "line":
            points.extend(_sample_line_link_points(motion.link_name, motion.length))
        elif geometry == "box":
            points.extend(_sample_box_surface_points(motion.link_name, motion.length, width, height))
        else:
            raise ValueError(f"Unknown geometry: {geometry}")
    return points


def _sample_line_link_points(link_name: str, length: float) -> list[SurfacePoint]:
    return [
        SurfacePoint(link_name, (x, 0.0, 0.0), link_name)
        for x in _linspace_midpoints(0.0, length, 240)
    ]


def _sample_box_surface_points(
    link_name: str,
    length: float,
    width: float,
    height: float,
) -> list[SurfacePoint]:
    points: list[SurfacePoint] = []
    x_values = _linspace_midpoints(0.0, length, 48)
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


def _linspace_midpoints(start: float, stop: float, count: int) -> list[float]:
    step = (stop - start) / count
    return [start + step * (index + 0.5) for index in range(count)]


def _joint_motion(motion: LinkMotion, t: float) -> tuple[float, float, float]:
    phase = motion.frequency * t + motion.phase
    angle = motion.base_angle + motion.amplitude * math.sin(phase)
    angular_velocity = motion.amplitude * motion.frequency * math.cos(phase)
    angular_acceleration = -motion.amplitude * motion.frequency**2 * math.sin(phase)
    return angle, angular_velocity, angular_acceleration


def _draw_robot_objects(
    meshcat: MeshcatVisualizer,
    model: RobotSurfaceModel,
    provider: MovingPlanarLinkStateProvider,
    state: RobotState,
    geometry: str,
) -> None:
    link_transforms = provider.link_transforms(model, state)
    if geometry == "box":
        meshcat.draw_surface_model(model, link_transforms=link_transforms)
        return

    line_segments = _line_segments_world(provider, model, state)
    meshcat.draw_cylinder_links(line_segments, radius=0.018)
    meshcat.draw_joint_spheres(_joint_positions_world(line_segments), radius=0.035)


def _update_robot(
    meshcat: MeshcatVisualizer,
    model: RobotSurfaceModel,
    provider: MovingPlanarLinkStateProvider,
    state: RobotState,
    geometry: str,
) -> None:
    if geometry == "box":
        for link_name, transform in provider.link_transforms(model, state).items():
            meshcat.vis[f"robot/{link_name}"].set_transform(make_transform(transform))
        return

    line_segments = _line_segments_world(provider, model, state)
    meshcat.update_cylinder_link_transforms(line_segments)
    meshcat.update_joint_sphere_transforms(_joint_positions_world(line_segments))


def _line_segments_world(
    provider: MovingPlanarLinkStateProvider,
    model: RobotSurfaceModel,
    state: RobotState,
) -> list[tuple[Vector3, Vector3]]:
    states = provider.compute_link_states(model, state)
    segments: list[tuple[Vector3, Vector3]] = []
    for motion in provider.motions:
        link_state = states[motion.link_name]
        start = link_state.position
        end = _add(start, _matvec(link_state.rotation, (motion.length, 0.0, 0.0)))
        segments.append((start, end))
    return segments


def _joint_positions_world(
    segments: list[tuple[Vector3, Vector3]],
) -> list[Vector3]:
    positions: list[Vector3] = []
    for start, end in segments:
        for position in [start, end]:
            if not any(_norm(_sub(position, known)) < 1e-9 for known in positions):
                positions.append(position)
    return positions


def _filter(values: list[QuantityValue], name: str) -> list[QuantityValue]:
    return [value for value in values if value.spec.name == name]


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
