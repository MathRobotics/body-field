"""Animate obstacle distances on robot body points in Meshcat.

Run:
    uv run python examples/realtime_obstacle_distance_meshcat.py

The sample drives the same planar six-link robot as realtime_point_field_meshcat.py.
Each frame computes signed distances from sampled link points to fixed obstacles,
then colors the body points by clearance.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from body_field import (
    AxisAlignedBoxObstacle,
    BodyField,
    QuantitySpec,
    QuantityValue,
    RobotState,
    SphereObstacle,
    Vector3,
)
from body_field.backends import NumpyObstacleDistanceBackend
from body_field.obstacles import DistanceObstacle
from body_field.visualization import MeshcatVisualizer
from realtime_point_field_meshcat import (
    MovingPlanarLinkStateProvider,
    _build_demo_robot,
    _demo_link_motions,
    _draw_robot_objects,
    _sample_points,
    _update_robot,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-open", action="store_true", help="Do not open the Meshcat browser tab.")
    parser.add_argument("--duration", type=float, default=None, help="Seconds to run. Default: run until Ctrl-C.")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--geometry", choices=["line", "box"], default="box")
    parser.add_argument("--point-size", type=float, default=0.007)
    parser.add_argument("--point-stride", type=int, default=2)
    parser.add_argument("--closest-stride", type=int, default=48)
    parser.add_argument("--warning-distance", type=float, default=0.18)
    parser.add_argument("--safe-distance", type=float, default=0.45)
    parser.add_argument("--show-closest-lines", action="store_true")
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
    if args.point_stride <= 0:
        raise SystemExit("--point-stride must be positive.")
    if args.closest_stride <= 0:
        raise SystemExit("--closest-stride must be positive.")
    if args.warning_distance <= 0.0:
        raise SystemExit("--warning-distance must be positive.")
    if args.safe_distance <= args.warning_distance:
        raise SystemExit("--safe-distance must be greater than --warning-distance.")

    provider = MovingPlanarLinkStateProvider(_demo_link_motions())
    model = _build_demo_robot(provider.motions)
    obstacles = _demo_obstacles()

    field = BodyField(model)
    field.register_backend(NumpyObstacleDistanceBackend(provider, obstacles))

    points = _sample_points(provider.motions, args.geometry)
    distance_quantity = QuantitySpec(
        "geometry.obstacle.distance",
        output_type="scalar",
        frame="world",
        unit="m",
    )

    server_args = [f"--zmq-url={args.server_zmq_url}"] if args.server_zmq_url else None
    meshcat = MeshcatVisualizer(zmq_url=args.zmq_url, server_args=server_args)
    meshcat.clear()
    _draw_robot_objects(meshcat, model, provider, RobotState(time=0.0), args.geometry)
    _draw_obstacles(meshcat, obstacles)

    print(f"Meshcat URL: {meshcat.url()}")
    print("Obstacle-distance backend: obstacle_distance.numpy")
    print("Point colors: red <= collision, yellow near, blue clear.")
    print("Updating robot motion and signed obstacle distances. Press Ctrl-C to stop.")
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
            distance_values = field.evaluate(
                points,
                [distance_quantity],
                state=state,
                backend="obstacle_distance.numpy",
            )

            _update_robot(meshcat, model, provider, state, args.geometry)
            _draw_distance_points(
                meshcat,
                distance_values[:: args.point_stride],
                size=args.point_size,
                warning_distance=args.warning_distance,
                safe_distance=args.safe_distance,
            )
            if args.show_closest_lines:
                _draw_closest_lines(
                    meshcat,
                    distance_values[:: args.closest_stride],
                    max_abs_distance=args.safe_distance,
                )

            if frame % max(1, int(args.fps)) == 0:
                closest = min(distance_values, key=lambda value: float(value.value))
                print(
                    "min_distance_m="
                    f"{float(closest.value): .4f} "
                    f"link={closest.point.link_name} "
                    f"obstacle={closest.metadata['closest_obstacle']}"
                )

            frame += 1
            sleep_s = start + frame * frame_period - time.perf_counter()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
    except KeyboardInterrupt:
        pass


def _demo_obstacles() -> tuple[DistanceObstacle, ...]:
    return (
        SphereObstacle("fixture", center=(0.95, 0.42, 0.0), radius=0.18),
        SphereObstacle("post", center=(1.55, -0.22, 0.0), radius=0.14),
        AxisAlignedBoxObstacle(
            "table_edge",
            min_corner=(0.35, -0.78, -0.10),
            max_corner=(1.45, -0.55, 0.16),
        ),
    )


def _draw_obstacles(
    meshcat: MeshcatVisualizer,
    obstacles: tuple[DistanceObstacle, ...],
    *,
    path: str = "obstacles",
) -> None:
    import meshcat.geometry as g

    for obstacle in obstacles:
        obstacle_path = f"{path}/{obstacle.name}"
        if isinstance(obstacle, SphereObstacle):
            meshcat.vis[obstacle_path].set_object(
                g.Sphere(obstacle.radius),
                g.MeshLambertMaterial(color=0xE15759, transparent=True, opacity=0.38),
            )
            meshcat.vis[obstacle_path].set_transform(_translation(obstacle.center))
        elif isinstance(obstacle, AxisAlignedBoxObstacle):
            min_corner = np.asarray(obstacle.min_corner, dtype=float)
            max_corner = np.asarray(obstacle.max_corner, dtype=float)
            lengths = max_corner - min_corner
            center = (min_corner + max_corner) * 0.5
            meshcat.vis[obstacle_path].set_object(
                g.Box(lengths),
                g.MeshLambertMaterial(color=0xF28E2B, transparent=True, opacity=0.32),
            )
            meshcat.vis[obstacle_path].set_transform(_translation(_tuple3(center)))
        else:
            raise TypeError(f"Unsupported obstacle type for drawing: {type(obstacle).__name__}")


def _draw_distance_points(
    meshcat: MeshcatVisualizer,
    values: list[QuantityValue],
    *,
    size: float,
    warning_distance: float,
    safe_distance: float,
    path: str = "field/obstacle_distance_points",
) -> None:
    import meshcat.geometry as g

    if not values:
        return

    positions = [_metadata_vector3(value, "world_position") for value in values]
    colors = [
        _distance_color(float(value.value), warning_distance, safe_distance) for value in values
    ]
    meshcat.vis[path].set_object(
        g.Points(
            g.PointsGeometry(
                np.asarray(positions, dtype=float).T,
                np.asarray(colors, dtype=float).T,
            ),
            g.PointsMaterial(size=size),
        )
    )


def _draw_closest_lines(
    meshcat: MeshcatVisualizer,
    values: list[QuantityValue],
    *,
    max_abs_distance: float,
    path: str = "field/closest_obstacle_lines",
) -> None:
    import meshcat.geometry as g

    vertices: list[Vector3] = []
    for value in values:
        distance = abs(float(value.value))
        if distance > max_abs_distance:
            continue
        start = _metadata_vector3(value, "world_position")
        end = _metadata_vector3(value, "closest_point")
        vertices.extend([start, end])

    if not vertices:
        meshcat.vis[path].delete()
        return

    meshcat.vis[path].set_object(
        g.LineSegments(
            g.PointsGeometry(np.asarray(vertices, dtype=float).T),
            g.LineBasicMaterial(color=0x2F4B7C, linewidth=2.0),
        )
    )


def _distance_color(
    distance: float,
    warning_distance: float,
    safe_distance: float,
) -> tuple[float, float, float]:
    if distance <= 0.0:
        return (0.86, 0.08, 0.12)
    if distance < warning_distance:
        t = distance / warning_distance
        return (0.95, 0.18 + 0.62 * t, 0.10)

    t = min(1.0, (distance - warning_distance) / (safe_distance - warning_distance))
    return (0.95 * (1.0 - t) + 0.12 * t, 0.80 * (1.0 - t) + 0.42 * t, 0.10 + 0.78 * t)


def _metadata_vector3(value: QuantityValue, key: str) -> Vector3:
    raw = value.metadata[key]
    return (float(raw[0]), float(raw[1]), float(raw[2]))


def _translation(position: Vector3) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, 3] = np.asarray(position, dtype=float)
    return matrix


def _tuple3(array: np.ndarray) -> Vector3:
    return (float(array[0]), float(array[1]), float(array[2]))


if __name__ == "__main__":
    main()
