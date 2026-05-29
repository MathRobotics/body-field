"""Evaluate robot body-point distances to known obstacles.

Run:
    uv run python examples/obstacle_distance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from body_field import AxisAlignedBoxObstacle, BodyField, QuantitySpec, SphereObstacle
from body_field.backends import NumpyObstacleDistanceBackend
from meshcat_robot_quantities import build_demo_robot, sample_points


def main() -> None:
    model, link_state_provider = build_demo_robot()
    points = sample_points("box")
    obstacles = [
        SphereObstacle("fixture", center=(0.9, 0.45, 0.0), radius=0.18),
        AxisAlignedBoxObstacle(
            "table_edge",
            min_corner=(0.35, -0.75, -0.10),
            max_corner=(1.25, -0.55, 0.18),
        ),
    ]

    field = BodyField(model)
    field.register_backend(NumpyObstacleDistanceBackend(link_state_provider, obstacles))
    values = field.evaluate(
        points,
        [
            QuantitySpec(
                "geometry.obstacle.distance",
                output_type="scalar",
                frame="world",
                unit="m",
            )
        ],
        backend="obstacle_distance.numpy",
    )

    closest = min(values, key=lambda value: float(value.value))
    print(f"points: {len(points)}")
    print(f"minimum_signed_distance_m: {float(closest.value):.6f}")
    print(f"point_link: {closest.point.link_name}")
    print(f"world_position: {closest.metadata['world_position']}")
    print(f"closest_obstacle: {closest.metadata['closest_obstacle']}")
    print(f"closest_point: {closest.metadata['closest_point']}")


if __name__ == "__main__":
    main()
