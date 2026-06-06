from __future__ import annotations

import math

import numpy as np
import pytest

from body_field import (
    AxisAlignedBoxObstacle,
    BodyField,
    LinkState,
    LinkSurface,
    QuantitySpec,
    RobotSurfaceModel,
    SphereObstacle,
    SurfaceMesh,
    SurfacePoint,
)
from body_field.backends import NumpyObstacleDistanceBackend


class StaticLinkStateProvider:
    name = "static"

    def compute_link_states(self, model, state=None):
        return {
            "link": LinkState(
                link_name="link",
                position=(1.0, 2.0, 3.0),
                rotation=(
                    (0.0, -1.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0),
                ),
                angular_velocity=(0.0, 0.0, 0.0),
                linear_velocity=(0.0, 0.0, 0.0),
            )
        }


def sample_model():
    return RobotSurfaceModel(
        name="sample",
        links={
            "link": LinkSurface(
                link_name="link",
                mesh=SurfaceMesh(
                    vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
                    faces=[(0, 1, 2)],
                ),
            )
        },
    )


def test_obstacle_distance_backend_evaluates_sphere_distance():
    field = BodyField(sample_model())
    field.register_backend(
        NumpyObstacleDistanceBackend(
            StaticLinkStateProvider(),
            [SphereObstacle("sphere", center=(1.0, 2.5, 4.0), radius=0.25)],
        )
    )

    values = field.evaluate(
        [SurfacePoint("link", (0.5, 0.0, 0.0), "link")],
        [QuantitySpec("geometry.obstacle.distance", output_type="scalar", frame="world")],
        backend="obstacle_distance.numpy",
    )

    assert values[0].value == pytest.approx(0.75)
    assert values[0].unit == "m"
    assert values[0].metadata["world_position"] == pytest.approx((1.0, 2.5, 3.0))
    assert values[0].metadata["closest_obstacle"] == "sphere"
    assert values[0].metadata["closest_point"] == pytest.approx((1.0, 2.5, 3.75))


def test_obstacle_distance_backend_evaluates_axis_aligned_box_distance():
    field = BodyField(sample_model())
    field.register_backend(
        NumpyObstacleDistanceBackend(
            StaticLinkStateProvider(),
            [
                AxisAlignedBoxObstacle(
                    "box",
                    min_corner=(1.5, 2.0, 2.0),
                    max_corner=(2.0, 3.0, 4.0),
                )
            ],
        )
    )

    values = field.evaluate(
        [SurfacePoint("link", (0.5, 0.0, 0.0), "link")],
        [
            QuantitySpec("geometry.obstacle.distance", output_type="scalar", frame="world"),
            QuantitySpec("geometry.obstacle.closest_point", output_type="vector3", frame="world"),
            QuantitySpec("geometry.obstacle.vector", output_type="vector3", frame="world"),
        ],
        backend="obstacle_distance.numpy",
    )

    assert values[0].value == pytest.approx(0.5)
    assert values[1].value == pytest.approx((1.5, 2.5, 3.0))
    assert values[2].value == pytest.approx((-0.5, 0.0, 0.0))


def test_obstacle_distance_is_signed_inside_obstacle():
    obstacle = AxisAlignedBoxObstacle(
        "box",
        min_corner=(0.0, 0.0, 0.0),
        max_corner=(2.0, 2.0, 2.0),
    )

    distances = obstacle.signed_distances(points=np.asarray([(1.0, 1.0, 1.0)], dtype=float))

    assert distances[0] == pytest.approx(-1.0)


def test_obstacle_distance_query_helper():
    field = BodyField(sample_model())
    field.register_backend(
        NumpyObstacleDistanceBackend(
            StaticLinkStateProvider(),
            [SphereObstacle("sphere", center=(1.0, 2.5, 4.0), radius=0.25)],
        )
    )

    values = field.at(SurfacePoint("link", (1.0, 2.5, 3.0), "world")).obstacle_distance()

    assert len(values) == 1
    assert math.isclose(values[0].value, 0.75)


def test_obstacle_vector_query_helper():
    field = BodyField(sample_model())
    field.register_backend(
        NumpyObstacleDistanceBackend(
            StaticLinkStateProvider(),
            [SphereObstacle("sphere", center=(1.0, 2.5, 4.0), radius=0.25)],
        )
    )

    values = field.at(SurfacePoint("link", (1.0, 2.5, 3.0), "world")).obstacle_vector()

    assert len(values) == 1
    assert values[0].value == pytest.approx((0.0, 0.0, -0.75))
