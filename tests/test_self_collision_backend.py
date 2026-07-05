from __future__ import annotations

import importlib.util
import math

import numpy as np
import pytest

from body_field import (
    BodyField,
    LinkState,
    LinkSurface,
    QuantitySpec,
    RobotSurfaceModel,
    SurfaceMesh,
    SurfacePoint,
    evaluate_minimum_self_collision,
)
from body_field.backends import (
    NumpyPointJacobianBackend,
    NumpySelfCollisionBackend,
    TaichiSelfCollisionBackend,
)


class StaticTwoLinkStateProvider:
    name = "static-two-link"

    def compute_link_states(self, model, state=None):
        return {
            "arm": LinkState(
                link_name="arm",
                position=(0.0, 0.0, 0.0),
                rotation=(
                    (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                ),
                angular_velocity=(0.0, 0.0, 0.0),
                linear_velocity=(0.0, 0.0, 0.0),
            ),
            "forearm": LinkState(
                link_name="forearm",
                position=(1.0, 0.0, 0.0),
                rotation=(
                    (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                ),
                angular_velocity=(0.0, 0.0, 0.0),
                linear_velocity=(0.0, 0.0, 0.0),
            ),
        }


class LinkDependentJacobianProvider:
    name = "link-dependent-jacobian"

    def compute_point_jacobians(self, model, points, state=None):
        jacobians = []
        for point in points:
            if point.link_name == "arm":
                jacobians.append(
                    np.asarray(
                        [
                            [1.0, 0.0],
                            [0.0, 0.0],
                            [0.0, 0.0],
                        ],
                        dtype=float,
                    )
                )
            else:
                jacobians.append(
                    np.asarray(
                        [
                            [0.0, 1.0],
                            [0.0, 0.0],
                            [0.0, 0.0],
                        ],
                        dtype=float,
                    )
                )
        return jacobians


def sample_model():
    mesh = SurfaceMesh(
        vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        faces=[(0, 1, 2)],
    )
    return RobotSurfaceModel(
        name="sample",
        links={
            "arm": LinkSurface(link_name="arm", mesh=mesh),
            "forearm": LinkSurface(link_name="forearm", mesh=mesh),
        },
    )


def test_self_collision_backend_finds_nearest_other_link_point():
    field = BodyField(sample_model())
    field.register_backend(NumpySelfCollisionBackend(StaticTwoLinkStateProvider()))
    points = [
        SurfacePoint("arm", (0.0, 0.0, 0.0), "link"),
        SurfacePoint("arm", (0.1, 0.0, 0.0), "link"),
        SurfacePoint("forearm", (-0.6, 0.0, 0.0), "link"),
    ]

    values = field.evaluate(
        points,
        [
            QuantitySpec(
                "geometry.self_collision.distance", output_type="scalar", frame="world"
            ),
            QuantitySpec(
                "geometry.self_collision.closest_point",
                output_type="vector3",
                frame="world",
            ),
            QuantitySpec(
                "geometry.self_collision.vector", output_type="vector3", frame="world"
            ),
        ],
        backend="self_collision.numpy",
    )

    assert values[0].value == pytest.approx(0.4)
    assert values[0].metadata["closest_link"] == "forearm"
    assert values[0].metadata["closest_point_index"] == 2
    assert values[1].value == pytest.approx(0.3)
    assert values[2].value == pytest.approx(0.3)
    assert values[3].value == pytest.approx((0.4, 0.0, 0.0))
    assert values[6].value == pytest.approx((-0.4, 0.0, 0.0))


def test_self_collision_backend_can_include_same_link_points():
    field = BodyField(sample_model())
    field.register_backend(
        NumpySelfCollisionBackend(StaticTwoLinkStateProvider(), include_same_link=True)
    )
    points = [
        SurfacePoint("arm", (0.0, 0.0, 0.0), "link"),
        SurfacePoint("arm", (0.1, 0.0, 0.0), "link"),
        SurfacePoint("forearm", (-0.6, 0.0, 0.0), "link"),
    ]

    values = field.evaluate(
        points,
        [
            QuantitySpec(
                "geometry.self_collision.distance", output_type="scalar", frame="world"
            )
        ],
        backend="self_collision.numpy",
    )

    assert values[0].value == pytest.approx(0.1)
    assert values[0].metadata["closest_link"] == "arm"


def test_self_collision_backend_applies_point_radius_to_signed_distance():
    field = BodyField(sample_model())
    field.register_backend(
        NumpySelfCollisionBackend(StaticTwoLinkStateProvider(), point_radius=0.25)
    )
    points = [
        SurfacePoint("arm", (0.0, 0.0, 0.0), "link"),
        SurfacePoint("forearm", (-0.6, 0.0, 0.0), "link"),
    ]

    values = field.evaluate(
        points,
        [
            QuantitySpec(
                "geometry.self_collision.distance", output_type="scalar", frame="world"
            )
        ],
        backend="self_collision.numpy",
    )

    assert values[0].value == pytest.approx(-0.1)


def test_self_collision_query_helper():
    field = BodyField(sample_model())
    field.register_backend(NumpySelfCollisionBackend(StaticTwoLinkStateProvider()))

    values = field.at_points(
        [
            SurfacePoint("arm", (0.0, 0.0, 0.0), "link"),
            SurfacePoint("forearm", (-0.6, 0.0, 0.0), "link"),
        ]
    ).self_collision_distance()

    assert len(values) == 2
    assert values[0].value == pytest.approx(0.4)


def test_self_collision_backend_reports_no_candidate_as_infinite_distance():
    field = BodyField(sample_model())
    field.register_backend(NumpySelfCollisionBackend(StaticTwoLinkStateProvider()))

    values = field.evaluate(
        [SurfacePoint("arm", (0.0, 0.0, 0.0), "link")],
        [
            QuantitySpec(
                "geometry.self_collision.distance", output_type="scalar", frame="world"
            )
        ],
        backend="self_collision.numpy",
    )

    assert math.isinf(values[0].value)
    assert values[0].metadata["closest_link"] is None
    assert values[0].metadata["closest_point_index"] == -1


@pytest.mark.skipif(importlib.util.find_spec("taichi") is None, reason="taichi is not installed")
def test_taichi_self_collision_backend_matches_numpy():
    points = [
        SurfacePoint("arm", (0.0, 0.0, 0.0), "link"),
        SurfacePoint("arm", (0.1, 0.0, 0.0), "link"),
        SurfacePoint("forearm", (-0.6, 0.0, 0.0), "link"),
    ]
    quantities = [
        QuantitySpec("geometry.self_collision.distance", output_type="scalar", frame="world"),
        QuantitySpec(
            "geometry.self_collision.closest_point", output_type="vector3", frame="world"
        ),
        QuantitySpec("geometry.self_collision.vector", output_type="vector3", frame="world"),
    ]
    numpy_values = _evaluate_self_collision_backend(
        NumpySelfCollisionBackend(
            StaticTwoLinkStateProvider(),
            excluded_link_pairs=[("arm", "unused_link")],
            point_radius=0.02,
        ),
        points,
        quantities,
    )
    taichi_values = _evaluate_self_collision_backend(
        TaichiSelfCollisionBackend(
            StaticTwoLinkStateProvider(),
            excluded_link_pairs=[("arm", "unused_link")],
            point_radius=0.02,
            arch="cpu",
        ),
        points,
        quantities,
    )

    assert [value.spec.name for value in taichi_values] == [
        value.spec.name for value in numpy_values
    ]
    for taichi_value, numpy_value in zip(taichi_values, numpy_values, strict=True):
        assert taichi_value.value == pytest.approx(numpy_value.value, rel=1e-5, abs=1e-5)
        assert taichi_value.metadata["closest_point_index"] == numpy_value.metadata[
            "closest_point_index"
        ]
        assert taichi_value.metadata["closest_link"] == numpy_value.metadata["closest_link"]


@pytest.mark.skipif(importlib.util.find_spec("taichi") is None, reason="taichi is not installed")
def test_taichi_self_collision_backend_honors_same_link_and_excluded_pairs():
    points = [
        SurfacePoint("arm", (0.0, 0.0, 0.0), "link"),
        SurfacePoint("arm", (0.1, 0.0, 0.0), "link"),
        SurfacePoint("forearm", (-0.6, 0.0, 0.0), "link"),
    ]
    values = _evaluate_self_collision_backend(
        TaichiSelfCollisionBackend(
            StaticTwoLinkStateProvider(),
            excluded_link_pairs=[("arm", "forearm")],
            include_same_link=True,
            arch="cpu",
        ),
        points,
        [
            QuantitySpec(
                "geometry.self_collision.distance", output_type="scalar", frame="world"
            )
        ],
    )

    assert values[0].value == pytest.approx(0.1)
    assert values[0].metadata["closest_link"] == "arm"
    assert values[2].metadata["closest_link"] is None
    assert values[2].metadata["closest_point_index"] == -1
    assert math.isinf(values[2].value)


def test_evaluate_minimum_self_collision_returns_search_quantities():
    field = BodyField(sample_model())
    field.register_backend(NumpySelfCollisionBackend(StaticTwoLinkStateProvider()))
    field.register_backend(NumpyPointJacobianBackend(LinkDependentJacobianProvider()))
    points = [
        SurfacePoint("arm", (0.0, 0.0, 0.0), "link"),
        SurfacePoint("arm", (0.1, 0.0, 0.0), "link"),
        SurfacePoint("forearm", (-0.6, 0.0, 0.0), "link"),
    ]

    minimum = evaluate_minimum_self_collision(
        field,
        points,
        jacobian_backend="jacobian.numpy",
    )

    assert minimum.signed_distance == pytest.approx(0.3)
    assert minimum.point_index == 1
    assert minimum.closest_point_index == 2
    assert minimum.point_link_name == "arm"
    assert minimum.closest_link_name == "forearm"
    assert minimum.vector == pytest.approx((-0.3, 0.0, 0.0))
    assert minimum.normal == pytest.approx((-1.0, 0.0, 0.0))
    assert minimum.distance_gradient == pytest.approx((-1.0, 1.0))


def test_evaluate_minimum_self_collision_handles_no_candidate():
    field = BodyField(sample_model())
    field.register_backend(NumpySelfCollisionBackend(StaticTwoLinkStateProvider()))

    minimum = evaluate_minimum_self_collision(
        field,
        [SurfacePoint("arm", (0.0, 0.0, 0.0), "link")],
    )

    assert math.isinf(minimum.signed_distance)
    assert minimum.point_index == -1
    assert minimum.closest_point_index == -1
    assert minimum.normal is None


def _evaluate_self_collision_backend(backend, points, quantities):
    field = BodyField(sample_model())
    field.register_backend(backend)
    return field.evaluate(points, quantities, backend=backend.name)
