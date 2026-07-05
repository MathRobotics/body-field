from __future__ import annotations

import importlib.util
import math

import pytest

from body_field import (
    BodyField,
    LinkState,
    LinkSurface,
    QuantitySpec,
    RobotSurfaceModel,
    SurfaceMesh,
    SurfacePoint,
)
from body_field.backends import NumpyPointFieldBackend, TaichiPointFieldBackend


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
                angular_velocity=(0.0, 0.0, 2.0),
                linear_velocity=(10.0, 20.0, 30.0),
                angular_acceleration=(0.0, 0.0, 3.0),
                linear_acceleration=(100.0, 200.0, 300.0),
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


def evaluate_with(backend):
    field = BodyField(sample_model())
    field.register_backend(backend)
    return field.evaluate(
        [SurfacePoint("link", (0.5, 0.0, 0.0), "link")],
        [
            QuantitySpec("geometry.position", output_type="vector3", frame="world"),
            QuantitySpec("kinematics.velocity", output_type="vector3", frame="world"),
            QuantitySpec("kinematics.acceleration", output_type="vector3", frame="world"),
            QuantitySpec("kinematics.speed", output_type="scalar", frame="world"),
        ],
        backend=backend.name,
    )


def test_numpy_point_field_backend_evaluates_link_surface_values():
    values = evaluate_with(NumpyPointFieldBackend(StaticLinkStateProvider()))

    assert values[0].value == pytest.approx((1.0, 2.5, 3.0))
    assert values[1].value == pytest.approx((9.0, 20.0, 30.0))
    assert values[2].value == pytest.approx((98.5, 198.0, 300.0))
    assert values[3].value == pytest.approx(math.sqrt(9.0**2 + 20.0**2 + 30.0**2))


def test_point_field_query_helpers():
    field = BodyField(sample_model())
    field.register_backend(NumpyPointFieldBackend(StaticLinkStateProvider()))
    query = field.at(SurfacePoint("link", (0.5, 0.0, 0.0), "link"))

    assert query.position()[0].value == pytest.approx((1.0, 2.5, 3.0))
    assert query.velocity()[0].value == pytest.approx((9.0, 20.0, 30.0))
    assert query.acceleration()[0].value == pytest.approx((98.5, 198.0, 300.0))
    assert query.speed()[0].value == pytest.approx(math.sqrt(9.0**2 + 20.0**2 + 30.0**2))


@pytest.mark.skipif(importlib.util.find_spec("taichi") is None, reason="taichi is not installed")
def test_taichi_point_field_backend_matches_numpy():
    numpy_values = evaluate_with(NumpyPointFieldBackend(StaticLinkStateProvider()))
    taichi_values = evaluate_with(TaichiPointFieldBackend(StaticLinkStateProvider(), arch="cpu"))

    assert [value.spec.name for value in taichi_values] == [
        value.spec.name for value in numpy_values
    ]
    for taichi_value, numpy_value in zip(taichi_values, numpy_values, strict=True):
        assert taichi_value.value == pytest.approx(numpy_value.value, rel=1e-5, abs=1e-5)
