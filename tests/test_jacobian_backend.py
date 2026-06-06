from __future__ import annotations

import numpy as np
import pytest

from body_field import BodyField, LinkSurface, QuantitySpec, RobotSurfaceModel, SurfaceMesh, SurfacePoint
from body_field.backends import NumpyPointJacobianBackend


class FakePointJacobianProvider:
    name = "fake_jacobian"

    def compute_point_jacobians(self, model, points, state=None):
        return [
            np.asarray(
                [
                    [0.0, -1.0],
                    [1.0, 0.0],
                    [0.0, 0.0],
                ],
                dtype=float,
            )
            for _ in points
        ]


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


def test_jacobian_backend_evaluates_point_jacobian():
    field = BodyField(sample_model())
    field.register_backend(NumpyPointJacobianBackend(FakePointJacobianProvider()))

    values = field.evaluate(
        [SurfacePoint("link", (0.5, 0.0, 0.0), "link")],
        [QuantitySpec("kinematics.jacobian", output_type="matrix", frame="world")],
        backend="jacobian.numpy",
    )

    np.testing.assert_allclose(
        values[0].value,
        ((0.0, -1.0), (1.0, 0.0), (0.0, 0.0)),
    )
    assert values[0].unit == "m/rad"
    assert values[0].metadata["columns"] == "joint velocities"


def test_jacobian_backend_evaluates_manipulability_axes():
    field = BodyField(sample_model())
    field.register_backend(NumpyPointJacobianBackend(FakePointJacobianProvider()))

    values = field.evaluate(
        [SurfacePoint("link", (0.5, 0.0, 0.0), "link")],
        [QuantitySpec("kinematics.manipulability.axes", output_type="tensor3", frame="world")],
        backend="jacobian.numpy",
    )

    axis_lengths = [np.linalg.norm(axis) for axis in values[0].value]
    assert axis_lengths == pytest.approx([1.0, 1.0, 0.0])
