from __future__ import annotations

import numpy as np
import pytest

from body_field import LinkState, LinkSurface, RobotSurfaceModel, SurfaceMesh, SurfacePoint
from body_field.backends._surface_points import pack_surface_point_transforms


def sample_model():
    mesh = SurfaceMesh(
        vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        faces=[(0, 1, 2)],
    )
    return RobotSurfaceModel(
        name="sample",
        links={
            "base": LinkSurface(link_name="base", mesh=mesh),
            "tool": LinkSurface(link_name="tool", mesh=mesh),
        },
    )


def sample_link_states():
    return {
        "base": LinkState(
            link_name="base",
            position=(1.0, 2.0, 3.0),
            rotation=(
                (0.0, -1.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
            angular_velocity=(0.0, 0.0, 0.0),
            linear_velocity=(0.0, 0.0, 0.0),
        ),
        "tool": LinkState(
            link_name="tool",
            position=(-1.0, 0.5, 2.0),
            rotation=(
                (1.0, 0.0, 0.0),
                (0.0, 0.0, -1.0),
                (0.0, 1.0, 0.0),
            ),
            angular_velocity=(0.0, 0.0, 0.0),
            linear_velocity=(0.0, 0.0, 0.0),
        ),
    }


def test_pack_surface_point_transforms_batches_by_link_and_preserves_order():
    points = [
        SurfacePoint("tool", (0.0, 1.0, 0.0), "link"),
        SurfacePoint("base", (0.5, 0.0, 0.0), "base"),
        SurfacePoint("tool", (2.0, 0.5, 2.5), "world"),
        SurfacePoint("base", (1.0, 2.0, 4.0), "world"),
    ]

    arrays = pack_surface_point_transforms(sample_model(), points, sample_link_states())

    assert arrays.link_names == ("base", "tool")
    assert arrays.point_link_ids.tolist() == [1, 0, 1, 0]
    np.testing.assert_allclose(
        arrays.world_position,
        [
            [-1.0, 0.5, 3.0],
            [1.0, 2.5, 3.0],
            [2.0, 0.5, 2.5],
            [1.0, 2.0, 4.0],
        ],
    )
    np.testing.assert_allclose(
        arrays.link_offset,
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.5, 0.0],
            [3.0, 0.0, 0.5],
            [0.0, 0.0, 1.0],
        ],
    )


def test_pack_surface_point_transforms_validates_frames():
    with pytest.raises(ValueError, match="point.frame"):
        pack_surface_point_transforms(
            sample_model(),
            [SurfacePoint("base", (0.0, 0.0, 0.0), "camera")],
            sample_link_states(),
        )
