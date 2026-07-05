import numpy as np
import pytest

from body_field import GaussianProcessDistanceField


def test_gaussian_process_distance_field_returns_euclidean_distance():
    field = GaussianProcessDistanceField(
        np.asarray([(0.0, 0.0, 0.0)]),
        a=16.0,
        noise=0.0,
    )

    result = field.query(np.asarray([(0.25, 0.0, 0.0)]))

    assert result.distance[0] == pytest.approx(0.25, abs=1e-6)
    assert result.normal[0] == pytest.approx((1.0, 0.0, 0.0), abs=1e-6)
    assert result.nearest[0] == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)
