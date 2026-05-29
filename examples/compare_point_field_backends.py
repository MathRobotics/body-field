"""Compare NumPy and Taichi point-field backends.

Run:
    uv run python examples/compare_point_field_backends.py
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from body_field import BodyField, QuantitySpec
from body_field.backends import NumpyPointFieldBackend, TaichiPointFieldBackend
from meshcat_robot_quantities import build_demo_robot, sample_points


def main() -> None:
    if importlib.util.find_spec("taichi") is None:
        raise SystemExit("taichi is not installed. Run `uv sync` or `uv run ...` first.")

    model, link_state_provider = build_demo_robot()
    points = sample_points("box")
    quantities = [
        QuantitySpec("geometry.position", output_type="vector3", frame="world"),
        QuantitySpec("kinematics.velocity", output_type="vector3", frame="world"),
        QuantitySpec("kinematics.acceleration", output_type="vector3", frame="world"),
        QuantitySpec("kinematics.speed", output_type="scalar", frame="world"),
    ]

    numpy_backend = NumpyPointFieldBackend(link_state_provider)
    taichi_backend = TaichiPointFieldBackend(link_state_provider, arch="cpu")

    numpy_values, numpy_first_s = _timed_evaluate(model, points, quantities, numpy_backend)
    taichi_values, taichi_first_s = _timed_evaluate(model, points, quantities, taichi_backend)

    numpy_avg_s = _average_time(model, points, quantities, numpy_backend, repeats=20)
    taichi_avg_s = _average_time(model, points, quantities, taichi_backend, repeats=20)

    diffs = {"geometry.position": 0.0, "kinematics.velocity": 0.0, "kinematics.acceleration": 0.0, "kinematics.speed": 0.0}
    for numpy_value, taichi_value in zip(numpy_values, taichi_values, strict=True):
        name = numpy_value.spec.name
        diffs[name] = max(diffs[name], _abs_diff(numpy_value.value, taichi_value.value))

    print(f"points: {len(points)}")
    print(f"numpy first_eval_s: {numpy_first_s:.6f}")
    print(f"taichi first_eval_s: {taichi_first_s:.6f}")
    print(f"numpy avg_eval_s: {numpy_avg_s:.6f}")
    print(f"taichi avg_eval_s: {taichi_avg_s:.6f}")
    for name, diff in diffs.items():
        print(f"{name}: max_abs_diff={diff:.6g}")


def _evaluate(model, points, quantities, backend):
    field = BodyField(model)
    field.register_backend(backend)
    return field.evaluate(points, quantities, backend=backend.name)


def _timed_evaluate(model, points, quantities, backend):
    start = time.perf_counter()
    values = _evaluate(model, points, quantities, backend)
    return values, time.perf_counter() - start


def _average_time(model, points, quantities, backend, repeats: int) -> float:
    # Run once before timing so Taichi JIT compilation is not mixed into steady-state timing.
    _evaluate(model, points, quantities, backend)

    start = time.perf_counter()
    for _ in range(repeats):
        _evaluate(model, points, quantities, backend)
    return (time.perf_counter() - start) / repeats


def _abs_diff(left, right) -> float:
    if isinstance(left, tuple):
        return max(abs(float(l) - float(r)) for l, r in zip(left, right, strict=True))
    return abs(float(left) - float(right))


if __name__ == "__main__":
    main()
