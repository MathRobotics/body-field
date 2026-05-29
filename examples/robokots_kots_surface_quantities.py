"""Sample: evaluate quantities on RoboKots links through KotsBackend.

This sample uses the local RoboKots checkout by default:
    /Users/a896/Documents/MathRobotics/RoboKots

Set ROBOKOTS_ROOT to use a different checkout.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROBOKOTS_ROOT = Path(os.environ.get("ROBOKOTS_ROOT", "/Users/a896/Documents/MathRobotics/RoboKots"))
sys.path.insert(0, str(ROBOKOTS_ROOT))

from body_field import BodyField, QuantitySpec, RobotState, SurfacePoint
from body_field.backends import KotsBackend, robot_surface_model_from_kots

try:
    from robokots.kots import Kots
except ImportError as exc:
    raise SystemExit(
        "Failed to import RoboKots. Check that RoboKots and its mathrobo dependency "
        f"are installed for this Python environment. ROBOKOTS_ROOT={ROBOKOTS_ROOT}"
    ) from exc


def main() -> None:
    model_path = ROBOKOTS_ROOT / "tests" / "test_model" / "sample_robot.json"
    kots = Kots.from_json_file(str(model_path), order=3)

    field = BodyField(robot_surface_model_from_kots(kots, name="sample_robot"))
    field.register_backend(KotsBackend(kots))

    state = RobotState(
        q=[0.0] * kots.dof(),
        dq=[0.0] * kots.dof(),
        ddq=[0.0] * kots.dof(),
        time=0.0,
    )
    point = SurfacePoint(
        link_name="arm3",
        position=(0.1, 0.0, 0.0),
        frame="arm3",
    )

    quantities = [
        QuantitySpec("geometry.position", output_type="vector3", frame="world", unit="m"),
        QuantitySpec("kinematics.velocity", output_type="vector3", frame="world", unit="m/s"),
        QuantitySpec("kinematics.acceleration", output_type="vector3", frame="world", unit="m/s^2"),
        QuantitySpec("dynamics.force", output_type="wrench", frame="world"),
    ]

    for value in field.evaluate([point], quantities, state, backend="robokots.kots"):
        print(value.spec.name, value.value, value.unit)


if __name__ == "__main__":
    main()
