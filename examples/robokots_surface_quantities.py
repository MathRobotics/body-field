"""Sample: evaluate quantities at points on robot links through a RoboKots adapter.

Replace DemoRoboKotsRobot with the actual RoboKots robot/session object. The rest of
the body_field interface should stay unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from body_field import (
    BodyField,
    LinkSurface,
    QuantitySpec,
    RobotState,
    RobotSurfaceModel,
    SurfaceMesh,
    SurfacePoint,
)
from body_field.backends import RoboKotsBackend, robokots_method


class RoboKotsRobotAdapter:
    """Small adapter layer for the real RoboKots API.

    Keep RoboKots-specific method names here so BodyField and RoboKotsBackend stay
    independent from a particular RoboKots release.
    """

    def __init__(self, robot: object) -> None:
        self.robot = robot

    def surface_normal(
        self,
        *,
        link_name: str,
        position: tuple[float, float, float],
        frame: str,
        state: RobotState | None,
    ) -> tuple[float, float, float]:
        # Example shape:
        # return self.robot.surface.normal(link_name, position, frame=frame, q=state.q)
        return self.robot.surface_normal(link_name, position, frame, state)

    def point_velocity(
        self,
        *,
        link_name: str,
        position: tuple[float, float, float],
        frame: str,
        state: RobotState | None,
    ) -> tuple[float, float, float]:
        # Example shape:
        # return self.robot.kinematics.point_velocity(link_name, position, q=state.q, dq=state.dq)
        return self.robot.point_velocity(link_name, position, frame, state)


def build_field(robokots_robot: object) -> BodyField:
    model = RobotSurfaceModel(
        name="sample_robot",
        links={
            "left_foot": LinkSurface(
                link_name="left_foot",
                mesh=SurfaceMesh(
                    vertices=[
                        (0.0, 0.0, 0.0),
                        (0.2, 0.0, 0.0),
                        (0.0, 0.1, 0.0),
                    ],
                    faces=[(0, 1, 2)],
                ),
            )
        },
    )

    adapter = RoboKotsRobotAdapter(robokots_robot)
    field = BodyField(model)
    field.register_backend(
        RoboKotsBackend(
            adapter,
            handlers={
                "geometry.normal": robokots_method("surface_normal"),
                "kinematics.velocity": robokots_method("point_velocity"),
            },
            units={
                "geometry.normal": None,
                "kinematics.velocity": "m/s",
            },
        )
    )
    return field


def evaluate_left_foot(field: BodyField) -> None:
    state = RobotState(q=[0.0, 0.1], dq=[0.2, 0.0], time=0.0)
    points = [
        SurfacePoint(
            link_name="left_foot",
            position=(0.04, 0.02, 0.0),
            frame="left_foot",
            triangle_id=0,
            barycentric=(0.6, 0.2, 0.2),
        )
    ]
    quantities = [
        QuantitySpec("geometry.normal", output_type="vector3", frame="world"),
        QuantitySpec("kinematics.velocity", output_type="vector3", frame="world", unit="m/s"),
    ]

    for value in field.evaluate(points, quantities, state):
        print(value.spec.name, value.point.link_name, value.value, value.unit)


class DemoRoboKotsRobot:
    """Runnable stand-in for the real RoboKots object."""

    def surface_normal(
        self,
        link_name: str,
        position: tuple[float, float, float],
        frame: str,
        state: RobotState | None,
    ) -> tuple[float, float, float]:
        return (0.0, 0.0, 1.0)

    def point_velocity(
        self,
        link_name: str,
        position: tuple[float, float, float],
        frame: str,
        state: RobotState | None,
    ) -> tuple[float, float, float]:
        return (0.0, 0.0, 0.0)


if __name__ == "__main__":
    evaluate_left_foot(build_field(DemoRoboKotsRobot()))
