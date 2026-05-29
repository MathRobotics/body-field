from body_field import (
    BodyField,
    LinkSurface,
    QuantitySpec,
    RobotState,
    RobotSurfaceModel,
    SurfaceMesh,
    SurfacePoint,
)
from body_field.backends import KotsBackend, RoboKotsBackend, robokots_method, robot_surface_model_from_kots


class FakeRoboKots:
    def surface_normal(self, *, link_name, position, frame, state):
        return (0.0, 0.0, 1.0)


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


def test_robokots_backend_maps_quantity_to_method():
    field = BodyField(sample_model())
    field.register_backend(
        RoboKotsBackend(
            FakeRoboKots(),
            handlers={"geometry.normal": robokots_method("surface_normal")},
        )
    )

    values = field.evaluate(
        [SurfacePoint("link", (0.2, 0.1, 0.0), "link")],
        [QuantitySpec("geometry.normal", output_type="vector3", frame="world")],
    )

    assert values[0].value == (0.0, 0.0, 1.0)
    assert values[0].metadata["backend"] == "robokots"
    assert not field.registry.get("robokots").parallel_profile().point_parallel


class FakeStateType:
    def __init__(self, owner_type, owner_name, data_type, frame_name=None):
        self.owner_type = owner_type
        self.owner_name = owner_name
        self.data_type = data_type
        self.frame_name = frame_name


class FakeFrame:
    def pos(self):
        return (1.0, 2.0, 3.0)

    def rot(self):
        return (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )


class FakeKots:
    def __init__(self):
        self.imported_motion = None
        self.update_calls = []

    def link_name_list(self):
        return ["link"]

    def dof(self):
        return 2

    def order(self):
        return 3

    def import_motions(self, motion):
        self.imported_motion = motion

    def update_state_dict(self, order, is_dynamics=False, backend=None):
        self.update_calls.append((order, is_dynamics))

    def state_info(self, state_type):
        if state_type.data_type == "frame":
            return FakeFrame()
        if state_type.data_type == "vel":
            return (0.0, 0.0, 2.0, 10.0, 20.0, 30.0)
        if state_type.data_type == "acc":
            return (0.0, 0.0, 3.0, 100.0, 200.0, 300.0)
        if state_type.data_type == "force":
            return (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        raise AssertionError(state_type.data_type)

    def jacobian(self, state_type):
        return [[1.0, 0.0], [0.0, 1.0]]


def test_kots_backend_evaluates_link_point_quantities():
    kots = FakeKots()
    field = BodyField(robot_surface_model_from_kots(kots))
    field.register_backend(KotsBackend(kots, state_type_cls=FakeStateType))

    point = SurfacePoint("link", (0.5, 0.0, 0.0), "link")
    state = RobotState(q=[0.1, 0.2], dq=[0.3, 0.4], ddq=[0.5, 0.6])

    values = field.evaluate(
        [point],
        [
            QuantitySpec("geometry.position", output_type="vector3", frame="world"),
            QuantitySpec("kinematics.velocity", output_type="vector3", frame="world"),
            QuantitySpec("kinematics.acceleration", output_type="vector3", frame="world"),
            QuantitySpec("dynamics.force", output_type="wrench", frame="world"),
        ],
        state,
        backend="robokots.kots",
    )

    assert kots.imported_motion == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert values[0].value == (1.5, 2.0, 3.0)
    assert values[1].value == (10.0, 21.0, 30.0)
    assert values[2].value == (98.0, 201.5, 300.0)
    assert values[3].value == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
