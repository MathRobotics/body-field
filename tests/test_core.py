from body_field import (
    BackendRegistry,
    BodyField,
    LinkSurface,
    ParallelProfile,
    QuantitySpec,
    QuantityValue,
    RobotSurfaceModel,
    SurfaceMesh,
    SurfacePoint,
)


class ConstantPreparedBackend:
    def evaluate(self, points, quantities, state=None):
        return [
            QuantityValue(
                spec=quantity,
                point=point,
                value=(1.0, 0.0, 0.0),
                frame=quantity.frame,
                unit=quantity.unit,
            )
            for point in points
            for quantity in quantities
        ]


class ConstantBackend:
    name = "constant"

    def supported_quantities(self):
        return {"geometry.normal"}

    def supports(self, model, quantity):
        return quantity.name in self.supported_quantities()

    def prepare(self, model):
        return ConstantPreparedBackend()

    def parallel_profile(self):
        return ParallelProfile(point_parallel=True, backend_kind="test")


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


def test_registry_selects_backend():
    registry = BackendRegistry()
    registry.register(ConstantBackend())

    backend = registry.select(
        sample_model(),
        QuantitySpec("geometry.normal", output_type="vector3"),
    )

    assert backend.name == "constant"


def test_body_field_evaluates_quantity():
    field = BodyField(sample_model())
    field.register_backend(ConstantBackend())

    values = field.at(SurfacePoint("link", (0.1, 0.1, 0.0), "link")).normal()

    assert len(values) == 1
    assert values[0].value == (1.0, 0.0, 0.0)


def test_registry_can_require_point_parallel_backend():
    registry = BackendRegistry()
    registry.register(ConstantBackend())

    backend = registry.select(
        sample_model(),
        QuantitySpec("geometry.normal", output_type="vector3"),
        require_parallel_axis="points",
    )

    assert backend.name == "constant"
