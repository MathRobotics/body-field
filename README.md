# body-field

Common surface-field interface for evaluating quantities on robot body points.

`body-field` separates three concerns:

- `RobotSurfaceModel`: robot links and their surface meshes
- `SurfacePoint`: query points on links, in local/link/world coordinates
- `Backend`: an evaluator for quantities such as position, velocity, or obstacle distance

Example scripts live in [`examples/`](examples/README.md).

## Setup

```bash
uv sync
uv run pytest
```

## Basic Usage

Create a robot surface model, register a backend, then evaluate quantity specs at
surface points.

```python
from body_field import (
    BodyField,
    LinkState,
    LinkSurface,
    QuantitySpec,
    RobotSurfaceModel,
    SurfaceMesh,
    SurfacePoint,
)
from body_field.backends import NumpyPointFieldBackend


class StaticLinkStateProvider:
    name = "static"

    def compute_link_states(self, model, state=None):
        return {
            "link": LinkState(
                link_name="link",
                position=(1.0, 2.0, 3.0),
                rotation=(
                    (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                ),
                angular_velocity=(0.0, 0.0, 1.0),
                linear_velocity=(0.1, 0.0, 0.0),
            )
        }


model = RobotSurfaceModel(
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

field = BodyField(model)
field.register_backend(NumpyPointFieldBackend(StaticLinkStateProvider()))

points = [SurfacePoint("link", (0.5, 0.0, 0.0), "link")]
quantities = [
    QuantitySpec("geometry.position", output_type="vector3", frame="world"),
    QuantitySpec("kinematics.velocity", output_type="vector3", frame="world"),
]

values = field.evaluate(points, quantities, backend="point_field.numpy")
for value in values:
    print(value.spec.name, value.value, value.unit)
```

## Query Helper

For one point or a small set of points, use `field.at(...)` or
`field.at_points(...)`.

```python
point = SurfacePoint("link", (0.5, 0.0, 0.0), "link")
velocity_values = field.at(point).velocity()
```

## Obstacle Distance

Obstacle distance backends evaluate signed distances and closest points in the
world frame.

```python
from body_field import SphereObstacle
from body_field.backends import NumpyObstacleDistanceBackend

field.register_backend(
    NumpyObstacleDistanceBackend(
        StaticLinkStateProvider(),
        [SphereObstacle("fixture", center=(1.0, 2.5, 4.0), radius=0.25)],
    )
)

distance = field.at(SurfacePoint("link", (0.5, 0.0, 0.0), "link")).obstacle_distance()
print(distance[0].value)
print(distance[0].metadata["closest_obstacle"])
print(distance[0].metadata["closest_point"])
```

## Available Backends

- `NumpyPointFieldBackend`: CPU point-field quantities
- `TaichiPointFieldBackend`: Taichi point-field quantities
- `NumpyObstacleDistanceBackend`: signed distances to simple obstacles
- `RoboKotsBackend`: adapter for user-supplied RoboKots-style methods
- `KotsBackend`: direct integration with RoboKots `Kots`

## Supported Built-In Quantities

Point-field backends:

- `geometry.position`
- `kinematics.velocity`
- `kinematics.acceleration`
- `kinematics.speed`

Obstacle-distance backend:

- `geometry.obstacle.distance`
- `geometry.obstacle.closest_point`

RoboKots `KotsBackend`:

- `robokots.link.frame`
- `robokots.link.position`
- `robokots.link.rotation`
- `robokots.link.velocity`
- `robokots.link.acceleration`
- `robokots.link.jacobian`
- `geometry.position`
- `kinematics.velocity`
- `kinematics.acceleration`
- `kinematics.jacobian`
- `dynamics.momentum`
- `dynamics.force`

For backend design details, see [`docs/backend-interface-design.md`](docs/backend-interface-design.md).
