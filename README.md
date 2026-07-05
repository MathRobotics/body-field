# body-field

Common surface-field interface for evaluating quantities on robot body points.

`body-field` separates three concerns:

- `RobotSurfaceModel`: robot links and their surface meshes
- `SurfacePoint`: query points on links, in local/link/world coordinates
- `Backend`: an evaluator for quantities such as position, velocity, or obstacle distance

## Features

- Point kinematics are implemented in both NumPy and Taichi backends.
- The NumPy / Taichi point-field backends evaluate world position, velocity,
  acceleration, and speed for many body points from link states.
- The Jacobian backend evaluates point Jacobians and manipulability axes from a
  provider that supplies one Jacobian per body point.
- The obstacle-distance backend evaluates signed distances and closest points to
  simple obstacles. It can also return the vector field from the closest obstacle
  surface point to each body point.
- Gaussian-process Euclidean distance fields approximate distances from sparse
  surface samples and are useful for fast local distance queries.
- The NumPy / Taichi self-collision backends evaluate nearest-neighbor distances
  between sampled body points, excluding points on the same link by default.

Example scripts live in [`examples/`](examples/README.md).

## Setup

```bash
uv sync --extra test
uv run pytest
```

Taichi backends and Meshcat examples are optional:

```bash
uv sync --extra test --extra taichi --extra viz
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

Obstacle distance backends evaluate signed distances, closest points, and
obstacle-to-point vectors in the world frame.

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
vector = field.at(SurfacePoint("link", (0.5, 0.0, 0.0), "link")).obstacle_vector()
normal = field.at(SurfacePoint("link", (0.5, 0.0, 0.0), "link")).obstacle_normal()
print(distance[0].value)
print(vector[0].value)
print(normal[0].value)
print(distance[0].metadata["closest_obstacle"])
print(distance[0].metadata["closest_point"])
```

## Gaussian Process Distance Field

Use `GaussianProcessDistanceField` when you want a fast Euclidean distance
approximation from sparse surface samples:

```python
from body_field import GaussianProcessDistanceField

field = GaussianProcessDistanceField(surface_samples, a=400.0)
result = field.query(query_points)
print(result.distance)
print(result.normal)
print(result.nearest)
```

## Self Collision

Self-collision distances are computed between the query body points you pass in.
By default, points on the same link are ignored. Set `point_radius` to return a
signed clearance between equal-radius point proxies.

```python
from body_field import evaluate_minimum_self_collision
from body_field.backends import NumpySelfCollisionBackend

field.register_backend(NumpySelfCollisionBackend(StaticLinkStateProvider(), point_radius=0.02))

points = [
    SurfacePoint("left_link", (0.0, 0.0, 0.0), "link"),
    SurfacePoint("right_link", (0.03, 0.0, 0.0), "link"),
]

distance = field.at_points(points).self_collision_distance()
vector = field.at_points(points).self_collision_vector()
print(distance[0].value)
print(vector[0].metadata["closest_link"])
print(vector[0].metadata["closest_point_index"])

minimum = evaluate_minimum_self_collision(field, points)
print(minimum.signed_distance)
print(minimum.point_index, minimum.closest_point_index)
print(minimum.normal)
```

For fast joint-angle searches, minimize `minimum.signed_distance`. If a Jacobian
backend is registered, pass `jacobian_backend="jacobian.numpy"` to also compute
`minimum.distance_gradient`, approximated as `n.T @ (J_point - J_closest)`.
Register `TaichiSelfCollisionBackend` and select `backend="self_collision.taichi"`
in `field.evaluate(...)` to use the Taichi implementation directly.

## Available Backends

- `NumpyPointFieldBackend`: CPU point-field quantities
- `TaichiPointFieldBackend`: Taichi point-field quantities; requires the `taichi` extra
- `NumpyPointJacobianBackend`: point Jacobians and manipulability axes
- `NumpyObstacleDistanceBackend`: signed distances to simple obstacles
- `GaussianProcessDistanceField`: approximate Euclidean distance queries from
  sparse surface samples
- `NumpySelfCollisionBackend`: nearest distances between sampled body points
- `TaichiSelfCollisionBackend`: Taichi nearest distances between sampled body points; requires the `taichi` extra

## Supported Built-In Quantities

Point-field backends:

- `geometry.position`
- `kinematics.velocity`
- `kinematics.acceleration`
- `kinematics.speed`

Jacobian backend:

- `kinematics.jacobian`
- `kinematics.manipulability.axes`

Obstacle-distance backend:

- `geometry.obstacle.distance`
- `geometry.obstacle.closest_point`
- `geometry.obstacle.normal`
- `geometry.obstacle.vector`

Self-collision backend:

- `geometry.self_collision.distance`
- `geometry.self_collision.closest_point`
- `geometry.self_collision.vector`

For backend design details, see [`docs/backend-interface-design.md`](docs/backend-interface-design.md).
