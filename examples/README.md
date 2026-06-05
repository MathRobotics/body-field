# Examples

Run examples from the repository root.

## Obstacle Distance

Compute signed distances from sampled robot body points to demo obstacles:

```bash
uv run python examples/obstacle_distance.py
```

## Point-Field Backend Comparison

Compare NumPy and Taichi point-field backends:

```bash
uv run python examples/compare_point_field_backends.py
```

## Meshcat Robot Quantities

Publish a static robot quantity scene to Meshcat:

```bash
uv run python examples/meshcat_robot_quantities.py
```

Useful options:

```bash
uv run python examples/meshcat_robot_quantities.py --no-open --no-block
uv run python examples/meshcat_robot_quantities.py --point-backend taichi
```

## Realtime Point Field

Animate a planar robot and update positions, velocities, speed colors, and
tensor ellipsoids:

```bash
uv run python examples/realtime_point_field_meshcat.py
```

Run for a fixed duration:

```bash
uv run python examples/realtime_point_field_meshcat.py --duration 10
```

## Realtime Obstacle Distance

Animate obstacle clearances in Meshcat:

```bash
uv run python examples/realtime_obstacle_distance_meshcat.py
```

Show closest-obstacle line segments:

```bash
uv run python examples/realtime_obstacle_distance_meshcat.py --show-closest-lines
```

## RoboKots Adapter Demo

Run the generic adapter demo with the built-in fake RoboKots object:

```bash
uv run python examples/robokots_surface_quantities.py
```

## RoboKots Kots Demo

Run against a local RoboKots checkout:

```bash
ROBOKOTS_ROOT=/path/to/RoboKots uv run python examples/robokots_kots_surface_quantities.py
```

If `ROBOKOTS_ROOT` is not set, the example uses:

```text
/Users/a896/Documents/MathRobotics/RoboKots
```

## Meshcat Notes

Meshcat examples print a `Meshcat URL` when they start. Use `--no-open` if you
do not want the script to open a browser tab automatically.
