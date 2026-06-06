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

Animate obstacle clearances in Meshcat. The scene shows colored clearance
points, closest-obstacle lines, and obstacle-to-body-point vectors by default:

```bash
uv run python examples/realtime_obstacle_distance_meshcat.py
```

Tune drawing density and scale in `VisualizationConfig` at the top of
`examples/realtime_obstacle_distance_meshcat.py`.

## Meshcat Notes

Meshcat examples print a `Meshcat URL` when they start. Use `--no-open` if you
do not want the script to open a browser tab automatically.
