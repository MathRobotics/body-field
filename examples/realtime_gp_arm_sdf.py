"""Realtime per-link GP distance field for a serial capsule arm.

Run:
    uv run --extra viser python examples/realtime_gp_arm_sdf.py
"""

from __future__ import annotations

import os
import sys
import time
import webbrowser
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples._gp_arm_demo import (  # noqa: E402
    COARSE_GRID_SPACING,
    DURATION_SECONDS,
    SerialCapsuleArmStateProvider,
    add_arm_meshes,
    arm_links,
    build_arm_robot,
    build_coarse_grid_points,
    build_link_distance_fields,
    coarse_distance_opacities,
    distance_colors,
    query_coarse_distances,
    unit_cube_mesh,
    update_arm_meshes,
    viser_port,
)


FPS = 15.0


def main() -> None:
    try:
        import viser
    except ImportError as exc:
        raise ImportError(
            "This example requires viser. "
            "Run: uv run --extra viser python examples/realtime_gp_arm_sdf.py"
        ) from exc

    links = arm_links()
    provider = SerialCapsuleArmStateProvider(links)
    model = build_arm_robot(links)
    link_fields = build_link_distance_fields(links)
    coarse_grid_points = build_coarse_grid_points(provider, model)

    server = viser.ViserServer(port=viser_port())
    server.scene.add_grid(
        "/ground",
        width=3.0,
        height=3.0,
        cell_size=0.1,
        cell_thickness=0.005,
        position=(0.25, 0.0, -0.05),
    )
    server.scene.add_light_ambient("/ambient", intensity=0.8)
    server.scene.add_light_directional("/key_light", intensity=1.7, position=(2.0, -3.0, 4.0))

    pause_button = server.gui.add_button("Pause")
    show_field = server.gui.add_checkbox("Show coarse space field", True)
    show_robot = server.gui.add_checkbox("Show arm meshes", True)
    server.gui.add_markdown("Realtime per-link GP SDF. Yellow: near / Blue: far.")

    initial_distances = query_coarse_distances(
        coarse_grid_points,
        link_fields,
        provider,
        model,
        0.0,
    )
    cube_vertices, cube_faces = unit_cube_mesh()
    field_handle = server.scene.add_batched_meshes_simple(
        "/realtime_arm_sdf",
        cube_vertices,
        cube_faces,
        batched_wxyzs=np.tile(
            np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            (len(coarse_grid_points), 1),
        ),
        batched_positions=coarse_grid_points,
        batched_scales=np.full(
            (len(coarse_grid_points), 3),
            COARSE_GRID_SPACING * 0.92,
            dtype=np.float32,
        ),
        batched_colors=distance_colors(initial_distances),
        batched_opacities=coarse_distance_opacities(initial_distances),
        flat_shading=True,
        side="double",
        cast_shadow=False,
        receive_shadow=False,
    )
    robot_handles = add_arm_meshes(server, links)

    playing = True

    @pause_button.on_click
    def _(_) -> None:
        nonlocal playing
        playing = not playing
        pause_button.label = "Pause" if playing else "Play"

    @show_field.on_update
    def _(_) -> None:
        field_handle.visible = bool(show_field.value)

    @show_robot.on_update
    def _(_) -> None:
        for handle in robot_handles.values():
            handle.visible = bool(show_robot.value)

    url = f"http://localhost:{server.get_port()}"
    print(f"viser URL: {url}")
    if os.environ.get("BODY_FIELD_VISER_NO_OPEN") != "1":
        webbrowser.open(url)

    start = time.perf_counter()
    frame_period = 1.0 / FPS
    duration_s = float(os.environ.get("BODY_FIELD_VISER_DURATION", "0.0"))
    frame_count = 0
    last_report = start
    try:
        while True:
            now = time.perf_counter()
            if duration_s > 0.0 and now - start >= duration_s:
                break
            if playing:
                t = (now - start) % DURATION_SECONDS
                distances = query_coarse_distances(
                    coarse_grid_points,
                    link_fields,
                    provider,
                    model,
                    t,
                )
                field_handle.batched_colors = distance_colors(distances)
                field_handle.batched_opacities = coarse_distance_opacities(distances)
                update_arm_meshes(robot_handles, provider, model, t)
                frame_count += 1
                if now - last_report >= 2.0:
                    elapsed = now - start
                    print(f"realtime update rate: {frame_count / elapsed:.1f} fps")
                    last_report = now
            time.sleep(frame_period)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
