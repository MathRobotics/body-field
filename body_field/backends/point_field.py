from dataclasses import dataclass
from typing import Literal

import numpy as np

from body_field.backends._surface_points import pack_surface_point_transforms
from body_field.backends._taichi import import_taichi, init_taichi
from body_field.core import (
    Backend,
    LinkState,
    LinkStateProvider,
    ParallelProfile,
    PreparedBackend,
    QuantitySpec,
    QuantityValue,
    RobotState,
    RobotSurfaceModel,
    SurfacePoint,
)


SUPPORTED_POINT_FIELD_QUANTITIES = {
    "geometry.position",
    "kinematics.velocity",
    "kinematics.acceleration",
    "kinematics.speed",
}


@dataclass(frozen=True)
class NumpyPointFieldBackend(Backend):
    link_state_provider: LinkStateProvider
    name: str = "point_field.numpy"

    def supported_quantities(self) -> set[str]:
        return set(SUPPORTED_POINT_FIELD_QUANTITIES)

    def supports(self, model: RobotSurfaceModel, quantity: QuantitySpec) -> bool:
        return quantity.name in self.supported_quantities()

    def prepare(self, model: RobotSurfaceModel) -> PreparedBackend:
        return PreparedNumpyPointFieldBackend(model, self.link_state_provider, self.name)

    def parallel_profile(self) -> ParallelProfile:
        return ParallelProfile(
            point_parallel=True,
            quantity_parallel=False,
            device="cpu",
            backend_kind="numpy-vectorized",
            preferred_min_points=1,
        )


@dataclass(frozen=True)
class TaichiPointFieldBackend(Backend):
    link_state_provider: LinkStateProvider
    arch: Literal["auto", "cpu", "gpu", "metal", "vulkan", "cuda"] = "auto"
    name: str = "point_field.taichi"

    def supported_quantities(self) -> set[str]:
        return set(SUPPORTED_POINT_FIELD_QUANTITIES)

    def supports(self, model: RobotSurfaceModel, quantity: QuantitySpec) -> bool:
        return quantity.name in self.supported_quantities()

    def prepare(self, model: RobotSurfaceModel) -> PreparedBackend:
        taichi = import_taichi()
        init_taichi(taichi, self.arch)
        return PreparedTaichiPointFieldBackend(model, self.link_state_provider, self.name)

    def parallel_profile(self) -> ParallelProfile:
        return ParallelProfile(
            point_parallel=True,
            quantity_parallel=False,
            device="auto" if self.arch == "auto" else self.arch,
            backend_kind="taichi",
            preferred_min_points=10_000,
        )


@dataclass
class PreparedNumpyPointFieldBackend:
    model: RobotSurfaceModel
    link_state_provider: LinkStateProvider
    backend_name: str

    def evaluate(
        self,
        points: list[SurfacePoint],
        quantities: list[QuantitySpec],
        state: RobotState | None = None,
    ) -> list[QuantityValue]:
        _require_world_quantities(quantities)
        link_states = self.link_state_provider.compute_link_states(self.model, state)
        arrays = _pack_inputs(self.model, points, link_states)
        outputs = _evaluate_numpy(arrays)
        return _collect_values(points, quantities, outputs, self.backend_name)


@dataclass
class PreparedTaichiPointFieldBackend:
    model: RobotSurfaceModel
    link_state_provider: LinkStateProvider
    backend_name: str

    def evaluate(
        self,
        points: list[SurfacePoint],
        quantities: list[QuantitySpec],
        state: RobotState | None = None,
    ) -> list[QuantityValue]:
        _require_world_quantities(quantities)
        link_states = self.link_state_provider.compute_link_states(self.model, state)
        arrays = _pack_inputs(self.model, points, link_states)
        outputs = _evaluate_taichi(arrays)
        return _collect_values(points, quantities, outputs, self.backend_name)


@dataclass(frozen=True)
class _PackedPointFieldInputs:
    point_link_ids: np.ndarray
    point_local: np.ndarray
    point_is_world: np.ndarray
    world_position: np.ndarray
    link_offset: np.ndarray
    link_position: np.ndarray
    link_rotation: np.ndarray
    link_angular_velocity: np.ndarray
    link_linear_velocity: np.ndarray
    link_angular_acceleration: np.ndarray
    link_linear_acceleration: np.ndarray


@dataclass(frozen=True)
class _PointFieldOutputs:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    speed: np.ndarray


def _pack_inputs(
    model: RobotSurfaceModel,
    points: list[SurfacePoint],
    link_states: dict[str, LinkState],
) -> _PackedPointFieldInputs:
    transform_arrays = pack_surface_point_transforms(model, points, link_states)
    link_names = list(transform_arrays.link_names)
    link_ids = {name: index for index, name in enumerate(link_names)}

    missing = [name for name in link_names if name not in link_states]
    if missing:
        raise ValueError(f"Missing link states for: {', '.join(missing)}")

    point_local = np.empty((len(points), 3), dtype=np.float32)
    point_is_world = np.empty(len(points), dtype=np.int32)

    for index, point in enumerate(points):
        point_local[index] = np.asarray(point.position, dtype=np.float32)
        point_is_world[index] = 1 if point.frame == "world" else 0

    def link_array(attr: str, shape: tuple[int, ...]) -> np.ndarray:
        out = np.empty((len(link_names), *shape), dtype=np.float32)
        for name, link_id in link_ids.items():
            out[link_id] = np.asarray(getattr(link_states[name], attr), dtype=np.float32)
        return out

    return _PackedPointFieldInputs(
        point_link_ids=transform_arrays.point_link_ids,
        point_local=point_local,
        point_is_world=point_is_world,
        world_position=transform_arrays.world_position,
        link_offset=transform_arrays.link_offset,
        link_position=link_array("position", (3,)),
        link_rotation=link_array("rotation", (3, 3)),
        link_angular_velocity=link_array("angular_velocity", (3,)),
        link_linear_velocity=link_array("linear_velocity", (3,)),
        link_angular_acceleration=link_array("angular_acceleration", (3,)),
        link_linear_acceleration=link_array("linear_acceleration", (3,)),
    )


def _evaluate_numpy(arrays: _PackedPointFieldInputs) -> _PointFieldOutputs:
    link_ids = arrays.point_link_ids
    position = arrays.world_position
    offset = arrays.link_offset

    omega = arrays.link_angular_velocity[link_ids]
    linear_velocity = arrays.link_linear_velocity[link_ids]
    alpha = arrays.link_angular_acceleration[link_ids]
    linear_acceleration = arrays.link_linear_acceleration[link_ids]

    velocity = linear_velocity + np.cross(omega, offset)
    acceleration = (
        linear_acceleration
        + np.cross(alpha, offset)
        + np.cross(omega, np.cross(omega, offset))
    )
    speed = np.linalg.norm(velocity, axis=1)
    return _PointFieldOutputs(position, velocity, acceleration, speed)


def _evaluate_taichi(arrays: _PackedPointFieldInputs) -> _PointFieldOutputs:
    n = arrays.point_link_ids.shape[0]
    position = np.empty((n, 3), dtype=np.float32)
    velocity = np.empty((n, 3), dtype=np.float32)
    acceleration = np.empty((n, 3), dtype=np.float32)
    speed = np.empty(n, dtype=np.float32)
    _taichi_eval_points(
        arrays.point_link_ids,
        arrays.point_local,
        arrays.point_is_world,
        arrays.link_position,
        arrays.link_rotation,
        arrays.link_angular_velocity,
        arrays.link_linear_velocity,
        arrays.link_angular_acceleration,
        arrays.link_linear_acceleration,
        position,
        velocity,
        acceleration,
        speed,
    )
    return _PointFieldOutputs(position, velocity, acceleration, speed)


def _collect_values(
    points: list[SurfacePoint],
    quantities: list[QuantitySpec],
    outputs: _PointFieldOutputs,
    backend_name: str,
) -> list[QuantityValue]:
    values: list[QuantityValue] = []
    for quantity in quantities:
        for index, point in enumerate(points):
            if quantity.name == "geometry.position":
                value = _tuple3(outputs.position[index])
                unit = quantity.unit or "m"
            elif quantity.name == "kinematics.velocity":
                value = _tuple3(outputs.velocity[index])
                unit = quantity.unit or "m/s"
            elif quantity.name == "kinematics.acceleration":
                value = _tuple3(outputs.acceleration[index])
                unit = quantity.unit or "m/s^2"
            elif quantity.name == "kinematics.speed":
                value = float(outputs.speed[index])
                unit = quantity.unit or "m/s"
            else:
                raise ValueError(f"{backend_name} does not support {quantity.name}")

            values.append(
                QuantityValue(
                    spec=quantity,
                    point=point,
                    value=value,
                    frame=quantity.frame or "world",
                    unit=unit,
                    metadata={
                        "backend": backend_name,
                        "world_position": _tuple3(outputs.position[index]),
                    },
                )
            )
    return values


def _tuple3(array: np.ndarray) -> tuple[float, float, float]:
    return (float(array[0]), float(array[1]), float(array[2]))


def _require_world_quantities(quantities: list[QuantitySpec]) -> None:
    for quantity in quantities:
        if quantity.frame not in {None, "world"}:
            raise ValueError(f"{quantity.name} supports frame='world' only")
        if quantity.name not in SUPPORTED_POINT_FIELD_QUANTITIES:
            raise ValueError(f"Unsupported point-field quantity: {quantity.name}")


def _taichi_eval_points(*args):
    ti = import_taichi()

    @ti.kernel
    def kernel(
        point_link_ids: ti.types.ndarray(dtype=ti.i32, ndim=1),
        point_local: ti.types.ndarray(dtype=ti.f32, ndim=2),
        point_is_world: ti.types.ndarray(dtype=ti.i32, ndim=1),
        link_position: ti.types.ndarray(dtype=ti.f32, ndim=2),
        link_rotation: ti.types.ndarray(dtype=ti.f32, ndim=3),
        link_angular_velocity: ti.types.ndarray(dtype=ti.f32, ndim=2),
        link_linear_velocity: ti.types.ndarray(dtype=ti.f32, ndim=2),
        link_angular_acceleration: ti.types.ndarray(dtype=ti.f32, ndim=2),
        link_linear_acceleration: ti.types.ndarray(dtype=ti.f32, ndim=2),
        out_position: ti.types.ndarray(dtype=ti.f32, ndim=2),
        out_velocity: ti.types.ndarray(dtype=ti.f32, ndim=2),
        out_acceleration: ti.types.ndarray(dtype=ti.f32, ndim=2),
        out_speed: ti.types.ndarray(dtype=ti.f32, ndim=1),
    ):
        for i in range(point_link_ids.shape[0]):
            link_id = point_link_ids[i]

            ox = 0.0
            oy = 0.0
            oz = 0.0
            if point_is_world[i] == 1:
                ox = point_local[i, 0] - link_position[link_id, 0]
                oy = point_local[i, 1] - link_position[link_id, 1]
                oz = point_local[i, 2] - link_position[link_id, 2]
            else:
                ox = (
                    link_rotation[link_id, 0, 0] * point_local[i, 0]
                    + link_rotation[link_id, 0, 1] * point_local[i, 1]
                    + link_rotation[link_id, 0, 2] * point_local[i, 2]
                )
                oy = (
                    link_rotation[link_id, 1, 0] * point_local[i, 0]
                    + link_rotation[link_id, 1, 1] * point_local[i, 1]
                    + link_rotation[link_id, 1, 2] * point_local[i, 2]
                )
                oz = (
                    link_rotation[link_id, 2, 0] * point_local[i, 0]
                    + link_rotation[link_id, 2, 1] * point_local[i, 1]
                    + link_rotation[link_id, 2, 2] * point_local[i, 2]
                )

            px = link_position[link_id, 0] + ox
            py = link_position[link_id, 1] + oy
            pz = link_position[link_id, 2] + oz

            wx = link_angular_velocity[link_id, 0]
            wy = link_angular_velocity[link_id, 1]
            wz = link_angular_velocity[link_id, 2]
            vx = link_linear_velocity[link_id, 0] + wy * oz - wz * oy
            vy = link_linear_velocity[link_id, 1] + wz * ox - wx * oz
            vz = link_linear_velocity[link_id, 2] + wx * oy - wy * ox

            ax = link_angular_acceleration[link_id, 0]
            ay = link_angular_acceleration[link_id, 1]
            az = link_angular_acceleration[link_id, 2]
            alpha_cross_x = ay * oz - az * oy
            alpha_cross_y = az * ox - ax * oz
            alpha_cross_z = ax * oy - ay * ox
            omega_cross_x = wy * oz - wz * oy
            omega_cross_y = wz * ox - wx * oz
            omega_cross_z = wx * oy - wy * ox
            omega_omega_x = wy * omega_cross_z - wz * omega_cross_y
            omega_omega_y = wz * omega_cross_x - wx * omega_cross_z
            omega_omega_z = wx * omega_cross_y - wy * omega_cross_x

            acc_x = link_linear_acceleration[link_id, 0] + alpha_cross_x + omega_omega_x
            acc_y = link_linear_acceleration[link_id, 1] + alpha_cross_y + omega_omega_y
            acc_z = link_linear_acceleration[link_id, 2] + alpha_cross_z + omega_omega_z

            out_position[i, 0] = px
            out_position[i, 1] = py
            out_position[i, 2] = pz
            out_velocity[i, 0] = vx
            out_velocity[i, 1] = vy
            out_velocity[i, 2] = vz
            out_acceleration[i, 0] = acc_x
            out_acceleration[i, 1] = acc_y
            out_acceleration[i, 2] = acc_z
            out_speed[i] = ti.sqrt(vx * vx + vy * vy + vz * vz)

    kernel(*args)
