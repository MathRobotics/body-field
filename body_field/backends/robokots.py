from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from body_field.core import (
    PreparedBackend,
    ParallelProfile,
    QuantitySpec,
    QuantityValue,
    RobotState,
    LinkSurface,
    RobotSurfaceModel,
    SurfaceMesh,
    SurfacePoint,
)


RoboKotsHandler = Callable[
    [object, RobotSurfaceModel, SurfacePoint, QuantitySpec, RobotState | None],
    Any,
]


def robokots_method(method_name: str) -> RoboKotsHandler:
    """Map a quantity to a method on the wrapped RoboKots object.

    The target method receives keyword arguments:
    link_name, position, frame, state, and QuantitySpec.params.
    """

    def handler(
        client: object,
        model: RobotSurfaceModel,
        point: SurfacePoint,
        spec: QuantitySpec,
        state: RobotState | None,
    ) -> Any:
        model.require_link(point.link_name)
        method = getattr(client, method_name)
        return method(
            link_name=point.link_name,
            position=point.position,
            frame=point.frame,
            state=state,
            **spec.params,
        )

    return handler


@dataclass(frozen=True)
class RoboKotsBackend:
    client: object
    handlers: Mapping[str, RoboKotsHandler]
    units: Mapping[str, str | None] | None = None
    name: str = "robokots"

    def supported_quantities(self) -> set[str]:
        return set(self.handlers)

    def supports(self, model: RobotSurfaceModel, quantity: QuantitySpec) -> bool:
        return quantity.name in self.handlers

    def prepare(self, model: RobotSurfaceModel) -> PreparedBackend:
        return PreparedRoboKotsBackend(
            model=model,
            client=self.client,
            handlers=self.handlers,
            units=self.units or {},
            backend_name=self.name,
        )

    def parallel_profile(self) -> ParallelProfile:
        return ParallelProfile(backend_kind="robokots-adapter")


@dataclass
class PreparedRoboKotsBackend:
    model: RobotSurfaceModel
    client: object
    handlers: Mapping[str, RoboKotsHandler]
    units: Mapping[str, str | None]
    backend_name: str

    def evaluate(
        self,
        points: list[SurfacePoint],
        quantities: list[QuantitySpec],
        state: RobotState | None = None,
    ) -> list[QuantityValue]:
        values: list[QuantityValue] = []

        for point in points:
            self.model.require_link(point.link_name)
            for quantity in quantities:
                handler = self.handlers.get(quantity.name)
                if handler is None:
                    raise ValueError(f"{self.backend_name} does not support {quantity.name}")

                raw_value = handler(self.client, self.model, point, quantity, state)
                values.append(
                    QuantityValue(
                        spec=quantity,
                        point=point,
                        value=raw_value,
                        frame=quantity.frame or point.frame,
                        unit=quantity.unit
                        if quantity.unit is not None
                        else self.units.get(quantity.name),
                        metadata={"backend": self.backend_name},
                    )
                )

        return values


def robot_surface_model_from_kots(
    kots: object,
    *,
    name: str = "robokots_robot",
    root_frame: str = "world",
) -> RobotSurfaceModel:
    """Create a minimal surface model from a RoboKots Kots instance.

    RoboKots' current model format primarily exposes links and kinematic/dynamic
    state. Mesh loading can be layered on later; this model is enough to validate
    link names and evaluate point quantities whose local coordinates are supplied
    by SurfacePoint.
    """

    link_names = list(kots.link_name_list())
    empty_mesh = SurfaceMesh(vertices=[], faces=[])
    return RobotSurfaceModel(
        name=name,
        root_frame=root_frame,
        links={
            link_name: LinkSurface(link_name=link_name, mesh=empty_mesh)
            for link_name in link_names
        },
        metadata={"source": "robokots"},
    )


@dataclass(frozen=True)
class KotsBackend:
    """RoboKots Kots backend for link and link-point quantities."""

    kots: object
    name: str = "robokots.kots"
    state_type_cls: type[Any] | None = None

    def supported_quantities(self) -> set[str]:
        return {
            "robokots.link.frame",
            "robokots.link.position",
            "robokots.link.rotation",
            "robokots.link.velocity",
            "robokots.link.acceleration",
            "robokots.link.jacobian",
            "geometry.position",
            "kinematics.velocity",
            "kinematics.acceleration",
            "kinematics.jacobian",
            "dynamics.momentum",
            "dynamics.force",
        }

    def supports(self, model: RobotSurfaceModel, quantity: QuantitySpec) -> bool:
        return quantity.name in self.supported_quantities()

    def prepare(self, model: RobotSurfaceModel) -> PreparedBackend:
        return PreparedKotsBackend(
            model=model,
            kots=self.kots,
            backend_name=self.name,
            state_type_cls=self.state_type_cls,
        )

    def parallel_profile(self) -> ParallelProfile:
        return ParallelProfile(backend_kind="robokots-kots")


@dataclass
class PreparedKotsBackend:
    model: RobotSurfaceModel
    kots: object
    backend_name: str
    state_type_cls: type[Any] | None = None

    def evaluate(
        self,
        points: list[SurfacePoint],
        quantities: list[QuantitySpec],
        state: RobotState | None = None,
    ) -> list[QuantityValue]:
        self._apply_state(state)
        values: list[QuantityValue] = []

        for point in points:
            self.model.require_link(point.link_name)
            for quantity in quantities:
                value = self._evaluate_one(point, quantity)
                values.append(
                    QuantityValue(
                        spec=quantity,
                        point=point,
                        value=value,
                        frame=quantity.frame or "world",
                        unit=quantity.unit,
                        metadata={"backend": self.backend_name},
                    )
                )

        return values

    def _evaluate_one(self, point: SurfacePoint, quantity: QuantitySpec) -> Any:
        if quantity.name == "robokots.link.frame":
            return self._link_state(point.link_name, "frame", quantity.frame)
        if quantity.name == "robokots.link.position":
            return self._link_state(point.link_name, "pos", quantity.frame)
        if quantity.name == "robokots.link.rotation":
            return self._link_state(point.link_name, "rot", quantity.frame)
        if quantity.name == "robokots.link.velocity":
            return self._link_state(point.link_name, "vel", quantity.frame)
        if quantity.name == "robokots.link.acceleration":
            return self._link_state(point.link_name, "acc", quantity.frame)
        if quantity.name in {"robokots.link.jacobian", "kinematics.jacobian"}:
            data_type = quantity.params.get("data_type", "frame")
            return self.kots.jacobian(self._state_type(point.link_name, data_type, quantity.frame))
        if quantity.name == "geometry.position":
            return self._point_position(point)
        if quantity.name == "kinematics.velocity":
            return self._point_velocity(point, quantity.frame)
        if quantity.name == "kinematics.acceleration":
            return self._point_acceleration(point, quantity.frame)
        if quantity.name == "dynamics.momentum":
            return self._link_state(point.link_name, "momentum", quantity.frame, is_dynamics=True)
        if quantity.name == "dynamics.force":
            return self._link_state(point.link_name, "force", quantity.frame, is_dynamics=True)

        raise ValueError(f"{self.backend_name} does not support {quantity.name}")

    def _apply_state(self, state: RobotState | None) -> None:
        if state is None:
            self.kots.update_state_dict(order=self.kots.order(), is_dynamics=False)
            return

        motion = _robot_state_to_kots_motion(state, dof=self.kots.dof(), order=self.kots.order())
        self.kots.import_motions(motion)
        self.kots.update_state_dict(order=self.kots.order(), is_dynamics=False)

    def _link_state(
        self,
        link_name: str,
        data_type: str,
        frame_name: str | None,
        *,
        is_dynamics: bool = False,
    ) -> Any:
        if is_dynamics:
            self.kots.update_state_dict(order=self.kots.order(), is_dynamics=True)
        state_type = self._state_type(link_name, data_type, frame_name)
        return self.kots.state_info(state_type)

    def _state_type(self, link_name: str, data_type: str, frame_name: str | None) -> Any:
        state_type_cls = self.state_type_cls or _import_robokots_state_type()
        return state_type_cls(
            owner_type="link",
            owner_name=link_name,
            data_type=data_type,
            frame_name=frame_name,
        )

    def _point_position(self, point: SurfacePoint) -> tuple[float, float, float]:
        if point.frame == "world":
            return point.position

        frame = self._link_state(point.link_name, "frame", None)
        origin = _se3_position(frame)
        rotation = _se3_rotation(frame)
        local = _point_local_vector(point)
        return _add(origin, _matvec(rotation, local))

    def _point_velocity(
        self,
        point: SurfacePoint,
        frame_name: str | None,
    ) -> tuple[float, float, float]:
        if frame_name not in {None, "world"}:
            raise ValueError("KotsBackend point velocity currently supports frame='world' only")
        twist = _as_six_vector(self._link_state(point.link_name, "vel", "world"))
        omega = twist[:3]
        linear = twist[3:]
        r_world = self._point_offset_world(point)
        return _add(linear, _cross(omega, r_world))

    def _point_acceleration(
        self,
        point: SurfacePoint,
        frame_name: str | None,
    ) -> tuple[float, float, float]:
        if frame_name not in {None, "world"}:
            raise ValueError("KotsBackend point acceleration currently supports frame='world' only")
        twist = _as_six_vector(self._link_state(point.link_name, "vel", "world"))
        accel = _as_six_vector(self._link_state(point.link_name, "acc", "world"))
        omega = twist[:3]
        alpha = accel[:3]
        linear_acc = accel[3:]
        r_world = self._point_offset_world(point)
        return _add(linear_acc, _add(_cross(alpha, r_world), _cross(omega, _cross(omega, r_world))))

    def _point_offset_world(self, point: SurfacePoint) -> tuple[float, float, float]:
        frame = self._link_state(point.link_name, "frame", None)
        origin = _se3_position(frame)

        if point.frame == "world":
            return _sub(point.position, origin)

        rotation = _se3_rotation(frame)
        return _matvec(rotation, _point_local_vector(point))


def _import_robokots_state_type() -> type[Any]:
    try:
        from robokots.core.state import StateType
    except ImportError as exc:
        raise ImportError(
            "KotsBackend requires RoboKots. Install it or pass state_type_cls for tests."
        ) from exc
    return StateType


def _robot_state_to_kots_motion(state: RobotState, *, dof: int, order: int) -> list[float]:
    if state.q is None:
        return [0.0] * (dof * order)

    q = _as_list(state.q)
    if len(q) == dof * order and state.dq is None and state.ddq is None:
        return q

    parts = [q, _as_list(state.dq), _as_list(state.ddq)]
    motion: list[float] = []
    for index in range(order):
        part = parts[index] if index < len(parts) and parts[index] else [0.0] * dof
        if len(part) != dof:
            raise ValueError(f"RobotState derivative {index} must have length {dof}, got {len(part)}")
        motion.extend(float(value) for value in part)
    return motion


def _as_list(value: Any | None) -> list[float]:
    if value is None:
        return []
    return [float(v) for v in value]


def _point_local_vector(point: SurfacePoint) -> tuple[float, float, float]:
    if point.frame not in {point.link_name, "local", "link"}:
        raise ValueError(
            "Point quantities expect SurfacePoint.position in the link frame "
            f"or world frame, got frame={point.frame!r}"
        )
    return point.position


def _se3_position(frame: Any) -> tuple[float, float, float]:
    if hasattr(frame, "pos"):
        return _as_vector3(frame.pos())
    if hasattr(frame, "mat"):
        mat = frame.mat()
        return (float(mat[0][3]), float(mat[1][3]), float(mat[2][3]))
    raise TypeError(f"Cannot extract position from {type(frame).__name__}")


def _se3_rotation(frame: Any) -> tuple[tuple[float, float, float], ...]:
    if hasattr(frame, "rot"):
        return _as_matrix3(frame.rot())
    if hasattr(frame, "mat"):
        mat = frame.mat()
        return _as_matrix3([row[:3] for row in mat[:3]])
    raise TypeError(f"Cannot extract rotation from {type(frame).__name__}")


def _as_vector3(value: Any) -> tuple[float, float, float]:
    return (float(value[0]), float(value[1]), float(value[2]))


def _as_six_vector(value: Any) -> tuple[float, float, float, float, float, float]:
    return (
        float(value[0]),
        float(value[1]),
        float(value[2]),
        float(value[3]),
        float(value[4]),
        float(value[5]),
    )


def _as_matrix3(value: Any) -> tuple[tuple[float, float, float], ...]:
    return (
        (float(value[0][0]), float(value[0][1]), float(value[0][2])),
        (float(value[1][0]), float(value[1][1]), float(value[1][2])),
        (float(value[2][0]), float(value[2][1]), float(value[2][2])),
    )


def _matvec(
    matrix: tuple[tuple[float, float, float], ...],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    )


def _add(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def _sub(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def _cross(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )
