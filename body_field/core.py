from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


Vector3 = tuple[float, float, float]
Matrix3 = tuple[Vector3, Vector3, Vector3]
OutputType = Literal["scalar", "vector3", "tensor3", "matrix", "wrench", "custom"]
MeshRole = Literal["visual", "collision", "custom"]
ParallelAxis = Literal["points", "quantities"]
DeviceKind = Literal["cpu", "gpu", "auto", "unknown"]


@dataclass(frozen=True)
class SurfaceMesh:
    vertices: list[Vector3]
    faces: list[tuple[int, int, int]]


@dataclass(frozen=True)
class LinkSurface:
    link_name: str
    mesh: SurfaceMesh
    role: MeshRole = "collision"


@dataclass(frozen=True)
class RobotSurfaceModel:
    name: str
    links: dict[str, LinkSurface]
    root_frame: str = "world"
    metadata: dict[str, Any] = field(default_factory=dict)

    def require_link(self, link_name: str) -> LinkSurface:
        try:
            return self.links[link_name]
        except KeyError as exc:
            known = ", ".join(sorted(self.links)) or "<none>"
            raise ValueError(f"Unknown link: {link_name}. Known links: {known}") from exc


@dataclass(frozen=True)
class SurfacePoint:
    link_name: str
    position: Vector3
    frame: str
    triangle_id: int | None = None
    barycentric: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class RobotState:
    q: Any | None = None
    dq: Any | None = None
    ddq: Any | None = None
    time: float | None = None


@dataclass(frozen=True)
class QuantitySpec:
    name: str
    output_type: OutputType
    frame: str | None = None
    unit: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QuantityValue:
    spec: QuantitySpec
    point: SurfacePoint
    value: Any
    frame: str | None = None
    unit: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LinkState:
    link_name: str
    position: Vector3
    rotation: Matrix3
    angular_velocity: Vector3
    linear_velocity: Vector3
    angular_acceleration: Vector3 = (0.0, 0.0, 0.0)
    linear_acceleration: Vector3 = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class ParallelProfile:
    point_parallel: bool = False
    quantity_parallel: bool = False
    device: DeviceKind = "unknown"
    backend_kind: str = "python"
    preferred_min_points: int = 0
    preferred_min_quantities: int = 0

    def supports_axis(self, axis: ParallelAxis) -> bool:
        if axis == "points":
            return self.point_parallel
        if axis == "quantities":
            return self.quantity_parallel
        raise ValueError(f"Unknown parallel axis: {axis}")


class PreparedBackend(Protocol):
    def evaluate(
        self,
        points: list[SurfacePoint],
        quantities: list[QuantitySpec],
        state: RobotState | None = None,
    ) -> list[QuantityValue]:
        ...


class Backend(Protocol):
    name: str

    def supported_quantities(self) -> set[str]:
        ...

    def supports(self, model: RobotSurfaceModel, quantity: QuantitySpec) -> bool:
        ...

    def prepare(self, model: RobotSurfaceModel) -> PreparedBackend:
        ...

    def parallel_profile(self) -> ParallelProfile:
        ...


class LinkStateProvider(Protocol):
    name: str

    def compute_link_states(
        self,
        model: RobotSurfaceModel,
        state: RobotState | None = None,
    ) -> dict[str, LinkState]:
        ...


class SerialBackendMixin:
    def parallel_profile(self) -> ParallelProfile:
        return ParallelProfile()


class BackendRegistry:
    def __init__(self) -> None:
        self._backends: dict[str, Backend] = {}

    def register(self, backend: Backend) -> None:
        if backend.name in self._backends:
            raise ValueError(f"Backend already registered: {backend.name}")
        self._backends[backend.name] = backend

    def get(self, name: str) -> Backend:
        try:
            return self._backends[name]
        except KeyError as exc:
            known = ", ".join(sorted(self._backends)) or "<none>"
            raise ValueError(f"Unknown backend: {name}. Known backends: {known}") from exc

    def select(
        self,
        model: RobotSurfaceModel,
        quantity: QuantitySpec,
        preferred: str | None = None,
        require_parallel_axis: ParallelAxis | None = None,
    ) -> Backend:
        if preferred is not None:
            backend = self.get(preferred)
            if not backend.supports(model, quantity):
                raise ValueError(f"{preferred} does not support {quantity.name}")
            if require_parallel_axis and not _backend_supports_axis(backend, require_parallel_axis):
                raise ValueError(f"{preferred} does not support {require_parallel_axis} parallelism")
            return backend

        for backend in self._backends.values():
            if backend.supports(model, quantity) and (
                require_parallel_axis is None
                or _backend_supports_axis(backend, require_parallel_axis)
            ):
                return backend

        raise ValueError(f"No backend supports {quantity.name}")

    def names(self) -> list[str]:
        return list(self._backends)


def default_backend_registry() -> BackendRegistry:
    return BackendRegistry()


class BodyField:
    def __init__(
        self,
        model: RobotSurfaceModel,
        registry: BackendRegistry | None = None,
    ) -> None:
        self.model = model
        self.registry = registry or default_backend_registry()
        self._prepared: dict[str, PreparedBackend] = {}

    def register_backend(self, backend: Backend) -> None:
        self.registry.register(backend)

    def at(
        self,
        point: SurfacePoint,
        state: RobotState | None = None,
    ) -> SurfaceFieldQuery:
        return SurfaceFieldQuery(self, [point], state)

    def at_points(
        self,
        points: list[SurfacePoint],
        state: RobotState | None = None,
    ) -> SurfaceFieldQuery:
        return SurfaceFieldQuery(self, points, state)

    def evaluate(
        self,
        points: list[SurfacePoint],
        quantities: list[QuantitySpec],
        state: RobotState | None = None,
        backend: str | None = None,
        require_parallel_axis: ParallelAxis | None = None,
    ) -> list[QuantityValue]:
        for point in points:
            self.model.require_link(point.link_name)

        values: list[QuantityValue] = []
        selected_batches: list[tuple[Backend, list[QuantitySpec]]] = []
        for quantity in quantities:
            selected = self.registry.select(
                self.model,
                quantity,
                preferred=backend,
                require_parallel_axis=require_parallel_axis,
            )
            if selected_batches and selected_batches[-1][0].name == selected.name:
                selected_batches[-1][1].append(quantity)
            else:
                selected_batches.append((selected, [quantity]))

        for selected, selected_quantities in selected_batches:
            prepared = self._prepared_backend(selected)
            values.extend(prepared.evaluate(points, selected_quantities, state))
        return values

    def _prepared_backend(self, backend: Backend) -> PreparedBackend:
        if backend.name not in self._prepared:
            self._prepared[backend.name] = backend.prepare(self.model)
        return self._prepared[backend.name]


class SurfaceFieldQuery:
    def __init__(
        self,
        field: BodyField,
        points: list[SurfacePoint],
        state: RobotState | None,
    ) -> None:
        self.field = field
        self.points = points
        self.state = state

    def quantity(
        self,
        name: str,
        output_type: OutputType = "custom",
        frame: str | None = None,
        unit: str | None = None,
        **params: Any,
    ) -> list[QuantityValue]:
        spec = QuantitySpec(
            name=name,
            output_type=output_type,
            frame=frame,
            unit=unit,
            params=params,
        )
        return self.field.evaluate(self.points, [spec], self.state)

    def normal(self, frame: str = "world") -> list[QuantityValue]:
        return self.quantity("geometry.normal", output_type="vector3", frame=frame)

    def velocity(self, frame: str = "world") -> list[QuantityValue]:
        return self.quantity(
            "kinematics.velocity",
            output_type="vector3",
            frame=frame,
            unit="m/s",
        )

    def obstacle_distance(self, frame: str = "world") -> list[QuantityValue]:
        return self.quantity(
            "geometry.obstacle.distance",
            output_type="scalar",
            frame=frame,
            unit="m",
        )

    def obstacle_vector(self, frame: str = "world") -> list[QuantityValue]:
        return self.quantity(
            "geometry.obstacle.vector",
            output_type="vector3",
            frame=frame,
            unit="m",
        )


def _backend_supports_axis(backend: Backend, axis: ParallelAxis) -> bool:
    profile_method = getattr(backend, "parallel_profile", None)
    if profile_method is None:
        return False
    return bool(profile_method().supports_axis(axis))
