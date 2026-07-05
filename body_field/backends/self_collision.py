from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from body_field.backends._surface_points import pack_surface_point_transforms
from body_field.backends._taichi import import_taichi, init_taichi
from body_field.core import (
    Backend,
    LinkStateProvider,
    ParallelProfile,
    PreparedBackend,
    QuantitySpec,
    QuantityValue,
    RobotState,
    RobotSurfaceModel,
    SurfacePoint,
)


SUPPORTED_SELF_COLLISION_QUANTITIES = {
    "geometry.self_collision.distance",
    "geometry.self_collision.closest_point",
    "geometry.self_collision.vector",
}


@dataclass(frozen=True)
class NumpySelfCollisionBackend(Backend):
    link_state_provider: LinkStateProvider
    excluded_link_pairs: Sequence[tuple[str, str]] = ()
    include_same_link: bool = False
    point_radius: float = 0.0
    name: str = "self_collision.numpy"

    def __post_init__(self) -> None:
        if self.point_radius < 0.0:
            raise ValueError("point_radius must be non-negative")

    def supported_quantities(self) -> set[str]:
        return set(SUPPORTED_SELF_COLLISION_QUANTITIES)

    def supports(self, model: RobotSurfaceModel, quantity: QuantitySpec) -> bool:
        return quantity.name in self.supported_quantities()

    def prepare(self, model: RobotSurfaceModel) -> PreparedBackend:
        return PreparedNumpySelfCollisionBackend(
            model=model,
            link_state_provider=self.link_state_provider,
            excluded_link_pairs=_normalize_excluded_pairs(self.excluded_link_pairs),
            include_same_link=self.include_same_link,
            point_radius=float(self.point_radius),
            backend_name=self.name,
        )

    def parallel_profile(self) -> ParallelProfile:
        return ParallelProfile(
            point_parallel=True,
            quantity_parallel=False,
            device="cpu",
            backend_kind="numpy-self-collision",
            preferred_min_points=1,
        )


@dataclass(frozen=True)
class TaichiSelfCollisionBackend(Backend):
    link_state_provider: LinkStateProvider
    excluded_link_pairs: Sequence[tuple[str, str]] = ()
    include_same_link: bool = False
    point_radius: float = 0.0
    arch: Literal["auto", "cpu", "gpu", "metal", "vulkan", "cuda"] = "auto"
    name: str = "self_collision.taichi"

    def __post_init__(self) -> None:
        if self.point_radius < 0.0:
            raise ValueError("point_radius must be non-negative")

    def supported_quantities(self) -> set[str]:
        return set(SUPPORTED_SELF_COLLISION_QUANTITIES)

    def supports(self, model: RobotSurfaceModel, quantity: QuantitySpec) -> bool:
        return quantity.name in self.supported_quantities()

    def prepare(self, model: RobotSurfaceModel) -> PreparedBackend:
        taichi = import_taichi()
        init_taichi(taichi, self.arch)
        return PreparedTaichiSelfCollisionBackend(
            model=model,
            link_state_provider=self.link_state_provider,
            excluded_link_pairs=_normalize_excluded_pairs(self.excluded_link_pairs),
            include_same_link=self.include_same_link,
            point_radius=float(self.point_radius),
            backend_name=self.name,
        )

    def parallel_profile(self) -> ParallelProfile:
        return ParallelProfile(
            point_parallel=True,
            quantity_parallel=False,
            device="auto" if self.arch == "auto" else self.arch,
            backend_kind="taichi-self-collision",
            preferred_min_points=1_000,
        )


@dataclass(frozen=True)
class PreparedNumpySelfCollisionBackend:
    model: RobotSurfaceModel
    link_state_provider: LinkStateProvider
    excluded_link_pairs: frozenset[frozenset[str]]
    include_same_link: bool
    point_radius: float
    backend_name: str

    def evaluate(
        self,
        points: list[SurfacePoint],
        quantities: list[QuantitySpec],
        state: RobotState | None = None,
    ) -> list[QuantityValue]:
        _require_supported_quantities(quantities)
        link_states = self.link_state_provider.compute_link_states(self.model, state)
        positions = pack_surface_point_transforms(
            self.model, points, link_states
        ).world_position
        outputs = _evaluate_self_collision_distances(
            points=points,
            positions=positions,
            excluded_link_pairs=self.excluded_link_pairs,
            include_same_link=self.include_same_link,
            point_radius=_point_radius(quantities, self.point_radius),
        )
        return _collect_values(points, quantities, outputs, self.backend_name)


@dataclass(frozen=True)
class PreparedTaichiSelfCollisionBackend:
    model: RobotSurfaceModel
    link_state_provider: LinkStateProvider
    excluded_link_pairs: frozenset[frozenset[str]]
    include_same_link: bool
    point_radius: float
    backend_name: str

    def evaluate(
        self,
        points: list[SurfacePoint],
        quantities: list[QuantitySpec],
        state: RobotState | None = None,
    ) -> list[QuantityValue]:
        _require_supported_quantities(quantities)
        link_states = self.link_state_provider.compute_link_states(self.model, state)
        transform_arrays = pack_surface_point_transforms(self.model, points, link_states)
        outputs = _evaluate_taichi_self_collision_distances(
            points=points,
            positions=transform_arrays.world_position,
            point_link_ids=transform_arrays.point_link_ids,
            link_names=transform_arrays.link_names,
            excluded_link_pairs=self.excluded_link_pairs,
            include_same_link=self.include_same_link,
            point_radius=_point_radius(quantities, self.point_radius),
        )
        return _collect_values(points, quantities, outputs, self.backend_name)


@dataclass(frozen=True)
class _SelfCollisionOutputs:
    world_position: np.ndarray
    signed_distance: np.ndarray
    closest_point: np.ndarray
    closest_point_index: np.ndarray
    closest_link_name: list[str | None]


def _evaluate_self_collision_distances(
    points: list[SurfacePoint],
    positions: np.ndarray,
    excluded_link_pairs: frozenset[frozenset[str]],
    include_same_link: bool,
    point_radius: float,
) -> _SelfCollisionOutputs:
    point_count = positions.shape[0]
    if point_count == 0:
        return _SelfCollisionOutputs(
            world_position=positions,
            signed_distance=np.empty(0, dtype=np.float32),
            closest_point=np.empty((0, 3), dtype=np.float32),
            closest_point_index=np.empty(0, dtype=np.int32),
            closest_link_name=[],
        )

    deltas = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(deltas, axis=2).astype(np.float32, copy=False)
    valid = np.ones((point_count, point_count), dtype=bool)
    np.fill_diagonal(valid, False)

    for i, point_i in enumerate(points):
        for j, point_j in enumerate(points):
            if i == j:
                continue
            if not include_same_link and point_i.link_name == point_j.link_name:
                valid[i, j] = False
                continue
            if frozenset((point_i.link_name, point_j.link_name)) in excluded_link_pairs:
                valid[i, j] = False

    masked_distances = np.where(valid, distances, np.inf)
    closest_indices = np.argmin(masked_distances, axis=1).astype(np.int32)
    best_distances = masked_distances[np.arange(point_count), closest_indices].astype(np.float32)

    has_candidate = np.isfinite(best_distances)
    signed_distances = best_distances - float(point_radius * 2.0)
    signed_distances = np.where(has_candidate, signed_distances, np.inf).astype(np.float32)

    closest_points = np.full((point_count, 3), np.nan, dtype=np.float32)
    closest_points[has_candidate] = positions[closest_indices[has_candidate]]

    closest_link_names: list[str | None] = [None] * point_count
    for index in np.nonzero(has_candidate)[0]:
        closest_link_names[int(index)] = points[int(closest_indices[index])].link_name

    closest_indices = np.where(has_candidate, closest_indices, -1).astype(np.int32)
    return _SelfCollisionOutputs(
        world_position=positions,
        signed_distance=signed_distances,
        closest_point=closest_points,
        closest_point_index=closest_indices,
        closest_link_name=closest_link_names,
    )


def _evaluate_taichi_self_collision_distances(
    points: list[SurfacePoint],
    positions: np.ndarray,
    point_link_ids: np.ndarray,
    link_names: tuple[str, ...],
    excluded_link_pairs: frozenset[frozenset[str]],
    include_same_link: bool,
    point_radius: float,
) -> _SelfCollisionOutputs:
    point_count = positions.shape[0]
    if point_count == 0:
        return _SelfCollisionOutputs(
            world_position=positions,
            signed_distance=np.empty(0, dtype=np.float32),
            closest_point=np.empty((0, 3), dtype=np.float32),
            closest_point_index=np.empty(0, dtype=np.int32),
            closest_link_name=[],
        )

    excluded_link_pair_matrix = _excluded_link_pair_matrix(
        link_names, excluded_link_pairs
    )
    signed_distances = np.empty(point_count, dtype=np.float32)
    closest_indices = np.empty(point_count, dtype=np.int32)
    _taichi_nearest_self_collision(
        positions,
        point_link_ids,
        excluded_link_pair_matrix,
        np.int32(1 if include_same_link else 0),
        np.float32(point_radius),
        signed_distances,
        closest_indices,
    )

    has_candidate = closest_indices >= 0
    signed_distances = np.where(has_candidate, signed_distances, np.inf).astype(np.float32)
    closest_points = np.full((point_count, 3), np.nan, dtype=np.float32)
    closest_points[has_candidate] = positions[closest_indices[has_candidate]]

    closest_link_names: list[str | None] = [None] * point_count
    for index in np.nonzero(has_candidate)[0]:
        closest_link_names[int(index)] = points[int(closest_indices[index])].link_name

    return _SelfCollisionOutputs(
        world_position=positions,
        signed_distance=signed_distances,
        closest_point=closest_points,
        closest_point_index=closest_indices,
        closest_link_name=closest_link_names,
    )


def _collect_values(
    points: list[SurfacePoint],
    quantities: list[QuantitySpec],
    outputs: _SelfCollisionOutputs,
    backend_name: str,
) -> list[QuantityValue]:
    values: list[QuantityValue] = []
    for quantity in quantities:
        for index, point in enumerate(points):
            if quantity.name == "geometry.self_collision.distance":
                value = float(outputs.signed_distance[index])
                unit = quantity.unit or "m"
            elif quantity.name == "geometry.self_collision.closest_point":
                value = _tuple3(outputs.closest_point[index])
                unit = quantity.unit or "m"
            elif quantity.name == "geometry.self_collision.vector":
                value = _tuple3(outputs.world_position[index] - outputs.closest_point[index])
                unit = quantity.unit or "m"
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
                        "world_position": _tuple3(outputs.world_position[index]),
                        "closest_point": _tuple3(outputs.closest_point[index]),
                        "closest_point_index": int(outputs.closest_point_index[index]),
                        "closest_link": outputs.closest_link_name[index],
                    },
                )
            )
    return values


def _require_supported_quantities(quantities: list[QuantitySpec]) -> None:
    for quantity in quantities:
        if quantity.frame not in {None, "world"}:
            raise ValueError(f"{quantity.name} supports frame='world' only")
        if quantity.name not in SUPPORTED_SELF_COLLISION_QUANTITIES:
            raise ValueError(f"Unsupported self-collision quantity: {quantity.name}")


def _point_radius(quantities: list[QuantitySpec], default_radius: float) -> float:
    radii: set[float] = set()
    for quantity in quantities:
        if "point_radius" not in quantity.params:
            continue
        quantity_radius = float(quantity.params["point_radius"])
        if quantity_radius < 0.0:
            raise ValueError("point_radius must be non-negative")
        radii.add(quantity_radius)
    if len(radii) > 1:
        raise ValueError("Batched self-collision quantities must use the same point_radius")
    return next(iter(radii), default_radius)


def _normalize_excluded_pairs(
    excluded_link_pairs: Sequence[tuple[str, str]],
) -> frozenset[frozenset[str]]:
    normalized: set[frozenset[str]] = set()
    for first, second in excluded_link_pairs:
        normalized.add(frozenset((first, second)))
    return frozenset(normalized)


def _excluded_link_pair_matrix(
    link_names: tuple[str, ...],
    excluded_link_pairs: frozenset[frozenset[str]],
) -> np.ndarray:
    link_ids = {name: index for index, name in enumerate(link_names)}
    matrix = np.zeros((len(link_names), len(link_names)), dtype=np.int32)
    for pair in excluded_link_pairs:
        ids = [link_ids[name] for name in pair if name in link_ids]
        if len(ids) == 1:
            matrix[ids[0], ids[0]] = 1
        elif len(ids) == 2:
            matrix[ids[0], ids[1]] = 1
            matrix[ids[1], ids[0]] = 1
    return matrix


def _taichi_nearest_self_collision(*args):
    ti = import_taichi()

    @ti.kernel
    def kernel(
        positions: ti.types.ndarray(dtype=ti.f32, ndim=2),
        point_link_ids: ti.types.ndarray(dtype=ti.i32, ndim=1),
        excluded_link_pair_matrix: ti.types.ndarray(dtype=ti.i32, ndim=2),
        include_same_link: ti.i32,
        point_radius: ti.f32,
        out_signed_distance: ti.types.ndarray(dtype=ti.f32, ndim=1),
        out_closest_index: ti.types.ndarray(dtype=ti.i32, ndim=1),
    ):
        for i in range(positions.shape[0]):
            link_i = point_link_ids[i]
            best_distance_sq = 3.4028234663852886e38
            best_index = -1
            for j in range(positions.shape[0]):
                link_j = point_link_ids[j]
                valid = i != j
                if include_same_link == 0 and link_i == link_j:
                    valid = False
                if excluded_link_pair_matrix[link_i, link_j] != 0:
                    valid = False
                if valid:
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    dz = positions[i, 2] - positions[j, 2]
                    distance_sq = dx * dx + dy * dy + dz * dz
                    if distance_sq < best_distance_sq:
                        best_distance_sq = distance_sq
                        best_index = j

            out_closest_index[i] = best_index
            if best_index >= 0:
                out_signed_distance[i] = ti.sqrt(best_distance_sq) - point_radius * 2.0
            else:
                out_signed_distance[i] = best_distance_sq

    kernel(*args)


def _tuple3(array: np.ndarray) -> tuple[float, float, float]:
    return (float(array[0]), float(array[1]), float(array[2]))
