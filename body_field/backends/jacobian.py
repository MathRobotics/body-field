from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from body_field.core import (
    Backend,
    ParallelProfile,
    PreparedBackend,
    QuantitySpec,
    QuantityValue,
    RobotState,
    RobotSurfaceModel,
    SurfacePoint,
    Vector3,
)


class PointJacobianProvider(Protocol):
    name: str

    def compute_point_jacobians(
        self,
        model: RobotSurfaceModel,
        points: list[SurfacePoint],
        state: RobotState | None = None,
    ) -> Sequence[Any]:
        ...


SUPPORTED_JACOBIAN_QUANTITIES = {
    "kinematics.jacobian",
    "kinematics.manipulability.axes",
}


@dataclass(frozen=True)
class NumpyPointJacobianBackend(Backend):
    jacobian_provider: PointJacobianProvider
    name: str = "jacobian.numpy"

    def supported_quantities(self) -> set[str]:
        return set(SUPPORTED_JACOBIAN_QUANTITIES)

    def supports(self, model: RobotSurfaceModel, quantity: QuantitySpec) -> bool:
        return quantity.name in self.supported_quantities()

    def prepare(self, model: RobotSurfaceModel) -> PreparedBackend:
        return PreparedNumpyPointJacobianBackend(model, self.jacobian_provider, self.name)

    def parallel_profile(self) -> ParallelProfile:
        return ParallelProfile(
            point_parallel=True,
            quantity_parallel=False,
            device="cpu",
            backend_kind="numpy-jacobian",
            preferred_min_points=1,
        )


@dataclass(frozen=True)
class PreparedNumpyPointJacobianBackend:
    model: RobotSurfaceModel
    jacobian_provider: PointJacobianProvider
    backend_name: str

    def evaluate(
        self,
        points: list[SurfacePoint],
        quantities: list[QuantitySpec],
        state: RobotState | None = None,
    ) -> list[QuantityValue]:
        _require_supported_quantities(quantities)
        for point in points:
            self.model.require_link(point.link_name)

        jacobians = [
            np.asarray(jacobian, dtype=float)
            for jacobian in self.jacobian_provider.compute_point_jacobians(
                self.model,
                points,
                state,
            )
        ]
        if len(jacobians) != len(points):
            raise ValueError(
                "PointJacobianProvider.compute_point_jacobians must return one Jacobian per point"
            )

        values: list[QuantityValue] = []
        for quantity in quantities:
            for point, jacobian in zip(points, jacobians, strict=True):
                if quantity.name == "kinematics.jacobian":
                    value = _matrix_as_tuple(jacobian)
                    unit = quantity.unit or "m/rad"
                    metadata = {"backend": self.backend_name, "columns": "joint velocities"}
                elif quantity.name == "kinematics.manipulability.axes":
                    value = _manipulability_axes(jacobian)
                    unit = quantity.unit or "m/rad"
                    metadata = {
                        "backend": self.backend_name,
                        "meaning": "columns are sqrt(eigenvalue) * eigenvector of J J^T",
                    }
                else:
                    raise ValueError(f"{self.backend_name} does not support {quantity.name}")

                values.append(
                    QuantityValue(
                        spec=quantity,
                        point=point,
                        value=value,
                        frame=quantity.frame or "world",
                        unit=unit,
                        metadata=metadata,
                    )
                )

        return values


def _require_supported_quantities(quantities: list[QuantitySpec]) -> None:
    for quantity in quantities:
        if quantity.frame not in {None, "world"}:
            raise ValueError(f"{quantity.name} supports frame='world' only")
        if quantity.name not in SUPPORTED_JACOBIAN_QUANTITIES:
            raise ValueError(f"Unsupported Jacobian quantity: {quantity.name}")


def _matrix_as_tuple(matrix: np.ndarray) -> tuple[tuple[float, ...], ...]:
    if matrix.ndim != 2:
        raise ValueError(f"Jacobian must be a matrix, got shape {matrix.shape}")
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _manipulability_axes(jacobian: np.ndarray) -> tuple[Vector3, Vector3, Vector3]:
    if jacobian.ndim != 2 or jacobian.shape[0] != 3:
        raise ValueError(f"Manipulability axes expect a 3xN Jacobian, got shape {jacobian.shape}")

    manipulability = jacobian @ jacobian.T
    eigenvalues, eigenvectors = np.linalg.eigh(manipulability)
    order = np.argsort(eigenvalues)[::-1]
    axes: list[Vector3] = []
    for index in order:
        length = float(np.sqrt(max(float(eigenvalues[index]), 0.0)))
        axis = eigenvectors[:, index] * length
        axes.append((float(axis[0]), float(axis[1]), float(axis[2])))
    return (axes[0], axes[1], axes[2])
