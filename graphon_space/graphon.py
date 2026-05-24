"""Step graphon representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import densities


@dataclass
class StepGraphon:
    """A symmetric ``k``-podal graphon represented by pode sizes and a matrix."""

    sizes: ArrayLike
    matrix: ArrayLike
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        c = np.asarray(self.sizes, dtype=float).reshape(-1)
        p = np.asarray(self.matrix, dtype=float)

        if c.size == 0:
            raise ValueError("sizes must be non-empty")
        if p.shape != (c.size, c.size):
            raise ValueError(f"matrix must have shape {(c.size, c.size)}, got {p.shape}")
        if np.any(~np.isfinite(c)) or np.any(~np.isfinite(p)):
            raise ValueError("sizes and matrix entries must be finite")
        if np.any(c < -1e-12):
            raise ValueError("sizes must be non-negative")

        c = np.maximum(c, 0.0)
        total = float(np.sum(c))
        if total <= 0.0:
            raise ValueError("sizes must have positive sum")
        c = c / total

        if not np.allclose(p, p.T, atol=1e-10, rtol=1e-10):
            raise ValueError("matrix must be symmetric")
        p = 0.5 * (p + p.T)
        if np.any(p < -1e-12) or np.any(p > 1.0 + 1e-12):
            raise ValueError("matrix entries must lie in [0, 1]")
        p = np.clip(p, 0.0, 1.0)

        self.sizes = c
        self.matrix = p
        self.metadata = dict(self.metadata or {})

    @property
    def k(self) -> int:
        return int(self.sizes.size)

    def edge_density(self) -> float:
        return densities.edge_density(self.sizes, self.matrix)

    def triangle_density(self) -> float:
        return densities.triangle_density(self.sizes, self.matrix)

    def two_star_density(self) -> float:
        return densities.two_star_density(self.sizes, self.matrix)

    def reduced_two_star_density(self) -> float:
        return densities.reduced_two_star_density(self.sizes, self.matrix)

    def entropy(self) -> float:
        return densities.entropy(self.sizes, self.matrix)

    def degrees(self) -> NDArray[np.float64]:
        return densities.degrees(self.sizes, self.matrix)

    def complement(self, metadata: Mapping[str, Any] | None = None) -> "StepGraphon":
        new_metadata = {**self.metadata, "transform": "complement"}
        if metadata:
            new_metadata.update(metadata)
        return StepGraphon(self.sizes.copy(), 1.0 - self.matrix, new_metadata)

    def canonicalize(self) -> "StepGraphon":
        """Return a relabeled copy sorted by degree, then size, then row profile."""

        degree = self.degrees()
        rounded_rows = np.round(self.matrix, decimals=12)
        row_keys = [tuple(row.tolist()) for row in rounded_rows]
        order = sorted(
            range(self.k),
            key=lambda i: (-degree[i], -self.sizes[i], row_keys[i]),
        )
        idx = np.asarray(order, dtype=int)
        metadata = {**self.metadata, "canonicalized": True}
        return StepGraphon(self.sizes[idx], self.matrix[np.ix_(idx, idx)], metadata)

    def as_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "sizes": self.sizes.tolist(),
            "matrix": self.matrix.tolist(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StepGraphon":
        return cls(data["sizes"], data["matrix"], dict(data.get("metadata", {})))

    def copy(self, **metadata: Any) -> "StepGraphon":
        new_metadata = {**self.metadata, **metadata}
        return StepGraphon(self.sizes.copy(), self.matrix.copy(), new_metadata)
