"""Density and entropy formulas for step graphons."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

ENTROPY_CONVENTION = (
    "edge-triangle: S(c,P)=sum_ij c_i c_j[-p log p-(1-p) log(1-p)]"
)


def coerce_step_arrays(sizes: ArrayLike, matrix: ArrayLike) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return normalized sizes and a float matrix after basic validation."""

    c = np.asarray(sizes, dtype=float).reshape(-1)
    p = np.asarray(matrix, dtype=float)
    if c.ndim != 1 or c.size == 0:
        raise ValueError("sizes must be a non-empty one-dimensional array")
    if p.shape != (c.size, c.size):
        raise ValueError(f"matrix must have shape {(c.size, c.size)}, got {p.shape}")
    if np.any(c < 0):
        raise ValueError("sizes must be non-negative")
    total = float(np.sum(c))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("sizes must have positive finite sum")
    if not np.all(np.isfinite(p)):
        raise ValueError("matrix entries must be finite")
    return c / total, p


def binary_entropy(u: ArrayLike) -> NDArray[np.float64] | float:
    """Compute ``-u log u - (1-u) log(1-u)`` with ``0 log 0 = 0``."""

    x = np.asarray(u, dtype=float)
    out = np.zeros_like(x, dtype=float)
    interior = (x > 0.0) & (x < 1.0)
    xi = x[interior]
    out[interior] = -xi * np.log(xi) - (1.0 - xi) * np.log(1.0 - xi)
    out[x < 0.0] = np.nan
    out[x > 1.0] = np.nan
    if out.ndim == 0:
        return float(out)
    return out


def edge_density(sizes: ArrayLike, matrix: ArrayLike) -> float:
    """Compute ``sum_ij c_i c_j P_ij``."""

    c, p = coerce_step_arrays(sizes, matrix)
    return float(np.einsum("i,j,ij->", c, c, p, optimize=True))


def triangle_density(sizes: ArrayLike, matrix: ArrayLike) -> float:
    """Compute triangle density ``sum_ijk c_i c_j c_k P_ij P_jk P_ki``."""

    c, p = coerce_step_arrays(sizes, matrix)
    return float(np.einsum("i,j,k,ij,jk,ki->", c, c, c, p, p, p, optimize=True))


def degrees(sizes: ArrayLike, matrix: ArrayLike) -> NDArray[np.float64]:
    """Return pode degrees ``d_i = sum_j c_j P_ij``."""

    c, p = coerce_step_arrays(sizes, matrix)
    return p @ c


def two_star_density(sizes: ArrayLike, matrix: ArrayLike) -> float:
    """Compute 2-star density ``sum_i c_i d_i^2``."""

    c, p = coerce_step_arrays(sizes, matrix)
    d = p @ c
    return float(np.dot(c, d * d))


def reduced_two_star_density(sizes: ArrayLike, matrix: ArrayLike) -> float:
    """Compute reduced 2-star density ``t_2star - e^2``."""

    e = edge_density(sizes, matrix)
    return float(two_star_density(sizes, matrix) - e * e)


def entropy(sizes: ArrayLike, matrix: ArrayLike) -> float:
    """Compute graphon entropy using :data:`ENTROPY_CONVENTION`."""

    c, p = coerce_step_arrays(sizes, matrix)
    h = binary_entropy(p)
    return float(np.einsum("i,j,ij->", c, c, h, optimize=True))
