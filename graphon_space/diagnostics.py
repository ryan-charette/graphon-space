"""Numerical diagnostics for graphon optimizer candidates."""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from .graphon import StepGraphon
from .phase import duplicate_pode_groups


def constraint_residuals(graphon: StepGraphon, model: str, edge: float, target_t: float) -> dict[str, float]:
    if model in {"triangle", "edge-triangle"}:
        observed_t = graphon.triangle_density()
    elif model in {"two-star", "2star", "edge-2star"}:
        observed_t = graphon.two_star_density()
    else:
        raise ValueError(f"unknown model {model!r}")
    return {
        "edge_residual": abs(graphon.edge_density() - edge),
        "t_residual": abs(observed_t - target_t),
    }


def symmetry_residuals(graphon: StepGraphon) -> dict[str, float]:
    if graphon.k != 2:
        return {}
    return {
        "size_residual": abs(float(graphon.sizes[0] - graphon.sizes[1])),
        "diagonal_residual": abs(float(graphon.matrix[0, 0] - graphon.matrix[1, 1])),
    }


def pode_merging_diagnostics(graphon: StepGraphon, tol: float = 1e-6) -> dict:
    groups = duplicate_pode_groups(graphon, tol=tol)
    return {
        "groups": groups,
        "n_effective_podes": len(groups),
        "merged": len(groups) < graphon.k,
    }


def finite_difference_hessian(
    fun: Callable[[NDArray[np.float64]], float],
    x: NDArray[np.float64],
    eps: float = 1e-4,
) -> NDArray[np.float64]:
    """Central finite-difference Hessian for small local diagnostics."""

    x = np.asarray(x, dtype=float)
    n = x.size
    hessian = np.zeros((n, n), dtype=float)
    f0 = float(fun(x))
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = eps
        f_plus = float(fun(x + ei))
        f_minus = float(fun(x - ei))
        hessian[i, i] = (f_plus - 2.0 * f0 + f_minus) / (eps * eps)
        for j in range(i + 1, n):
            ej = np.zeros(n)
            ej[j] = eps
            f_pp = float(fun(x + ei + ej))
            f_pm = float(fun(x + ei - ej))
            f_mp = float(fun(x - ei + ej))
            f_mm = float(fun(x - ei - ej))
            value = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps * eps)
            hessian[i, j] = value
            hessian[j, i] = value
    return hessian


def hessian_eigenvalues(
    fun: Callable[[NDArray[np.float64]], float],
    x: NDArray[np.float64],
    eps: float = 1e-4,
) -> NDArray[np.float64]:
    return np.linalg.eigvalsh(finite_difference_hessian(fun, x, eps=eps))
