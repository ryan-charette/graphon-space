"""Fixed-edge boundary optimization and known envelope helpers."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .families import GraphonFamily, KPodalFamily, default_optimization_families
from .optimize import OptimizationResult, model_t


def known_triangle_upper(edge: float) -> float:
    edge = float(np.clip(edge, 0.0, 1.0))
    return edge ** 1.5


def known_two_star_upper(edge: float) -> float:
    edge = float(np.clip(edge, 0.0, 1.0))
    if np.isclose(edge, 0.5):
        return float(np.sqrt(2.0) / 4.0)
    if edge > 0.5:
        return edge ** 1.5
    return (1.0 - edge) ** 1.5 + 2.0 * edge - 1.0


def known_two_star_reduced_upper(edge: float) -> float:
    return known_two_star_upper(edge) - float(edge) ** 2


def optimize_fixed_edge_boundary_for_family(
    model: str,
    edge: float,
    family: GraphonFamily,
    direction: str = "max",
    starts: int = 20,
    seed: int | None = None,
    ftol: float = 1e-10,
    maxiter: int = 1000,
) -> OptimizationResult:
    if not family.optimizable:
        raise ValueError(f"family {family.name} is not optimizable")
    if direction not in {"min", "max"}:
        raise ValueError("direction must be 'min' or 'max'")

    rng = np.random.default_rng(seed)
    vectors = family.initial_vectors(starts, seed=rng, target={"edge": edge})
    bounds = family.bounds()
    constraints = [{"type": "eq", "fun": lambda x: family.unpack_params(x).edge_density() - edge}]
    sign = -1.0 if direction == "max" else 1.0

    def objective(x):
        return sign * model_t(family.unpack_params(x), model)

    raw_results = []
    total_nfev = 0
    messages: list[str] = []
    for x0 in vectors:
        if len(x0) < len(constraints):
            continue
        result = minimize(
            objective,
            np.asarray(x0, dtype=float),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": maxiter, "ftol": ftol, "disp": False},
        )
        total_nfev += int(getattr(result, "nfev", 0))
        messages.append(str(result.message))
        graphon = family.unpack_params(result.x)
        raw_results.append((objective(result.x), graphon, result))

    if not raw_results:
        graphon = family.unpack_params(family.random_vector(rng, target={"edge": edge}))
        return OptimizationResult.from_graphon(
            graphon,
            model,
            target_e=edge,
            target_t=model_t(graphon, model),
            family=family.name,
            success=False,
            objective_value=np.inf,
            nfev=total_nfev,
            message="no boundary run attempted",
            notes=direction,
        )

    _, best_graphon, best_result = min(raw_results, key=lambda item: item[0])
    observed_t = model_t(best_graphon, model)
    success = bool(best_result.success) and abs(best_graphon.edge_density() - edge) <= 1e-6
    return OptimizationResult.from_graphon(
        best_graphon,
        model,
        target_e=edge,
        target_t=observed_t,
        family=family.name,
        success=success,
        objective_value=float(best_result.fun),
        nfev=total_nfev,
        message=" | ".join(messages[-3:]),
        notes=direction,
    )


def optimize_fixed_edge_boundary(
    model: str,
    edge: float,
    direction: str = "max",
    families: Iterable[GraphonFamily] | None = None,
    kmax: int = 4,
    starts: int = 20,
    seed: int | None = None,
) -> OptimizationResult | None:
    families = list(families) if families is not None else default_optimization_families(kmax, include_er=True)
    rng = np.random.default_rng(seed)
    results: list[OptimizationResult] = []
    for family in families:
        if not family.optimizable:
            continue
        child_seed = int(rng.integers(0, 2**32 - 1))
        results.append(
            optimize_fixed_edge_boundary_for_family(
                model=model,
                edge=edge,
                family=family,
                direction=direction,
                starts=starts,
                seed=child_seed,
            )
        )

    successful = [result for result in results if result.success]
    if not successful:
        successful = results
    if not successful:
        return None
    key = (lambda r: r.target_t if r.target_t is not None else model_t(r.graphon(), model))
    return max(successful, key=key) if direction == "max" else min(successful, key=key)


def boundary_grid(
    model: str,
    e_values: Sequence[float],
    kmax: int = 4,
    starts: int = 20,
    seed: int | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    families = default_optimization_families(kmax, include_er=True)
    if kmax >= 1 and not any(isinstance(family, KPodalFamily) and family.k == kmax for family in families):
        families.append(KPodalFamily(kmax))
    for edge in e_values:
        for direction in ("min", "max"):
            child_seed = int(rng.integers(0, 2**32 - 1))
            result = optimize_fixed_edge_boundary(
                model=model,
                edge=float(edge),
                direction=direction,
                families=families,
                kmax=kmax,
                starts=starts,
                seed=child_seed,
            )
            if result is None:
                continue
            record = result.as_dict()
            record["direction"] = direction
            record["observed_t"] = result.target_t
            if model in {"triangle", "edge-triangle"}:
                record["known_upper"] = known_triangle_upper(edge)
            elif model in {"two-star", "2star", "edge-2star"}:
                record["known_upper"] = known_two_star_upper(edge)
                record["known_reduced_upper"] = known_two_star_reduced_upper(edge)
            records.append(record)
    return pd.DataFrame.from_records(records)
