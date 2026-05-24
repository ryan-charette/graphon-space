"""Constrained and penalty optimizers for graphon exploration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .families import GraphonFamily, default_optimization_families
from .graphon import StepGraphon
from .phase import classify_symmetry


def model_t(graphon: StepGraphon, model: str) -> float:
    if model in {"triangle", "edge-triangle"}:
        return graphon.triangle_density()
    if model in {"two-star", "2star", "edge-2star"}:
        return graphon.two_star_density()
    raise ValueError(f"unknown model {model!r}")


@dataclass
class OptimizationResult:
    target_e: float
    target_t: float | None
    model: str
    family: str
    k: int
    success: bool
    entropy: float
    edge_residual: float
    t_residual: float | None
    sizes: list[float]
    matrix: list[list[float]]
    symmetry_class: str
    objective_value: float
    nfev: int
    message: str
    notes: str = ""

    @classmethod
    def from_graphon(
        cls,
        graphon: StepGraphon,
        model: str,
        target_e: float,
        target_t: float | None,
        family: str,
        success: bool,
        objective_value: float,
        nfev: int,
        message: str,
        notes: str = "",
    ) -> "OptimizationResult":
        observed_t = model_t(graphon, model)
        t_residual = None if target_t is None else abs(observed_t - target_t)
        return cls(
            target_e=float(target_e),
            target_t=None if target_t is None else float(target_t),
            model=model,
            family=family,
            k=graphon.k,
            success=bool(success),
            entropy=graphon.entropy(),
            edge_residual=abs(graphon.edge_density() - target_e),
            t_residual=t_residual,
            sizes=graphon.sizes.tolist(),
            matrix=graphon.matrix.tolist(),
            symmetry_class=classify_symmetry(graphon),
            objective_value=float(objective_value),
            nfev=int(nfev),
            message=str(message),
            notes=notes,
        )

    def as_dict(self) -> dict:
        return asdict(self)

    def graphon(self) -> StepGraphon:
        return StepGraphon(self.sizes, self.matrix, {"family": self.family})


def _constraints(
    family: GraphonFamily,
    model: str,
    target_e: float,
    target_t: float | None,
) -> list[dict]:
    constraints = [
        {
            "type": "eq",
            "fun": lambda x: family.unpack_params(x).edge_density() - target_e,
        }
    ]
    if target_t is not None:
        constraints.append(
            {
                "type": "eq",
                "fun": lambda x: model_t(family.unpack_params(x), model) - target_t,
            }
        )
    return constraints


def _best_result(results: list[tuple[float, object]]) -> object | None:
    if not results:
        return None
    return min(results, key=lambda item: item[0])[1]


def _clip_to_bounds(x: Sequence[float], bounds: Sequence[tuple[float, float]]) -> np.ndarray:
    values = np.asarray(x, dtype=float)
    low = np.asarray([bound[0] for bound in bounds], dtype=float)
    high = np.asarray([bound[1] for bound in bounds], dtype=float)
    return np.minimum(np.maximum(values, low), high)


def optimize_entropy_for_family(
    model: str,
    target_e: float,
    target_t: float,
    family: GraphonFamily,
    starts: int = 20,
    seed: int | None = None,
    mode: str = "hybrid",
    penalty_edge: float = 1e4,
    penalty_t: float = 1e4,
    ftol: float = 1e-10,
    maxiter: int = 300,
) -> OptimizationResult:
    """Maximize entropy over one optimizer-parameterized family."""

    if not family.optimizable:
        raise ValueError(f"family {family.name} is not optimizable")

    target = {"edge": target_e, "t": target_t}
    rng = np.random.default_rng(seed)
    vectors = family.initial_vectors(starts, seed=rng, target=target)
    bounds = family.bounds()
    constrained = _constraints(family, model, target_e, target_t)

    def entropy_objective(x: Sequence[float]) -> float:
        return -family.unpack_params(x).entropy()

    def penalty_objective(x: Sequence[float]) -> float:
        graphon = family.unpack_params(x)
        e_res = graphon.edge_density() - target_e
        t_res = model_t(graphon, model) - target_t
        return -graphon.entropy() + penalty_edge * e_res * e_res + penalty_t * t_res * t_res

    raw_results: list[tuple[float, object]] = []
    total_nfev = 0
    messages: list[str] = []

    for x0 in vectors:
        start = _clip_to_bounds(x0, bounds)
        if mode in {"penalty", "hybrid"}:
            penalty_result = minimize(
                penalty_objective,
                start,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": maxiter, "ftol": ftol},
            )
            total_nfev += int(getattr(penalty_result, "nfev", 0))
            messages.append(str(penalty_result.message))
            start = _clip_to_bounds(penalty_result.x, bounds)
            if mode == "penalty":
                graphon = family.unpack_params(start)
                score = penalty_objective(start)
                raw_results.append((score, (graphon, penalty_result, False)))
                continue

        if len(start) < len(constrained):
            continue
        constrained_result = minimize(
            entropy_objective,
            start,
            method="SLSQP",
            bounds=bounds,
            constraints=constrained,
            options={"maxiter": maxiter, "ftol": ftol, "disp": False},
        )
        total_nfev += int(getattr(constrained_result, "nfev", 0))
        messages.append(str(constrained_result.message))
        graphon = family.unpack_params(constrained_result.x)
        score = entropy_objective(constrained_result.x)
        raw_results.append((score, (graphon, constrained_result, True)))

    feasible_results = [
        item
        for item in raw_results
        if abs(item[1][0].edge_density() - target_e) <= 1e-6
        and abs(model_t(item[1][0], model) - target_t) <= 1e-6
    ]
    best = _best_result(feasible_results or raw_results)
    if best is None:
        graphon = family.unpack_params(family.random_vector(np.random.default_rng(seed), target=target))
        return OptimizationResult.from_graphon(
            graphon,
            model,
            target_e,
            target_t,
            family=family.name,
            success=False,
            objective_value=np.inf,
            nfev=total_nfev,
            message="no feasible optimizer run was attempted",
        )

    graphon, scipy_result, used_constraints = best
    edge_residual = abs(graphon.edge_density() - target_e)
    t_residual = abs(model_t(graphon, model) - target_t)
    residual_ok = edge_residual <= 1e-6 and t_residual <= 1e-6
    success = residual_ok
    notes = "constrained refinement" if used_constraints else "penalty result"
    return OptimizationResult.from_graphon(
        graphon,
        model,
        target_e,
        target_t,
        family=family.name,
        success=success,
        objective_value=float(getattr(scipy_result, "fun", np.nan)),
        nfev=total_nfev,
        message=" | ".join(messages[-3:]),
        notes=notes,
    )


def optimize_entropy_for_target(
    model: str,
    target_e: float,
    target_t: float,
    families: Iterable[GraphonFamily] | None = None,
    kmax: int = 4,
    starts: int = 20,
    seed: int | None = None,
    mode: str = "hybrid",
) -> list[OptimizationResult]:
    """Run target entropy maximization across several families."""

    families = list(families) if families is not None else default_optimization_families(kmax)
    rng = np.random.default_rng(seed)
    results: list[OptimizationResult] = []
    for family in families:
        if not family.optimizable:
            continue
        child_seed = int(rng.integers(0, 2**32 - 1))
        result = optimize_entropy_for_family(
            model=model,
            target_e=target_e,
            target_t=target_t,
            family=family,
            starts=starts,
            seed=child_seed,
            mode=mode,
        )
        results.append(result)
    return results


def best_entropy_result(results: Sequence[OptimizationResult]) -> OptimizationResult | None:
    successful = [result for result in results if result.success]
    if not successful:
        return None
    return max(successful, key=lambda result: result.entropy)


def compare_families(
    model: str,
    target_e: float,
    target_t: float,
    kmax: int = 4,
    starts: int = 20,
    seed: int | None = None,
    mode: str = "hybrid",
) -> pd.DataFrame:
    results = optimize_entropy_for_target(
        model=model,
        target_e=target_e,
        target_t=target_t,
        kmax=kmax,
        starts=starts,
        seed=seed,
        mode=mode,
    )
    return pd.DataFrame.from_records(result.as_dict() for result in results)


def optimize_grid(
    model: str,
    e_values: Sequence[float],
    t_values: Sequence[float],
    kmax: int = 4,
    starts: int = 10,
    seed: int | None = None,
    mode: str = "hybrid",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    for edge in e_values:
        for target_t in t_values:
            child_seed = int(rng.integers(0, 2**32 - 1))
            for result in optimize_entropy_for_target(
                model=model,
                target_e=float(edge),
                target_t=float(target_t),
                kmax=kmax,
                starts=starts,
                seed=child_seed,
                mode=mode,
            ):
                records.append(result.as_dict())
    return pd.DataFrame.from_records(records)
