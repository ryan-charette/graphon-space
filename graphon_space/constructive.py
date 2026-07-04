"""Constructive fixed-edge tripodal search.

This module implements the handwritten ``A, B, c`` algorithm as a concrete
numerical search.  The tripodal graphon has pode sizes ``(c/2, c/2, 1-c)`` and
block probabilities

``e - A + B(1-c)``, ``e + A + B(1-c)``, ``e - cB``, and
``e + c^2 B / (1-c)``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Literal, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .densities import binary_entropy
from .graphon import StepGraphon

BipodalMode = Literal["optimized", "merged"]
CSampler = Literal["grid", "uniform"]
BipodalEntropyCallback = Callable[[StepGraphon, float], float]


@dataclass(frozen=True)
class ConstructiveSearchConfig:
    """Configuration for the fixed-edge ``A, B, c`` search."""

    edge: float
    ab_samples: int = 1_000
    c_samples: int = 100
    seed: int | None = None
    min_ratio: float = 10.0
    c_sampler: CSampler = "grid"
    bipodal_mode: BipodalMode = "optimized"
    bipodal_starts: int = 8
    optimizer_mode: str = "hybrid"
    entropy_gap_tol: float = 0.0
    max_ab_attempts: int | None = None


@dataclass(frozen=True)
class CSearchResult:
    """Best sampled ``c`` for one accepted ``(A, B)`` pair."""

    edge: float
    A: float
    B: float
    F: float
    H_second: float
    c_max: float | None
    t_min: float | None
    tripodal_entropy: float | None
    bipodal_entropy: float | None
    entropy_gap: float | None
    checked_c: int
    feasible_c: int
    successful_c: int
    bipodal_mode: str

    def as_dict(self) -> dict[str, float | int | str | None]:
        return asdict(self)


def binary_entropy_prime(edge: float) -> float:
    """Return ``H'(e)`` for ``H(u) = -u log u - (1-u) log(1-u)``."""

    edge = _validate_edge(edge)
    return float(np.log((1.0 - edge) / edge))


def binary_entropy_second_derivative(edge: float) -> float:
    """Return ``H''(e)`` for the binary entropy convention used by the package."""

    edge = _validate_edge(edge)
    return float(-1.0 / edge - 1.0 / (1.0 - edge))


def tripodal_entropy_curvature_score(edge: float, A: float, B: float) -> float:
    """Compute the ``F(A, B)`` screening score for the tripodal perturbation.

    The score is the normalized ``c -> 0`` entropy-curvature coefficient induced
    by the fixed-edge tripodal construction:

    ``F = 2 * (0.5 * (H(e-A+B) + H(e+A+B)) - H(e) - B H'(e)) / (A^2 + B^2)``.
    """

    edge = _validate_edge(edge)
    if not is_valid_ab_pair(edge, A, B, min_ratio=1.0):
        raise ValueError("A and B must satisfy 0 < B < A < e and keep probabilities in [0, 1]")
    numerator = (
        0.5 * (_entropy_scalar(edge - A + B) + _entropy_scalar(edge + A + B))
        - _entropy_scalar(edge)
        - B * binary_entropy_prime(edge)
    )
    return float(2.0 * numerator / (A * A + B * B))


def passes_entropy_curvature_screen(edge: float, A: float, B: float) -> bool:
    """Return whether ``F(A, B) > H''(e)``."""

    return tripodal_entropy_curvature_score(edge, A, B) > binary_entropy_second_derivative(edge)


def tripodal_perturbation(edge: float, A: float, B: float, c: float) -> StepGraphon:
    """Build the tripodal graphon from the handwritten construction."""

    edge = _validate_edge(edge)
    _validate_c(c)
    if not is_valid_ab_pair(edge, A, B, min_ratio=1.0):
        raise ValueError("invalid A, B pair")
    one_minus_c = 1.0 - c
    matrix = np.array(
        [
            [edge - A + B * one_minus_c, edge + A + B * one_minus_c, edge - c * B],
            [edge + A + B * one_minus_c, edge - A + B * one_minus_c, edge - c * B],
            [edge - c * B, edge - c * B, edge + c * c * B / one_minus_c],
        ],
        dtype=float,
    )
    return StepGraphon(
        [0.5 * c, 0.5 * c, one_minus_c],
        matrix,
        {"family": "constructive-tripodal", "edge": edge, "A": A, "B": B, "c": c},
    )


def merged_bipodal_reference(edge: float, B: float, c: float) -> StepGraphon:
    """Return the bipodal graphon obtained by merging the two small tripodal podes."""

    edge = _validate_edge(edge)
    _validate_c(c)
    one_minus_c = 1.0 - c
    matrix = np.array(
        [
            [edge + B * one_minus_c, edge - c * B],
            [edge - c * B, edge + c * c * B / one_minus_c],
        ],
        dtype=float,
    )
    return StepGraphon(
        [c, one_minus_c],
        matrix,
        {"family": "constructive-merged-bipodal", "edge": edge, "B": B, "c": c},
    )


def constructive_triangle_density(edge: float, A: float, B: float, c: float) -> float:
    """Return the closed-form triangle density ``e^3 - c^3(A^3 - B^3)``."""

    edge = _validate_edge(edge)
    _validate_c(c)
    if not (0.0 <= B < A):
        raise ValueError("expected 0 <= B < A")
    return float(edge**3 - c**3 * (A**3 - B**3))


def sample_ab_pairs(
    edge: float,
    samples: int,
    seed: int | np.random.Generator | None = None,
    min_ratio: float = 10.0,
    max_attempts: int | None = None,
) -> list[tuple[float, float]]:
    """Sample ``(A, B)`` pairs with ``B << A < e`` and valid probabilities."""

    edge = _validate_edge(edge)
    if samples < 0:
        raise ValueError("samples must be non-negative")
    if min_ratio <= 1.0:
        raise ValueError("min_ratio must be greater than 1")

    rng = _rng(seed)
    attempts = 0
    max_attempts = max_attempts if max_attempts is not None else max(100, samples * 100)
    pairs: list[tuple[float, float]] = []
    upper_A = min(edge, 1.0 - edge)
    if upper_A <= 0.0:
        return pairs

    while len(pairs) < samples and attempts < max_attempts:
        attempts += 1
        A = float(rng.uniform(np.finfo(float).eps, upper_A))
        upper_B = min(A / min_ratio, 1.0 - edge - A)
        if upper_B <= 0.0:
            continue
        B = float(rng.uniform(np.finfo(float).eps, upper_B))
        if is_valid_ab_pair(edge, A, B, min_ratio=min_ratio):
            pairs.append((A, B))
    return pairs


def is_valid_ab_pair(edge: float, A: float, B: float, min_ratio: float = 10.0) -> bool:
    """Return whether an ``(A, B)`` pair satisfies the sketch constraints."""

    try:
        edge = _validate_edge(edge)
    except ValueError:
        return False
    if min_ratio < 1.0:
        return False
    if not (np.isfinite(A) and np.isfinite(B)):
        return False
    if not (0.0 < B < A < edge):
        return False
    if min_ratio > 1.0 and B * min_ratio > A:
        return False
    return bool(0.0 <= edge - A + B <= 1.0 and 0.0 <= edge + A + B <= 1.0)


def find_max_c_for_pair(
    edge: float,
    A: float,
    B: float,
    c_samples: int = 100,
    seed: int | np.random.Generator | None = None,
    c_sampler: CSampler = "grid",
    bipodal_mode: BipodalMode = "optimized",
    bipodal_starts: int = 8,
    optimizer_mode: str = "hybrid",
    entropy_gap_tol: float = 0.0,
    c_values: Sequence[float] | None = None,
    bipodal_entropy: BipodalEntropyCallback | None = None,
) -> CSearchResult:
    """Find the largest sampled ``c`` for which tripodal entropy beats bipodal entropy."""

    edge = _validate_edge(edge)
    F = tripodal_entropy_curvature_score(edge, A, B)
    H_second = binary_entropy_second_derivative(edge)
    rng = _rng(seed)
    values = _candidate_c_values(c_samples, rng, c_sampler) if c_values is None else np.asarray(c_values, dtype=float)

    checked_c = 0
    feasible_c = 0
    successful_c = 0
    best: tuple[float, float, float, float, float] | None = None
    for c in values:
        checked_c += 1
        try:
            graphon = tripodal_perturbation(edge, A, B, float(c))
        except ValueError:
            continue
        feasible_c += 1
        target_t = constructive_triangle_density(edge, A, B, float(c))
        tripodal_entropy = graphon.entropy()
        reference_entropy = _bipodal_reference_entropy(
            edge=edge,
            B=B,
            c=float(c),
            target_t=target_t,
            seed=int(rng.integers(0, 2**32 - 1)),
            bipodal_mode=bipodal_mode,
            bipodal_starts=bipodal_starts,
            optimizer_mode=optimizer_mode,
            tripodal=graphon,
            callback=bipodal_entropy,
        )
        if not np.isfinite(reference_entropy):
            continue
        entropy_gap = tripodal_entropy - reference_entropy
        if entropy_gap > entropy_gap_tol:
            successful_c += 1
            if best is None or c > best[0]:
                best = (float(c), target_t, tripodal_entropy, reference_entropy, entropy_gap)

    if best is None:
        return CSearchResult(
            edge=edge,
            A=A,
            B=B,
            F=F,
            H_second=H_second,
            c_max=None,
            t_min=None,
            tripodal_entropy=None,
            bipodal_entropy=None,
            entropy_gap=None,
            checked_c=checked_c,
            feasible_c=feasible_c,
            successful_c=successful_c,
            bipodal_mode=bipodal_mode,
        )

    c_max, t_min, tripodal_entropy, reference_entropy, entropy_gap = best
    return CSearchResult(
        edge=edge,
        A=A,
        B=B,
        F=F,
        H_second=H_second,
        c_max=c_max,
        t_min=t_min,
        tripodal_entropy=tripodal_entropy,
        bipodal_entropy=reference_entropy,
        entropy_gap=entropy_gap,
        checked_c=checked_c,
        feasible_c=feasible_c,
        successful_c=successful_c,
        bipodal_mode=bipodal_mode,
    )


def constructive_tripodal_search(config: ConstructiveSearchConfig) -> pd.DataFrame:
    """Run the full fixed-edge constructive search and return one row per accepted pair."""

    edge = _validate_edge(config.edge)
    rng = _rng(config.seed)
    pairs = sample_ab_pairs(
        edge=edge,
        samples=config.ab_samples,
        seed=rng,
        min_ratio=config.min_ratio,
        max_attempts=config.max_ab_attempts,
    )
    records: list[dict[str, float | int | str | None | bool]] = []
    for pair_index, (A, B) in enumerate(pairs):
        F = tripodal_entropy_curvature_score(edge, A, B)
        H_second = binary_entropy_second_derivative(edge)
        passes_screen = F > H_second
        if not passes_screen:
            continue
        result = find_max_c_for_pair(
            edge=edge,
            A=A,
            B=B,
            c_samples=config.c_samples,
            seed=rng,
            c_sampler=config.c_sampler,
            bipodal_mode=config.bipodal_mode,
            bipodal_starts=config.bipodal_starts,
            optimizer_mode=config.optimizer_mode,
            entropy_gap_tol=config.entropy_gap_tol,
        )
        record = result.as_dict()
        record["pair_index"] = pair_index
        record["passes_screen"] = passes_screen
        records.append(record)
    return pd.DataFrame.from_records(records)


def _bipodal_reference_entropy(
    *,
    edge: float,
    B: float,
    c: float,
    target_t: float,
    seed: int,
    bipodal_mode: BipodalMode,
    bipodal_starts: int,
    optimizer_mode: str,
    tripodal: StepGraphon,
    callback: BipodalEntropyCallback | None,
) -> float:
    if callback is not None:
        return float(callback(tripodal, target_t))
    if bipodal_mode == "merged":
        return merged_bipodal_reference(edge, B, c).entropy()
    if bipodal_mode != "optimized":
        raise ValueError("bipodal_mode must be 'optimized' or 'merged'")
    from .families import BipodalFamily
    from .optimize import optimize_entropy_for_family

    result = optimize_entropy_for_family(
        model="triangle",
        target_e=edge,
        target_t=target_t,
        family=BipodalFamily(),
        starts=bipodal_starts,
        seed=seed,
        mode=optimizer_mode,
    )
    return float(result.entropy) if result.success else float("nan")


def _candidate_c_values(c_samples: int, rng: np.random.Generator, c_sampler: CSampler) -> NDArray[np.float64]:
    if c_samples < 0:
        raise ValueError("c_samples must be non-negative")
    if c_sampler == "grid":
        return np.linspace(0.0, 1.0, c_samples + 2, dtype=float)[1:-1]
    if c_sampler == "uniform":
        return np.sort(rng.uniform(0.0, 1.0, size=c_samples))
    raise ValueError("c_sampler must be 'grid' or 'uniform'")


def _entropy_scalar(value: float) -> float:
    out = binary_entropy(float(value))
    return float(out)


def _validate_edge(edge: float) -> float:
    if not np.isfinite(edge) or not (0.0 < edge < 1.0):
        raise ValueError("edge must lie strictly between 0 and 1")
    return float(edge)


def _validate_c(c: float) -> None:
    if not np.isfinite(c) or not (0.0 <= c < 1.0):
        raise ValueError("c must satisfy 0 <= c < 1")


def _rng(seed: int | np.random.Generator | None) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)
