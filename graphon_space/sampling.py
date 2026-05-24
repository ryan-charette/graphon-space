"""Sampling engines for feasible-region exploration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import qmc

from .families import (
    GraphonFamily,
    KPodalFamily,
    default_sampling_families,
    family_by_name,
    symmetric_from_upper,
    upper_triangle_count,
)
from .graphon import StepGraphon
from .io import graphon_record, records_to_dataframe

SamplerName = Literal["uniform", "boundary", "beta", "sobol", "latin"]


@dataclass
class SamplingConfig:
    model: str
    kmax: int = 4
    samples: int = 10_000
    sampler: SamplerName = "uniform"
    seed: int | None = None
    family: str = "k-podal"


def _simplex_from_unit_cube(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    exp_values = -np.log(np.clip(values, 1e-12, 1.0))
    total = np.sum(exp_values, axis=-1, keepdims=True)
    return exp_values / total


def sample_kpodal(
    k: int,
    n: int,
    sampler: SamplerName = "uniform",
    seed: int | None = None,
) -> list[StepGraphon]:
    """Sample general ``k``-podal graphons using random or low-discrepancy draws."""

    rng = np.random.default_rng(seed)
    dim = k + upper_triangle_count(k)
    if sampler == "sobol":
        engine = qmc.Sobol(d=dim, scramble=True, seed=seed)
        raw = engine.random(n)
    elif sampler == "latin":
        engine = qmc.LatinHypercube(d=dim, seed=seed)
        raw = engine.random(n)
    else:
        raw = rng.uniform(0.0, 1.0, size=(n, dim))

    graphons: list[StepGraphon] = []
    for row in raw:
        sizes = _simplex_from_unit_cube(row[:k])
        values = row[k:]
        if sampler == "boundary":
            values = rng.beta(0.35, 0.35, size=upper_triangle_count(k))
        elif sampler == "beta":
            values = rng.beta(2.0, 2.0, size=upper_triangle_count(k))
        graphons.append(StepGraphon(sizes, symmetric_from_upper(k, values), {"family": f"{k}-podal"}))
    return graphons


def sample_feasible_region(config: SamplingConfig) -> pd.DataFrame:
    """Generate graphons and return records with all densities."""

    rng = np.random.default_rng(config.seed)
    records: list[dict] = []
    if config.family == "all":
        families = default_sampling_families(config.kmax)
        per_family = max(1, config.samples // len(families))
        for family in families:
            child_seed = int(rng.integers(0, 2**32 - 1))
            for graphon in family.sample(per_family, seed=child_seed):
                records.append(graphon_record(graphon, model=config.model, seed=child_seed))
    elif config.family in {"k-podal", "kpodal", "podal"}:
        per_k = max(1, config.samples // config.kmax)
        for k in range(1, config.kmax + 1):
            child_seed = int(rng.integers(0, 2**32 - 1))
            for graphon in sample_kpodal(k, per_k, sampler=config.sampler, seed=child_seed):
                records.append(graphon_record(graphon, model=config.model, seed=child_seed))
    else:
        family: GraphonFamily = family_by_name(config.family, k=config.kmax)
        for graphon in family.sample(config.samples, seed=config.seed):
            records.append(graphon_record(graphon, model=config.model, seed=config.seed))
    return records_to_dataframe(records)


def bin_feasible_envelope(
    df: pd.DataFrame,
    e_bins: int = 100,
    t_column: str = "t",
) -> pd.DataFrame:
    """Track observed lower and upper ``t`` values in edge-density bins."""

    data = df[["e", t_column]].dropna().copy()
    data["e_bin"] = pd.cut(data["e"], bins=e_bins, include_lowest=True)
    grouped = data.groupby("e_bin", observed=True)
    envelope = grouped[t_column].agg(["min", "max", "count"]).reset_index()
    envelope["e_mid"] = envelope["e_bin"].apply(lambda interval: interval.mid)
    return envelope[["e_mid", "min", "max", "count"]]
