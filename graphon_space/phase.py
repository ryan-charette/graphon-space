"""Symmetry and phase classification helpers."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd

from .graphon import StepGraphon


def classify_symmetry(graphon: StepGraphon, tol: float = 1e-6) -> str:
    """Classify a candidate by simple numerical symmetry signatures."""

    p = graphon.matrix
    c = graphon.sizes
    if graphon.k == 1:
        return "erdos-renyi"
    if np.allclose(p, p[0, 0], atol=tol) and np.allclose(c, c[0], atol=tol):
        return "constant"
    if graphon.k == 2:
        equal_sizes = abs(c[0] - c[1]) <= tol
        equal_diagonal = abs(p[0, 0] - p[1, 1]) <= tol
        if equal_sizes and equal_diagonal:
            return "symmetric-bipodal"
        return "nonsymmetric-bipodal"

    groups = duplicate_pode_groups(graphon, tol=tol)
    group_sizes = sorted((len(group) for group in groups), reverse=True)
    if len(groups) < graphon.k:
        signature = ",".join(str(size) for size in group_sizes)
        return f"multipodal-with-merged-podes({signature})"
    return f"{graphon.k}-podal"


def duplicate_pode_groups(graphon: StepGraphon, tol: float = 1e-6) -> list[list[int]]:
    """Group podes with nearly identical sizes and matrix rows."""

    groups: list[list[int]] = []
    used: set[int] = set()
    for i in range(graphon.k):
        if i in used:
            continue
        group = [i]
        used.add(i)
        for j in range(i + 1, graphon.k):
            if j in used:
                continue
            row_distance = float(np.linalg.norm(graphon.matrix[i] - graphon.matrix[j], ord=np.inf))
            size_distance = abs(float(graphon.sizes[i] - graphon.sizes[j]))
            if row_distance <= tol and size_distance <= tol:
                group.append(j)
                used.add(j)
        groups.append(group)
    return groups


def summarize_winners(results: Iterable[dict]) -> pd.DataFrame:
    """Return best result per target from serialized optimization records."""

    df = pd.DataFrame.from_records(list(results))
    if df.empty:
        return df
    ok = df[df["success"].astype(bool)].copy()
    if ok.empty:
        return ok
    idx = ok.groupby(["target_e", "target_t"])["entropy"].idxmax()
    winners = ok.loc[idx].reset_index(drop=True)
    winner_counts = Counter(winners["family"])
    winners.attrs["winner_counts"] = dict(winner_counts)
    return winners


def entropy_gap_to_family(results: Iterable[dict], family: str = "symmetric-bipodal") -> pd.DataFrame:
    """Compute ``best entropy - selected-family entropy`` per target."""

    df = pd.DataFrame.from_records(list(results))
    if df.empty:
        return df
    ok = df[df["success"].astype(bool)].copy()
    best = ok.groupby(["target_e", "target_t"])["entropy"].max().rename("best_entropy")
    baseline = (
        ok[ok["family"] == family]
        .groupby(["target_e", "target_t"])["entropy"]
        .max()
        .rename("baseline_entropy")
    )
    joined = pd.concat([best, baseline], axis=1).reset_index()
    joined["entropy_gap"] = joined["best_entropy"] - joined["baseline_entropy"]
    return joined
