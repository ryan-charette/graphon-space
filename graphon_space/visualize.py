"""Matplotlib visualizations for graphon-space experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .graphon import StepGraphon


def _finish(fig, out: str | Path | None = None):
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=180, bbox_inches="tight")
    return fig


def feasible_scatter(
    df: pd.DataFrame,
    t_column: str = "t",
    color: str = "k",
    out: str | Path | None = None,
):
    fig, ax = plt.subplots(figsize=(7, 5))
    if color in df.columns:
        scatter = ax.scatter(df["e"], df[t_column], c=df[color], s=8, alpha=0.45, cmap="viridis")
        fig.colorbar(scatter, ax=ax, label=color)
    else:
        ax.scatter(df["e"], df[t_column], s=8, alpha=0.45)
    ax.set_xlabel("edge density e")
    ax.set_ylabel(t_column)
    ax.set_title("Sampled feasible region")
    return _finish(fig, out)


def boundary_plot(
    df: pd.DataFrame,
    out: str | Path | None = None,
):
    fig, ax = plt.subplots(figsize=(7, 5))
    for direction, group in df.groupby("direction"):
        ax.plot(group["target_e"], group["observed_t"], marker="o", linewidth=1.5, label=direction)
    if "known_upper" in df.columns:
        upper = df[df["direction"] == "max"].sort_values("target_e")
        ax.plot(upper["target_e"], upper["known_upper"], "--", color="black", linewidth=1, label="known upper")
    ax.set_xlabel("edge density e")
    ax.set_ylabel("t")
    ax.set_title("Estimated fixed-edge boundary")
    ax.legend()
    return _finish(fig, out)


def entropy_heatmap(
    df: pd.DataFrame,
    bins: int = 60,
    out: str | Path | None = None,
):
    ok = df[df["success"].astype(bool)] if "success" in df.columns else df
    pivot = ok.pivot_table(index="target_t", columns="target_e", values="entropy", aggfunc="max")
    if pivot.empty:
        pivot = ok.pivot_table(index="t", columns="e", values="entropy", aggfunc="max")
    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(
        pivot.values,
        origin="lower",
        aspect="auto",
        extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()],
        cmap="magma",
    )
    fig.colorbar(image, ax=ax, label="best entropy")
    ax.set_xlabel("edge density e")
    ax.set_ylabel("t")
    ax.set_title("Entropy heatmap")
    return _finish(fig, out)


def winner_map(
    df: pd.DataFrame,
    out: str | Path | None = None,
):
    ok = df[df["success"].astype(bool)].copy()
    idx = ok.groupby(["target_e", "target_t"])["entropy"].idxmax()
    winners = ok.loc[idx]
    families = {name: i for i, name in enumerate(sorted(winners["family"].unique()))}
    winners["family_code"] = winners["family"].map(families)
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(
        winners["target_e"],
        winners["target_t"],
        c=winners["family_code"],
        s=30,
        cmap="tab20",
    )
    handles, _ = scatter.legend_elements()
    labels = list(families.keys())
    ax.legend(handles, labels, title="winner", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_xlabel("edge density e")
    ax.set_ylabel("t")
    ax.set_title("Winning family map")
    return _finish(fig, out)


def symmetric_gap_plot(
    df: pd.DataFrame,
    out: str | Path | None = None,
):
    ok = df[df["success"].astype(bool)].copy()
    best = ok.groupby(["target_e", "target_t"])["entropy"].max().rename("best")
    sym = (
        ok[ok["family"] == "symmetric-bipodal"]
        .groupby(["target_e", "target_t"])["entropy"]
        .max()
        .rename("sym")
    )
    gap = pd.concat([best, sym], axis=1).dropna().reset_index()
    gap["delta"] = gap["best"] - gap["sym"]
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(gap["target_e"], gap["target_t"], c=gap["delta"], s=30, cmap="coolwarm")
    fig.colorbar(scatter, ax=ax, label="S_best - S_sym")
    ax.set_xlabel("edge density e")
    ax.set_ylabel("t")
    ax.set_title("Symmetric bipodal entropy gap")
    return _finish(fig, out)


def pode_heatmap(
    graphon: StepGraphon,
    out: str | Path | None = None,
):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    image = ax.imshow(graphon.matrix, vmin=0.0, vmax=1.0, cmap="viridis")
    fig.colorbar(image, ax=ax, label="P_ij")
    labels = [f"{i}: {size:.3f}" for i, size in enumerate(graphon.sizes)]
    ax.set_xticks(np.arange(graphon.k), labels=labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(graphon.k), labels=labels)
    ax.set_title("Pode matrix")
    return _finish(fig, out)
