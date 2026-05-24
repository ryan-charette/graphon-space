from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs" / "final" / "data"
FIG = ROOT / "outputs" / "final" / "figures"
REPORT = ROOT / "outputs" / "final" / "final_report.md"


def read(name: str) -> pd.DataFrame:
    return pd.read_parquet(DATA / name)


def savefig(name: str):
    path = FIG / name
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def triangle_upper(e):
    return np.asarray(e) ** 1.5


def two_star_upper(e):
    e = np.asarray(e)
    return np.where(
        np.isclose(e, 0.5),
        np.sqrt(2.0) / 4.0,
        np.where(e > 0.5, e**1.5, (1 - e) ** 1.5 + 2 * e - 1),
    )


def plot_feasible_triangle():
    df = read("samples_triangle.parquet")
    curve = np.linspace(0, 1, 500)
    plt.figure(figsize=(7.2, 5.2))
    plt.scatter(df["e"], df["t_triangle"], c=df["k"], s=3, alpha=0.25, cmap="viridis", rasterized=True)
    plt.plot(curve, curve**3, color="black", linewidth=1.6, label="ER: t=e^3")
    plt.plot(curve, triangle_upper(curve), color="#b2182b", linewidth=1.6, label="upper: t=e^(3/2)")
    plt.xlabel("edge density e")
    plt.ylabel("triangle density t")
    plt.title("Edge-triangle sampled feasible region")
    plt.colorbar(label="k")
    plt.legend(loc="upper left")
    savefig("final_triangle_feasible_with_curves.png")


def plot_feasible_two_star():
    df = read("samples_2star.parquet")
    curve = np.linspace(0, 1, 500)
    plt.figure(figsize=(7.2, 5.2))
    plt.scatter(df["e"], df["t_2star"], c=df["k"], s=3, alpha=0.25, cmap="viridis", rasterized=True)
    plt.plot(curve, curve**2, color="black", linewidth=1.6, label="constant degree / ER: t=e^2")
    plt.plot(curve, two_star_upper(curve), color="#b2182b", linewidth=1.6, label="known upper envelope")
    plt.xlabel("edge density e")
    plt.ylabel("2-star density t")
    plt.title("Edge-2-star sampled feasible region")
    plt.colorbar(label="k")
    plt.legend(loc="upper left")
    savefig("final_2star_feasible_with_curves.png")


def plot_boundary_validation():
    for model, filename, ylabel in [
        ("triangle", "boundary_triangle.parquet", "triangle density t"),
        ("2star", "boundary_2star.parquet", "2-star density t"),
    ]:
        df = read(filename)
        max_df = df[df["direction"] == "max"].sort_values("target_e")
        min_df = df[df["direction"] == "min"].sort_values("target_e")
        curve = np.linspace(0, 1, 500)
        known = triangle_upper(curve) if model == "triangle" else two_star_upper(curve)
        plt.figure(figsize=(7.2, 5.2))
        plt.plot(min_df["target_e"], min_df["observed_t"], "o-", markersize=3, label="estimated lower")
        plt.plot(max_df["target_e"], max_df["observed_t"], "o-", markersize=3, label="estimated upper")
        plt.plot(curve, known, "--", color="black", linewidth=1.4, label="known upper")
        if model == "2star":
            plt.plot(curve, curve**2, ":", color="gray", linewidth=1.4, label="ER / lower")
        plt.xlabel("edge density e")
        plt.ylabel(ylabel)
        plt.title(f"{model} fixed-edge boundary validation")
        plt.legend()
        savefig(f"final_{model}_boundary_validation.png")


def plot_compare_bars():
    for stem, title in [
        ("compare_triangle_e035_t002.parquet", "Triangle compare: e=0.35, t=0.02"),
        ("compare_2star_e05_t029.parquet", "2-star compare: e=0.5, t=0.29"),
    ]:
        df = read(stem).copy()
        df["label"] = df["family"] + "\n" + df["success"].map({True: "ok", False: "miss"})
        plt.figure(figsize=(7.4, 4.4))
        colors = np.where(df["success"], "#2166ac", "#b2182b")
        plt.bar(df["label"], df["entropy"], color=colors)
        plt.ylabel("entropy")
        plt.title(title)
        plt.xticks(rotation=0)
        savefig(f"final_{stem.replace('.parquet', '_entropy_bars.png')}")


def plot_stability():
    df = read("stability_2star_e05.parquet").copy()
    ok = df[df["success"]].copy()
    plt.figure(figsize=(7.4, 5.0))
    if not ok.empty:
        idx = ok.groupby("target_t_tilde")["entropy"].idxmax()
        winners = ok.loc[idx].sort_values("target_t_tilde")
        families = {name: i for i, name in enumerate(sorted(winners["family"].unique()))}
        plt.scatter(
            winners["target_t_tilde"],
            winners["entropy"],
            c=winners["family"].map(families),
            s=48,
            cmap="tab10",
            label="best successful candidate",
        )
        for family, group in winners.groupby("family"):
            plt.plot(group["target_t_tilde"], group["entropy"], linewidth=1, label=family)
    plt.axvline(0.03727637, color="#b2182b", linestyle="--", linewidth=1.4, label="reported crossover")
    plt.xlabel("reduced 2-star density t - e^2 at e=1/2")
    plt.ylabel("best entropy found")
    plt.title("Edge-2-star e=1/2 stability scan")
    plt.legend(loc="best")
    savefig("final_2star_stability_scan.png")


def summarize() -> str:
    samples_triangle = read("samples_triangle.parquet")
    samples_2star = read("samples_2star.parquet")
    boundary_triangle = read("boundary_triangle.parquet")
    boundary_2star = read("boundary_2star.parquet")
    grid_triangle = read("entropy_grid_triangle.parquet")
    grid_2star = read("entropy_grid_2star.parquet")
    stability = read("stability_2star_e05.parquet")
    compare_triangle = read("compare_triangle_e035_t002.parquet")
    compare_2star = read("compare_2star_e05_t029.parquet")

    tri_max = boundary_triangle[boundary_triangle["direction"] == "max"].copy()
    tri_max["upper_error"] = (tri_max["observed_t"] - tri_max["known_upper"]).abs()
    star_max = boundary_2star[boundary_2star["direction"] == "max"].copy()
    star_max["upper_error"] = (star_max["observed_t"] - star_max["known_upper"]).abs()

    def winner_line(df):
        ok = df[df["success"]]
        if ok.empty:
            return "no successful candidates"
        row = ok.loc[ok["entropy"].idxmax()]
        return f"{row['family']} (S={row['entropy']:.6f}, class={row['symmetry_class']})"

    grid_tri_success = int(grid_triangle["success"].sum()) if "success" in grid_triangle else 0
    grid_star_success = int(grid_2star["success"].sum()) if "success" in grid_2star else 0
    stability_success = int(stability["success"].sum()) if "success" in stability else 0

    return "\n".join(
        [
            "# Final Graphon-Space Numerical Run",
            "",
            "## Data Products",
            "",
            f"- Triangle samples: {len(samples_triangle):,} rows, k=1..{samples_triangle['k'].max()}.",
            f"- Edge-2-star samples: {len(samples_2star):,} rows, k=1..{samples_2star['k'].max()}.",
            f"- Triangle boundary records: {len(boundary_triangle):,}.",
            f"- Edge-2-star boundary records: {len(boundary_2star):,}.",
            f"- Triangle optimize-grid records: {len(grid_triangle):,}, successful family records: {grid_tri_success:,}.",
            f"- Edge-2-star optimize-grid records: {len(grid_2star):,}, successful family records: {grid_star_success:,}.",
            f"- Edge-2-star stability records: {len(stability):,}, successful family records: {stability_success:,}.",
            "",
            "## Validation Metrics",
            "",
            f"- Triangle max-boundary median absolute error vs e^(3/2): {tri_max['upper_error'].median():.4g}.",
            f"- Triangle max-boundary worst absolute error vs e^(3/2): {tri_max['upper_error'].max():.4g}.",
            f"- Edge-2-star max-boundary median absolute error vs known envelope: {star_max['upper_error'].median():.4g}.",
            f"- Edge-2-star max-boundary worst absolute error vs known envelope: {star_max['upper_error'].max():.4g}.",
            "",
            "## Target Comparisons",
            "",
            f"- Triangle target e=0.35, t=0.02 winner: {winner_line(compare_triangle)}.",
            f"- Edge-2-star target e=0.5, t=0.29 winner: {winner_line(compare_2star)}.",
            "",
            "## Notes",
            "",
            "- The optimize-grid runs are intentionally coarse first-pass maps; dense phase maps require much longer SLSQP runs.",
            "- The figures overlay known ER and upper-boundary curves where available.",
        ]
    )


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    plot_feasible_triangle()
    plot_feasible_two_star()
    plot_boundary_validation()
    plot_compare_bars()
    plot_stability()
    REPORT.write_text(summarize(), encoding="utf-8")
    print(f"wrote figures to {FIG}")
    print(f"wrote report to {REPORT}")


if __name__ == "__main__":
    main()
