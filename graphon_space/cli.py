"""Command-line interface for graphon-space."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .boundary import boundary_grid
from .constructive import ConstructiveSearchConfig, constructive_tripodal_search
from .io import graphon_from_record, load_dataframe, save_dataframe
from .optimize import compare_families, optimize_grid
from .sampling import SamplingConfig, sample_feasible_region


def _parse_grid(text: str) -> tuple[int, int]:
    if "x" in text:
        left, right = text.lower().split("x", 1)
        return int(left), int(right)
    value = int(text)
    return value, value


def _edge_grid(count: int) -> np.ndarray:
    return np.linspace(1e-4, 1.0 - 1e-4, count)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="graphon-space")
    sub = parser.add_subparsers(dest="command", required=True)

    sample = sub.add_parser("sample", help="sample feasible graphon points")
    sample.add_argument("--model", choices=["triangle", "two-star"], required=True)
    sample.add_argument("--kmax", type=int, default=4)
    sample.add_argument("--samples", type=int, default=10_000)
    sample.add_argument("--sampler", choices=["uniform", "boundary", "beta", "sobol", "latin"], default="uniform")
    sample.add_argument("--family", default="k-podal")
    sample.add_argument("--seed", type=int, default=None)
    sample.add_argument("--out", required=True)

    boundary = sub.add_parser("boundary", help="optimize min/max t at fixed edge densities")
    boundary.add_argument("--model", choices=["triangle", "two-star"], required=True)
    boundary.add_argument("--kmax", type=int, default=4)
    boundary.add_argument("--e-grid", type=int, default=50)
    boundary.add_argument("--starts", type=int, default=20)
    boundary.add_argument("--seed", type=int, default=None)
    boundary.add_argument("--out", required=True)

    compare = sub.add_parser("compare", help="compare families at one target (e,t)")
    compare.add_argument("--model", choices=["triangle", "two-star"], required=True)
    compare.add_argument("--edge", type=float, required=True)
    compare.add_argument("--t", type=float, required=True)
    compare.add_argument("--kmax", type=int, default=4)
    compare.add_argument("--starts", type=int, default=20)
    compare.add_argument("--seed", type=int, default=None)
    compare.add_argument("--mode", choices=["constrained", "penalty", "hybrid"], default="hybrid")
    compare.add_argument("--out", default=None)

    opt_grid = sub.add_parser("optimize-grid", help="run entropy search on a rectangular target grid")
    opt_grid.add_argument("--model", choices=["triangle", "two-star"], required=True)
    opt_grid.add_argument("--kmax", type=int, default=4)
    opt_grid.add_argument("--grid", default="20x20")
    opt_grid.add_argument("--starts", type=int, default=8)
    opt_grid.add_argument("--seed", type=int, default=None)
    opt_grid.add_argument("--mode", choices=["constrained", "penalty", "hybrid"], default="hybrid")
    opt_grid.add_argument("--out", required=True)

    stability = sub.add_parser("stability", help="edge-2-star e=1/2 symmetry-line scan")
    stability.add_argument("--model", choices=["two-star"], default="two-star")
    stability.add_argument("--edge", type=float, default=0.5)
    stability.add_argument("--t-tilde-grid", type=int, default=50)
    stability.add_argument("--kmax", type=int, default=3)
    stability.add_argument("--starts", type=int, default=10)
    stability.add_argument("--seed", type=int, default=None)
    stability.add_argument("--mode", choices=["constrained", "penalty", "hybrid"], default="hybrid")
    stability.add_argument("--out", required=True)

    constructive = sub.add_parser("constructive", help="run the fixed-edge A,B,c tripodal search")
    constructive.add_argument("--edge", type=float, required=True)
    constructive.add_argument("--ab-samples", type=int, default=1_000)
    constructive.add_argument("--c-samples", type=int, default=100)
    constructive.add_argument("--seed", type=int, default=None)
    constructive.add_argument("--min-ratio", type=float, default=10.0)
    constructive.add_argument("--c-sampler", choices=["grid", "uniform"], default="grid")
    constructive.add_argument("--bipodal-mode", choices=["optimized", "merged"], default="optimized")
    constructive.add_argument("--bipodal-starts", type=int, default=8)
    constructive.add_argument("--mode", choices=["constrained", "penalty", "hybrid"], default="hybrid")
    constructive.add_argument("--entropy-gap-tol", type=float, default=0.0)
    constructive.add_argument("--out", required=True)

    plot = sub.add_parser("plot", help="plot saved experiment data")
    plot.add_argument("--input", required=True)
    plot.add_argument(
        "--kind",
        choices=["scatter", "boundary", "entropy", "phase-map", "sym-gap", "pode"],
        required=True,
    )
    plot.add_argument("--out", required=True)
    plot.add_argument("--row", type=int, default=0)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "sample":
        config = SamplingConfig(
            model=args.model,
            kmax=args.kmax,
            samples=args.samples,
            sampler=args.sampler,
            seed=args.seed,
            family=args.family,
        )
        df = sample_feasible_region(config)
        save_dataframe(df, args.out)
        print(f"wrote {len(df)} samples to {args.out}")
        return 0

    if args.command == "boundary":
        df = boundary_grid(
            model=args.model,
            e_values=_edge_grid(args.e_grid),
            kmax=args.kmax,
            starts=args.starts,
            seed=args.seed,
        )
        save_dataframe(df, args.out)
        print(f"wrote {len(df)} boundary records to {args.out}")
        return 0

    if args.command == "compare":
        df = compare_families(
            model=args.model,
            target_e=args.edge,
            target_t=args.t,
            kmax=args.kmax,
            starts=args.starts,
            seed=args.seed,
            mode=args.mode,
        )
        if args.out:
            save_dataframe(df, args.out)
            print(f"wrote {len(df)} comparison records to {args.out}")
        columns = ["family", "success", "entropy", "edge_residual", "t_residual", "symmetry_class"]
        print(df[columns].to_string(index=False))
        return 0

    if args.command == "optimize-grid":
        e_count, t_count = _parse_grid(args.grid)
        e_values = _edge_grid(e_count)
        # A broad normalized t-grid. Users can post-filter infeasible failures.
        t_values = np.linspace(1e-5, 1.0 - 1e-5, t_count)
        df = optimize_grid(
            model=args.model,
            e_values=e_values,
            t_values=t_values,
            kmax=args.kmax,
            starts=args.starts,
            seed=args.seed,
            mode=args.mode,
        )
        save_dataframe(df, args.out)
        print(f"wrote {len(df)} optimization records to {args.out}")
        return 0

    if args.command == "stability":
        t_tilde = np.linspace(0.0, 0.125, args.t_tilde_grid)
        t_values = t_tilde + args.edge * args.edge
        df = optimize_grid(
            model="two-star",
            e_values=[args.edge],
            t_values=t_values,
            kmax=args.kmax,
            starts=args.starts,
            seed=args.seed,
        )
        df["target_t_tilde"] = df["target_t"] - args.edge * args.edge
        df["known_crossover_t_tilde"] = 0.03727637
        save_dataframe(df, args.out)
        print(f"wrote {len(df)} stability scan records to {args.out}")
        return 0

    if args.command == "constructive":
        config = ConstructiveSearchConfig(
            edge=args.edge,
            ab_samples=args.ab_samples,
            c_samples=args.c_samples,
            seed=args.seed,
            min_ratio=args.min_ratio,
            c_sampler=args.c_sampler,
            bipodal_mode=args.bipodal_mode,
            bipodal_starts=args.bipodal_starts,
            optimizer_mode=args.mode,
            entropy_gap_tol=args.entropy_gap_tol,
        )
        df = constructive_tripodal_search(config)
        save_dataframe(df, args.out)
        print(f"wrote {len(df)} constructive search records to {args.out}")
        return 0

    if args.command == "plot":
        from .visualize import (
            boundary_plot,
            entropy_heatmap,
            feasible_scatter,
            pode_heatmap,
            symmetric_gap_plot,
            winner_map,
        )

        df = load_dataframe(args.input)
        out = Path(args.out)
        if args.kind == "scatter":
            feasible_scatter(df, out=out)
        elif args.kind == "boundary":
            boundary_plot(df, out=out)
        elif args.kind == "entropy":
            entropy_heatmap(df, out=out)
        elif args.kind == "phase-map":
            winner_map(df, out=out)
        elif args.kind == "sym-gap":
            symmetric_gap_plot(df, out=out)
        elif args.kind == "pode":
            graphon = graphon_from_record(json.loads(df.iloc[args.row].to_json()))
            pode_heatmap(graphon, out=out)
        print(f"wrote plot to {args.out}")
        return 0

    parser.error("unhandled command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
