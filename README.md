# graphon-space

`graphon-space` is a numerical discovery tool for exploring feasible `(e,t)`
regions and entropy-maximizing step graphons. It supports the edge-triangle and
edge-2-star models, symmetric and unrestricted bipodal baselines, general
`k`-podal graphons, structured clique-like and anti-clique-like starts, plotting,
and a command-line interface.

The implementation is intentionally numerical rather than a proof engine. It is
designed to test symmetric bipodal candidates against broader families.

## Quick Start

From this workspace:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .[dev]
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\graphon-space.exe sample --model triangle --kmax 3 --samples 1000 --out outputs\samples_triangle.parquet
.\.venv\Scripts\graphon-space.exe boundary --model two-star --kmax 3 --e-grid 25 --starts 8 --out outputs\boundary_2star.parquet
.\.venv\Scripts\graphon-space.exe compare --model triangle --edge 0.35 --t 0.04 --kmax 4 --starts 12
```

## Package Layout

- `graphon_space.graphon`: `StepGraphon` representation.
- `graphon_space.densities`: edge, triangle, 2-star, reduced 2-star, and entropy.
- `graphon_space.families`: ER, symmetric bipodal, bipodal, `k`-podal, and structured constructors.
- `graphon_space.sampling`: random, boundary-biased, Latin hypercube, and Sobol sampling.
- `graphon_space.optimize`: fixed-target entropy search and family comparison.
- `graphon_space.boundary`: fixed-edge min/max boundary searches and known envelope helpers.
- `graphon_space.phase`: symmetry classification and winner summaries.
- `graphon_space.diagnostics`: residuals, duplicate-pode checks, finite-difference Hessians.
- `graphon_space.visualize`: scatter, boundary, entropy, phase, gap, and pode heatmap plots.
- `graphon_space.cli`: command-line entry point.

## Entropy Convention

Entropy is stored using the edge-triangle convention:

```text
S(c,P) = sum_ij c_i c_j H(P_ij)
H(u) = -u log u - (1-u) log(1-u)
```

Some edge-2-star papers differ by a constant normalization. That does not
change optimizers, but it does affect reported entropy values.
