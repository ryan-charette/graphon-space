"""Compare the 2023 screening normalization with the current curvature score."""

from __future__ import annotations

import argparse
import json

from graphon_space.constructive import (
    binary_entropy_second_derivative,
    tripodal_entropy_curvature_score,
)


def compare_scores(edge: float, a: float, b: float) -> dict[str, float | bool]:
    """Evaluate the two normalizations for a valid perturbation pair.

    The historical ``graphons_take2.py`` used ``(A**3 - B**3)**(2/3)``
    instead of ``A**2 + B**2``. Reuse the current validated numerator;
    this preserves just that distinct experiment without a second optimizer.
    """
    curvature = tripodal_entropy_curvature_score(edge, a, b)
    historical = curvature * (a * a + b * b) / ((a**3 - b**3) ** (2.0 / 3.0))
    threshold = binary_entropy_second_derivative(edge)
    return {
        "edge": edge,
        "A": a,
        "B": b,
        "curvature_score": curvature,
        "historical_score": historical,
        "H_second": threshold,
        "curvature_passes": curvature > threshold,
        "historical_passes": historical > threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge", type=float, default=0.3)
    parser.add_argument("--a", type=float, default=0.1)
    parser.add_argument("--b", type=float, default=0.01)
    args = parser.parse_args()
    try:
        result = compare_scores(args.edge, args.a, args.b)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
