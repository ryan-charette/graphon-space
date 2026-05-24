"""Numerical exploration of graphon feasible and entropy-maximizing space."""

from .densities import (
    ENTROPY_CONVENTION,
    binary_entropy,
    edge_density,
    entropy,
    reduced_two_star_density,
    triangle_density,
    two_star_density,
)
from .graphon import StepGraphon

__all__ = [
    "ENTROPY_CONVENTION",
    "StepGraphon",
    "binary_entropy",
    "edge_density",
    "entropy",
    "reduced_two_star_density",
    "triangle_density",
    "two_star_density",
]

__version__ = "0.1.0"
