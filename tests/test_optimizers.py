import pytest

from graphon_space.boundary import known_triangle_upper, optimize_fixed_edge_boundary_for_family
from graphon_space.families import KPodalFamily
from graphon_space.optimize import compare_families


def test_boundary_optimizer_finds_triangle_clique_upper_start():
    edge = 0.36
    result = optimize_fixed_edge_boundary_for_family(
        model="triangle",
        edge=edge,
        family=KPodalFamily(2),
        direction="max",
        starts=2,
        seed=4,
    )

    assert result.edge_residual <= 1e-8
    assert result.target_t == pytest.approx(known_triangle_upper(edge), abs=1e-7)


def test_compare_families_returns_records():
    df = compare_families(
        model="triangle",
        target_e=0.4,
        target_t=0.4**3,
        kmax=3,
        starts=2,
        seed=3,
        mode="hybrid",
    )

    assert {"family", "success", "entropy", "edge_residual", "t_residual"}.issubset(df.columns)
    assert len(df) >= 2
