import pytest

from graphon_space.constructive import (
    binary_entropy_second_derivative,
    constructive_triangle_density,
    find_max_c_for_pair,
    is_valid_ab_pair,
    passes_entropy_curvature_screen,
    sample_ab_pairs,
    tripodal_entropy_curvature_score,
    tripodal_perturbation,
)


def test_tripodal_construction_fixes_edge_and_matches_triangle_formula():
    edge, A, B, c = 0.4, 0.1, 0.03, 0.2

    graphon = tripodal_perturbation(edge, A, B, c)

    assert graphon.sizes.tolist() == pytest.approx([0.1, 0.1, 0.8])
    assert graphon.edge_density() == pytest.approx(edge, abs=1e-12)
    assert graphon.triangle_density() == pytest.approx(
        constructive_triangle_density(edge, A, B, c),
        abs=1e-12,
    )
    assert graphon.triangle_density() == pytest.approx(edge**3 - c**3 * (A**3 - B**3), abs=1e-12)


def test_entropy_curvature_screen_compares_to_entropy_hessian():
    edge, A, B = 0.4, 0.1, 0.03

    assert tripodal_entropy_curvature_score(edge, A, B) > binary_entropy_second_derivative(edge)
    assert passes_entropy_curvature_screen(edge, A, B)


def test_ab_sampling_respects_b_much_smaller_than_a_constraint():
    pairs = sample_ab_pairs(0.4, samples=25, seed=11, min_ratio=8.0)

    assert len(pairs) == 25
    for A, B in pairs:
        assert is_valid_ab_pair(0.4, A, B, min_ratio=8.0)
        assert B * 8.0 <= A


def test_find_max_c_uses_tripodal_greater_than_bipodal_rule():
    edge, A, B = 0.4, 0.1, 0.03

    result = find_max_c_for_pair(
        edge,
        A,
        B,
        c_values=[0.1, 0.2, 0.3],
        bipodal_mode="merged",
        bipodal_entropy=lambda graphon, _target_t: graphon.entropy() - 0.01,
    )

    assert result.c_max == pytest.approx(0.3)
    assert result.t_min == pytest.approx(constructive_triangle_density(edge, A, B, 0.3))
    assert result.successful_c == 3
