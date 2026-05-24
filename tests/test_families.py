import pytest

from graphon_space.families import (
    anti_clique_like,
    clique_like,
    symmetric_bipodal,
    turan_like,
)


def test_clique_like_hits_triangle_upper_boundary():
    edge = 0.36
    graphon = clique_like(edge)

    assert graphon.edge_density() == pytest.approx(edge, abs=1e-12)
    assert graphon.triangle_density() == pytest.approx(edge**1.5, abs=1e-12)


def test_anti_clique_like_edge_density():
    edge = 0.64
    graphon = anti_clique_like(edge)

    assert graphon.edge_density() == pytest.approx(edge, abs=1e-12)


def test_symmetric_bipodal_classic_constraints():
    graphon = symmetric_bipodal(0.1, 0.7)

    assert graphon.sizes[0] == pytest.approx(0.5)
    assert graphon.sizes[1] == pytest.approx(0.5)
    assert graphon.matrix[0, 0] == pytest.approx(graphon.matrix[1, 1])


def test_turan_equal_sizes_edge_density():
    graphon = turan_like(3)

    assert graphon.edge_density() == pytest.approx(2 / 3, abs=1e-12)
    assert graphon.triangle_density() == pytest.approx(2 / 9, abs=1e-12)
