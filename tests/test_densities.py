import math

import numpy as np

from graphon_space.densities import binary_entropy
from graphon_space.families import bipodal, erdos_renyi


def test_erdos_renyi_densities():
    p = 0.37
    graphon = erdos_renyi(p)

    assert graphon.edge_density() == pytest_approx(p)
    assert graphon.triangle_density() == pytest_approx(p**3)
    assert graphon.two_star_density() == pytest_approx(p**2)
    assert graphon.reduced_two_star_density() == pytest_approx(0.0)
    assert graphon.entropy() == pytest_approx(binary_entropy(p))


def test_bipodal_closed_formulas():
    c, a, b, d = 0.31, 0.2, 0.8, 0.47
    graphon = bipodal(c, a, b, d)

    expected_e = c**2 * a + (1 - c) ** 2 * b + 2 * c * (1 - c) * d
    expected_triangle = (
        c**3 * a**3
        + (1 - c) ** 3 * b**3
        + 3 * c**2 * (1 - c) * a * d**2
        + 3 * c * (1 - c) ** 2 * b * d**2
    )
    d1 = c * a + (1 - c) * d
    d2 = c * d + (1 - c) * b
    expected_two_star = c * d1**2 + (1 - c) * d2**2

    assert graphon.edge_density() == pytest_approx(expected_e)
    assert graphon.triangle_density() == pytest_approx(expected_triangle)
    assert graphon.two_star_density() == pytest_approx(expected_two_star)


def test_binary_entropy_endpoints_are_stable():
    values = np.array([0.0, 1.0, 0.5])
    observed = binary_entropy(values)
    assert observed[0] == 0.0
    assert observed[1] == 0.0
    assert observed[2] == pytest_approx(math.log(2.0))


def pytest_approx(value, tol=1e-12):
    import pytest

    return pytest.approx(value, abs=tol, rel=tol)
