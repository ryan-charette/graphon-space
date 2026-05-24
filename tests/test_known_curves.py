import pytest

from graphon_space.boundary import (
    known_triangle_upper,
    known_two_star_reduced_upper,
    known_two_star_upper,
)


@pytest.mark.parametrize("edge", [0.0, 0.25, 0.5, 0.81, 1.0])
def test_triangle_upper_curve(edge):
    assert known_triangle_upper(edge) == pytest.approx(edge**1.5)


def test_two_star_upper_curve_branches():
    assert known_two_star_upper(0.25) == pytest.approx((1 - 0.25) ** 1.5 + 2 * 0.25 - 1)
    assert known_two_star_upper(0.5) == pytest.approx(2**0.5 / 4)
    assert known_two_star_upper(0.81) == pytest.approx(0.81**1.5)


def test_reduced_two_star_upper():
    edge = 0.81
    assert known_two_star_reduced_upper(edge) == pytest.approx(known_two_star_upper(edge) - edge**2)
