"""Tests for PoincareBall model."""

import numpy as np
import pytest

from flux_hyperbolic.poincare import PoincareBall


class TestPoincareBallCreation:
    def test_default_curvature(self):
        ball = PoincareBall()
        assert ball.c == 1.0
        assert ball.radius == pytest.approx(1.0)

    def test_custom_curvature(self):
        ball = PoincareBall(curvature=2.0)
        assert ball.radius == pytest.approx(1.0 / np.sqrt(2.0))

    def test_invalid_curvature(self):
        with pytest.raises(ValueError):
            PoincareBall(curvature=0.0)


class TestDistance:
    def test_zero_distance(self):
        ball = PoincareBall()
        x = np.array([0.1, 0.2, 0.3])
        assert ball.distance(x, x) == pytest.approx(0.0, abs=1e-10)

    def test_symmetry(self):
        ball = PoincareBall()
        u = np.array([0.1, 0.2, 0.3])
        v = np.array([0.4, 0.1, 0.2])
        assert ball.distance(u, v) == pytest.approx(ball.distance(v, u))

    def test_triangle_inequality(self):
        ball = PoincareBall()
        a = np.array([0.1, 0.0, 0.0])
        b = np.array([0.0, 0.2, 0.0])
        c = np.array([0.0, 0.0, 0.3])
        assert ball.distance(a, c) <= ball.distance(a, b) + ball.distance(b, c) + 1e-10

    def test_origin_distance(self):
        ball = PoincareBall()
        origin = np.zeros(3)
        x = np.array([0.5, 0.0, 0.0])
        d = ball.distance(origin, x)
        assert d > 0
        # d(origin, x) = arccosh(1 + 2*0.25 / (1*0.75)) = arccosh(1 + 2/3) ≈ 0.9624
        assert d == pytest.approx(np.arccosh(1.0 + 2.0 / 3.0), rel=1e-6)


class TestProjection:
    def test_inside_unchanged(self):
        ball = PoincareBall()
        x = np.array([0.1, 0.2, 0.3])
        p = ball.project(x)
        np.testing.assert_array_almost_equal(p, x)

    def test_outside_clamped(self):
        ball = PoincareBall()
        x = np.array([0.9, 0.9, 0.9])  # norm > 1
        p = ball.project(x)
        assert np.linalg.norm(p) < 1.0


class TestExpLogMap:
    def test_exp_log_roundtrip(self):
        ball = PoincareBall()
        base = np.array([0.1, 0.1, 0.1])
        tangent = np.array([0.05, -0.03, 0.02])
        point = ball.exp_map(base, tangent)
        recovered = ball.log_map(base, point)
        np.testing.assert_array_almost_equal(tangent, recovered, decimal=3)

    def test_exp_preserves_ball(self):
        ball = PoincareBall()
        base = np.array([0.1, 0.0, 0.0])
        tangent = np.array([0.1, 0.1, 0.1])
        point = ball.exp_map(base, tangent)
        assert np.linalg.norm(point) < 1.0


class TestMobility:
    def test_self_mobility(self):
        ball = PoincareBall()
        x = np.array([0.1, 0.2, 0.3])
        assert ball.mobility(x, x) == pytest.approx(1.0, abs=1e-10)

    def test_distant_low_mobility(self):
        ball = PoincareBall()
        u = np.array([0.9, 0.0, 0.0])
        v = np.array([-0.9, 0.0, 0.0])
        assert ball.mobility(u, v) < 0.1
