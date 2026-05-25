"""Tests for distance functions."""

import numpy as np
import pytest

from flux_hyperbolic.distance import poincare_distance, lorentz_distance, hyperbolic_distance


class TestPoincareDistance:
    def test_zero(self):
        x = np.array([0.1, 0.2, 0.3])
        assert poincare_distance(x, x) == pytest.approx(0.0, abs=1e-10)

    def test_symmetric(self):
        u = np.array([0.1, 0.2, 0.3])
        v = np.array([0.4, 0.1, 0.2])
        assert poincare_distance(u, v) == pytest.approx(poincare_distance(v, u))


class TestLorentzDistance:
    def test_same_point(self):
        # Point on unit hyperboloid: x0 = sqrt(1 + |x_s|^2)
        xs = np.array([0.1, 0.2, 0.3])
        x0 = np.sqrt(1.0 + np.dot(xs, xs))
        x = np.concatenate([[x0], xs])
        assert lorentz_distance(x, x) == pytest.approx(0.0, abs=1e-10)

    def test_positive_for_different(self):
        xs1 = np.array([0.1, 0.2])
        x1 = np.concatenate([[np.sqrt(1 + np.dot(xs1, xs1))], xs1])
        xs2 = np.array([0.5, 0.6])
        x2 = np.concatenate([[np.sqrt(1 + np.dot(xs2, xs2))], xs2])
        d = lorentz_distance(x1, x2)
        assert d > 0


class TestHyperbolicDistance:
    def test_poincare_dispatch(self):
        u = np.array([0.1, 0.2])
        v = np.array([0.3, 0.4])
        assert hyperbolic_distance(u, v, model="poincare") == pytest.approx(
            poincare_distance(u, v)
        )

    def test_lorentz_dispatch(self):
        xs1 = np.array([0.1])
        x1 = np.concatenate([[np.sqrt(1 + np.dot(xs1, xs1))], xs1])
        xs2 = np.array([0.5])
        x2 = np.concatenate([[np.sqrt(1 + np.dot(xs2, xs2))], xs2])
        assert hyperbolic_distance(x1, x2, model="lorentz") == pytest.approx(
            lorentz_distance(x1, x2)
        )

    def test_invalid_model(self):
        with pytest.raises(ValueError):
            hyperbolic_distance(np.zeros(3), np.ones(3), model="invalid")
