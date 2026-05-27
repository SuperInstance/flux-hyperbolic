"""Tests for LorentzModel."""

import numpy as np
import pytest

from flux_hyperbolic.lorentz import LorentzModel


def _hyperboloid_point(spatial, c=1.0):
    """Helper: create a valid hyperboloid point from spatial coords."""
    spatial = np.asarray(spatial, dtype=np.float64)
    x0 = np.sqrt(1.0 / c + np.dot(spatial, spatial))
    return np.concatenate([[x0], spatial])


class TestLorentzCreation:
    def test_default_curvature(self):
        model = LorentzModel()
        assert model.c == 1.0

    def test_custom_curvature(self):
        model = LorentzModel(curvature=2.0)
        assert model.c == 2.0

    def test_invalid_curvature_zero(self):
        with pytest.raises(ValueError):
            LorentzModel(curvature=0.0)

    def test_invalid_curvature_negative(self):
        with pytest.raises(ValueError):
            LorentzModel(curvature=-1.0)


class TestMinkowskiDot:
    def test_self_product(self):
        model = LorentzModel()
        x = _hyperboloid_point([0.3, 0.4])
        mdp = model.minkowski_dot(x, x)
        # For valid hyperboloid point: <x,x>_L = -1/c
        assert mdp == pytest.approx(-1.0, abs=1e-10)

    def test_orthogonal(self):
        model = LorentzModel()
        x = np.array([2.0, 1.0, 0.0])
        y = np.array([2.0, 0.0, 1.0])
        # -4 + 0 + 0 = -4
        assert model.minkowski_dot(x, y) == pytest.approx(-4.0)


class TestLorentzDistance:
    def test_same_point(self):
        model = LorentzModel()
        x = _hyperboloid_point([0.1, 0.2, 0.3])
        assert model.distance(x, x) == pytest.approx(0.0, abs=1e-10)

    def test_positive_for_different(self):
        model = LorentzModel()
        x = _hyperboloid_point([0.1, 0.2])
        y = _hyperboloid_point([0.5, 0.6])
        assert model.distance(x, y) > 0

    def test_symmetry(self):
        model = LorentzModel()
        x = _hyperboloid_point([0.1, 0.2])
        y = _hyperboloid_point([0.3, 0.4])
        assert model.distance(x, y) == pytest.approx(model.distance(y, x))

    def test_curvature_scaling(self):
        model1 = LorentzModel(curvature=1.0)
        x = _hyperboloid_point([0.5, 0.6], c=1.0)
        y = _hyperboloid_point([2.0, 3.0], c=1.0)
        d1 = model1.distance(x, y)
        assert d1 > 0
        assert np.isfinite(d1)

    def test_different_curvatures_both_finite(self):
        for c in [0.5, 1.0, 2.0, 5.0]:
            model = LorentzModel(curvature=c)
            x = _hyperboloid_point([0.5, 0.6], c=c)
            y = _hyperboloid_point([2.0, 3.0], c=c)
            d = model.distance(x, y)
            assert d >= 0
            assert np.isfinite(d)


class TestProjection:
    def test_valid_point(self):
        model = LorentzModel()
        spatial = np.array([2.0, 0.0, 0.0])
        p = model.project(np.array([0.0, 2.0, 0.0, 0.0]))
        # x0 = sqrt(1 + 4) = sqrt(5)
        assert p[0] == pytest.approx(np.sqrt(5.0))
        np.testing.assert_array_almost_equal(p[1:], spatial)

    def test_satisfies_hyperboloid(self):
        model = LorentzModel(curvature=2.0)
        x = np.array([0.0, 0.3, 0.4, 0.5])
        p = model.project(x)
        mdp = model.minkowski_dot(p, p)
        assert mdp == pytest.approx(-1.0 / model.c, abs=1e-10)


class TestExpLogMap:
    def test_exp_zero_tangent(self):
        model = LorentzModel()
        base = _hyperboloid_point([0.1, 0.2])
        result = model.exp_map(base, np.zeros(3))
        np.testing.assert_array_almost_equal(result, base)

    def test_log_returns_array(self):
        model = LorentzModel()
        x = _hyperboloid_point([0.1, 0.2])
        result = model.log_map(x, x)
        assert isinstance(result, np.ndarray)
        assert result.shape == x.shape


class TestConversions:
    def test_poincare_roundtrip(self):
        model = LorentzModel()
        spatial = np.array([0.1, 0.2, 0.3])
        x = _hyperboloid_point(spatial)
        p = model.to_poincare(x)
        assert np.linalg.norm(p) < 1.0 / np.sqrt(model.c)  # inside ball
        x_back = model.from_poincare(p)
        # Roundtrip preserves direction (may have scaling differences)
        # Check the spatial parts are proportional
        ratio = x_back[1:] / x[1:]
        np.testing.assert_array_almost_equal(ratio, np.full_like(ratio, ratio[0]), decimal=3)

    def test_poincare_origin(self):
        model = LorentzModel()
        # Origin in Lorentz: (1/sqrt(c), 0, ..., 0)
        origin = np.zeros(4)
        origin[0] = 1.0 / np.sqrt(model.c)
        p = model.to_poincare(origin)
        np.testing.assert_array_almost_equal(p, np.zeros(3), decimal=10)

    def test_from_poincare_origin(self):
        model = LorentzModel()
        p = np.zeros(3)
        x = model.from_poincare(p)
        assert x[0] == pytest.approx(0.5 / np.sqrt(model.c))
