"""Tests for RiemannianGradientDescent."""

import numpy as np
import pytest

from flux_hyperbolic.optimization import RiemannianGradientDescent


class TestRiemannianGD:
    def test_basic_convergence(self):
        """Optimise toward the origin using squared-distance loss."""
        opt = RiemannianGradientDescent(dim=3, lr=0.05, max_iter=500, tol=1e-8)

        start = np.array([0.3, 0.2, 0.1])

        def loss(x):
            return np.dot(x, x)

        def grad(x):
            return 2.0 * x

        result, losses = opt.optimize(start, loss, grad)
        # Should have moved closer to origin
        assert np.linalg.norm(result) < np.linalg.norm(start)
        # Loss should decrease
        assert losses[-1] < losses[0]

    def test_returns_decreasing_loss(self):
        opt = RiemannianGradientDescent(dim=2, lr=0.01, max_iter=200, tol=1e-10)
        start = np.array([0.4, -0.3])
        target = np.array([0.1, 0.1])

        def loss(x):
            return np.sum((x - target) ** 2)

        def grad(x):
            return 2.0 * (x - target)

        result, losses = opt.optimize(start, loss, grad)
        assert losses[-1] < losses[0]
        assert len(losses) > 0

    def test_max_iter_respected(self):
        opt = RiemannianGradientDescent(dim=2, lr=0.1, max_iter=5, tol=0.0)
        start = np.array([0.5, 0.0])

        def loss(x):
            return np.dot(x, x)

        def grad(x):
            return 2.0 * x

        _, losses = opt.optimize(start, loss, grad)
        assert len(losses) <= 5

    def test_result_stays_in_ball(self):
        opt = RiemannianGradientDescent(dim=3, curvature=1.0, lr=0.01, max_iter=200)
        start = np.array([0.3, -0.2, 0.4])

        def loss(x):
            return np.dot(x, x)

        def grad(x):
            return 2.0 * x

        result, _ = opt.optimize(start, loss, grad)
        assert np.linalg.norm(result) < 1.0

    def test_custom_curvature(self):
        opt = RiemannianGradientDescent(dim=2, curvature=2.0, lr=0.01, max_iter=100)
        assert opt.ball.c == 2.0
        start = np.array([0.2, 0.1])

        def loss(x):
            return np.dot(x, x)

        def grad(x):
            return 2.0 * x

        result, _ = opt.optimize(start, loss, grad)
        assert np.linalg.norm(result) < opt.ball.radius
