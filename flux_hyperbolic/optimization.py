"""Riemannian gradient descent in the Poincaré ball.

Optimises point positions by following Riemannian gradient directions,
accounting for the curved geometry of hyperbolic space.

The Riemannian gradient is related to the Euclidean gradient by:

    ∇ᴿf(x) = (1 / λ_x²) · ∇ᴱf(x)

where λ_x = 2/(1 − ‖x‖²) is the conformal factor.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .poincare import PoincareBall


class RiemannianGradientDescent:
    """Riemannian gradient descent in the Poincaré ball.

    Parameters
    ----------
    dim : int
        Dimensionality of the ball.
    curvature : float
        Curvature parameter (> 0).
    lr : float
        Learning rate.
    max_iter : int
        Maximum iterations.
    tol : float
        Convergence tolerance on gradient norm.
    """

    def __init__(
        self,
        dim: int = 3,
        curvature: float = 1.0,
        lr: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> None:
        self.dim = dim
        self.ball = PoincareBall(curvature=curvature)
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol

    def optimize(
        self,
        start: np.ndarray,
        loss_fn: Callable[[np.ndarray], float],
        grad_fn: Callable[[np.ndarray], np.ndarray],
    ) -> tuple[np.ndarray, list[float]]:
        """Run Riemannian gradient descent.

        At each step:
            1. Compute Euclidean gradient: g = ∇loss(x)
            2. Convert to Riemannian gradient: gᴿ = g / λ_x²
            3. Take a step: x ← exp_x(−lr · gᴿ)
            4. Project back into ball

        Parameters
        ----------
        start : np.ndarray
            Initial point in the ball.
        loss_fn : callable
            Loss function x → scalar.
        grad_fn : callable
            Gradient function x → np.ndarray (Euclidean gradient).

        Returns
        -------
        tuple[np.ndarray, list[float]]
            Optimised point and loss history.
        """
        x = self.ball.project(np.asarray(start, dtype=np.float64).copy())
        losses: list[float] = []

        for _ in range(self.max_iter):
            loss = loss_fn(x)
            losses.append(loss)

            grad = np.asarray(grad_fn(x), dtype=np.float64)
            # Riemannian gradient: divide by conformal factor squared
            x_norm_sq = np.dot(x, x)
            lam = 2.0 / (1.0 - self.ball.c * x_norm_sq)
            r_grad = grad / (lam ** 2)

            if np.linalg.norm(r_grad) < self.tol:
                break

            # Exponential map step
            x = self.ball.exp_map(x, -self.lr * r_grad)
            x = self.ball.project(x)

        return x, losses
