"""Poincaré ball model of hyperbolic space.

The Poincaré ball model represents hyperbolic space as the open unit ball
Bⁿ = {x ∈ ℝⁿ : ‖x‖ < 1} with the Riemannian metric:

    g_x = λ_x² · I,    λ_x = 2 / (1 − ‖x‖²)

where λ_x is the conformal factor.

Distance
--------
The hyperbolic distance between two points u, v ∈ Bⁿ is:

    d(u, v) = arccosh(1 + 2‖u − v‖² / ((1 − ‖u‖²)(1 − ‖v‖²)))

This distance grows rapidly near the boundary ‖x‖ → 1, making it ideal
for representing hierarchical data: leaves live near the boundary,
root nodes live near the origin.
"""

from __future__ import annotations

import numpy as np


class PoincareBall:
    """Poincaré ball model of hyperbolic space.

    Parameters
    ----------
    curvature : float
        Negative curvature parameter *c* > 0.  Higher *c* → more curved
        (more "room" near boundary).  Distance formula becomes:

            d(u, v) = (1/√c) · arccosh(1 + 2c‖u−v‖² / ((1−c‖u‖²)(1−c‖v‖²)))

        When ``c = 1`` the ball has radius 1.
    """

    def __init__(self, curvature: float = 1.0) -> None:
        if curvature <= 0:
            raise ValueError("curvature must be positive")
        self.c = curvature

    @property
    def radius(self) -> float:
        """Radius of the ball (= 1/√c)."""
        return 1.0 / np.sqrt(self.c)

    # ------------------------------------------------------------------
    # Distance
    # ------------------------------------------------------------------

    def distance(self, u: np.ndarray, v: np.ndarray) -> float:
        """Hyperbolic distance between points *u* and *v* in the ball.

        .. math::

            d(u, v) = \\frac{1}{\\sqrt{c}} \\operatorname{arccosh}\\!\\left(
                1 + \\frac{2c\\,\\|u - v\\|^2}{(1 - c\\|u\\|^2)(1 - c\\|v\\|^2)}
            \\right)

        Parameters
        ----------
        u, v : np.ndarray
            Points in the ball (‖·‖ < 1/√c).

        Returns
        -------
        float
        """
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        u_norm_sq = np.dot(u, u)
        v_norm_sq = np.dot(v, v)
        diff_norm_sq = np.dot(u - v, u - v)

        denom = (1.0 - self.c * u_norm_sq) * (1.0 - self.c * v_norm_sq)
        if denom <= 0:
            # Numerical edge case: project and retry
            return self.distance(self.project(u), self.project(v))

        arg = 1.0 + 2.0 * self.c * diff_norm_sq / denom
        arg = max(arg, 1.0)  # numerical safety for arccosh
        return float(np.arccosh(arg) / np.sqrt(self.c))

    # ------------------------------------------------------------------
    # Projection
    # ------------------------------------------------------------------

    def project(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Project point back inside the ball (clamp norm < radius).

        Parameters
        ----------
        x : np.ndarray
        eps : float
            Small margin from boundary.

        Returns
        -------
        np.ndarray
        """
        x = np.asarray(x, dtype=np.float64)
        max_norm = self.radius - eps
        norm = np.linalg.norm(x)
        if norm >= max_norm:
            return x * (max_norm / norm)
        return x.copy()

    # ------------------------------------------------------------------
    # Exponential / logarithmic maps
    # ------------------------------------------------------------------

    def exp_map(self, base: np.ndarray, tangent: np.ndarray) -> np.ndarray:
        """Exponential map: map a tangent vector at *base* to a point.

        .. math::

            \\exp_{\\text{base}}(v) = \\text{base} \\oplus
            \\tanh\\!\\left(\\frac{\\lambda_{\\text{base}} \\|v\\|}{2}\\right)
            \\frac{v}{\\|v\\|}

        where λ_base = 2/(1 − c·‖base‖²) is the conformal factor.

        Parameters
        ----------
        base : np.ndarray
            Point on the manifold.
        tangent : np.ndarray
            Tangent vector at *base*.

        Returns
        -------
        np.ndarray
        """
        base = np.asarray(base, dtype=np.float64)
        tangent = np.asarray(tangent, dtype=np.float64)
        base_norm_sq = np.dot(base, base)
        lam = 2.0 / (1.0 - self.c * base_norm_sq)
        t_norm = np.linalg.norm(tangent)
        if t_norm < 1e-15:
            return base.copy()
        return self.project(
            base + np.tanh(lam * t_norm / 2.0) * (tangent / t_norm)
        )

    def log_map(self, base: np.ndarray, point: np.ndarray) -> np.ndarray:
        """Logarithmic map: map a point to a tangent vector at *base*.

        Inverse of ``exp_map``.

        Parameters
        ----------
        base : np.ndarray
        point : np.ndarray

        Returns
        -------
        np.ndarray
        """
        base = np.asarray(base, dtype=np.float64)
        point = np.asarray(point, dtype=np.float64)
        diff = point - base
        diff_norm = np.linalg.norm(diff)
        if diff_norm < 1e-15:
            return np.zeros_like(base)
        base_norm_sq = np.dot(base, base)
        lam = 2.0 / (1.0 - self.c * base_norm_sq)
        return (2.0 / lam) * np.arctanh(diff_norm) * (diff / diff_norm)

    # ------------------------------------------------------------------
    # Mobility
    # ------------------------------------------------------------------

    def mobility(self, u: np.ndarray, v: np.ndarray) -> float:
        """Mobility: how 'easily' information flows between two points.

        Defined as exp(−d(u, v)), so mobility ∈ (0, 1].

        - mobility = 1 when u = v
        - mobility → 0 as distance → ∞

        Parameters
        ----------
        u, v : np.ndarray

        Returns
        -------
        float
        """
        return float(np.exp(-self.distance(u, v)))

    # ------------------------------------------------------------------
    # Möbius addition
    # ------------------------------------------------------------------

    def mobius_add(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Möbius addition in the Poincaré ball.

        .. math::

            x \\oplus_c y = \\frac{
                (1 + 2c\\langle x, y\\rangle + c\\|y\\|^2) x
                + (1 - c\\|x\\|^2) y
            }{
                1 + 2c\\langle x, y\\rangle + c^2\\|x\\|^2\\|y\\|^2
            }

        Parameters
        ----------
        x, y : np.ndarray

        Returns
        -------
        np.ndarray
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        x_norm_sq = np.dot(x, x)
        y_norm_sq = np.dot(y, y)
        xy = np.dot(x, y)
        denom = 1.0 + 2.0 * self.c * xy + self.c ** 2 * x_norm_sq * y_norm_sq
        num = (
            (1.0 + 2.0 * self.c * xy + self.c * y_norm_sq) * x
            + (1.0 - self.c * x_norm_sq) * y
        )
        return num / denom
