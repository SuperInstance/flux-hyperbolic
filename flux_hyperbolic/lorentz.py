"""Lorentz (hyperboloid) model of hyperbolic space.

The Lorentz model embeds hyperbolic space on the upper sheet of a
two-sheeted hyperboloid in Minkowski space:

    Hⁿ = {x ∈ ℝⁿ⁺¹ : ⟨x, x⟩_L = −1, x₀ > 0}

where the Minkowski inner product is:

    ⟨x, y⟩_L = −x₀y₀ + Σᵢ xᵢyᵢ

Distance
--------
    d(x, y) = arccosh(−⟨x, y⟩_L)

The Lorentz model is numerically more stable than the Poincaré ball
for points far from the origin, and is useful for optimization.
"""

from __future__ import annotations

import numpy as np


class LorentzModel:
    """Lorentz (hyperboloid) model of hyperbolic space.

    Parameters
    ----------
    curvature : float
        Curvature parameter *c* > 0.
    """

    def __init__(self, curvature: float = 1.0) -> None:
        if curvature <= 0:
            raise ValueError("curvature must be positive")
        self.c = curvature

    # ------------------------------------------------------------------
    # Minkowski inner product
    # ------------------------------------------------------------------

    @staticmethod
    def minkowski_dot(x: np.ndarray, y: np.ndarray) -> float:
        """Minkowski inner product ⟨x, y⟩_L = −x₀y₀ + Σᵢ xᵢyᵢ.

        Parameters
        ----------
        x, y : np.ndarray
            Vectors in ℝⁿ⁺¹ (first component is the "time" coordinate).

        Returns
        -------
        float
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        return float(-x[0] * y[0] + np.dot(x[1:], y[1:]))

    # ------------------------------------------------------------------
    # Distance
    # ------------------------------------------------------------------

    def distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Hyperbolic distance on the hyperboloid.

        .. math::

            d(x, y) = \\frac{1}{\\sqrt{c}} \\operatorname{arccosh}(-\\langle x, y \\rangle_L)

        Parameters
        ----------
        x, y : np.ndarray

        Returns
        -------
        float
        """
        mdp = self.minkowski_dot(x, y)
        arg = max(-mdp, 1.0)  # -⟨x,y⟩_L ≥ 1 for valid hyperboloid points
        return float(np.arccosh(arg) / np.sqrt(self.c))

    # ------------------------------------------------------------------
    # Projection to hyperboloid
    # ------------------------------------------------------------------

    def project(self, x: np.ndarray) -> np.ndarray:
        """Project a point onto the hyperboloid.

        Given a spatial component x_s = x[1:], sets:

            x₀ = √(1/c + ‖x_s‖²)

        Parameters
        ----------
        x : np.ndarray
            Point whose spatial part defines the projection.

        Returns
        -------
        np.ndarray
        """
        x = np.asarray(x, dtype=np.float64)
        spatial = x[1:]
        x0 = np.sqrt(1.0 / self.c + np.dot(spatial, spatial))
        return np.concatenate([[x0], spatial])

    # ------------------------------------------------------------------
    # Exponential / logarithmic maps
    # ------------------------------------------------------------------

    def exp_map(self, base: np.ndarray, tangent: np.ndarray) -> np.ndarray:
        """Exponential map at *base* along *tangent*.

        .. math::

            \\exp_{\\text{base}}(v) = \\cosh(\\sqrt{c}\\,\\|v\\|_L)\\,\\text{base}
                + \\frac{\\sinh(\\sqrt{c}\\,\\|v\\|_L)}{\\sqrt{c}\\,\\|v\\|_L}\\,v

        Parameters
        ----------
        base : np.ndarray
            Point on the hyperboloid.
        tangent : np.ndarray
            Tangent vector at *base* (must satisfy ⟨base, v⟩_L = 0).

        Returns
        -------
        np.ndarray
        """
        base = np.asarray(base, dtype=np.float64)
        tangent = np.asarray(tangent, dtype=np.float64)
        norm_sq = self.minkowski_dot(tangent, tangent)
        if norm_sq < 0:
            norm_sq = abs(norm_sq)
        norm = np.sqrt(max(norm_sq, 0.0))
        sqc = np.sqrt(self.c)
        if norm < 1e-15:
            return base.copy()
        return np.cosh(sqc * norm) * base + (np.sinh(sqc * norm) / (sqc * norm)) * tangent

    def log_map(self, base: np.ndarray, point: np.ndarray) -> np.ndarray:
        """Logarithmic map: point → tangent vector at *base*.

        Parameters
        ----------
        base, point : np.ndarray

        Returns
        -------
        np.ndarray
        """
        d = self.distance(base, point)
        if d < 1e-15:
            return np.zeros_like(base)
        sqc = np.sqrt(self.c)
        return d * (point - self.minkowski_dot(base, point) * base) / (
            sqc * np.sinh(sqc * d) / sqc
        )

    # ------------------------------------------------------------------
    # Conversions
    # ------------------------------------------------------------------

    def to_poincare(self, x: np.ndarray) -> np.ndarray:
        """Convert Lorentz → Poincaré ball coordinates.

            p = x[1:] / (x[0] + 1)  (scaled by 1/√c for general curvature)
        """
        x = np.asarray(x, dtype=np.float64)
        return x[1:] / (np.sqrt(self.c) * (x[0] + 1.0 / np.sqrt(self.c)))

    def from_poincare(self, p: np.ndarray) -> np.ndarray:
        """Convert Poincaré ball → Lorentz coordinates.

            x₀ = (1 + c·‖p‖²) / (2√c)
            x_i = p_i · √c · x₀
        """
        p = np.asarray(p, dtype=np.float64)
        p_norm_sq = np.dot(p, p)
        sqc = np.sqrt(self.c)
        denom = 1.0 - self.c * p_norm_sq
        x0 = (1.0 + self.c * p_norm_sq) / (2.0 * sqc * denom)
        spatial = p * sqc / denom
        return np.concatenate([[x0], spatial])
