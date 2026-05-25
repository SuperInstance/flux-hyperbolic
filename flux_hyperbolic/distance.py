"""Distance functions for hyperbolic geometry.

Provides standalone distance functions for Poincaré ball, Lorentz model,
and a generic interface that delegates to the appropriate model.
"""

from __future__ import annotations

import numpy as np


def poincare_distance(
    u: np.ndarray,
    v: np.ndarray,
    curvature: float = 1.0,
) -> float:
    """Distance in the Poincaré ball model.

    .. math::

        d(u, v) = \\frac{1}{\\sqrt{c}} \\operatorname{arccosh}\\left(
            1 + \\frac{2c\\,\\|u - v\\|^2}{(1 - c\\|u\\|^2)(1 - c\\|v\\|^2)}
        \\right)

    Parameters
    ----------
    u, v : np.ndarray
    curvature : float

    Returns
    -------
    float
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    c = curvature
    u_sq = np.dot(u, u)
    v_sq = np.dot(v, v)
    diff_sq = np.dot(u - v, u - v)
    denom = (1.0 - c * u_sq) * (1.0 - c * v_sq)
    if denom <= 0:
        return float("inf")
    arg = 1.0 + 2.0 * c * diff_sq / denom
    return float(np.arccosh(max(arg, 1.0)) / np.sqrt(c))


def lorentz_distance(
    x: np.ndarray,
    y: np.ndarray,
    curvature: float = 1.0,
) -> float:
    """Distance in the Lorentz (hyperboloid) model.

    .. math::

        d(x, y) = \\frac{1}{\\sqrt{c}} \\operatorname{arccosh}(-\\langle x, y \\rangle_L)

    Parameters
    ----------
    x, y : np.ndarray
        Points on the hyperboloid.
    curvature : float

    Returns
    -------
    float
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mdp = -x[0] * y[0] + np.dot(x[1:], y[1:])
    arg = max(-mdp, 1.0)  # -⟨x,y⟩_L should be ≥ 1 for valid hyperboloid points
    return float(np.arccosh(arg) / np.sqrt(curvature))


def hyperbolic_distance(
    u: np.ndarray,
    v: np.ndarray,
    model: str = "poincare",
    curvature: float = 1.0,
) -> float:
    """Generic hyperbolic distance (dispatches by model).

    Parameters
    ----------
    u, v : np.ndarray
    model : str
        ``"poincare"`` or ``"lorentz"``.
    curvature : float

    Returns
    -------
    float
    """
    if model == "poincare":
        return poincare_distance(u, v, curvature)
    elif model == "lorentz":
        return lorentz_distance(u, v, curvature)
    else:
        raise ValueError(f"Unknown model: {model!r}")
