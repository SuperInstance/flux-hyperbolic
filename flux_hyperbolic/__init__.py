"""flux-hyperbolic: Hyperbolic geometry toolkit for embedding musical traditions."""

from .poincare import PoincareBall
from .lorentz import LorentzModel
from .embedding import TraditionEmbedding
from .distance import hyperbolic_distance, poincare_distance, lorentz_distance
from .tree import HyperbolicTree
from .optimization import RiemannianGradientDescent

__all__ = [
    "PoincareBall",
    "LorentzModel",
    "TraditionEmbedding",
    "hyperbolic_distance",
    "poincare_distance",
    "lorentz_distance",
    "HyperbolicTree",
    "RiemannianGradientDescent",
]
