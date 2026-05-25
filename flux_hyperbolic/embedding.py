"""TraditionEmbedding: embed musical traditions in hyperbolic space.

Uses a hierarchical taxonomy to place traditions in the Poincaré ball:

    Root: "Music"
    ├── Western
    │   ├── Jazz
    │   ├── Classical
    │   ├── Rock
    │   └── Blues
    ├── Eastern
    │   ├── Hindustani
    │   ├── Gamelan
    │   └── Gagaku
    ├── African
    │   └── WestAfrican
    └── Electronic
        └── Electronic

Hyperbolic space naturally captures this hierarchy: children are close
to parents, siblings are at similar distances, cousins are far apart.
"""

from __future__ import annotations

import numpy as np

from .poincare import PoincareBall
from .tree import HyperbolicTree


# Tree structure for tradition taxonomy.
_TRADITION_TREE: dict = {
    "name": "Music",
    "children": [
        {
            "name": "Western",
            "children": [
                {"name": "Jazz"},
                {"name": "Classical"},
                {"name": "Rock"},
                {"name": "Blues"},
            ],
        },
        {
            "name": "Eastern",
            "children": [
                {"name": "Hindustani"},
                {"name": "Gamelan"},
                {"name": "Gagaku"},
            ],
        },
        {
            "name": "African",
            "children": [
                {"name": "WestAfrican"},
            ],
        },
        {
            "name": "Electronic",
            "children": [
                {"name": "Electronic"},
            ],
        },
    ],
}

# Approximate dial centres for mapping.
_DIAL_CENTRES: dict[str, tuple[float, float, float]] = {
    "Jazz": (3.2, 2.8, 2.5),
    "Classical": (1.8, 1.2, 1.5),
    "Rock": (3.5, 3.8, 3.0),
    "Blues": (3.0, 2.5, 2.0),
    "Electronic": (3.8, 4.0, 4.5),
    "Hindustani": (2.5, 3.2, 1.8),
    "Gamelan": (2.0, 3.5, 2.2),
    "Gagaku": (1.5, 1.8, 1.0),
    "WestAfrican": (2.8, 4.2, 2.8),
    "FreeImprovisation": (4.0, 3.5, 3.8),
}


class TraditionEmbedding:
    """Embed musical traditions in hyperbolic space.

    Parameters
    ----------
    dim : int
        Dimensionality of the Poincaré ball.
    curvature : float
        Negative curvature parameter (> 0).
    """

    def __init__(self, dim: int = 3, curvature: float = 1.0) -> None:
        self.dim = dim
        self.ball = PoincareBall(curvature=curvature)
        self._embeddings: dict[str, np.ndarray] = {}
        self._tree_embedded: dict | None = None

    def embed_traditions(
        self,
        tradition_dials: dict[str, tuple[float, float, float]] | None = None,
    ) -> dict[str, np.ndarray]:
        """Map tradition dial positions to hyperbolic coordinates.

        Uses the hierarchical tree structure and Sarkar's algorithm to
        place traditions in the Poincaré ball.

        Parameters
        ----------
        tradition_dials : dict or None
            Mapping of tradition name → (h, r, s) dial.  If ``None``,
            uses the built-in catalogue.

        Returns
        -------
        dict[str, np.ndarray]
            Tradition name → point in Poincaré ball.
        """
        tree_embedder = HyperbolicTree(dim=self.dim, curvature=self.ball.c)
        self._tree_embedded = tree_embedder.embed(_TRADITION_TREE)

        # Extract leaf embeddings.
        if tradition_dials is None:
            tradition_dials = _DIAL_CENTRES

        self._embeddings = {}
        self._collect_leaves(self._tree_embedded, self._embeddings)

        # Add any traditions from dials not in the tree.
        for name, dial in tradition_dials.items():
            if name not in self._embeddings:
                # Map dial to Poincaré ball coordinate.
                dial_arr = np.array(dial, dtype=np.float64)
                # Normalise to ball: map [0,5]^3 → ball of radius ~0.8
                point = (dial_arr / 5.0) * 0.8
                self._embeddings[name] = self.ball.project(point)

        return self._embeddings

    def _collect_leaves(self, node: dict, result: dict) -> None:
        """Recursively collect leaf embeddings."""
        if "point" in node:
            name = node["name"]
            if "children" not in node or not node["children"]:
                result[name] = node["point"]
            else:
                for child in node.get("children", []):
                    self._collect_leaves(child, result)

    def tradition_distance(self, t1: str, t2: str) -> float:
        """Hyperbolic distance between two traditions.

        Parameters
        ----------
        t1, t2 : str

        Returns
        -------
        float

        Raises
        ------
        RuntimeError
            If embeddings haven't been computed yet.
        """
        if not self._embeddings:
            raise RuntimeError("Call embed_traditions() first")
        return self.ball.distance(self._embeddings[t1], self._embeddings[t2])

    def nearest_traditions(self, point: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        """Find *k* nearest traditions to a hyperbolic point.

        Parameters
        ----------
        point : np.ndarray
        k : int

        Returns
        -------
        list[tuple[str, float]]
            Sorted by distance (ascending).
        """
        if not self._embeddings:
            raise RuntimeError("Call embed_traditions() first")
        dists = [
            (name, self.ball.distance(point, emb))
            for name, emb in self._embeddings.items()
        ]
        dists.sort(key=lambda x: x[1])
        return dists[:k]

    def tradition_tree(self) -> dict:
        """Return the hierarchical tree structure with embedded points.

        Returns
        -------
        dict
        """
        if self._tree_embedded is None:
            raise RuntimeError("Call embed_traditions() first")
        return self._tree_embedded
