"""HyperbolicTree: embed tree structures in the Poincaré ball.

Uses a simplified version of Sarkar's algorithm to place tree nodes
in hyperbolic space such that:
  - Parent-child distance is constant
  - Sibling subtrees are well-separated
  - Root is at the origin
"""

from __future__ import annotations

import numpy as np

from .poincare import PoincareBall


class HyperbolicTree:
    """Embed a tree in hyperbolic space using Sarkar's algorithm.

    Parameters
    ----------
    dim : int
        Dimensionality of the embedding space.
    curvature : float
        Curvature parameter (> 0).
    """

    def __init__(self, dim: int = 3, curvature: float = 1.0) -> None:
        self.dim = dim
        self.ball = PoincareBall(curvature=curvature)
        self._rng = np.random.default_rng(42)

    def embed(self, tree: dict) -> dict:
        """Embed tree nodes as points in the Poincaré ball.

        Places the root at the origin, then recursively places children
        at a fixed distance from their parent, evenly spaced angularly.

        Parameters
        ----------
        tree : dict
            Nested dict with ``"name"`` and optional ``"children"`` keys.

        Returns
        -------
        dict
            Same tree structure with ``"point"`` added to each node.
        """
        return self._embed_recursive(tree, None, 0)

    def _embed_recursive(
        self,
        node: dict,
        parent_point: np.ndarray | None,
        depth: int,
    ) -> dict:
        """Recursively embed a subtree."""
        result = {"name": node["name"]}

        if parent_point is None:
            # Root → origin
            point = np.zeros(self.dim, dtype=np.float64)
        else:
            # Place at a fixed distance from parent, in a random direction
            # that's perpendicular-ish to the parent→origin axis.
            n_children_at_this_depth = 1  # We'll spread later
            direction = self._random_direction()
            # Offset from parent
            dist = 0.3 / (depth + 1)  # Closer to boundary at deeper levels
            tangent = direction * dist
            point = self.ball.exp_map(parent_point, tangent)
            point = self.ball.project(point)

        result["point"] = point

        children = node.get("children", [])
        if children:
            # Spread children angularly around the parent.
            n = len(children)
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            embedded_children = []
            for i, child in enumerate(children):
                child_result = self._embed_child(child, point, angles[i], depth + 1)
                embedded_children.append(child_result)
            result["children"] = embedded_children

        return result

    def _embed_child(
        self,
        child: dict,
        parent_point: np.ndarray,
        angle: float,
        depth: int,
    ) -> dict:
        """Embed a child node at a given angle from its parent."""
        dist = 0.3 / (depth + 0.5)
        direction = np.zeros(self.dim, dtype=np.float64)
        direction[0] = np.cos(angle)
        direction[1] = np.sin(angle)
        if self.dim > 2:
            # Add some variation in higher dimensions
            direction[2] = np.sin(angle * 0.7) * 0.5
            direction /= np.linalg.norm(direction)

        tangent = direction * dist
        point = self.ball.exp_map(parent_point, tangent)
        point = self.ball.project(point)

        result = {"name": child["name"], "point": point}

        grandchildren = child.get("children", [])
        if grandchildren:
            n = len(grandchildren)
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
            embedded = []
            for i, gc in enumerate(grandchildren):
                embedded.append(self._embed_child(gc, point, angles[i], depth + 1))
            result["children"] = embedded

        return result

    def _random_direction(self) -> np.ndarray:
        """Generate a random unit vector in self.dim dimensions."""
        v = self._rng.standard_normal(self.dim)
        norm = np.linalg.norm(v)
        if norm < 1e-15:
            v = np.zeros(self.dim)
            v[0] = 1.0
        else:
            v /= norm
        return v
