"""Tests for TraditionEmbedding."""

import numpy as np
import pytest

from flux_hyperbolic.embedding import TraditionEmbedding


class TestTraditionEmbedding:
    def test_embed_traditions(self):
        te = TraditionEmbedding(dim=3)
        embs = te.embed_traditions()
        assert len(embs) >= 9  # at least the traditions from the tree

    def test_tradition_distance(self):
        te = TraditionEmbedding()
        te.embed_traditions()
        # Same tradition → zero
        names = list(te._embeddings.keys())
        if len(names) >= 2:
            d = te.tradition_distance(names[0], names[0])
            assert d == pytest.approx(0.0, abs=1e-10)
            # Different traditions → positive
            d2 = te.tradition_distance(names[0], names[1])
            assert d2 > 0

    def test_nearest_traditions(self):
        te = TraditionEmbedding()
        embs = te.embed_traditions()
        names = list(embs.keys())
        if names:
            point = embs[names[0]]
            nearest = te.nearest_traditions(point, k=3)
            assert len(nearest) <= 3
            # First should be the tradition itself
            assert nearest[0][0] == names[0]

    def test_tradition_tree(self):
        te = TraditionEmbedding()
        te.embed_traditions()
        tree = te.tradition_tree()
        assert "name" in tree
        assert "point" in tree

    def test_without_embed_raises(self):
        te = TraditionEmbedding()
        with pytest.raises(RuntimeError):
            te.tradition_distance("Jazz", "Classical")
