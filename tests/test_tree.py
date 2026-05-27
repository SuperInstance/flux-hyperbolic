"""Tests for HyperbolicTree."""

import numpy as np
import pytest

from flux_hyperbolic.tree import HyperbolicTree


class TestHyperbolicTree:
    def test_single_root(self):
        ht = HyperbolicTree(dim=3)
        tree = {"name": "Root"}
        result = ht.embed(tree)
        assert result["name"] == "Root"
        assert "point" in result
        # Root should be at origin
        np.testing.assert_array_almost_equal(result["point"], np.zeros(3))

    def test_tree_with_children(self):
        ht = HyperbolicTree(dim=3)
        tree = {
            "name": "Root",
            "children": [
                {"name": "A"},
                {"name": "B"},
                {"name": "C"},
            ],
        }
        result = ht.embed(tree)
        assert result["name"] == "Root"
        assert len(result["children"]) == 3
        names = [c["name"] for c in result["children"]]
        assert names == ["A", "B", "C"]

    def test_points_inside_ball(self):
        ht = HyperbolicTree(dim=3)
        tree = {
            "name": "Root",
            "children": [
                {"name": "A", "children": [{"name": "A1"}, {"name": "A2"}]},
                {"name": "B"},
            ],
        }
        result = ht.embed(tree)

        def check_inside(node):
            pt = node["point"]
            assert np.linalg.norm(pt) < 1.0, f"{node['name']} outside ball"
            for child in node.get("children", []):
                check_inside(child)

        check_inside(result)

    def test_all_points_have_embeddings(self):
        ht = HyperbolicTree(dim=2)
        tree = {
            "name": "Root",
            "children": [
                {"name": "X"},
                {"name": "Y"},
            ],
        }
        result = ht.embed(tree)
        assert "point" in result
        for child in result["children"]:
            assert "point" in child

    def test_deep_tree(self):
        ht = HyperbolicTree(dim=3)
        tree = {"name": "L0", "children": [
            {"name": "L1", "children": [
                {"name": "L2", "children": [
                    {"name": "L3"}
                ]}
            ]}
        ]}
        result = ht.embed(tree)
        node = result
        for depth in range(4):
            assert "point" in node
            assert "name" in node
            if "children" in node:
                node = node["children"][0]

    def test_higher_dimension(self):
        ht = HyperbolicTree(dim=5)
        tree = {"name": "Root", "children": [{"name": "A"}]}
        result = ht.embed(tree)
        assert result["point"].shape == (5,)
        assert result["children"][0]["point"].shape == (5,)

    def test_custom_curvature(self):
        ht = HyperbolicTree(dim=3, curvature=2.0)
        assert ht.ball.c == 2.0
        tree = {"name": "Root", "children": [{"name": "A"}]}
        result = ht.embed(tree)
        # All points should be within radius = 1/sqrt(2)
        assert np.linalg.norm(result["children"][0]["point"]) < ht.ball.radius

    def test_deterministic(self):
        """Same input → same output (seeded RNG)."""
        ht1 = HyperbolicTree(dim=3)
        ht2 = HyperbolicTree(dim=3)
        tree = {"name": "Root", "children": [{"name": "A"}, {"name": "B"}]}
        r1 = ht1.embed(tree)
        r2 = ht2.embed(tree)
        np.testing.assert_array_almost_equal(r1["children"][0]["point"], r2["children"][0]["point"])
