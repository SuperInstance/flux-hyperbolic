# flux-hyperbolic

Hyperbolic geometry for embedding musical tradition hierarchies.

Music traditions nest like trees: Jazz → Bebop → Charlie Parker's style. Euclidean space can't faithfully represent deep hierarchies because ball volume grows polynomially. Hyperbolic space has exponential volume growth — a natural fit for tree-like data where each level branches into many children.

This library provides two models of hyperbolic geometry (Poincaré ball, Lorentz/hyperboloid), Riemannian gradient descent, Sarkar's tree construction, and a `TraditionEmbedding` that maps the 10 SuperInstance traditions into hyperbolic coordinates.

## Installation

```bash
pip install flux-hyperbolic
```

Requires Python ≥ 3.11, NumPy ≥ 1.24.

## Poincaré Ball

The Poincaré ball is the open unit ball **B**ⁿ = {x : ‖x‖ < 1/√c} with the metric:

```
g_x = λ_x² · I,    λ_x = 2 / (1 − c·‖x‖²)
```

Distance grows rapidly near the boundary — root nodes sit near the origin, leaves near the edge.

```python
from flux_hyperbolic import PoincareBall

ball = PoincareBall(curvature=1.0)

# Two points in the ball (‖·‖ < 1 when c=1)
jazz = np.array([0.2, 0.1, 0.3])
blues = np.array([0.25, 0.15, 0.28])

# Hyperbolic distance
d = ball.distance(jazz, blues)

# Möbius addition (non-commutative group operation)
moved = ball.mobius_add(jazz, blues)

# Move along a tangent vector via exponential map
tangent = np.array([0.01, -0.02, 0.005])
new_point = ball.exp_map(jazz, tangent)

# Recover the tangent via logarithmic map
recovered = ball.log_map(jazz, new_point)  # ≈ tangent

# Mobility: exp(−d(u,v)), measures information flow between points
mob = ball.mobility(jazz, blues)  # in (0, 1]

# Project a point back inside the ball
outside = np.array([1.5, 0.0, 0.0])
inside = ball.project(outside)  # norm clamped to < radius
```

## Lorentz (Hyperboloid) Model

The Lorentz model lives on the upper sheet of a hyperboloid in Minkowski space:

```
Hⁿ = {x ∈ ℝⁿ⁺¹ : ⟨x,x⟩_L = −1/c, x₀ > 0}
```

Numerically more stable than Poincaré for points far from the origin.

```python
from flux_hyperbolic import LorentzModel

model = LorentzModel(curvature=1.0)

# Project a spatial vector onto the hyperboloid
spatial = np.array([0.1, 0.2, 0.05])  # 3D spatial part
x = model.project(spatial)  # returns [x0, 0.1, 0.2, 0.05] where x0 = √(1 + ‖spatial‖²)

# Distance via Minkowski inner product: d = arccosh(−⟨x,y⟩_L)
y = model.project(np.array([0.15, 0.18, 0.1]))
d = model.distance(x, y)

# Convert between Poincaré ball and Lorentz coordinates
ball = PoincareBall(curvature=1.0)
p = model.to_poincare(x)       # Lorentz → Poincaré
x_back = model.from_poincare(p) # Poincaré → Lorentz
```

### Which model to use?

| Property | Poincaré Ball | Lorentz |
|---|---|---|
| Stable near origin | ✅ | ✅ |
| Stable near boundary | ❌ | ✅ |
| Conformal (preserves angles) | ✅ | ❌ |
| Visualization | ✅ | ❌ |
| Deep embeddings / optimization | ❌ | ✅ |

## Standalone Distance Functions

```python
from flux_hyperbolic import poincare_distance, lorentz_distance, hyperbolic_distance

# Direct distance computation without instantiating a model
d = poincare_distance(u, v, curvature=1.0)
d = lorentz_distance(x, y, curvature=1.0)

# Generic dispatcher
d = hyperbolic_distance(u, v, model="poincare")  # or model="lorentz"
```

## Tradition Embedding

`TraditionEmbedding` maps the 10 built-in traditions into the Poincaré ball using the hierarchical tree:

```
Music
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
```

```python
from flux_hyperbolic import TraditionEmbedding
import numpy as np

emb = TraditionEmbedding(dim=3, curvature=1.0)
points = emb.embed_traditions()

# Get the hyperbolic point for a tradition
jazz_point = points["Jazz"]       # np.ndarray of shape (3,)
blues_point = points["Blues"]

# Distance between two traditions in hyperbolic space
d = emb.tradition_distance("Jazz", "Blues")

# Find nearest traditions to a point
query = np.array([0.15, 0.1, 0.2])
neighbors = emb.nearest_traditions(query, k=5)
# [("Jazz", 0.42), ("Blues", 0.58), ...]

# Get the full tree with embedded points
tree = emb.tradition_tree()
```

## Hyperbolic Tree Construction

`HyperbolicTree` embeds arbitrary tree structures using a simplified Sarkar's algorithm — root at origin, children placed at decreasing distances with angular separation.

```python
from flux_hyperbolic import HyperbolicTree

tree = HyperbolicTree(dim=3, curvature=1.0)

# Define a tree as nested dicts
taxonomy = {
    "name": "Music",
    "children": [
        {
            "name": "Jazz",
            "children": [
                {"name": "Bebop"},
                {"name": "Cool Jazz"},
            ]
        },
        {
            "name": "Classical",
            "children": [
                {"name": "Baroque"},
                {"name": "Romantic"},
            ]
        },
    ],
}

# Embed: adds a "point" field to each node
embedded = tree.embed(taxonomy)
# embedded["children"][0]["point"]  →  Jazz's coordinates
# embedded["children"][0]["children"][0]["point"]  →  Bebop's coordinates
```

Child placement uses distance `0.3 / (depth + 1)` from the parent, with children evenly spaced angularly. This ensures:
- Parent-child distance decreases with depth
- Siblings are well-separated
- Deeper nodes are closer to the ball boundary

## Riemannian Gradient Descent

Standard Euclidean gradient descent doesn't account for curvature. The Riemannian gradient corrects for the conformal factor:

```
∇ᴿf(x) = (1 / λ_x²) · ∇ᴱf(x)
```

Each step applies the exponential map to move along the manifold:

```python
from flux_hyperbolic import RiemannianGradientDescent, PoincareBall
import numpy as np

optimizer = RiemannianGradientDescent(
    dim=3,
    curvature=1.0,
    lr=0.01,
    max_iter=1000,
    tol=1e-6,
)

# Minimize distance to a target point
target = np.array([0.3, 0.2, 0.1])
start = np.array([0.5, 0.4, 0.3])

def loss_fn(x):
    diff = x - target
    return float(np.dot(diff, diff))

def grad_fn(x):
    return 2.0 * (x - target)

optimized, losses = optimizer.optimize(start, loss_fn, grad_fn)
# optimized ≈ target, losses decreases over iterations
```

The optimizer:
1. Computes the Euclidean gradient
2. Divides by λ² (conformal factor squared) to get the Riemannian gradient
3. Steps via `exp_map(x, −lr · gᴿ)`
4. Projects back into the ball

## Testing

```bash
pytest                          # all tests
pytest tests/test_poincare.py   # Poincaré model
pytest tests/test_lorentz.py    # Lorentz model
pytest tests/test_embedding.py  # TraditionEmbedding
pytest tests/test_distance.py   # standalone distance functions
pytest -v --cov=flux_hyperbolic # with coverage
```

## API Reference

### `PoincareBall(curvature=1.0)`

| Method | Description |
|---|---|
| `.distance(u, v)` | Hyperbolic distance d(u,v) = (1/√c)·arccosh(1 + 2c‖u−v‖²/((1−c‖u‖²)(1−c‖v‖²))) |
| `.mobius_add(x, y)` | Möbius addition (non-commutative) |
| `.exp_map(base, tangent)` | Exponential map: tangent vector → point |
| `.log_map(base, point)` | Logarithmic map: point → tangent vector |
| `.project(x, eps=1e-5)` | Clamp norm < radius |
| `.mobility(u, v)` | exp(−d(u,v)), information flow metric |

### `LorentzModel(curvature=1.0)`

| Method | Description |
|---|---|
| `.distance(x, y)` | d = (1/√c)·arccosh(−⟨x,y⟩_L) |
| `.minkowski_dot(x, y)` | Minkowski inner product −x₀y₀ + Σxᵢyᵢ |
| `.project(x)` | Project onto hyperboloid: x₀ ← √(1/c + ‖x_spatial‖²) |
| `.exp_map(base, tangent)` | Exponential map on hyperboloid |
| `.log_map(base, point)` | Logarithmic map |
| `.to_poincare(x)` | Lorentz → Poincaré coordinates |
| `.from_poincare(p)` | Poincaré → Lorentz coordinates |

### `TraditionEmbedding(dim=3, curvature=1.0)`

| Method | Description |
|---|---|
| `.embed_traditions(dials=None)` | Map traditions to hyperbolic points, returns dict |
| `.tradition_distance(t1, t2)` | Hyperbolic distance between two traditions |
| `.nearest_traditions(point, k=5)` | k-nearest traditions to a point |
| `.tradition_tree()` | Full tree with embedded coordinates |

### `HyperbolicTree(dim=3, curvature=1.0)`

| Method | Description |
|---|---|
| `.embed(tree_dict)` | Embed a nested dict tree, adds `"point"` to each node |

### `RiemannianGradientDescent(dim=3, curvature=1.0, lr=0.01, max_iter=1000, tol=1e-6)`

| Method | Description |
|---|---|
| `.optimize(start, loss_fn, grad_fn)` | Run RGD, returns (optimized_point, loss_history) |

### Standalone Functions

| Function | Description |
|---|---|
| `poincare_distance(u, v, curvature=1.0)` | Direct Poincaré distance |
| `lorentz_distance(x, y, curvature=1.0)` | Direct Lorentz distance |
| `hyperbolic_distance(u, v, model="poincare", curvature=1.0)` | Dispatch by model name |

## Related

- [flux-genome](https://github.com/SuperInstance/flux-genome) — Genetic evolution of musical genomes
- [constraint-dsl](https://github.com/SuperInstance/constraint-dsl) — Declarative constraint language
- [superinstance-live](https://github.com/SuperInstance/superinstance-live) — Live session controller

## License

MIT
