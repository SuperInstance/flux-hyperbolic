# flux-hyperbolic

> Hyperbolic geometry toolkit for embedding musical tradition hierarchies.

`flux-hyperbolic` provides two models of hyperbolic space (Poincaré ball and Lorentz hyperboloid), tree embedding via Sarkar's algorithm, Riemannian gradient descent, and a `TraditionEmbedding` class that maps 10 musical traditions into hyperbolic coordinates — preserving their hierarchical relationships.

## Why Hyperbolic Space?

Music traditions form a **tree-like hierarchy** — Western music branches into Jazz, Classical, Rock, Blues; Eastern into Hindustani, Gamelan, Gagaku. This structure has a critical property: **the number of sub-traditions grows exponentially with depth**. A genre spawns subgenres, which spawn sub-subgenres.

Euclidean space struggles with this. To faithfully embed an exponential tree in ℝⁿ, you need dimensions that grow with tree depth, and distances between leaves at the same depth balloon.

**Hyperbolic space solves this elegantly.** In the Poincaré ball, the area of a sphere of radius *r* grows as *eʳ* — exponentially. This means:

- **Root nodes** sit near the origin
- **Leaf nodes** live near the boundary
- **Siblings** are close; **cousins** are far apart
- The hierarchy is preserved with **low distortion** even in 2–3 dimensions

For music traditions, this means Jazz and Blues are naturally close (same parent: Western), while Jazz and Gagaku are far apart — all without manual distance tuning.

## Installation

```bash
pip install flux-hyperbolic
```

Requires Python ≥ 3.10, NumPy ≥ 1.24, SciPy ≥ 1.10.

## Quick Start

### Embed traditions in hyperbolic space

```python
from flux_hyperbolic import TraditionEmbedding

te = TraditionEmbedding(dim=3, curvature=1.0)
embeddings = te.embed_traditions()

# How far apart are Jazz and Blues?
d = te.tradition_distance("Jazz", "Blues")

# How far apart are Jazz and Gagaku?
d2 = te.tradition_distance("Jazz", "Gagaku")
assert d2 > d  # Cross-family is farther

# Nearest traditions to a point
nearest = te.nearest_traditions(embeddings["Jazz"], k=3)
```

### Poincaré ball operations

```python
from flux_hyperbolic import PoincareBall
import numpy as np

ball = PoincareBall(curvature=1.0)

# Distance between two points
u = np.array([0.1, 0.2, 0.3])
v = np.array([0.4, 0.1, 0.2])
d = ball.distance(u, v)

# Exponential / logarithmic maps
tangent = np.array([0.05, -0.03, 0.02])
point = ball.exp_map(u, tangent)        # tangent → manifold
recovered = ball.log_map(u, point)       # manifold → tangent

# Möbius addition
result = ball.mobius_add(u, v)

# Mobility (information flow): exp(-d(u,v))
mob = ball.mobility(u, v)  # ∈ (0, 1]

# Project back into the ball
p = ball.project(np.array([0.9, 0.9, 0.9]))
```

### Lorentz (hyperboloid) model

```python
from flux_hyperbolic import LorentzModel
import numpy as np

model = LorentzModel(curvature=1.0)

# Points on the hyperboloid: x₀ = √(1 + ‖x_s‖²)
xs = np.array([0.1, 0.2])
x = np.concatenate([[np.sqrt(1.0 + np.dot(xs, xs))], xs])

# Distance
d = model.distance(x, y)

# Convert between models
p = model.to_poincare(x)    # Lorentz → Poincaré
x2 = model.from_poincare(p)  # Poincaré → Lorentz
```

### Riemannian gradient descent

```python
from flux_hyperbolic import RiemannianGradientDescent, PoincareBall
import numpy as np

opt = RiemannianGradientDescent(dim=3, lr=0.01, max_iter=1000)

start = np.array([0.1, 0.2, 0.3])

def loss(x):
    target = np.array([0.5, 0.0, 0.0])
    return 0.5 * np.sum((x - target) ** 2)

def grad(x):
    target = np.array([0.5, 0.0, 0.0])
    return x - target

optimized, history = opt.optimize(start, loss, grad)
```

### Tree embedding

```python
from flux_hyperbolic import HyperbolicTree

tree_data = {
    "name": "Root",
    "children": [
        {"name": "A", "children": [{"name": "A1"}, {"name": "A2"}]},
        {"name": "B", "children": [{"name": "B1"}]},
    ],
}

ht = HyperbolicTree(dim=3, curvature=1.0)
embedded = ht.embed(tree_data)
# Each node now has a "point" key with its Poincaré ball coordinate
```

### Standalone distance functions

```python
from flux_hyperbolic import hyperbolic_distance, poincare_distance, lorentz_distance

d = hyperbolic_distance(u, v, model="poincare", curvature=1.0)
```

## Architecture

### Tradition Hierarchy

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

Sarkar's algorithm places the root at the origin and recursively positions children at fixed distances from their parent, evenly spaced angularly. Deeper nodes are placed closer to the boundary, naturally encoding the hierarchy.

### Models

| Model | Representation | Best For |
|---|---|---|
| **Poincaré ball** | Open unit ball Bⁿ | Visualization, embeddings, tree construction |
| **Lorentz hyperboloid** | Upper sheet of Hⁿ | Numerical stability at large distances, optimization |

The models are interconvertible via `LorentzModel.to_poincare()` and `LorentzModel.from_poincare()`.

## API Reference

### `PoincareBall(curvature)`

| Method | Description |
|---|---|
| `.distance(u, v)` | Hyperbolic distance |
| `.project(x, eps)` | Clamp point inside ball |
| `.exp_map(base, tangent)` | Tangent vector → manifold point |
| `.log_map(base, point)` | Manifold point → tangent vector |
| `.mobility(u, v)` | exp(−d(u,v)), information flow measure |
| `.mobius_add(x, y)` | Möbius addition |
| `.radius` | 1/√c |

### `LorentzModel(curvature)`

| Method | Description |
|---|---|
| `.minkowski_dot(x, y)` | Minkowski inner product |
| `.distance(x, y)` | Hyperbolic distance |
| `.project(x)` | Project to hyperboloid |
| `.exp_map(base, tangent)` | Exponential map |
| `.log_map(base, point)` | Logarithmic map |
| `.to_poincare(x)` | Convert to Poincaré coordinates |
| `.from_poincare(p)` | Convert from Poincaré coordinates |

### `TraditionEmbedding(dim, curvature)`

| Method | Description |
|---|---|
| `.embed_traditions(dials)` | Map traditions to hyperbolic points |
| `.tradition_distance(t1, t2)` | Distance between two traditions |
| `.nearest_traditions(point, k)` | k-nearest traditions to a point |
| `.tradition_tree()` | Full embedded tree structure |

### `HyperbolicTree(dim, curvature)`

| Method | Description |
|---|---|
| `.embed(tree_dict)` | Embed arbitrary tree, adds `"point"` to each node |

### `RiemannianGradientDescent(dim, curvature, lr, max_iter, tol)`

| Method | Description |
|---|---|
| `.optimize(start, loss_fn, grad_fn)` | Returns (optimized_point, loss_history) |

### Distance Functions

| Function | Description |
|---|---|
| `poincare_distance(u, v, curvature)` | Standalone Poincaré distance |
| `lorentz_distance(x, y, curvature)` | Standalone Lorentz distance |
| `hyperbolic_distance(u, v, model, curvature)` | Dispatch to either model |

## Related Repos

- **[flux-genome](https://github.com/SuperInstance/flux-genome)** — Genetic algorithm framework for evolving musical traditions
- **[constraint-toolkit](https://github.com/SuperInstance/constraint-toolkit)** — Dial space definitions and constraint solving
- **[superinstance-live](https://github.com/SuperInstance/superinstance-live)** — Live session controller
- **[plato-client](https://github.com/SuperInstance/plato-client)** — Client for the Plato optimization backend

## License

MIT
