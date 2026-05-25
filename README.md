# flux-hyperbolic

> Hyperbolic geometry for music tradition hierarchy — because traditions nest like trees, not flat lists

Part of the [SuperInstance](https://github.com/SuperInstance) music constraint theory ecosystem. Embeds music traditions in hyperbolic space where hierarchical relationships (Blues → Delta Blues → Robert Johnson) are naturally preserved through exponential distance growth, something Euclidean space fundamentally cannot do.

## What It Does

Music traditions form hierarchies: "Jazz" contains "Bebop" which contains "Charlie Parker's style." In Euclidean space, representing these nested relationships faithfully is impossible — the volume of a Euclidean ball grows polynomially, so you run out of room for deep hierarchies. Hyperbolic space solves this: volume grows exponentially, mirroring how each level of a music taxonomy branches into many sub-traditions.

**flux-hyperbolic** provides two models of hyperbolic geometry (Poincaré ball and Lorentz/Hyperboloid), Riemannian gradient descent for optimization, and Sarkar's algorithm for constructing hyperbolic trees from pairwise distance data. Traditions get embedded as points where distance encodes similarity and hierarchy depth.

## Key Features

- **PoincaréBall model** — conformal ball model with Möbius addition, exponential/logarithmic maps
- **LorentzModel** — hyperboloid model with closed-form distance and projection
- **TraditionEmbedding** — maps music traditions to hyperbolic points with similarity queries
- **HyperbolicTree** — constructs embeddings from tree structure via Sarkar's algorithm
- **RiemannianGradientDescent** — optimizes embeddings using Riemannian (not Euclidean) gradients
- **25 tests** — full coverage of both models and tree operations

## Installation

```bash
git clone https://github.com/SuperInstance/flux-hyperbolic.git
cd flux-hyperbolic
pip install -e ".[dev]"
```

Requires Python 3.11+.

## Quick Start

### Compute hyperbolic distances

```python
from flux_hyperbolic.poincare import PoincareBall

ball = PoincareBall(dimension=5)

# Two tradition embeddings in the Poincaré ball
tradition_a = ball.random_point()
tradition_b = ball.random_point()

# Hyperbolic distance (not Euclidean!)
dist = ball.distance(tradition_a, tradition_b)
```

### Embed a tradition hierarchy

```python
from flux_hyperbolic.embedding import TraditionEmbedding

embedding = TraditionEmbedding(dimension=5)

# Define hierarchy: parent → children
embedding.add_tradition("jazz", parent=None)
embedding.add_tradition("bebop", parent="jazz")
embedding.add_tradition("cool_jazz", parent="jazz")
embedding.add_tradition("hard_bop", parent="jazz")

# Find nearest traditions
neighbors = embedding.nearest("bebop", k=3)
```

### Build a hyperbolic tree

```python
from flux_hyperbolic.tree import HyperbolicTree

tree = HyperbolicTree(dimension=5)
tree.insert("root", parent=None)
tree.insert("jazz", parent="root")
tree.insert("classical", parent="root")
tree.insert("bebop", parent="jazz")

# Get embedding for any node
point = tree.get_embedding("bebop")

# Sarkar's algorithm: construct from distances
tree = HyperbolicTree.from_distances(dimension=5, distance_matrix=dist_matrix)
```

### Optimize embeddings

```python
from flux_hyperbolic.optim import RiemannianGradientDescent
from flux_hyperbolic.poincare import PoincareBall

ball = PoincareBall(dimension=5)
optimizer = RiemannianGradientDescent(ball, learning_rate=0.01)

# Optimize points to minimize a loss
for step in range(1000):
    grads = compute_gradients(points, targets)
    points = optimizer.step(points, grads)
```

## Architecture

```
flux_hyperbolic/
├── poincare.py     # PoincaréBall: Möbius ops, exp/log maps, distance
├── lorentz.py      # LorentzModel: hyperboloid projection, distance
├── embedding.py    # TraditionEmbedding: tradition → point mapping
├── tree.py         # HyperbolicTree: Sarkar's algorithm
└── optim.py        # RiemannianGradientDescent
```

### Why Two Models?

| Feature | Poincaré Ball | Lorentz (Hyperboloid) |
|---|---|---|
| Numerically stable near origin | ✅ | ✅ |
| Numerically stable near boundary | ❌ | ✅ |
| Conformal (angles preserved) | ✅ | ❌ |
| Closed-form projection | ✅ | ✅ |
| Use case | Visualization, interpolation | Deep embeddings, optimization |

## API Reference

### `PoincareBall(dimension)`

```python
ball.distance(u, v)          # Hyperbolic distance
ball.mobius_add(a, b)        # Möbius addition
ball.exp_map(base, tangent)  # Exponential map
ball.log_map(base, point)    # Logarithmic map
ball.project(x)              # Project to ball (clamp norm < 1)
ball.random_point()          # Uniform random point in ball
```

### `LorentzModel(dimension)`

```python
model.distance(u, v)         # Hyperbolic distance on hyperboloid
model.inner_product(u, v)    # Minkowski inner product
model.project(x)             # Project onto hyperboloid
model.to_poincare(x)         # Convert to Poincaré ball coordinates
model.from_poincare(x)       # Convert from Poincaré ball coordinates
```

### `TraditionEmbedding(dimension)`

```python
emb.add_tradition(name, parent)        # Add to hierarchy
emb.get_point(name)                    # Get hyperbolic coordinates
emb.nearest(name, k)                   # k-nearest traditions
emb.similarity(name_a, name_b)         # Similarity score in [0, 1]
```

### `HyperbolicTree(dimension)`

```python
tree.insert(name, parent)              # Insert node
tree.get_embedding(name)               # Get point for node
tree.from_distances(dim, dist_matrix)  # Sarkar's construction
tree.distance(name_a, name_b)          # Tree distance
```

## Testing

```bash
pytest                        # Run all 25 tests
pytest tests/test_poincare.py # Poincaré model only
pytest tests/test_lorentz.py  # Lorentz model only
pytest -v --cov=flux_hyperbolic
```

## Related Repos

- [**flux-genome**](https://github.com/SuperInstance/flux-genome) — Genetic evolution of musical genomes
- [**constraint-toolkit**](https://github.com/SuperInstance/constraint-toolkit) — Core constraint satisfaction engine
- [**constraint-dsl**](https://github.com/SuperInstance/constraint-dsl) — YAML DSL for constraint pipelines
- [**superinstance-live**](https://github.com/SuperInstance/superinstance-live) — Live performance session controller
- [**flux-ffi**](https://github.com/SuperInstance/flux-ffi) — FFI bindings for shared LLVM backend

## License

MIT
