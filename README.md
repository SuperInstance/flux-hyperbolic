# flux-hyperbolic

Hyperbolic geometry toolkit for embedding musical traditions in curved space.

## Overview

`flux-hyperbolic` provides tools for representing musical traditions as points in hyperbolic space, where hierarchical relationships are naturally captured: similar traditions are close together, dissimilar ones are far apart (near the boundary).

### Models

- **Poincaré ball** — points in the open unit ball, ideal for visualization
- **Lorentz (hyperboloid)** — numerically stable, good for optimization

## Install

```bash
pip install flux-hyperbolic
```

## Quick start

### Poincaré ball

```python
from flux_hyperbolic import PoincareBall
import numpy as np

ball = PoincareBall(curvature=1.0)

# Distance between two points
u = np.array([0.1, 0.2, 0.3])
v = np.array([0.4, 0.1, 0.2])
d = ball.distance(u, v)

# Exponential/logarithmic maps (tangent space ↔ manifold)
tangent = ball.log_map(u, v)
recovered = ball.exp_map(u, tangent)  # ≈ v

# Mobility: how easily information flows
mob = ball.mobility(u, v)  # exp(-d), in (0, 1]
```

### Tradition embeddings

```python
from flux_hyperbolic import TraditionEmbedding

te = TraditionEmbedding(dim=3)
embeddings = te.embed_traditions()

# Distance between traditions
d = te.tradition_distance("Jazz", "Classical")

# Nearest traditions to a point
nearest = te.nearest_traditions(embeddings["Jazz"], k=5)
```

### Distance functions

```python
from flux_hyperbolic import hyperbolic_distance, poincare_distance, lorentz_distance

d = poincare_distance(u, v, curvature=1.0)
d = lorentz_distance(x, y, curvature=1.0)
d = hyperbolic_distance(u, v, model="poincare")
```

### Riemannian optimization

```python
from flux_hyperbolic import RiemannianGradientDescent
import numpy as np

opt = RiemannianGradientDescent(dim=3, lr=0.01, max_iter=500)
start = np.array([0.1, 0.2, 0.3])

def loss(x):
    return np.sum(x ** 2)

def grad(x):
    return 2 * x

result, history = opt.optimize(start, loss, grad)
```

## Module reference

| Module | Description |
|--------|-------------|
| `poincare` | Poincaré ball model (distance, exp/log maps, Möbius addition) |
| `lorentz` | Lorentz hyperboloid model (distance, projection, conversions) |
| `distance` | Standalone distance functions for both models |
| `embedding` | Tradition embeddings with hierarchical tree structure |
| `tree` | Sarkar's algorithm for tree embedding in hyperbolic space |
| `optimization` | Riemannian gradient descent |

## Dependencies

- numpy ≥ 1.24
- scipy ≥ 1.10

## License

MIT
