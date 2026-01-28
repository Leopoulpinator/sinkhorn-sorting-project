# sinkhorn-sorting-project
# Differentiable Sorting and Ranking via Optimal Transport

Implementation of Cuturi et al.'s Sinkhorn-based differentiable sorting operators.

## Installation
```bash
pip install numpy matplotlib scipy torch
```

## Usage
```python
from src.sinkhorn import sinkhorn_rank_sort
import numpy as np

x = np.array([0.5, 0.2, 0.4, 0.8, 0.7])
r_eps, s_eps = sinkhorn_rank_sort(x, epsilon=1e-2)
```

## Experiments

Run experiments with:
```bash
python experiments/exp1_convergence.py
```

## Author

Léo-Paul Delsaux - Master's Project - 2026

