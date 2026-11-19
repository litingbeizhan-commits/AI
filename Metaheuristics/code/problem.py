"""Read the coursework's whitespace-delimited multidimensional knapsack format."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Instance:
    profits: np.ndarray
    weights: np.ndarray       # rows are resources, columns are models
    capacities: np.ndarray

    @property
    def n(self):
        return len(self.profits)

    @property
    def m(self):
        return len(self.capacities)

    def evaluate(self, bits):
        """Return utility and resource totals for one binary candidate."""
        return int(self.profits @ bits), self.weights @ bits


def load_instance(path):
    """Read n, m, n utilities, m rows of n draws, then m capacities."""
    path = Path(path)
    try:
        entries = [int(token) for token in path.read_text(encoding="utf-8-sig").split()]
    except ValueError as exc:
        raise ValueError(f"{path}: expected whitespace-separated integers") from exc
    if len(entries) < 2 or min(entries[:2]) < 1:
        raise ValueError(f"{path}: expected positive model and resource counts")
    n, m = entries[:2]
    expected = 2 + n + m * n + m
    if len(entries) != expected:
        raise ValueError(f"{path}: expected {expected} integers, found {len(entries)}")
    values = np.array(entries[2:], dtype=np.int64)
    profits = values[:n]
    weights = values[n:n + m * n].reshape(m, n)
    capacities = values[n + m * n:]
    if np.any(weights < 0) or np.any(capacities < 0):
        raise ValueError(f"{path}: resource draws and capacities must be nonnegative")
    return Instance(profits, weights, capacities)
