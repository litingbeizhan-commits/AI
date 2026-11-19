"""Offline 0/1 knapsack bound using only the challenge's second resource.

Run with: python code/verify_challenge.py
Or provide the path to the original 90-model, five-resource instance.
"""

import argparse
import json
from pathlib import Path

from problem import load_instance


def check_bound(path):
    instance = load_instance(path)
    if (instance.n, instance.m) != (90, 5):
        raise ValueError("Expected the 90-model, five-resource instance.")
    capacity = int(instance.capacities[1])
    # Each capacity holds the best score using items examined so far and a
    # matching bit mask. Descending capacities allow each item at most once.
    value = [0] * (capacity + 1)
    witness = [0] * (capacity + 1)
    for j, (gain, cost) in enumerate(zip(instance.profits, instance.weights[1])):
        gain, cost = int(gain), int(cost)
        for c in range(capacity, cost - 1, -1):
            candidate = value[c - cost] + gain
            if candidate > value[c]:
                value[c] = candidate
                witness[c] = witness[c - cost] | (1 << j)

    selected = [j for j in range(instance.n) if witness[capacity] & (1 << j)]
    loads = [sum(int(row[j]) for j in selected) for row in instance.weights]
    assert sum(int(instance.profits[j]) for j in selected) == value[capacity]
    assert loads[1] <= capacity
    feasible = all(v <= b for v, b in zip(loads, instance.capacities))
    # Removing four constraints gives an upper bound. If this maximiser also
    # satisfies all five, its objective attains the bound in the original model.
    return {"upper_bound": value[capacity], "feasible_for_all_resources": bool(feasible),
            "certified_optimum": value[capacity] if feasible else None,
            "models_1based": [j + 1 for j in selected], "resource_loads": loads,
            "capacities": instance.capacities.tolist()}


if __name__ == "__main__":
    default = Path(__file__).resolve().parents[1] / "data" / "multi_resource_n90_m5.txt"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", nargs="?", type=Path, default=default)
    print(json.dumps(check_bound(parser.parse_args().instance), indent=2))
