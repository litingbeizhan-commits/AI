"""Independent exact check for the 28-model, two-resource instance.

Run with: python verify_small.py PATH_TO_multi_resource_n28_m2.txt
This offline reference is independent of the search and its evaluation budget.
"""

import argparse
import json
from pathlib import Path


def exact_reference(path):
    """Return an optimal witness by sparse dynamic programming over exact loads."""
    tokens = list(map(int, Path(path).read_text(encoding="utf-8-sig").split()))
    if tokens[:2] != [28, 2] or len(tokens) != 88:
        raise ValueError("Expected the 28-model, 2-resource integer instance.")
    utilities = tokens[2:30]
    draws = [tokens[30:58], tokens[58:86]]
    limits = tokens[86:88]
    if any(value < 0 for value in tokens[30:]):
        raise ValueError("Resource draws and capacities must be nonnegative.")

    # After item j, each exact load pair keeps its highest utility and a witness.
    states = {(0, 0): (0, 0)}
    for j, (utility, a, b) in enumerate(zip(utilities, *draws)):
        next_states = states.copy()  # Skip the current model.
        for (u, v), (score, mask) in states.items():
            new_load = (u + a, v + b)
            if new_load[0] > limits[0] or new_load[1] > limits[1]:
                continue
            previous = next_states.get(new_load)
            if previous is None or score + utility > previous[0]:
                next_states[new_load] = (score + utility, mask | (1 << j))
        states = next_states

    loads, (best, mask) = max(states.items(), key=lambda state: state[1][0])
    chosen = [j for j in range(len(utilities)) if mask & (1 << j)]
    assert best == sum(utilities[j] for j in chosen)
    assert loads == tuple(sum(row[j] for j in chosen) for row in draws)
    assert all(load <= limit for load, limit in zip(loads, limits))
    return {"optimal_utility": best, "models_1based": [j + 1 for j in chosen],
            "resource_loads": list(loads), "capacities": limits}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", type=Path)
    print(json.dumps(exact_reference(parser.parse_args().instance), indent=2))
