"""Elitist genetic search with two discrete recombination operators."""

from time import perf_counter

import numpy as np

from problem import Instance


POPULATION_SIZE = 40
VARIANTS = ("uniform", "feasible_union")


def pack(instance, child, ordered_items):
    """Insert items in the supplied order if every residual capacity permits it."""
    remaining = instance.capacities - instance.weights @ child
    for j in ordered_items:
        demand = instance.weights[:, j]
        if np.all(demand <= remaining):
            child[j] = 1
            remaining -= demand
    return child


def uniform(parent_a, parent_b, rng):
    mask = rng.random(parent_a.size) < 0.5
    return np.where(mask, parent_a, parent_b)


def feasible_union(instance, parent_a, parent_b, density, rng):
    child = parent_a & parent_b
    candidates = np.flatnonzero(parent_a ^ parent_b)
    priorities = density[candidates] * rng.uniform(0.5, 1.5, size=candidates.size)
    order = candidates[np.argsort(-priorities, kind="stable")]
    return pack(instance, child, order)


def repair(instance, child, scales):
    """Delete an item maximizing normalized excess relieved per unit utility lost."""
    usage = instance.weights @ child
    deleted = 0
    while np.any(usage > instance.capacities):
        excess = np.maximum(usage - instance.capacities, 0)
        active = np.flatnonzero(child)
        relief = np.minimum(instance.weights[:, active], excess[:, None]) / scales[:, None]
        # Prefer an item with nonpositive utility if it relieves excess.
        relief = relief.sum(axis=0)
        profits = instance.profits[active]
        ratio = np.divide(relief, profits, out=np.zeros_like(relief), where=profits > 0)
        ratio[(profits <= 0) & (relief > 0)] = np.inf
        j = active[np.argmax(ratio)]
        child[j] = 0
        usage -= instance.weights[:, j]
        deleted += 1
    return deleted


def tournament(population, scores, rng):
    contestants = rng.choice(len(population), size=3, replace=False)
    return population[contestants[np.argmax(scores[contestants])]]


def run_ga(instance: Instance, variant: str, seed: int, budget: int):
    """One run; each initialized individual and each child uses one fitness call."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown crossover {variant!r}")
    if budget < POPULATION_SIZE:
        raise ValueError(f"Budget must allow at least {POPULATION_SIZE} initial individuals")
    start = perf_counter()
    rng = np.random.default_rng(seed)
    scales = np.maximum(instance.capacities, 1)
    total_draw = (instance.weights / scales[:, None]).sum(axis=0)
    density = np.divide(instance.profits, total_draw, out=np.zeros_like(total_draw),
                        where=total_draw > 0)
    density[(total_draw == 0) & (instance.profits > 0)] = np.inf

    population = []
    scores = []
    best_score = -np.inf
    trace = []
    rejected_before_repair = 0
    deletions = 0

    def evaluate(bits):
        nonlocal best_score, best_bits
        score = int(instance.profits @ bits)
        if score > best_score:
            best_score = score
            best_bits = bits.copy()
        trace.append(best_score)
        return score

    best_bits = np.zeros(instance.n, dtype=np.int8)
    for _ in range(POPULATION_SIZE):
        child = pack(instance, np.zeros_like(best_bits), rng.permutation(instance.n))
        scores.append(evaluate(child))
        population.append(child)

    while len(trace) < budget:
        scores = np.asarray(scores)
        elite = int(np.argmax(scores))
        next_population = [population[elite]]
        next_scores = [scores[elite]]
        children_needed = min(POPULATION_SIZE - 1, budget - len(trace))
        for _ in range(children_needed):
            a = tournament(population, scores, rng)
            b = tournament(population, scores, rng)
            if variant == "uniform":
                child = uniform(a, b, rng)
            else:
                child = feasible_union(instance, a, b, density, rng)
            child ^= (rng.random(instance.n) < 1 / instance.n).astype(np.int8)
            removed = repair(instance, child, scales)
            rejected_before_repair += int(removed > 0)
            deletions += removed
            next_scores.append(evaluate(child))
            next_population.append(child)
        population, scores = next_population, next_scores

    return dict(seed=seed, score=best_score, bits=best_bits.tolist(),
                usage=(instance.weights @ best_bits).tolist(), trace=trace,
                evaluations=len(trace), elapsed_s=perf_counter() - start,
                offspring=budget - POPULATION_SIZE,
                infeasible_proposals=rejected_before_repair, repair_deletions=deletions)
