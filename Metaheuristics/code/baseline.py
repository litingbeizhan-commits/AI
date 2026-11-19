"""Budgeted iterated hill climbing for a binary multi-resource deployment."""

from time import perf_counter

import numpy as np


def _neighbor(bits, rng):
    """Add one absent model or exchange one deployed model for an absent one."""
    selected = np.flatnonzero(bits)
    absent = np.flatnonzero(bits == 0)
    candidate = bits.copy()
    if absent.size:
        candidate[rng.choice(absent)] = 1
    if selected.size and (not absent.size or rng.random() < 0.5):
        candidate[rng.choice(selected)] = 0
    return candidate


def _perturb(bits, rng):
    """Release capacity at a stalled point by dropping up to two models."""
    selected = np.flatnonzero(bits)
    candidate = bits.copy()
    if selected.size:
        dropped = rng.choice(selected, size=min(2, selected.size), replace=False)
        candidate[dropped] = 0
    return candidate


def run_baseline(instance, seed, budget):
    """Return one run, recording the best feasible utility after every evaluation."""
    if not isinstance(budget, (int, np.integer)) or budget < 1:
        raise ValueError("budget must be a positive integer")
    start = perf_counter()
    rng = np.random.default_rng(seed)
    current = np.zeros(instance.n, dtype=np.int8)
    score, usage = instance.evaluate(current)
    best, best_score, best_usage = current.copy(), score, usage.copy()
    trace = [best_score]
    stalls = kicks = infeasible_proposals = 0
    while len(trace) < budget:
        kick = stalls >= 2 * instance.n
        candidate = _perturb(current, rng) if kick else _neighbor(current, rng)
        candidate_score, candidate_usage = instance.evaluate(candidate)
        feasible = bool(np.all(candidate_usage <= instance.capacities))
        if not feasible:
            infeasible_proposals += 1
        if feasible and (kick or candidate_score > score):
            current, score = candidate, candidate_score
            stalls = 0
            kicks += int(kick)
            if score > best_score:
                best, best_score, best_usage = current.copy(), score, candidate_usage.copy()
        else:
            stalls += 1
        trace.append(best_score)
    return {
        "seed": int(seed),
        "score": best_score,
        "bits": best.tolist(),
        "usage": best_usage.tolist(),
        "trace": trace,
        "evaluations": len(trace),
        "elapsed_s": perf_counter() - start,
        "infeasible_proposals": infeasible_proposals,
        "kicks": kicks,
    }
