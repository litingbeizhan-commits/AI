"""Run the coursework experiments from any working directory.

    python code/experiments.py baseline
    python code/experiments.py challenge
    python code/experiments.py smoke

Place the official instances in data/, or pass --instance PATH.
"""

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

from baseline import run_baseline
from population import VARIANTS, POPULATION_SIZE, run_ga
from problem import load_instance
from workload_data_loader import load_workload_instance


ROOT = Path(__file__).resolve().parents[1]
LABELS = {"baseline": "Hill climber", "uniform": "Uniform", "feasible_union": "Feasible union"}
COLOURS = {"baseline": "#16646c", "uniform": "#a54c21", "feasible_union": "#16646c"}
SEED_ROOT = 20261119


def independent_seed(algorithm, run_index):
    """Derive a reproducible seed from a fixed algorithm ID and replicate index."""
    group = {"baseline": 1, "uniform": 2, "feasible_union": 3}[algorithm]
    seed = np.random.SeedSequence([SEED_ROOT, group, run_index])
    return int(seed.generate_state(1, dtype=np.uint64)[0])


def verify(instance, result, budget):
    bits = result["bits"]
    trace = result["trace"]
    # Recompute the winning portfolio without using the search's vector operations.
    selected = [j for j, value in enumerate(bits) if value == 1]
    uses = [sum(int(row[j]) for j in selected) for row in instance.weights]
    value = sum(int(instance.profits[j]) for j in selected)
    valid = (len(bits) == instance.n and all(v in (0, 1) for v in bits)
             and value == result["score"] and uses == list(result["usage"])
             and all(used <= cap for used, cap in zip(uses, instance.capacities))
             and budget == result["evaluations"] == len(trace)
             and trace[-1] == value and all(x <= y for x, y in zip(trace, trace[1:])))
    if not valid:
        raise RuntimeError(f"Invalid portfolio, trace or evaluation count for seed {result['seed']}")


def compare(groups):
    """Primary endpoint: final utility; direction fixed as feasible union > uniform."""
    from scipy.stats import mannwhitneyu

    x = np.array([r["score"] for r in groups["feasible_union"]])
    y = np.array([r["score"] for r in groups["uniform"]])
    test = mannwhitneyu(x, y, alternative="greater", method="asymptotic", use_continuity=True)
    # Fraction of cross-group wins with half credit for ties. Bootstrap whole runs.
    wins = (x[:, None] > y).astype(float) + 0.5 * (x[:, None] == y)
    rand = np.random.default_rng(19311)
    bootstrap = []
    for _ in range(5000):
        ix = rand.integers(len(x), size=len(x))
        iy = rand.integers(len(y), size=len(y))
        bootstrap.append(wins[np.ix_(ix, iy)].mean())
    return {"comparison": "feasible_union > uniform", "alpha": 0.05,
            "method": "one-sided Mann-Whitney U, asymptotic, tie and continuity corrected",
            "U": float(test.statistic), "p": float(test.pvalue),
            "superiority_probability": float(wins.mean()),
            "superiority_bootstrap_95pct": np.percentile(bootstrap, [2.5, 97.5]).tolist(),
            "median_score_difference": float(np.median(x) - np.median(y))}


def summarise(runs, instance):
    scores = np.array([r["score"] for r in runs])
    best = runs[int(np.argmax(scores))]
    selected = [idx + 1 for idx, v in enumerate(best["bits"]) if v]
    summary = {"runs": len(runs), "best": int(scores.max()), "worst": int(scores.min()),
               "mean": float(scores.mean()), "sd": float(scores.std(ddof=1)),
               "q1": float(np.percentile(scores, 25)), "median": float(np.median(scores)),
               "q3": float(np.percentile(scores, 75)), "best_frequency": int(sum(scores == scores.max())),
               "median_runtime_s": float(np.median([r["elapsed_s"] for r in runs])),
               "median_first_best_eval": float(np.median([r["trace"].index(r["score"]) + 1 for r in runs])),
               "best_seed": best["seed"], "best_selected_1based": selected, "best_usage": best["usage"],
               "best_slack": (instance.capacities - best["usage"]).tolist(),
               "infeasible_proposals": sum(r["infeasible_proposals"] for r in runs)}
    if "offspring" in best:
        children = sum(r["offspring"] for r in runs)
        summary.update(offspring=children, repair_rate=summary["infeasible_proposals"] / children,
                       deletions_per_child=sum(r["repair_deletions"] for r in runs) / children)
    if "kicks" in best:
        summary["mean_kicks"] = float(np.mean([r["kicks"] for r in runs]))
    return summary


def write_tables(output, groups, checkpoints):
    with (output / "runs.csv").open("w", newline="", encoding="utf-8") as f:
        cols = ["algorithm", "run", "seed", "evaluations", "score", "elapsed_s",
                "infeasible_proposals", "kicks", "offspring", "repair_deletions", "bits", "usage"]
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for algorithm, runs in groups.items():
            for i, run in enumerate(runs, 1):
                row = {k: run.get(k, "") for k in cols}
                row.update(algorithm=algorithm, run=i, bits="".join(map(str, run["bits"])),
                           usage=" ".join(map(str, run["usage"])))
                writer.writerow(row)
    with (output / "convergence.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["algorithm", "run", "evaluations", "best_utility"])
        for algorithm, runs in groups.items():
            for i, run in enumerate(runs, 1):
                writer.writerows((algorithm, i, k, run["trace"][k - 1]) for k in checkpoints)


def plot(output, groups, checkpoints):
    fig, ax = plt.subplots(figsize=(7.8, 3.25), constrained_layout=True)
    for algorithm, runs in groups.items():
        curves = np.array([r["trace"] for r in runs])[:, checkpoints - 1]
        q1, median, q3 = np.percentile(curves, [25, 50, 75], axis=0)
        ax.fill_between(checkpoints, q1, q3, color=COLOURS[algorithm], alpha=0.16)
        ax.plot(checkpoints, median, color=COLOURS[algorithm], label=LABELS[algorithm], lw=2.2)
    ax.set(xlabel="Fitness evaluations", ylabel="Best feasible utility", xlim=(1, checkpoints[-1]))
    ax.grid(axis="y", alpha=0.24)
    ax.legend(frameon=False)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    fig.savefig(output / "convergence.png", dpi=180)
    plt.close(fig)
    if len(groups) > 1:
        fig, ax = plt.subplots(figsize=(6.7, 3.5), constrained_layout=True)
        for i, (algorithm, runs) in enumerate(groups.items(), 1):
            scores = np.array([r["score"] for r in runs])
            artists = ax.boxplot(scores, positions=[i], widths=0.35, patch_artist=True,
                                 showfliers=False, medianprops={"color":"#252a30", "linewidth":2})
            artists["boxes"][0].set(facecolor=COLOURS[algorithm], alpha=0.32)
            jitter = np.random.default_rng(i).uniform(-0.06, 0.06, len(scores))
            ax.scatter(i + 0.28 + jitter, scores, s=15, color=COLOURS[algorithm], alpha=0.65)
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels([LABELS[k] for k in groups])
        ax.set(ylabel="Final feasible utility", xlim=(0.55, len(groups) + 0.55))
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.grid(axis="y", alpha=0.24)
        fig.savefig(output / "boxplot.png", dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=["baseline", "challenge", "smoke"],
                        help="smoke verifies both GA variants on the small input")
    parser.add_argument("--instance", type=Path)
    parser.add_argument("--out", type=Path, help="directory for this study's outputs")
    parser.add_argument("--runs", type=int, help="replicates per method (default 30, or 1 for smoke)")
    parser.add_argument("--budget", type=int, default=2000)
    args = parser.parse_args()
    challenge = args.study == "challenge"
    smoke = args.study == "smoke"
    if args.runs is None:
        args.runs = 1 if smoke else 30
    shape = (90, 5) if challenge else (28, 2)
    input_path = args.instance or ROOT / "data" / f"multi_resource_n{shape[0]}_m{shape[1]}.txt"
    if not input_path.is_file():
        parser.error(f"Official input missing: {input_path}. Supply the course file with --instance.")
    min_runs = 1 if smoke else 30
    if args.runs < min_runs or args.budget <= POPULATION_SIZE:
        parser.error(f"Use at least {min_runs} runs and more than {POPULATION_SIZE} evaluations.")
    instance = load_instance(input_path)
    if (instance.n, instance.m) != shape:
        parser.error(f"{args.study} requires n={shape[0]}, m={shape[1]}")
    # Cross-check the course loader once per input. Use the validated integer
    # representation during search so objectives and capacity checks are exact.
    n, m, utilities, demands, capacities = load_workload_instance(input_path)
    if ((n, m) != shape or not np.array_equal(utilities, instance.profits)
            or not np.array_equal(demands, instance.weights)
            or not np.array_equal(capacities, instance.capacities)):
        parser.error("Course loader and integer parser disagree on the supplied instance.")
    output = args.out or ROOT / "results" / args.study
    groups = {k: [] for k in (("baseline",) if args.study == "baseline" else VARIANTS)}
    for i in range(args.runs):
        # Alternate run order to reduce timing bias; RNG streams remain independent.
        order = list(groups) if i % 2 == 0 else list(reversed(groups))
        for algorithm in order:
            seed = independent_seed(algorithm, i)
            if challenge or smoke:
                run = run_ga(instance, algorithm, seed, args.budget)
            else:
                run = run_baseline(instance, seed, args.budget)
            verify(instance, run, args.budget)
            if smoke:
                replay = run_ga(instance, algorithm, seed, args.budget)
                verify(instance, replay, args.budget)
                if any(run[k] != replay[k] for k in run if k != "elapsed_s"):
                    raise RuntimeError("Seed replay changed solver output")
            groups[algorithm].append(run)
    seeds = [r["seed"] for runs in groups.values() for r in runs]
    if len(set(seeds)) != len(seeds):
        raise RuntimeError("Duplicate random seeds in the study")
    checkpoints = np.unique(np.r_[1, np.arange(25, args.budget + 1, 25), args.budget])
    output.mkdir(parents=True, exist_ok=True)
    write_tables(output, groups, checkpoints)
    if smoke:
        print(f"Verified both GA variants and their deterministic replays on {input_path.name}.")
        return
    plot(output, groups, checkpoints)
    summary = {"instance": input_path.name, "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
               "n": instance.n, "m": instance.m, "capacity": instance.capacities.tolist(),
               "budget": args.budget, "master_seed": SEED_ROOT,
               "environment": {"python": platform.python_version(), "numpy": np.__version__,
                               "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
                               "platform": platform.platform()},
               "algorithms": {k: summarise(rs, instance) for k, rs in groups.items()}}
    if challenge:
        summary["test"] = compare(groups)
    with (output / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    print(f"{len(seeds)} verified runs written to {output}")


if __name__ == "__main__":
    main()
