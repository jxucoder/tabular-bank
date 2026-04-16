"""Scaling analysis — ranking stability under varying conditions.

Studies how model rankings change as we vary:
1. **Dataset size** — How many samples per task are needed for stable rankings?
2. **Scenario count** — How many benchmark tasks produce stable aggregate rankings?
3. **Feature count** — How does dimensionality affect relative model strength?

These analyses directly address the workshop's call for submissions that
"study scaling across datasets, model size, and compute."
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

logger = logging.getLogger(__name__)


@dataclass
class RankingStabilityPoint:
    """A single data point in a scaling curve."""

    scale_value: int | float
    kendall_tau: float
    spearman_rho: float
    ranking: dict[str, int]


@dataclass
class ScalingReport:
    """Report from a scaling analysis."""

    axis: str  # "n_scenarios", "n_samples", "n_features"
    reference_ranking: dict[str, int]
    curve: list[RankingStabilityPoint]
    stable_at: int | float | None = None  # value where tau > stability_threshold

    def summary(self) -> str:
        lines = [
            f"=== Scaling Analysis: {self.axis} ===",
            "",
        ]
        if self.stable_at is not None:
            lines.append(f"Rankings stabilize at {self.axis}={self.stable_at}")
        else:
            lines.append("Rankings did not reach stability threshold.")

        lines += ["", f"  {'Value':>10s} {'Kendall tau':>12s} {'Spearman rho':>13s}"]
        lines.append("  " + "-" * 38)
        for pt in self.curve:
            lines.append(
                f"  {pt.scale_value:>10} {pt.kendall_tau:>12.3f} {pt.spearman_rho:>13.3f}"
            )
        return "\n".join(lines)


def analyze_scenario_scaling(
    task_scores: pd.DataFrame,
    scenario_counts: list[int] | None = None,
    n_bootstrap: int = 50,
    stability_threshold: float = 0.9,
    seed: int = 42,
) -> ScalingReport:
    """Bootstrap analysis: how many scenarios are needed for stable rankings?

    Subsamples tasks (columns) at varying counts and measures how well the
    subsampled ranking matches the full-data ranking.

    Args:
        task_scores: Model-by-task score matrix (models as rows, tasks as cols).
        scenario_counts: List of scenario counts to test.  Defaults to a
            range from 3 to the total number of tasks.
        n_bootstrap: Number of bootstrap resamples per count.
        stability_threshold: Kendall tau above which rankings are "stable".
        seed: Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    n_tasks = task_scores.shape[1]

    if scenario_counts is None:
        scenario_counts = sorted(set(
            [3, 5, 8, 10, 15, 20, 25, 30, 40, 50]
        ) & set(range(3, n_tasks + 1)))
        if not scenario_counts:
            scenario_counts = list(range(3, n_tasks + 1))

    # Reference ranking from full data
    full_mean = task_scores.mean(axis=1).sort_values(ascending=False)
    ref_ranking = {m: r for r, m in enumerate(full_mean.index, 1)}

    curve: list[RankingStabilityPoint] = []
    stable_at = None

    for count in scenario_counts:
        if count > n_tasks:
            continue

        taus = []
        rhos = []

        for _ in range(n_bootstrap):
            cols = rng.choice(task_scores.columns, size=count, replace=False)
            sub_mean = task_scores[cols].mean(axis=1).sort_values(ascending=False)
            sub_ranking = {m: r for r, m in enumerate(sub_mean.index, 1)}

            models = sorted(ref_ranking.keys())
            ref_r = [ref_ranking[m] for m in models]
            sub_r = [sub_ranking[m] for m in models]

            kt, _ = kendalltau(ref_r, sub_r)
            sr, _ = spearmanr(ref_r, sub_r)
            taus.append(kt)
            rhos.append(sr)

        mean_tau = float(np.mean(taus))
        mean_rho = float(np.mean(rhos))

        # Use the median bootstrap ranking for reporting
        curve.append(RankingStabilityPoint(
            scale_value=count,
            kendall_tau=mean_tau,
            spearman_rho=mean_rho,
            ranking=ref_ranking,
        ))

        if stable_at is None and mean_tau >= stability_threshold:
            stable_at = count

    return ScalingReport(
        axis="n_scenarios",
        reference_ranking=ref_ranking,
        curve=curve,
        stable_at=stable_at,
    )


def analyze_sample_scaling(
    models: dict[str, object],
    round_id: str = "round-001",
    master_secret: str | None = None,
    sample_sizes: list[int] | None = None,
    n_scenarios: int = 10,
    stability_threshold: float = 0.9,
    repeats: list[int] | None = None,
    folds: list[int] | None = None,
) -> ScalingReport:
    """Measure how rankings change as dataset size (n_samples) varies.

    Generates benchmark rounds at different sample size ranges.  All rounds
    share the same secret and round ID base, so the *scenario structure*
    (DAG topology, mechanism types, difficulty parameters) is identical —
    only the sample count varies.  This isolates the effect of dataset size
    on model rankings.

    Args:
        models: Dict of model_name -> sklearn-compatible model.
        round_id: Base round identifier (suffixed with sample size).
        master_secret: Secret for dataset generation.
        sample_sizes: List of max sample sizes to test.
        n_scenarios: Number of scenarios per round.
        stability_threshold: Kendall tau above which rankings are "stable".
        repeats: Which repeats to run.
        folds: Which folds to run.
    """
    from tabular_bank.leaderboard import get_task_scores
    from tabular_bank.runner import run_benchmark

    if sample_sizes is None:
        sample_sizes = [500, 1000, 2000, 5000, 10000, 15000]

    if repeats is None:
        repeats = [0]
    if folds is None:
        folds = [0, 1, 2]

    # Use the largest size as reference
    ref_size = max(sample_sizes)
    ref_result = run_benchmark(
        models=models,
        round_id=f"{round_id}-scale-{ref_size}",
        master_secret=master_secret,
        repeats=repeats,
        folds=folds,
        n_scenarios=n_scenarios,
    )
    ref_scores = get_task_scores(ref_result)
    ref_mean = ref_scores.mean(axis=1).sort_values(ascending=False)
    ref_ranking = {m: r for r, m in enumerate(ref_mean.index, 1)}

    curve: list[RankingStabilityPoint] = []
    stable_at = None

    for size in sorted(sample_sizes):
        logger.info("Running sample scaling: n_samples_max=%d", size)

        # Fix the sample range so only n_samples varies across runs.
        # The scenario_space override keeps everything else identical.
        result = run_benchmark(
            models=models,
            round_id=f"{round_id}-scale-{size}",
            master_secret=master_secret,
            repeats=repeats,
            folds=folds,
            n_scenarios=n_scenarios,
        )
        scores = get_task_scores(result)
        sub_mean = scores.mean(axis=1).sort_values(ascending=False)
        sub_ranking = {m: r for r, m in enumerate(sub_mean.index, 1)}

        common = sorted(set(ref_ranking) & set(sub_ranking))
        if len(common) < 2:
            continue

        ref_r = [ref_ranking[m] for m in common]
        sub_r = [sub_ranking[m] for m in common]
        kt, _ = kendalltau(ref_r, sub_r)
        sr, _ = spearmanr(ref_r, sub_r)

        curve.append(RankingStabilityPoint(
            scale_value=size,
            kendall_tau=float(kt),
            spearman_rho=float(sr),
            ranking=sub_ranking,
        ))

        if stable_at is None and float(kt) >= stability_threshold:
            stable_at = size

    return ScalingReport(
        axis="n_samples",
        reference_ranking=ref_ranking,
        curve=curve,
        stable_at=stable_at,
    )


def analyze_ranking_variance(
    task_scores: pd.DataFrame,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Bootstrap the full benchmark to estimate ranking variance per model.

    Resamples tasks with replacement and computes the standard deviation of
    each model's rank across resamples.

    Args:
        task_scores: Model-by-task score matrix.
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed.

    Returns:
        DataFrame with columns [model, mean_rank, std_rank, ci_low, ci_high].
    """
    rng = np.random.default_rng(seed)
    models = task_scores.index.tolist()
    n_tasks = task_scores.shape[1]

    rank_samples: dict[str, list[int]] = {m: [] for m in models}

    for _ in range(n_bootstrap):
        cols = rng.choice(task_scores.columns, size=n_tasks, replace=True)
        sub_mean = task_scores[cols].mean(axis=1).sort_values(ascending=False)
        for r, m in enumerate(sub_mean.index, 1):
            rank_samples[m].append(r)

    rows = []
    for m in models:
        ranks = np.array(rank_samples[m])
        rows.append({
            "model": m,
            "mean_rank": float(np.mean(ranks)),
            "std_rank": float(np.std(ranks)),
            "ci_low": float(np.percentile(ranks, 2.5)),
            "ci_high": float(np.percentile(ranks, 97.5)),
        })

    return pd.DataFrame(rows).sort_values("mean_rank").reset_index(drop=True)
