"""Leaderboard generation from benchmark results.

Computes ELO ratings, win rates, and average ranks from pairwise comparisons
across all tasks and splits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tabular_bank.runner import BenchmarkResult


def get_task_scores(result: BenchmarkResult) -> pd.DataFrame:
    """Extract model-by-task mean score matrix.

    Returns a DataFrame with models as rows and tasks as columns, where
    each cell is the mean score across all repeats/folds for that pair.
    This is the canonical input format for meta-evaluation diagnostics.
    """
    df = result.to_dataframe()
    if df.empty:
        return pd.DataFrame()
    return (
        df.groupby(["model", "task"])["score"]
        .mean()
        .reset_index()
        .pivot(index="model", columns="task", values="score")
    )


def generate_leaderboard(result: BenchmarkResult) -> pd.DataFrame:
    """Generate a leaderboard from benchmark results.

    Computes:
    - Mean score per task (averaged over repeats/folds)
    - Average rank across tasks
    - Win rate (fraction of tasks where model is best)
    - ELO rating from pairwise comparisons

    Returns a DataFrame sorted by ELO rating.
    """
    task_scores = get_task_scores(result)
    if task_scores.empty:
        return pd.DataFrame()

    models = task_scores.index.tolist()
    tasks = task_scores.columns.tolist()

    # Average rank
    ranks = task_scores.rank(axis=0, ascending=False)
    avg_rank = ranks.mean(axis=1)

    # Win rate
    wins = (ranks == 1).sum(axis=1)
    win_rate = wins / len(tasks)

    # Mean score across all tasks
    mean_score = task_scores.mean(axis=1)

    # ELO ratings from pairwise comparisons
    elo = _compute_elo(task_scores)

    leaderboard = pd.DataFrame({
        "model": models,
        "elo": [elo[m] for m in models],
        "avg_rank": avg_rank.values,
        "win_rate": win_rate.values,
        "mean_score": mean_score.values,
        "n_tasks": len(tasks),
    }).sort_values("elo", ascending=False).reset_index(drop=True)

    leaderboard.index = leaderboard.index + 1  # 1-indexed rank
    leaderboard.index.name = "rank"

    return leaderboard


def _compute_elo(
    task_scores: pd.DataFrame,
    k: float = 32.0,
    initial_elo: float = 1500.0,
    n_iterations: int = 100,
) -> dict[str, float]:
    """Compute ELO ratings from pairwise task comparisons.

    For each task, every pair of models is compared. The winner gets ELO
    points transferred from the loser. We iterate multiple times for
    convergence.
    """
    models = task_scores.index.tolist()
    tasks = task_scores.columns.tolist()

    elo = {m: initial_elo for m in models}

    for _ in range(n_iterations):
        for task in tasks:
            scores = task_scores[task]
            for i, m1 in enumerate(models):
                for m2 in models[i + 1:]:
                    s1, s2 = scores[m1], scores[m2]
                    if pd.isna(s1) or pd.isna(s2):
                        continue

                    # Expected scores
                    e1 = 1.0 / (1.0 + 10 ** ((elo[m2] - elo[m1]) / 400.0))
                    e2 = 1.0 - e1

                    # Actual scores
                    if s1 > s2:
                        a1, a2 = 1.0, 0.0
                    elif s2 > s1:
                        a1, a2 = 0.0, 1.0
                    else:
                        a1, a2 = 0.5, 0.5

                    # Update
                    update = k * (a1 - e1) / len(tasks)
                    elo[m1] += update
                    elo[m2] -= update

    return elo


def format_leaderboard(leaderboard: pd.DataFrame) -> str:
    """Format leaderboard as a human-readable string."""
    if leaderboard.empty:
        return "No results to display."

    lines = []
    lines.append(f"{'Rank':<6} {'Model':<30} {'ELO':>8} {'Avg Rank':>10} {'Win Rate':>10} {'Mean Score':>12}")
    lines.append("-" * 80)

    for rank, row in leaderboard.iterrows():
        lines.append(
            f"{rank:<6} {row['model']:<30} {row['elo']:>8.1f} {row['avg_rank']:>10.2f} "
            f"{row['win_rate']:>10.1%} {row['mean_score']:>12.4f}"
        )

    return "\n".join(lines)


def generate_leaderboard_tabarena(
    round_id: str = "round-001",
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Generate leaderboard using TabArena's EndToEnd pipeline.

    Requires TabArena to be installed.
    """
    try:
        from tabarena.evaluation import TabArenaEvaluator
    except ImportError:
        raise ImportError(
            "TabArena is required for this operation. "
            "Install with: pip install tabular-bank[benchmark]"
        )

    # This would use TabArena's full evaluation pipeline
    # Left as a hook for when TabArena integration is fully wired up
    raise NotImplementedError(
        "Full TabArena leaderboard integration coming soon. "
        "Use generate_leaderboard() with BenchmarkResult for now."
    )
