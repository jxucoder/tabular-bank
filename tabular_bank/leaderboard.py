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
    initial_elo: float = 1500.0,
    n_iterations: int = 200,
    lr: float = 0.1,
) -> dict[str, float]:
    """Compute ELO-scale ratings via Bradley-Terry maximum likelihood.

    Instead of repeated ELO updates (which don't converge on static data),
    we use iterative MM (minorisation-maximisation) to find the MLE of the
    Bradley-Terry model.  The resulting strengths are then rescaled to the
    familiar ELO scale (mean 1500, 400-point logistic base).

    For each task, every pair of models is compared to produce win/loss/tie
    counts.  Ties contribute 0.5 wins to each side.
    """
    models = task_scores.index.tolist()
    tasks = task_scores.columns.tolist()
    n = len(models)
    idx = {m: i for i, m in enumerate(models)}

    # Accumulate pairwise win counts: wins[i, j] = #tasks where i beat j
    wins = np.zeros((n, n))
    for task in tasks:
        scores = task_scores[task]
        for i_idx, m1 in enumerate(models):
            for m2 in models[i_idx + 1:]:
                s1, s2 = scores[m1], scores[m2]
                if pd.isna(s1) or pd.isna(s2):
                    continue
                i, j = idx[m1], idx[m2]
                if s1 > s2:
                    wins[i, j] += 1.0
                elif s2 > s1:
                    wins[j, i] += 1.0
                else:
                    wins[i, j] += 0.5
                    wins[j, i] += 0.5

    # Total games between each pair
    games = wins + wins.T

    # Bradley-Terry MM iterations (Hunter 2004).
    # Models with zero wins get a small floor strength (epsilon) so the
    # ranking still reflects their losses correctly.
    eps = 1e-6
    strength = np.ones(n)
    for _ in range(n_iterations):
        new_strength = np.full(n, eps)
        for i in range(n):
            w_i = wins[i].sum()
            if w_i == 0:
                # No wins — keep at floor
                continue
            denom = 0.0
            for j in range(n):
                if i == j or games[i, j] == 0:
                    continue
                denom += games[i, j] / (strength[i] + strength[j])
            if denom > 0:
                new_strength[i] = w_i / denom
        # Normalise so geometric mean = 1
        log_mean = np.mean(np.log(np.maximum(new_strength, eps)))
        new_strength /= np.exp(log_mean)
        strength = new_strength

    # Convert to ELO scale: ELO = 1500 + 400 * log10(strength)
    elo_values = initial_elo + 400.0 * np.log10(np.maximum(strength, 1e-12))
    return {m: float(elo_values[idx[m]]) for m in models}


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
