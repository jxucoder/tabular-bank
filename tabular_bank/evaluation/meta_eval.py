"""Meta-evaluation: diagnostics for benchmark quality.

Answers the question "is this benchmark any good?" via four metrics:

1. Discriminability  — can each task separate strong from weak models?
2. Ranking concordance — do our rankings agree with a reference benchmark?
3. Task diversity     — are the tasks redundant or complementary?
4. Per-task IRT       — principled difficulty/discrimination (stretch).

All functions accept the same input: a model-by-task score matrix produced
by ``get_task_scores()`` from the leaderboard module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from tabular_bank.leaderboard import get_task_scores


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

@dataclass
class DiscriminabilityResult:
    """Per-task and aggregate discriminability scores."""

    per_task: dict[str, float]
    overall: float
    flagged_tasks: list[str] = field(default_factory=list)


@dataclass
class ConcordanceResult:
    """Ranking concordance against a reference leaderboard."""

    kendall_tau: float
    kendall_p: float
    spearman_rho: float
    spearman_p: float
    n_common_models: int
    our_ranking: dict[str, int]
    ref_ranking: dict[str, int]


@dataclass
class DiversityResult:
    """Inter-task score correlation matrix and summary statistics."""

    correlation_matrix: pd.DataFrame
    mean_correlation: float
    max_correlation: float
    redundant_pairs: list[tuple[str, str, float]]


@dataclass
class MetaEvalReport:
    """Complete meta-evaluation report for a benchmark round."""

    discriminability: DiscriminabilityResult
    diversity: DiversityResult
    concordance: ConcordanceResult | None = None

    def summary(self) -> str:
        lines = [
            "=== Meta-Evaluation Report ===",
            "",
            f"Discriminability (overall): {self.discriminability.overall:.4f}",
        ]
        if self.discriminability.flagged_tasks:
            lines.append(
                f"  Flagged tasks (low discriminability): "
                f"{', '.join(self.discriminability.flagged_tasks)}"
            )

        lines += [
            "",
            f"Task Diversity:",
            f"  Mean inter-task correlation: {self.diversity.mean_correlation:.3f}",
            f"  Max  inter-task correlation: {self.diversity.max_correlation:.3f}",
        ]
        if self.diversity.redundant_pairs:
            lines.append("  Redundant pairs (rho > 0.9):")
            for t1, t2, rho in self.diversity.redundant_pairs:
                lines.append(f"    {t1} <-> {t2}: {rho:.3f}")

        if self.concordance is not None:
            lines += [
                "",
                f"Ranking Concordance ({self.concordance.n_common_models} common models):",
                f"  Kendall tau:  {self.concordance.kendall_tau:.3f} "
                f"(p={self.concordance.kendall_p:.4f})",
                f"  Spearman rho: {self.concordance.spearman_rho:.3f} "
                f"(p={self.concordance.spearman_p:.4f})",
            ]

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. Discriminability
# ---------------------------------------------------------------------------

def compute_discriminability(
    task_scores: pd.DataFrame,
    low_threshold: float = 0.2,
) -> DiscriminabilityResult:
    """Measure how well each task separates models.

    For each task, compute a scale-normalised discriminability score (DS):
    the mean absolute pairwise score difference divided by the pooled
    standard deviation of scores (analogous to Cohen's d).  This makes DS
    comparable across metrics with different scales (e.g. ROC-AUC in [0,1]
    vs negative RMSE in [-100, 0]).

    Tasks where all models score nearly identically (DS < ``low_threshold``)
    are flagged.

    Args:
        task_scores: DataFrame with models as rows, tasks as columns.
        low_threshold: DS below this flags a task as non-discriminative.
            The default (0.2) corresponds roughly to a "small" effect size.
    """
    per_task: dict[str, float] = {}

    for task in task_scores.columns:
        scores = task_scores[task].dropna().values
        if len(scores) < 2:
            per_task[task] = 0.0
            continue
        diffs = [
            abs(scores[i] - scores[j])
            for i, j in combinations(range(len(scores)), 2)
        ]
        mean_diff = float(np.mean(diffs))
        std = float(np.std(scores, ddof=1))
        # Normalise by pooled std; fall back to raw diff if std is zero
        per_task[task] = mean_diff / std if std > 0 else 0.0

    overall = float(np.mean(list(per_task.values()))) if per_task else 0.0
    flagged = [t for t, ds in per_task.items() if ds < low_threshold]

    return DiscriminabilityResult(
        per_task=per_task,
        overall=overall,
        flagged_tasks=flagged,
    )


# ---------------------------------------------------------------------------
# 2. Ranking Concordance
# ---------------------------------------------------------------------------

def compute_ranking_concordance(
    task_scores: pd.DataFrame,
    reference_ranking: dict[str, int],
) -> ConcordanceResult:
    """Compare our model ranking against a reference (e.g. TabArena).

    Models are ranked by their mean score across tasks (higher = better).
    Only models present in *both* our results and the reference are compared.

    Args:
        task_scores: DataFrame with models as rows, tasks as columns.
        reference_ranking: Dict of model_name -> rank (1 = best) from the
            reference benchmark.
    """
    our_mean = task_scores.mean(axis=1).sort_values(ascending=False)
    our_ranking = {m: rank for rank, m in enumerate(our_mean.index, 1)}

    common = sorted(set(our_ranking) & set(reference_ranking))
    if len(common) < 2:
        return ConcordanceResult(
            kendall_tau=float("nan"),
            kendall_p=float("nan"),
            spearman_rho=float("nan"),
            spearman_p=float("nan"),
            n_common_models=len(common),
            our_ranking=our_ranking,
            ref_ranking=reference_ranking,
        )

    our_ranks = [our_ranking[m] for m in common]
    ref_ranks = [reference_ranking[m] for m in common]

    kt, kt_p = kendalltau(our_ranks, ref_ranks)
    sr, sr_p = spearmanr(our_ranks, ref_ranks)

    return ConcordanceResult(
        kendall_tau=float(kt),
        kendall_p=float(kt_p),
        spearman_rho=float(sr),
        spearman_p=float(sr_p),
        n_common_models=len(common),
        our_ranking=our_ranking,
        ref_ranking=reference_ranking,
    )


# ---------------------------------------------------------------------------
# 3. Task Diversity
# ---------------------------------------------------------------------------

def compute_task_diversity(
    task_scores: pd.DataFrame,
    redundancy_threshold: float = 0.9,
) -> DiversityResult:
    """Measure inter-task score correlations.

    For each pair of tasks, compute Spearman correlation of model scores.
    High correlation means the tasks rank models the same way (redundant).

    Args:
        task_scores: DataFrame with models as rows, tasks as columns.
        redundancy_threshold: Pairs with abs(rho) above this are flagged.
    """
    tasks = task_scores.columns.tolist()
    n = len(tasks)
    corr = pd.DataFrame(np.eye(n), index=tasks, columns=tasks)

    off_diag: list[float] = []
    redundant: list[tuple[str, str, float]] = []

    for i, j in combinations(range(n), 2):
        t1, t2 = tasks[i], tasks[j]
        vals1 = task_scores[t1].dropna()
        vals2 = task_scores[t2].dropna()
        common_idx = vals1.index.intersection(vals2.index)
        if len(common_idx) < 5:
            rho = float("nan")
        else:
            rho, _ = spearmanr(vals1[common_idx], vals2[common_idx])
            rho = float(rho)
        corr.loc[t1, t2] = rho
        corr.loc[t2, t1] = rho
        if not np.isnan(rho):
            off_diag.append(abs(rho))
            if abs(rho) > redundancy_threshold:
                redundant.append((t1, t2, rho))

    return DiversityResult(
        correlation_matrix=corr,
        mean_correlation=float(np.mean(off_diag)) if off_diag else 0.0,
        max_correlation=float(np.max(off_diag)) if off_diag else 0.0,
        redundant_pairs=redundant,
    )


# ---------------------------------------------------------------------------
# Full meta-eval pipeline
# ---------------------------------------------------------------------------

def run_meta_eval(
    result,
    reference_ranking: dict[str, int] | None = None,
    discriminability_threshold: float = 0.2,
    redundancy_threshold: float = 0.9,
) -> MetaEvalReport:
    """Run all meta-evaluation diagnostics on a BenchmarkResult.

    Args:
        result: A ``BenchmarkResult`` from the runner.
        reference_ranking: Optional reference ranking for concordance
            (e.g. TabArena model rankings as ``{model_name: rank}``).
        discriminability_threshold: DS below this flags a task.
        redundancy_threshold: Correlation above this flags a task pair.
    """
    scores = get_task_scores(result)

    disc = compute_discriminability(scores, discriminability_threshold)
    div = compute_task_diversity(scores, redundancy_threshold)

    conc = None
    if reference_ranking is not None:
        conc = compute_ranking_concordance(scores, reference_ranking)

    return MetaEvalReport(
        discriminability=disc,
        diversity=div,
        concordance=conc,
    )
