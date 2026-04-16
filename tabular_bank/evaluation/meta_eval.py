"""Meta-evaluation: diagnostics for benchmark quality.

Answers the question "is this benchmark any good?" via five metrics:

1. Discriminability  — can each task separate strong from weak models?
2. Ranking concordance — do our rankings agree with a reference benchmark?
3. Task diversity     — are the tasks redundant or complementary?
4. Per-task IRT       — principled difficulty/discrimination estimates.
5. Coverage profile   — distribution over problem types and difficulty.

All functions accept the same input: a model-by-task score matrix produced
by ``get_task_scores()`` from the leaderboard module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from tabular_bank.leaderboard import get_task_scores

logger = logging.getLogger(__name__)


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
class IRTItem:
    """IRT parameters for a single benchmark task."""

    task: str
    difficulty: float  # b — higher = harder
    discrimination: float  # a — higher = better at separating models


@dataclass
class IRTResult:
    """Item Response Theory analysis for the benchmark."""

    items: list[IRTItem]
    model_abilities: dict[str, float]  # theta per model
    converged: bool = True

    @property
    def difficulty_range(self) -> tuple[float, float]:
        diffs = [item.difficulty for item in self.items]
        return (min(diffs), max(diffs))

    @property
    def mean_discrimination(self) -> float:
        return float(np.mean([item.discrimination for item in self.items]))


@dataclass
class MetaEvalReport:
    """Complete meta-evaluation report for a benchmark round."""

    discriminability: DiscriminabilityResult
    diversity: DiversityResult
    concordance: ConcordanceResult | None = None
    irt: IRTResult | None = None

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

        if self.irt is not None:
            lo, hi = self.irt.difficulty_range
            lines += [
                "",
                f"IRT Analysis ({len(self.irt.items)} tasks, "
                f"{len(self.irt.model_abilities)} models):",
                f"  Difficulty range:       [{lo:.2f}, {hi:.2f}]",
                f"  Mean discrimination:    {self.irt.mean_discrimination:.3f}",
                f"  Converged:              {self.irt.converged}",
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
        if len(common_idx) < 3:
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
# 4. Item Response Theory (2PL)
# ---------------------------------------------------------------------------

def compute_irt(
    task_scores: pd.DataFrame,
    min_models: int = 4,
) -> IRTResult | None:
    """Fit a continuous IRT model to the benchmark score matrix.

    Uses a Gaussian IRT model (normal-ogive variant) that works directly
    with continuous scores rather than crude binarisation.  Each task has:
    - **difficulty** (b): tasks with lower mean performance are harder
    - **discrimination** (a): tasks that better separate strong from weak
      models have higher discrimination

    Each model has an **ability** (theta) parameter.  The model is:

        score_{ij} ~ Normal(a_j * (theta_i - b_j), sigma_j^2)

    This preserves the full information in continuous scores, unlike the
    binary 2PL model which loses ranking information within each half.

    Args:
        task_scores: Model-by-task score matrix.
        min_models: Minimum number of models required to fit IRT.

    Returns:
        IRTResult, or None if there are too few models.
    """
    models = task_scores.index.tolist()
    tasks = task_scores.columns.tolist()
    n_models = len(models)
    n_tasks = len(tasks)

    if n_models < min_models:
        logger.info(
            "IRT requires at least %d models, got %d — skipping.",
            min_models, n_models,
        )
        return None

    # Standardise scores per task so that difficulty and discrimination
    # are on comparable scales across metrics (ROC-AUC vs neg-RMSE).
    Z = task_scores.copy()
    for task in tasks:
        col = Z[task].dropna()
        if col.std() > 0:
            Z[task] = (Z[task] - col.mean()) / col.std()
        else:
            Z[task] = 0.0

    scores = Z.values  # (n_models, n_tasks)
    mask = ~np.isnan(scores)  # valid entries

    # Fit Gaussian IRT via alternating least squares:
    # score_{ij} ≈ a_j * theta_i + c_j
    # where difficulty b_j = -c_j / a_j
    #
    # This is equivalent to a rank-1 matrix factorization on the
    # standardised score matrix, which can be solved stably.

    # Initialize theta from row means (model strength proxy)
    theta = np.nanmean(scores, axis=1)
    theta_std = np.std(theta)
    if theta_std > 0:
        theta = (theta - np.mean(theta)) / theta_std

    # Alternating optimization with convergence checking
    a = np.ones(n_tasks)
    c = np.zeros(n_tasks)
    max_iterations = 15
    convergence_tol = 1e-6
    converged = False

    prev_residual = float("inf")
    for iteration in range(max_iterations):
        # Fix theta, solve for a_j, c_j per task (linear regression)
        for j in range(n_tasks):
            valid = mask[:, j]
            if valid.sum() < 2:
                continue
            y = scores[valid, j]
            X = np.column_stack([theta[valid], np.ones(valid.sum())])
            # Least squares: [a_j, c_j] = (X'X)^-1 X'y
            try:
                params, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                a[j] = max(params[0], 0.01)  # discrimination must be positive
                c[j] = params[1]
            except np.linalg.LinAlgError:
                pass

        # Fix a, c, solve for theta_i per model
        for i in range(n_models):
            valid = mask[i, :]
            if valid.sum() < 1:
                continue
            y = scores[i, valid]
            a_valid = a[valid]
            c_valid = c[valid]
            # theta_i = sum(a_j * (y_j - c_j)) / sum(a_j^2)
            denom = np.sum(a_valid ** 2)
            if denom > 0:
                theta[i] = np.sum(a_valid * (y - c_valid)) / denom

        # Re-centre theta for identifiability
        theta -= np.mean(theta)

        # Check convergence via reconstruction error
        predicted = np.outer(theta, a) + c[np.newaxis, :]
        residuals = np.where(mask, (scores - predicted) ** 2, 0.0)
        residual = float(np.sum(residuals))
        if abs(prev_residual - residual) < convergence_tol:
            converged = True
            break
        prev_residual = residual

    if not converged:
        logger.warning(
            "IRT did not converge after %d iterations (residual=%.6f).",
            max_iterations, prev_residual,
        )

    # Convert to IRT parameters: difficulty b_j = -c_j / a_j
    items = []
    for j in range(n_tasks):
        b_j = -c[j] / a[j] if a[j] > 0.01 else 0.0
        items.append(IRTItem(
            task=tasks[j],
            difficulty=float(b_j),
            discrimination=float(a[j]),
        ))

    abilities = {models[i]: float(theta[i]) for i in range(n_models)}

    return IRTResult(items=items, model_abilities=abilities, converged=converged)


# ---------------------------------------------------------------------------
# Full meta-eval pipeline
# ---------------------------------------------------------------------------

def run_meta_eval(
    result,
    reference_ranking: dict[str, int] | None = None,
    discriminability_threshold: float = 0.2,
    redundancy_threshold: float = 0.9,
    fit_irt: bool = True,
    irt_min_models: int = 4,
) -> MetaEvalReport:
    """Run all meta-evaluation diagnostics on a BenchmarkResult.

    Args:
        result: A ``BenchmarkResult`` from the runner.
        reference_ranking: Optional reference ranking for concordance
            (e.g. TabArena model rankings as ``{model_name: rank}``).
        discriminability_threshold: DS below this flags a task.
        redundancy_threshold: Correlation above this flags a task pair.
        fit_irt: Whether to fit a 2PL IRT model.
        irt_min_models: Minimum models required for IRT.
    """
    scores = get_task_scores(result)

    disc = compute_discriminability(scores, discriminability_threshold)
    div = compute_task_diversity(scores, redundancy_threshold)

    conc = None
    if reference_ranking is not None:
        conc = compute_ranking_concordance(scores, reference_ranking)

    irt = None
    if fit_irt and not scores.empty:
        irt = compute_irt(scores, min_models=irt_min_models)

    return MetaEvalReport(
        discriminability=disc,
        diversity=div,
        concordance=conc,
        irt=irt,
    )
