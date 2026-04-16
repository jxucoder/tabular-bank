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
from scipy.optimize import minimize
from scipy.special import expit
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
    """Fit a 2-parameter IRT model to the benchmark score matrix.

    Uses a 2-parameter logistic (2PL) model where each task has a
    difficulty (b) and discrimination (a) parameter, and each model has
    an ability (theta) parameter.  Scores are binarised per task: a model
    "passes" a task if its score exceeds the task median.

    The model is: P(pass | theta, a, b) = sigmoid(a * (theta - b))

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

    # Binarise: 1 if model score > task median, 0 otherwise
    responses = np.zeros((n_models, n_tasks))
    for j, task in enumerate(tasks):
        col = task_scores[task].dropna()
        if col.empty:
            continue
        median = col.median()
        for i, model in enumerate(models):
            val = task_scores.loc[model, task]
            if pd.notna(val):
                responses[i, j] = 1.0 if val > median else 0.0

    # Fit 2PL via joint MLE
    # Parameters: theta (n_models) + a, b (2 * n_tasks)
    n_params = n_models + 2 * n_tasks

    def neg_log_likelihood(params):
        theta = params[:n_models]
        a = params[n_models:n_models + n_tasks]
        b = params[n_models + n_tasks:]

        nll = 0.0
        for i in range(n_models):
            for j in range(n_tasks):
                logit = a[j] * (theta[i] - b[j])
                p = expit(logit)
                p = np.clip(p, 1e-10, 1 - 1e-10)
                if responses[i, j] == 1:
                    nll -= np.log(p)
                else:
                    nll -= np.log(1 - p)
        return nll

    # Initial values
    x0 = np.zeros(n_params)
    x0[:n_models] = 0.0  # abilities
    x0[n_models:n_models + n_tasks] = 1.0  # discriminations
    x0[n_models + n_tasks:] = 0.0  # difficulties

    result = minimize(
        neg_log_likelihood, x0,
        method="L-BFGS-B",
        bounds=(
            [(None, None)] * n_models  # theta unbounded
            + [(0.1, 5.0)] * n_tasks  # a > 0
            + [(None, None)] * n_tasks  # b unbounded
        ),
        options={"maxiter": 500, "ftol": 1e-8},
    )

    converged = result.success
    theta = result.x[:n_models]
    a = result.x[n_models:n_models + n_tasks]
    b = result.x[n_models + n_tasks:]

    items = [
        IRTItem(task=tasks[j], difficulty=float(b[j]), discrimination=float(a[j]))
        for j in range(n_tasks)
    ]
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
