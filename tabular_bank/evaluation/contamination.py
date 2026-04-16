"""Contamination analysis — detect memorization in tabular benchmarks.

Compares model performance on tabular-bank (contamination-proof) vs. a
reference benchmark (potentially contaminated) to quantify memorization
effects.  The core idea: models that memorize training data (e.g. LLMs)
will perform relatively *worse* on procedurally generated data than on
benchmarks whose datasets may have appeared in pretraining corpora.

Key metrics:

- **Contamination gap**: Per-model difference between reference-benchmark
  rank and tabular-bank rank.  Models with large positive gaps (better on
  the reference than on tabular-bank) may benefit from memorization.
- **Relative performance drop**: Normalised score difference between the
  two benchmarks for each model.
- **Memorization susceptibility index (MSI)**: Aggregate statistic that
  compares the rank displacement of memorization-prone models (foundation
  models, LLMs) against classical models.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


@dataclass
class ContaminationGap:
    """Per-model contamination gap between two benchmarks."""

    model: str
    ref_rank: int
    tbank_rank: int
    rank_gap: int  # tbank_rank - ref_rank (positive = better on reference)
    ref_mean_score: float
    tbank_mean_score: float
    relative_drop: float  # (ref - tbank) / |ref| if ref != 0


@dataclass
class ContaminationReport:
    """Full contamination analysis report."""

    per_model: list[ContaminationGap]
    overall_kendall_tau: float
    overall_kendall_p: float
    overall_spearman_rho: float
    overall_spearman_p: float
    n_common_models: int
    msi: float | None = None  # memorization susceptibility index
    flagged_models: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=== Contamination Analysis Report ===",
            "",
            f"Common models compared: {self.n_common_models}",
            f"Ranking concordance (Kendall tau):  {self.overall_kendall_tau:.3f} "
            f"(p={self.overall_kendall_p:.4f})",
            f"Ranking concordance (Spearman rho): {self.overall_spearman_rho:.3f} "
            f"(p={self.overall_spearman_p:.4f})",
        ]
        if self.msi is not None:
            lines.append(f"Memorization Susceptibility Index:  {self.msi:.3f}")

        if self.flagged_models:
            lines += [
                "",
                "Flagged models (rank gap >= 3, possible memorization):",
            ]
            for m in self.flagged_models:
                lines.append(f"  - {m}")

        lines += ["", "Per-model breakdown:"]
        lines.append(
            f"  {'Model':<30s} {'Ref Rank':>9s} {'TBank Rank':>11s} "
            f"{'Gap':>5s} {'Rel. Drop':>10s}"
        )
        lines.append("  " + "-" * 68)
        for g in sorted(self.per_model, key=lambda x: -x.rank_gap):
            flag = " *" if g.model in self.flagged_models else ""
            lines.append(
                f"  {g.model:<30s} {g.ref_rank:>9d} {g.tbank_rank:>11d} "
                f"{g.rank_gap:>+5d} {g.relative_drop:>+10.3f}{flag}"
            )

        return "\n".join(lines)


def analyze_contamination(
    tbank_scores: pd.DataFrame,
    ref_scores: pd.DataFrame,
    memorization_prone: list[str] | None = None,
    gap_threshold: int = 3,
) -> ContaminationReport:
    """Compare model rankings between tabular-bank and a reference benchmark.

    Args:
        tbank_scores: Model-by-task score matrix from tabular-bank
            (models as rows, tasks as columns; higher = better).
        ref_scores: Model-by-task score matrix from reference benchmark
            (same convention).
        memorization_prone: Model names suspected of memorization
            (e.g. foundation models, LLMs).  Used for MSI computation.
        gap_threshold: Rank gap above which a model is flagged.

    Returns:
        ContaminationReport with per-model gaps and aggregate metrics.
    """
    # Compute mean scores and rankings
    tbank_mean = tbank_scores.mean(axis=1).sort_values(ascending=False)
    ref_mean = ref_scores.mean(axis=1).sort_values(ascending=False)

    tbank_ranking = {m: r for r, m in enumerate(tbank_mean.index, 1)}
    ref_ranking = {m: r for r, m in enumerate(ref_mean.index, 1)}

    common = sorted(set(tbank_ranking) & set(ref_ranking))
    if len(common) < 2:
        return ContaminationReport(
            per_model=[],
            overall_kendall_tau=float("nan"),
            overall_kendall_p=float("nan"),
            overall_spearman_rho=float("nan"),
            overall_spearman_p=float("nan"),
            n_common_models=len(common),
        )

    # Per-model gaps
    # rank_gap = tbank_rank - ref_rank: positive means the model ranks
    # better on the reference than on tabular-bank (possible memorization).
    gaps: list[ContaminationGap] = []
    for m in common:
        ref_score = float(ref_mean[m])
        tbank_score = float(tbank_mean[m])
        denom = abs(ref_score) if abs(ref_score) > 1e-12 else 1.0
        gaps.append(ContaminationGap(
            model=m,
            ref_rank=ref_ranking[m],
            tbank_rank=tbank_ranking[m],
            rank_gap=tbank_ranking[m] - ref_ranking[m],
            ref_mean_score=ref_score,
            tbank_mean_score=tbank_score,
            relative_drop=(ref_score - tbank_score) / denom,
        ))

    # Rank correlation
    tbank_ranks = [tbank_ranking[m] for m in common]
    ref_ranks = [ref_ranking[m] for m in common]
    kt, kt_p = kendalltau(tbank_ranks, ref_ranks)
    sr, sr_p = spearmanr(tbank_ranks, ref_ranks)

    # Flag models with large rank gaps
    flagged = [g.model for g in gaps if g.rank_gap >= gap_threshold]

    # Memorization Susceptibility Index
    msi = None
    if memorization_prone:
        prone_set = set(memorization_prone) & set(common)
        classical_set = set(common) - prone_set
        if prone_set and classical_set:
            prone_gaps = [g.rank_gap for g in gaps if g.model in prone_set]
            classical_gaps = [g.rank_gap for g in gaps if g.model in classical_set]
            msi = float(np.mean(prone_gaps) - np.mean(classical_gaps))

    return ContaminationReport(
        per_model=gaps,
        overall_kendall_tau=float(kt),
        overall_kendall_p=float(kt_p),
        overall_spearman_rho=float(sr),
        overall_spearman_p=float(sr_p),
        n_common_models=len(common),
        msi=msi,
        flagged_models=flagged,
    )


def run_contamination_benchmark(
    models: dict[str, object],
    round_id: str = "round-001",
    master_secret: str | None = None,
    reference_results: pd.DataFrame | None = None,
    memorization_prone: list[str] | None = None,
    n_scenarios: int = 10,
    repeats: list[int] | None = None,
    folds: list[int] | None = None,
    gap_threshold: int = 3,
) -> ContaminationReport:
    """Run a full contamination benchmark: evaluate on tabular-bank, then compare.

    This is the high-level entry point. It:
    1. Runs all models on tabular-bank tasks
    2. Compares the resulting rankings against reference results
    3. Returns a contamination report

    Args:
        models: Dict of model_name -> sklearn-compatible model.
        round_id: Benchmark round identifier.
        master_secret: Secret for dataset generation.
        reference_results: DataFrame with columns [model, task, score]
            from the reference benchmark.  If None, only tabular-bank
            results are returned (no comparison).
        memorization_prone: Model names suspected of memorization.
        n_scenarios: Number of scenarios in the round.
        repeats: Which repeats to run.
        folds: Which folds to run.
        gap_threshold: Rank gap above which a model is flagged.
    """
    from tabular_bank.leaderboard import get_task_scores
    from tabular_bank.runner import run_benchmark

    result = run_benchmark(
        models=models,
        round_id=round_id,
        master_secret=master_secret,
        repeats=repeats,
        folds=folds,
        n_scenarios=n_scenarios,
    )

    tbank_scores = get_task_scores(result)

    if reference_results is None:
        # No reference — return a degenerate report
        return ContaminationReport(
            per_model=[],
            overall_kendall_tau=float("nan"),
            overall_kendall_p=float("nan"),
            overall_spearman_rho=float("nan"),
            overall_spearman_p=float("nan"),
            n_common_models=0,
        )

    # Build reference score matrix
    ref_scores = (
        reference_results
        .groupby(["model", "task"])["score"]
        .mean()
        .reset_index()
        .pivot(index="model", columns="task", values="score")
    )

    return analyze_contamination(
        tbank_scores=tbank_scores,
        ref_scores=ref_scores,
        memorization_prone=memorization_prone,
        gap_threshold=gap_threshold,
    )
