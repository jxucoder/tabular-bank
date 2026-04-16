"""Dimension-aware performance diagnostics.

Instead of just "Model X is rank 3", answer questions like:
- "Model X degrades specifically on high-nonlinearity tasks"
- "Tree models dominate when confounders are present"
- "Neural networks improve relative to classical models as sample size grows"

This transforms the benchmark from a leaderboard into a *diagnostic tool*
that reveals model strengths and weaknesses along specific data axes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass
class DimensionProfile:
    """Performance profile of a single model along one difficulty dimension."""

    model: str
    dimension: str
    # Each entry: (dimension_value, mean_score)
    curve: list[tuple[float, float]]
    correlation: float  # Spearman correlation between dimension value and score
    correlation_p: float


@dataclass
class DiagnosticReport:
    """Full diagnostic breakdown across models and difficulty dimensions."""

    profiles: list[DimensionProfile]
    dimension_importance: dict[str, float]  # dimension -> variance explained

    def get_model_strengths(self, model: str) -> list[tuple[str, float]]:
        """Dimensions where this model's correlation is most positive."""
        model_profiles = [p for p in self.profiles if p.model == model]
        return sorted(
            [(p.dimension, p.correlation) for p in model_profiles],
            key=lambda x: -x[1],
        )

    def get_model_weaknesses(self, model: str) -> list[tuple[str, float]]:
        """Dimensions where this model's correlation is most negative."""
        model_profiles = [p for p in self.profiles if p.model == model]
        return sorted(
            [(p.dimension, p.correlation) for p in model_profiles],
            key=lambda x: x[1],
        )

    def summary(self) -> str:
        lines = [
            "=== Dimension-Aware Diagnostics ===",
            "",
        ]

        if self.dimension_importance:
            lines.append("Dimension importance (variance explained in rankings):")
            for dim, imp in sorted(self.dimension_importance.items(), key=lambda x: -x[1]):
                lines.append(f"  {dim:<35s} {imp:.3f}")
            lines.append("")

        models = sorted(set(p.model for p in self.profiles))
        dims = sorted(set(p.dimension for p in self.profiles))

        # Build a model x dimension correlation matrix
        lines.append("Model performance correlations with difficulty dimensions:")
        lines.append(f"  {'Model':<25s} " + " ".join(f"{d[:12]:>13s}" for d in dims))
        lines.append("  " + "-" * (25 + 14 * len(dims)))

        for model in models:
            model_profiles = {p.dimension: p for p in self.profiles if p.model == model}
            vals = []
            for d in dims:
                p = model_profiles.get(d)
                if p is not None:
                    sig = "*" if p.correlation_p < 0.05 else " "
                    vals.append(f"{p.correlation:>+12.3f}{sig}")
                else:
                    vals.append(f"{'N/A':>13s}")
            lines.append(f"  {model:<25s} " + " ".join(vals))

        lines.append("")
        lines.append("  * = p < 0.05")

        return "\n".join(lines)


# Difficulty dimensions extractable from task metadata
DIFFICULTY_DIMENSIONS = [
    "noise_scale",
    "nonlinear_prob",
    "interaction_prob",
    "heteroscedastic_prob",
    "edge_density",
    "max_parents",
    "n_confounders",
    "confounder_strength",
    "temporal_prob",
    "root_correlation_strength",
]

# Non-difficulty structural dimensions
STRUCTURAL_DIMENSIONS = [
    "n_samples",
    "n_features",
    "n_informative_features",
    "n_noise_features",
    "n_categorical",
    "missing_rate",
]


def run_diagnostics(
    task_scores: pd.DataFrame,
    task_metadata: pd.DataFrame,
    dimensions: list[str] | None = None,
) -> DiagnosticReport:
    """Analyze model performance along task difficulty dimensions.

    Args:
        task_scores: Model-by-task score matrix (models as rows, tasks as cols).
        task_metadata: DataFrame with task-level metadata including difficulty
            parameters.  Must have a column that matches the task_scores columns
            (typically 'dataset' or the index).
        dimensions: Which difficulty dimensions to analyze.  Defaults to all
            available dimensions found in the metadata.
    """
    if dimensions is None:
        dimensions = _detect_dimensions(task_metadata)

    # Build a task -> dimension_values mapping
    task_dim_values = _extract_dimension_values(task_metadata, task_scores.columns.tolist())

    profiles: list[DimensionProfile] = []
    dim_importance: dict[str, float] = {}

    for dim in dimensions:
        dim_vals = task_dim_values.get(dim)
        if dim_vals is None or len(dim_vals) < 3:
            continue

        # For each model, compute correlation between dimension value and score
        models = task_scores.index.tolist()
        rank_variances = []

        for model in models:
            scores_for_model = []
            dim_for_model = []

            for task in task_scores.columns:
                if task in dim_vals and pd.notna(task_scores.loc[model, task]):
                    scores_for_model.append(float(task_scores.loc[model, task]))
                    dim_for_model.append(dim_vals[task])

            if len(scores_for_model) < 3:
                continue

            rho, p = spearmanr(dim_for_model, scores_for_model)

            # Build the curve
            pairs = sorted(zip(dim_for_model, scores_for_model))
            curve = [(float(d), float(s)) for d, s in pairs]

            profiles.append(DimensionProfile(
                model=model,
                dimension=dim,
                curve=curve,
                correlation=float(rho),
                correlation_p=float(p),
            ))

            rank_variances.append(abs(float(rho)))

        # Dimension importance: mean |correlation| across models
        if rank_variances:
            dim_importance[dim] = float(np.mean(rank_variances))

    return DiagnosticReport(
        profiles=profiles,
        dimension_importance=dim_importance,
    )


def compute_pareto_frontier(
    task_scores: pd.DataFrame,
    fit_times: pd.DataFrame,
) -> pd.DataFrame:
    """Compute the Pareto frontier of accuracy vs. computational cost.

    Args:
        task_scores: Model-by-task score matrix (higher = better).
        fit_times: Model-by-task fit time matrix (lower = better).

    Returns:
        DataFrame with columns [model, mean_score, mean_time, is_pareto].
    """
    models = task_scores.index.tolist()
    rows = []
    for model in models:
        mean_score = float(task_scores.loc[model].mean())
        mean_time = float(fit_times.loc[model].mean()) if model in fit_times.index else float("nan")
        rows.append({
            "model": model,
            "mean_score": mean_score,
            "mean_time": mean_time,
        })

    df = pd.DataFrame(rows)

    # Compute Pareto: a model is Pareto-optimal if no other model is both
    # more accurate and faster.
    df["is_pareto"] = False
    for i, row in df.iterrows():
        if np.isnan(row["mean_time"]):
            continue
        dominated = False
        for j, other in df.iterrows():
            if i == j or np.isnan(other["mean_time"]):
                continue
            if other["mean_score"] >= row["mean_score"] and other["mean_time"] <= row["mean_time"]:
                if other["mean_score"] > row["mean_score"] or other["mean_time"] < row["mean_time"]:
                    dominated = True
                    break
        if not dominated:
            df.at[i, "is_pareto"] = True

    return df.sort_values("mean_score", ascending=False).reset_index(drop=True)


def _detect_dimensions(task_metadata: pd.DataFrame) -> list[str]:
    """Detect which difficulty dimensions are available in the metadata."""
    available = []
    for dim in DIFFICULTY_DIMENSIONS + STRUCTURAL_DIMENSIONS:
        if dim in task_metadata.columns:
            available.append(dim)
    # Check nested difficulty dict
    if "difficulty" in task_metadata.columns:
        non_null = task_metadata["difficulty"].dropna()
        sample = non_null.iloc[0] if not non_null.empty else None
        if isinstance(sample, dict):
            for dim in DIFFICULTY_DIMENSIONS:
                if dim in sample and dim not in available:
                    available.append(dim)
    return available


def _extract_dimension_values(
    task_metadata: pd.DataFrame,
    task_names: list[str],
) -> dict[str, dict[str, float]]:
    """Extract dimension values per task from metadata."""
    # Build task name -> row mapping
    if "dataset" in task_metadata.columns:
        meta_by_task = task_metadata.set_index("dataset")
    elif "name" in task_metadata.columns:
        meta_by_task = task_metadata.set_index("name")
    else:
        meta_by_task = task_metadata

    result: dict[str, dict[str, float]] = {}

    for dim in DIFFICULTY_DIMENSIONS + STRUCTURAL_DIMENSIONS:
        dim_vals: dict[str, float] = {}

        for task_name in task_names:
            if task_name not in meta_by_task.index:
                continue
            row = meta_by_task.loc[task_name]

            # Try direct column first
            if dim in meta_by_task.columns:
                val = row[dim]
                if pd.notna(val):
                    try:
                        dim_vals[task_name] = float(val)
                    except (TypeError, ValueError):
                        pass
                    continue

            # Try nested difficulty dict
            if "difficulty" in meta_by_task.columns:
                diff = row["difficulty"]
                if isinstance(diff, dict) and dim in diff:
                    try:
                        dim_vals[task_name] = float(diff[dim])
                    except (TypeError, ValueError):
                        pass

        if dim_vals:
            result[dim] = dim_vals

    return result
