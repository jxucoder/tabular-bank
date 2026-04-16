"""Ground-truth feature importance from the causal DAG.

Unlike real-world benchmarks where true feature importance is unknown,
tabular-bank can extract *exact* causal importance from the procedurally
generated DAG.  This enables:

1. **Ground-truth importance ranking** — which features actually matter
   and how much, computed from DAG edge weights and structure.
2. **Importance estimation quality** — compare model-estimated importance
   (permutation, gain, SHAP) against the known ground truth.
3. **Feature importance fidelity** — a novel evaluation axis: does the
   model not only predict well, but also "understand" the data structure?

This is a unique capability of procedural benchmarks that cannot be
replicated on real-world datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from tabular_bank.generation.dag_builder import DAGSpec


@dataclass
class FeatureImportanceProfile:
    """Ground-truth feature importance for a single task."""

    task_name: str
    target: str
    # feature_name -> importance score (higher = more important)
    importance: dict[str, float]
    # feature_name -> number of causal paths to target
    n_paths_to_target: dict[str, int]
    # feature_name -> mechanism types on paths to target
    mechanism_types: dict[str, list[str]]
    # feature_name -> True if this is a noise/uninformative feature
    is_noise: dict[str, bool]

    @property
    def ranked_features(self) -> list[tuple[str, float]]:
        """Features sorted by importance (descending)."""
        return sorted(self.importance.items(), key=lambda x: -x[1])

    @property
    def informative_features(self) -> list[str]:
        return [f for f, noise in self.is_noise.items() if not noise]

    @property
    def noise_features(self) -> list[str]:
        return [f for f, noise in self.is_noise.items() if noise]


@dataclass
class ImportanceFidelityResult:
    """How well a model's estimated importance matches ground truth."""

    model_name: str
    task_name: str
    kendall_tau: float
    kendall_p: float
    spearman_rho: float
    spearman_p: float
    noise_detection_precision: float  # fraction of bottom-ranked features that are truly noise
    noise_detection_recall: float  # fraction of noise features in the bottom ranks
    top_k_overlap: float  # overlap between top-k true and estimated features


@dataclass
class ImportanceFidelityReport:
    """Aggregate importance fidelity across tasks and models."""

    per_model_task: list[ImportanceFidelityResult]

    def summary_by_model(self) -> pd.DataFrame:
        """Aggregate fidelity metrics per model."""
        rows = []
        models = sorted(set(r.model_name for r in self.per_model_task))
        for model in models:
            entries = [r for r in self.per_model_task if r.model_name == model]
            rows.append({
                "model": model,
                "mean_kendall_tau": float(np.mean([r.kendall_tau for r in entries])),
                "mean_spearman_rho": float(np.mean([r.spearman_rho for r in entries])),
                "mean_noise_precision": float(np.mean([r.noise_detection_precision for r in entries])),
                "mean_noise_recall": float(np.mean([r.noise_detection_recall for r in entries])),
                "mean_top_k_overlap": float(np.mean([r.top_k_overlap for r in entries])),
                "n_tasks": len(entries),
            })
        return pd.DataFrame(rows).sort_values("mean_kendall_tau", ascending=False).reset_index(drop=True)

    def summary(self) -> str:
        df = self.summary_by_model()
        lines = [
            "=== Feature Importance Fidelity ===",
            "",
            df.to_string(index=False),
        ]
        return "\n".join(lines)


def compute_ground_truth_importance(
    dag: DAGSpec,
    feature_names: list[str],
) -> FeatureImportanceProfile:
    """Extract ground-truth feature importance from the causal DAG.

    Importance is computed as the sum of absolute edge weights along all
    directed paths from each feature to the target.  This captures both
    direct effects and indirect effects mediated through other features.

    For features with no path to the target (noise features), importance
    is zero.
    """
    target = dag.target

    # Build adjacency: child -> list of (parent, |coefficient|)
    children_of: dict[str, list[tuple[str, float, str]]] = {}
    for edge in dag.edges:
        if edge.is_confounder:
            continue
        if edge.parent not in children_of:
            children_of[edge.parent] = []
        mech_type = edge.mechanism["type"] if edge.mechanism else edge.form
        children_of[edge.parent].append((edge.child, abs(edge.coefficient), mech_type))

    # For each feature, find all paths to the target via DFS and sum
    # the product of edge weights along each path.
    importance: dict[str, float] = {}
    n_paths: dict[str, int] = {}
    mech_types: dict[str, list[str]] = {}

    for feat in feature_names:
        paths = _find_all_paths(feat, target, children_of)
        n_paths[feat] = len(paths)

        if not paths:
            importance[feat] = 0.0
            mech_types[feat] = []
            continue

        # Importance = sum of path products (captures multiple independent paths)
        total = 0.0
        all_mechs: list[str] = []
        for path_edges in paths:
            path_weight = 1.0
            for _, weight, mech in path_edges:
                path_weight *= weight
                all_mechs.append(mech)
            total += path_weight

        importance[feat] = total
        mech_types[feat] = sorted(set(all_mechs))

    # Identify noise features: those with zero importance
    is_noise = {f: (importance.get(f, 0.0) == 0.0) for f in feature_names}

    return FeatureImportanceProfile(
        task_name=dag.target,
        target=target,
        importance=importance,
        n_paths_to_target=n_paths,
        mechanism_types=mech_types,
        is_noise=is_noise,
    )


def _find_all_paths(
    source: str,
    target: str,
    children_of: dict[str, list[tuple[str, float, str]]],
    max_depth: int = 20,
) -> list[list[tuple[str, float, str]]]:
    """Find all directed paths from source to target via DFS."""
    if source == target:
        return [[]]

    paths: list[list[tuple[str, float, str]]] = []
    stack: list[tuple[str, list[tuple[str, float, str]], set[str]]] = [
        (source, [], {source})
    ]

    while stack:
        node, path, visited = stack.pop()
        if len(path) >= max_depth:
            continue

        for child, weight, mech in children_of.get(node, []):
            if child in visited:
                continue
            new_path = path + [(child, weight, mech)]
            if child == target:
                paths.append(new_path)
            else:
                stack.append((child, new_path, visited | {child}))

    return paths


def evaluate_importance_fidelity(
    ground_truth: FeatureImportanceProfile,
    estimated_importance: dict[str, float],
    model_name: str,
    top_k: int | None = None,
) -> ImportanceFidelityResult:
    """Compare a model's estimated feature importance against ground truth.

    Args:
        ground_truth: Ground-truth importance from the DAG.
        estimated_importance: Model's estimated importance (feature -> score).
        model_name: Name of the model.
        top_k: Number of top features to check overlap for.  Defaults to
            the number of informative features.
    """
    common = sorted(set(ground_truth.importance) & set(estimated_importance))
    if len(common) < 2:
        return ImportanceFidelityResult(
            model_name=model_name,
            task_name=ground_truth.task_name,
            kendall_tau=float("nan"),
            kendall_p=float("nan"),
            spearman_rho=float("nan"),
            spearman_p=float("nan"),
            noise_detection_precision=0.0,
            noise_detection_recall=0.0,
            top_k_overlap=0.0,
        )

    true_vals = [ground_truth.importance[f] for f in common]
    est_vals = [estimated_importance[f] for f in common]

    kt, kt_p = kendalltau(true_vals, est_vals)
    sr, sr_p = spearmanr(true_vals, est_vals)

    # Noise detection: how well does the model identify unimportant features?
    n_noise = sum(1 for f in common if ground_truth.is_noise.get(f, False))
    if n_noise > 0 and n_noise < len(common):
        # Bottom-ranked features by the model's estimates
        ranked_by_model = sorted(common, key=lambda f: estimated_importance[f])
        bottom_k = set(ranked_by_model[:n_noise])
        true_noise = set(f for f in common if ground_truth.is_noise.get(f, False))
        correct_in_bottom = len(bottom_k & true_noise)
        precision = correct_in_bottom / len(bottom_k) if bottom_k else 0.0
        recall = correct_in_bottom / len(true_noise) if true_noise else 0.0
    else:
        precision = 1.0 if n_noise == 0 else 0.0
        recall = 1.0 if n_noise == 0 else 0.0

    # Top-k overlap
    n_informative = len(ground_truth.informative_features)
    if top_k is None:
        top_k = max(1, n_informative)
    top_k = min(top_k, len(common))

    true_top = set(sorted(common, key=lambda f: -ground_truth.importance[f])[:top_k])
    est_top = set(sorted(common, key=lambda f: -estimated_importance[f])[:top_k])
    overlap = len(true_top & est_top) / top_k if top_k > 0 else 0.0

    return ImportanceFidelityResult(
        model_name=model_name,
        task_name=ground_truth.task_name,
        kendall_tau=float(kt),
        kendall_p=float(kt_p),
        spearman_rho=float(sr),
        spearman_p=float(sr_p),
        noise_detection_precision=precision,
        noise_detection_recall=recall,
        top_k_overlap=overlap,
    )


def get_permutation_importance(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int = 5,
    seed: int = 42,
) -> dict[str, float]:
    """Compute permutation importance for a fitted model.

    A thin wrapper around sklearn's permutation_importance that returns
    a dict compatible with ``evaluate_importance_fidelity()``.
    """
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats, random_state=seed,
    )
    return {
        col: float(result.importances_mean[i])
        for i, col in enumerate(X_test.columns)
    }
