"""Evaluation diagnostics for benchmark quality, contamination, scaling, and interpretability."""

from tabular_bank.evaluation.contamination import (
    ContaminationReport,
    analyze_contamination,
    run_contamination_benchmark,
)
from tabular_bank.evaluation.diagnostics import (
    DiagnosticReport,
    compute_pareto_frontier,
    run_diagnostics,
)
from tabular_bank.evaluation.feature_importance import (
    FeatureImportanceProfile,
    ImportanceFidelityReport,
    ImportanceFidelityResult,
    compute_ground_truth_importance,
    evaluate_importance_fidelity,
    get_permutation_importance,
)
from tabular_bank.evaluation.meta_eval import (
    IRTResult,
    MetaEvalReport,
    compute_discriminability,
    compute_irt,
    compute_ranking_concordance,
    compute_task_diversity,
    run_meta_eval,
)
from tabular_bank.evaluation.scaling import (
    ScalingReport,
    analyze_ranking_variance,
    analyze_sample_scaling,
    analyze_scenario_scaling,
)

__all__ = [
    # Meta-eval
    "MetaEvalReport",
    "IRTResult",
    "compute_discriminability",
    "compute_irt",
    "compute_ranking_concordance",
    "compute_task_diversity",
    "run_meta_eval",
    # Contamination
    "ContaminationReport",
    "analyze_contamination",
    "run_contamination_benchmark",
    # Scaling
    "ScalingReport",
    "analyze_ranking_variance",
    "analyze_sample_scaling",
    "analyze_scenario_scaling",
    # Diagnostics
    "DiagnosticReport",
    "compute_pareto_frontier",
    "run_diagnostics",
    # Feature importance
    "FeatureImportanceProfile",
    "ImportanceFidelityReport",
    "ImportanceFidelityResult",
    "compute_ground_truth_importance",
    "evaluate_importance_fidelity",
    "get_permutation_importance",
]
