"""Evaluation diagnostics for benchmark quality, contamination, and scaling."""

from tabular_bank.evaluation.contamination import (
    ContaminationReport,
    analyze_contamination,
    run_contamination_benchmark,
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
]
