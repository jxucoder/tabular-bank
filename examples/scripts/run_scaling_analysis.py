"""Example: Run scaling analysis to find minimum stable scenario count.

Run:
    python examples/scripts/run_scaling_analysis.py

Demonstrates how the scenario scaling and ranking variance analyses work.
Uses bootstrap resampling to determine how many benchmark tasks are needed
for stable model rankings.
"""

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

import _bootstrap  # noqa: F401
from tabular_bank.context import TabularBankContext
from tabular_bank.evaluation.scaling import analyze_ranking_variance, analyze_scenario_scaling
from tabular_bank.leaderboard import format_leaderboard, generate_leaderboard, get_task_scores
from tabular_bank.runner import BenchmarkResult, evaluate_model

SECRET = "demo-secret-do-not-use-in-production"
ROUND = "round-001"
CACHE = "/tmp/tabular_bank_scaling"
N_SCENARIOS = 20  # More scenarios for meaningful scaling analysis

ctx = TabularBankContext(
    round_id=ROUND,
    master_secret=SECRET,
    cache_dir=CACHE,
    n_scenarios=N_SCENARIOS,
)

print(f"Generated {len(ctx.get_tasks())} tasks")
print()

CLASSIFIERS = {
    "GBM": GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=0),
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=0),
    "LogisticReg": LogisticRegression(max_iter=1000, random_state=0),
    "KNN": KNeighborsClassifier(n_neighbors=10),
    "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=0),
}
REGRESSORS = {
    "GBM": GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=0),
    "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=0),
    "Ridge": Ridge(alpha=1.0),
    "KNN": KNeighborsRegressor(n_neighbors=10),
    "MLP": MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=300, random_state=0),
}

# Run benchmark (first repeat only for speed)
print("Running benchmark...")
benchmark = BenchmarkResult()
for task in ctx.get_tasks():
    models = CLASSIFIERS if task.problem_type in ("binary", "multiclass") else REGRESSORS
    for model_name, model in models.items():
        results = evaluate_model(model=model, task=task, model_name=model_name, repeats=[0])
        benchmark.results.extend(results)

# Leaderboard
print("\n" + "=" * 80)
print("FULL LEADERBOARD")
print("=" * 80)
leaderboard = generate_leaderboard(benchmark)
print(format_leaderboard(leaderboard))

task_scores = get_task_scores(benchmark)

# --- Scenario scaling ---
print("\n" + "=" * 80)
print("SCENARIO SCALING ANALYSIS")
print("=" * 80)
print("How many tasks are needed for stable model rankings?")
print(f"Testing with {task_scores.shape[1]} total tasks, 200 bootstrap resamples each...\n")

scaling = analyze_scenario_scaling(
    task_scores,
    scenario_counts=[3, 5, 8, 10, 12, 15, 18, 20],
    n_bootstrap=200,
    stability_threshold=0.9,
)
print(scaling.summary())

# --- Ranking variance ---
print("\n" + "=" * 80)
print("RANKING VARIANCE (bootstrap confidence intervals)")
print("=" * 80)
variance_df = analyze_ranking_variance(task_scores, n_bootstrap=500)
print(variance_df.to_string(index=False))
print()
print("Models with std_rank > 1.0 have unstable rankings and may need more tasks.")
