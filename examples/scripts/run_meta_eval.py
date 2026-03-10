"""Example: Run a benchmark round and compute meta-evaluation diagnostics.

Run:
    python examples/scripts/run_meta_eval.py

Generates datasets, benchmarks several models, then reports on:
  - Discriminability: can each task separate models?
  - Task diversity: are tasks complementary or redundant?
  - Ranking concordance (when a reference ranking is provided)
"""

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

import _bootstrap  # noqa: F401
from tabular_bank.context import TabularBankContext
from tabular_bank.evaluation import run_meta_eval
from tabular_bank.leaderboard import format_leaderboard, generate_leaderboard
from tabular_bank.runner import BenchmarkResult, evaluate_model

SECRET = "demo-secret-do-not-use-in-production"
ROUND = "round-001"
CACHE = "/tmp/tabular_bank_meta_eval"

ctx = TabularBankContext(
    round_id=ROUND,
    master_secret=SECRET,
    cache_dir=CACHE,
)

print("Datasets:")
print(ctx.get_metadata().to_string(index=False))
print()

CLASSIFIERS = {
    "GBM": GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=0),
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=0),
    "LogisticReg": LogisticRegression(max_iter=1000, random_state=0),
    "KNN": KNeighborsClassifier(n_neighbors=5),
}
REGRESSORS = {
    "GBM": GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=0),
    "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=0),
    "Ridge": Ridge(alpha=1.0),
    "KNN": KNeighborsRegressor(n_neighbors=5),
}

benchmark = BenchmarkResult()
for task in ctx.get_tasks():
    print(f"Running task: {task.name} ({task.problem_type})")
    models = CLASSIFIERS if task.problem_type in ("binary", "multiclass") else REGRESSORS
    for model_name, model in models.items():
        print(f"  {model_name}...", end=" ", flush=True)
        results = evaluate_model(
            model=model,
            task=task,
            model_name=model_name,
            repeats=[0, 1],
        )
        benchmark.results.extend(results)
        avg = sum(r.metric_value for r in results) / len(results)
        print(f"avg={avg:.4f}")

# Leaderboard
print("\n" + "=" * 80)
print("LEADERBOARD")
print("=" * 80)
leaderboard = generate_leaderboard(benchmark)
print(format_leaderboard(leaderboard))

# Meta-evaluation
print("\n" + "=" * 80)
print("META-EVALUATION")
print("=" * 80)
report = run_meta_eval(benchmark)
print(report.summary())

# Per-task discriminability detail
print("\nPer-task discriminability:")
for task, ds in sorted(report.discriminability.per_task.items()):
    flag = " ** LOW **" if task in report.discriminability.flagged_tasks else ""
    print(f"  {task:>40s}: {ds:.4f}{flag}")

# Task correlation matrix
print("\nTask score correlation matrix:")
print(report.diversity.correlation_matrix.to_string(float_format="%.2f"))
