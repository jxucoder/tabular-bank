"""Example: Run a full benchmark with multiple models.

Run:
    python examples/scripts/run_benchmark.py

This generates datasets (if needed), runs multiple sklearn models across
all tasks/splits, and produces a leaderboard with ELO ratings.
"""

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge

import _bootstrap  # noqa: F401
from tabular_bank.context import TabularBankContext
from tabular_bank.leaderboard import format_leaderboard, generate_leaderboard
from tabular_bank.runner import BenchmarkResult, evaluate_model

# Setup
SECRET = "demo-secret-do-not-use-in-production"
ROUND = "round-001"
CACHE = "/tmp/tabular_bank_demo"

# Load context (generates datasets if needed)
ctx = TabularBankContext(
    round_id=ROUND,
    master_secret=SECRET,
    cache_dir=CACHE,
)

print("Datasets:")
print(ctx.get_metadata().to_string(index=False))
print()

# Define models per problem type
CLASSIFIERS = {
    "GBM": GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=0),
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=0),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=0),
}
REGRESSORS = {
    "GBM": GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=0),
    "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=0),
    "Ridge": Ridge(alpha=1.0),
}

# Run benchmark (using only first 2 repeats x 3 folds for speed)
benchmark = BenchmarkResult()

for task in ctx.get_tasks():
    print(f"Running task: {task.name} ({task.problem_type})")
    models = CLASSIFIERS if task.problem_type in ("binary", "multiclass") else REGRESSORS

    for model_name, model in models.items():
        print(f"  Model: {model_name}...", end=" ", flush=True)
        results = evaluate_model(
            model=model,
            task=task,
            model_name=model_name,
            repeats=[0, 1],  # First 2 repeats for speed
        )
        benchmark.results.extend(results)
        avg_score = sum(r.metric_value for r in results) / len(results)
        print(f"avg={avg_score:.4f}")

# Generate and display leaderboard
print("\n" + "=" * 80)
print("LEADERBOARD")
print("=" * 80)
leaderboard = generate_leaderboard(benchmark)
print(format_leaderboard(leaderboard))
