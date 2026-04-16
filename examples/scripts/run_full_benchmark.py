"""Example: Run a comprehensive benchmark with all available baselines.

Run:
    python examples/scripts/run_full_benchmark.py

Evaluates all official baselines (classical + optional boosting + foundation)
on tabular-bank tasks, produces a leaderboard, and runs full meta-evaluation
including IRT analysis.
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
from tabular_bank.evaluation import run_meta_eval, analyze_ranking_variance
from tabular_bank.leaderboard import format_leaderboard, generate_leaderboard, get_task_scores
from tabular_bank.runner import BenchmarkResult, evaluate_model

SECRET = "demo-secret-do-not-use-in-production"
ROUND = "round-001"
CACHE = "/tmp/tabular_bank_full_benchmark"

ctx = TabularBankContext(
    round_id=ROUND,
    master_secret=SECRET,
    cache_dir=CACHE,
    n_scenarios=10,
)

print("Datasets:")
print(ctx.get_metadata().to_string(index=False))
print()

# All always-available baselines (sklearn)
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

# Try to add optional boosting baselines
try:
    from xgboost import XGBClassifier, XGBRegressor
    CLASSIFIERS["XGBoost"] = XGBClassifier(n_estimators=100, random_state=0, eval_metric="logloss")
    REGRESSORS["XGBoost"] = XGBRegressor(n_estimators=100, random_state=0)
    print("XGBoost: available")
except ImportError:
    print("XGBoost: not installed (pip install xgboost)")

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    CLASSIFIERS["LightGBM"] = LGBMClassifier(n_estimators=100, random_state=0, verbose=-1)
    REGRESSORS["LightGBM"] = LGBMRegressor(n_estimators=100, random_state=0, verbose=-1)
    print("LightGBM: available")
except ImportError:
    print("LightGBM: not installed (pip install lightgbm)")

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CLASSIFIERS["CatBoost"] = CatBoostClassifier(iterations=100, random_seed=0, verbose=0)
    REGRESSORS["CatBoost"] = CatBoostRegressor(iterations=100, random_seed=0, verbose=0)
    print("CatBoost: available")
except ImportError:
    print("CatBoost: not installed (pip install catboost)")

# Try TabPFN
try:
    from tabpfn import TabPFNClassifier
    CLASSIFIERS["TabPFN"] = TabPFNClassifier()
    print("TabPFN: available")
except ImportError:
    print("TabPFN: not installed (pip install tabpfn)")

print(f"\nTotal classifiers: {len(CLASSIFIERS)}, regressors: {len(REGRESSORS)}")

# Run benchmark (first 2 repeats for speed)
benchmark = BenchmarkResult()
for task in ctx.get_tasks():
    print(f"\nTask: {task.name} ({task.problem_type}, {task.n_samples} rows)")
    models = CLASSIFIERS if task.problem_type in ("binary", "multiclass") else REGRESSORS
    for model_name, model in models.items():
        print(f"  {model_name}...", end=" ", flush=True)
        results = evaluate_model(model=model, task=task, model_name=model_name, repeats=[0, 1])
        benchmark.results.extend(results)
        avg = sum(r.metric_value for r in results) / len(results)
        print(f"avg={avg:.4f}")

# Leaderboard
print("\n" + "=" * 80)
print("LEADERBOARD")
print("=" * 80)
leaderboard = generate_leaderboard(benchmark)
print(format_leaderboard(leaderboard))

# Meta-evaluation with IRT
print("\n" + "=" * 80)
print("META-EVALUATION (with IRT)")
print("=" * 80)
report = run_meta_eval(benchmark, fit_irt=True, irt_min_models=4)
print(report.summary())

# IRT detail
if report.irt is not None:
    print("\nPer-task IRT parameters:")
    print(f"  {'Task':<40s} {'Difficulty':>11s} {'Discrimination':>15s}")
    print("  " + "-" * 68)
    for item in sorted(report.irt.items, key=lambda x: -x.difficulty):
        print(f"  {item.task:<40s} {item.difficulty:>11.3f} {item.discrimination:>15.3f}")

    print("\nModel abilities (theta):")
    for m, theta in sorted(report.irt.model_abilities.items(), key=lambda x: -x[1]):
        print(f"  {m:<30s} {theta:>8.3f}")

# Ranking variance
print("\n" + "=" * 80)
print("RANKING STABILITY (bootstrap)")
print("=" * 80)
task_scores = get_task_scores(benchmark)
variance = analyze_ranking_variance(task_scores, n_bootstrap=200)
print(variance.to_string(index=False))
