"""Example: Run contamination analysis comparing tabular-bank vs reference.

Run:
    python examples/scripts/run_contamination_analysis.py

Demonstrates how to use the contamination analysis module to detect
memorization effects by comparing model rankings on tabular-bank
(contamination-proof) vs. a reference benchmark.

In a real analysis you would:
1. Run the same models on both TabArena and tabular-bank
2. Compare the rankings to detect models that benefit from memorization
3. Use the Memorization Susceptibility Index (MSI) to quantify the effect

This example simulates a reference benchmark for demonstration purposes.
"""

import numpy as np
import pandas as pd
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
from tabular_bank.evaluation.contamination import analyze_contamination
from tabular_bank.leaderboard import get_task_scores
from tabular_bank.runner import BenchmarkResult, evaluate_model

SECRET = "demo-secret-do-not-use-in-production"
ROUND = "round-001"
CACHE = "/tmp/tabular_bank_contamination"

ctx = TabularBankContext(
    round_id=ROUND,
    master_secret=SECRET,
    cache_dir=CACHE,
    n_scenarios=8,
)

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

# --- Step 1: Run on tabular-bank ---
print("Running models on tabular-bank...")
benchmark = BenchmarkResult()
for task in ctx.get_tasks():
    models = CLASSIFIERS if task.problem_type in ("binary", "multiclass") else REGRESSORS
    for model_name, model in models.items():
        results = evaluate_model(model=model, task=task, model_name=model_name, repeats=[0, 1])
        benchmark.results.extend(results)

tbank_scores = get_task_scores(benchmark)
print(f"tabular-bank score matrix: {tbank_scores.shape[0]} models x {tbank_scores.shape[1]} tasks")

# --- Step 2: Simulate reference benchmark scores ---
# In practice, you would load real TabArena results here.
# We simulate by adding noise to the tabular-bank scores (with a slight
# boost for "MLP" to simulate what memorization might look like).
print("\nSimulating reference benchmark (for demonstration)...")
rng = np.random.default_rng(123)
ref_scores = tbank_scores.copy()
for col in ref_scores.columns:
    ref_scores[col] += rng.normal(0, 0.02, size=len(ref_scores))
    # Simulate memorization: MLP gets an artificial boost on reference
    if "MLP" in ref_scores.index:
        ref_scores.loc["MLP", col] += 0.05

# --- Step 3: Run contamination analysis ---
print("\nRunning contamination analysis...")
report = analyze_contamination(
    tbank_scores=tbank_scores,
    ref_scores=ref_scores,
    memorization_prone=["MLP"],  # In real use: ["TabPFN", "LLM-classifier", etc.]
    gap_threshold=2,
)
print(report.summary())

print("\n--- Interpretation ---")
print("A positive rank gap means the model ranks BETTER on the reference")
print("than on tabular-bank. Large positive gaps for foundation models")
print("(but not for classical models) suggest memorization.")
if report.msi is not None:
    if report.msi > 1.0:
        print(f"\nMSI = {report.msi:.2f}: memorization-prone models rank notably")
        print("better on the reference. Worth investigating further.")
    elif report.msi < -1.0:
        print(f"\nMSI = {report.msi:.2f}: memorization-prone models rank WORSE on")
        print("the reference. No evidence of memorization.")
    else:
        print(f"\nMSI = {report.msi:.2f}: no strong evidence of differential memorization.")
