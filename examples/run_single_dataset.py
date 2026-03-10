"""Example: Run a model on a single dataset with detailed output.

Run:
    python examples/run_single_dataset.py
"""

from sklearn.ensemble import GradientBoostingClassifier

from tabular_bank.context import TabularBankContext
from tabular_bank.runner import evaluate_model

# Setup
ctx = TabularBankContext(
    round_id="round-001",
    master_secret="demo-secret-do-not-use-in-production",
    cache_dir="/tmp/tabular_bank_demo",
)

# Pick the first task
task = ctx.get_tasks()[0]
print(f"Task: {task.name}")
print(f"  Problem type: {task.problem_type}")
print(f"  Samples: {task.n_samples}")
print(f"  Features: {task.n_features}")
print(f"  Target: {task.target}")
print(f"  Splits: {task.n_repeats} repeats x {task.n_folds} folds")
print()

# Show first few rows
print("First 5 rows:")
print(task.dataset.head().to_string())
print()

# Evaluate a model
model = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
results = evaluate_model(
    model=model,
    task=task,
    model_name="GBM",
    repeats=[0],  # Just first repeat
)

print("Results:")
for r in results:
    print(f"  Repeat {r.repeat}, Fold {r.fold}: {r.metric_name}={r.metric_value:.4f} "
          f"(fit={r.fit_time:.2f}s, predict={r.predict_time:.2f}s)")

avg = sum(r.metric_value for r in results) / len(results)
print(f"\nMean {results[0].metric_name}: {avg:.4f}")
