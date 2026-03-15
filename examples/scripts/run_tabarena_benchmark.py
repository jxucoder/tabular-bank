#!/usr/bin/env python3
"""Run tabular-bank datasets through TabArena's full evaluation pipeline.

This script demonstrates the complete workflow:
1. Generate contamination-proof synthetic datasets
2. Convert them to TabArena's task format
3. Run TabArena's ExperimentBatchRunner with 8-fold bagging
4. Generate a leaderboard (optionally compared against TabArena's reference)

Requirements:
    pip install "tabular-bank[benchmark]"
    # Plus AutoGluon (see TabArena install instructions)

Usage:
    python run_tabarena_benchmark.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# --- Step 1: Run the benchmark ---
from tabular_bank.runner import run_benchmark_tabarena
from tabular_bank.context import TabularBankContext

SECRET = "demo-secret-tabarena"
ROUND_ID = "demo-round"
N_SCENARIOS = 3  # Use fewer scenarios for a quick demo

# Generate datasets and run TabArena experiments.
# This will:
#   - Generate N_SCENARIOS synthetic datasets from the secret
#   - Convert each to a TabArena UserTask
#   - Run each experiment (LightGBM, RandomForest by default) on each task
#   - Return raw result dicts for leaderboard generation
results_lst = run_benchmark_tabarena(
    round_id=ROUND_ID,
    master_secret=SECRET,
    n_scenarios=N_SCENARIOS,
    folds=[0],  # Single fold for speed; use [0,1,2] for full evaluation
)

print(f"Collected {len(results_lst)} result entries")

# --- Step 2: Generate a leaderboard ---
# Option A: Standalone leaderboard (just your models, no TabArena reference)
from tabular_bank.leaderboard import generate_leaderboard_standalone

ctx = TabularBankContext(
    round_id=ROUND_ID,
    master_secret=SECRET,
    n_scenarios=N_SCENARIOS,
)
task_metadata = ctx.get_task_metadata()

leaderboard = generate_leaderboard_standalone(
    results_lst=results_lst,
    task_metadata=task_metadata,
)
print("\n=== Standalone Leaderboard ===")
with pd.option_context("display.max_columns", None, "display.width", 120):
    print(leaderboard)

# Option B: Compare against TabArena's published reference leaderboard
# (Requires cached TabArena results — downloads ~10GB on first run)
#
# from tabular_bank.leaderboard import generate_leaderboard_tabarena
#
# leaderboard_vs_tabarena = generate_leaderboard_tabarena(
#     results_lst=results_lst,
#     task_metadata=task_metadata,
#     eval_dir="./eval_output",
# )
# print("\n=== vs TabArena Reference ===")
# print(leaderboard_vs_tabarena)


# --- Step 3 (Advanced): Custom experiments ---
# You can pass any AutoGluon model or TabArena experiment:
#
# from tabarena.benchmark.experiment import AGModelBagExperiment
# from autogluon.tabular.models import LGBModel, XTModel
#
# custom_experiments = [
#     AGModelBagExperiment(
#         name="LightGBM_custom",
#         model_cls=LGBModel,
#         model_hyperparameters={"num_boost_round": 500, "learning_rate": 0.05},
#         num_bag_folds=8,
#         time_limit=1800,
#     ),
#     AGModelBagExperiment(
#         name="ExtraTrees_custom",
#         model_cls=XTModel,
#         model_hyperparameters={"n_estimators": 300},
#         num_bag_folds=8,
#         time_limit=1800,
#     ),
# ]
#
# results_custom = run_benchmark_tabarena(
#     round_id=ROUND_ID,
#     master_secret=SECRET,
#     n_scenarios=N_SCENARIOS,
#     experiments=custom_experiments,
# )
