"""Core procedural generation engine.

Ties together seed derivation, feature generation, DAG construction, and data
sampling into a single high-level API.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold

from tabular_bank.generation.dag_builder import DAGSpec, build_dag
from tabular_bank.generation.feature_generator import generate_features
from tabular_bank.generation.missing import inject_missing
from tabular_bank.generation.sampler import sample_dataset
from tabular_bank.generation.seed import (
    derive_dag_seed,
    derive_dataset_seed,
    derive_feature_seed,
    derive_round_seed,
    derive_split_seed,
)
from tabular_bank.templates.scenarios import sample_scenario


@dataclass
class GeneratedDataset:
    """A fully generated synthetic dataset with splits."""

    scenario_id: str
    dataset_id: str
    problem_type: str
    target_name: str
    data: pd.DataFrame
    splits: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]]
    metadata: dict
    dag: DAGSpec | None = None

    @property
    def n_samples(self) -> int:
        return len(self.data)

    @property
    def n_features(self) -> int:
        return len(self.data.columns) - 1  # Exclude target

    @property
    def feature_names(self) -> list[str]:
        return [c for c in self.data.columns if c != self.target_name]


def generate_single_dataset(
    master_secret: str,
    round_id: str,
    scenario_index: int,
    template_override: dict | None = None,
) -> GeneratedDataset:
    """Generate a single dataset from a scenario template + seeds.

    This is the main entry point for dataset generation. It:
    1. Derives all seeds from master_secret + round_id + scenario_index
    2. Generates features procedurally
    3. Builds a random DAG
    4. Samples data from the DAG
    5. Creates cross-validation splits

    Args:
        template_override: If provided, use this template dict instead of
            looking up by ``scenario_index``.  Enables parametric sampling.
    """
    if template_override is None:
        raise ValueError(
            "template_override is required. Use generate_sampled_datasets() or pass a template dict."
        )
    template = template_override
    round_seed = derive_round_seed(master_secret, round_id)

    # Derive independent seeds for each generation step
    feature_seed = derive_feature_seed(round_seed, scenario_index)
    dag_seed = derive_dag_seed(round_seed, scenario_index)
    data_seed = derive_dataset_seed(round_seed, scenario_index)
    split_seed = derive_split_seed(round_seed, scenario_index)

    # Step 1: Generate features
    feature_rng = np.random.default_rng(feature_seed)
    features, target = generate_features(feature_rng, template)

    # Step 2: Build DAG
    dag_rng = np.random.default_rng(dag_seed)
    dag = build_dag(dag_rng, features, target, template)

    # Step 3: Sample data
    data_rng = np.random.default_rng(data_seed)
    n_samples = max(1, int(data_rng.integers(
        template["n_samples_range"][0],
        template["n_samples_range"][1] + 1,
    )))
    df = sample_dataset(data_rng, dag, features, target, n_samples, template=template)

    # Step 4: For forecasting tasks, add lagged features before missing
    # value injection so that lags are computed from complete data.
    n_lags = 0
    forecast_horizon = 1
    if template["problem_type"] == "forecasting":
        n_lags = template.get("n_lags", 3)
        forecast_horizon = template.get("forecast_horizon", 1)
        df = _add_lagged_features(df, target["name"], features, n_lags, forecast_horizon)

    # Step 5: Inject missing values (if configured)
    missing_rate = template.get("missing_rate", 0.0)
    if missing_rate > 0:
        missing_mechanism = template.get("missing_mechanism", "MCAR")
        df = inject_missing(data_rng, df, target["name"], missing_rate, missing_mechanism)

    # Step 6: Create splits
    if template["problem_type"] == "forecasting":
        splits = _create_temporal_splits(df, split_seed)
    else:
        splits = _create_splits(df, target, split_seed)

    # Build metadata — use actual DataFrame shape for accuracy
    dataset_id = f"{round_id}_{template['id']}"
    n_samples = len(df)
    total_features = len(df.columns) - 1
    informative_features = len(features)
    noise_features = total_features - informative_features
    informative_continuous = sum(1 for f in features if f["type"] == "continuous")
    informative_categorical = sum(1 for f in features if f["type"] == "categorical")
    metadata = {
        "dataset_id": dataset_id,
        "scenario_id": template["id"],
        "round_id": round_id,
        "problem_type": template["problem_type"],
        "n_samples": n_samples,
        "n_features": total_features,
        "n_informative_features": informative_features,
        "n_noise_features": noise_features,
        "n_continuous": informative_continuous + noise_features,
        "n_categorical": informative_categorical,
        "target_name": target["name"],
        "domain": template["domain"],
        "difficulty": template["difficulty"],
    }
    if template["problem_type"] not in ("regression", "forecasting"):
        n_classes = template.get("n_classes", 2)
        if template["problem_type"] == "multiclass" and n_classes < 3:
            n_classes = 3
        metadata["n_classes"] = n_classes
    if template["problem_type"] == "forecasting":
        metadata["n_lags"] = n_lags
        metadata["forecast_horizon"] = forecast_horizon

    return GeneratedDataset(
        scenario_id=template["id"],
        dataset_id=dataset_id,
        problem_type=template["problem_type"],
        target_name=target["name"],
        data=df,
        splits=splits,
        metadata=metadata,
        dag=dag,
    )


def generate_sampled_datasets(
    master_secret: str,
    round_id: str = "round-001",
    n_scenarios: int = 10,
    scenario_space: dict | None = None,
) -> list[GeneratedDataset]:
    """Generate benchmark datasets by sampling from the scenario space.

    Instead of using the fixed 5 templates, draws ``n_scenarios`` random
    configurations from the continuous parameter space (CausalProfiler-style
    coverage guarantee).

    Args:
        scenario_space: Optional overrides for the default
            :data:`~tabular_bank.templates.scenarios.SCENARIO_SPACE`.
            Only the keys you provide are changed; everything else
            keeps its default value.
    """
    round_seed = derive_round_seed(master_secret, round_id)
    # round_seed is 32 bytes; convert to int for numpy RNG
    scenario_rng = np.random.default_rng(int.from_bytes(round_seed[:8], "big"))

    datasets = []
    for i in range(n_scenarios):
        tmpl = sample_scenario(scenario_rng, scenario_id=f"sampled_{i}",
                               scenario_space=scenario_space)
        ds = generate_single_dataset(master_secret, round_id, i, template_override=tmpl)
        datasets.append(ds)
    return datasets


def _create_splits(
    df: pd.DataFrame,
    target: dict,
    split_seed: int,
    n_repeats: int = 10,
    n_folds: int = 3,
) -> dict[int, dict[int, tuple[np.ndarray, np.ndarray]]]:
    """Create cross-validation splits matching TabArena's protocol.

    Returns a dict: repeat_idx -> fold_idx -> (train_indices, test_indices)
    """
    target_name = target["name"]

    if target["problem_type"] in ("binary", "multiclass"):
        splitter = RepeatedStratifiedKFold(
            n_splits=n_folds,
            n_repeats=n_repeats,
            random_state=split_seed,
        )
        y = df[target_name]
    else:
        splitter = RepeatedKFold(
            n_splits=n_folds,
            n_repeats=n_repeats,
            random_state=split_seed,
        )
        y = df[target_name]

    splits: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(df, y)):
        repeat = fold_idx // n_folds
        fold = fold_idx % n_folds
        if repeat not in splits:
            splits[repeat] = {}
        splits[repeat][fold] = (train_idx, test_idx)

    return splits


def _create_temporal_splits(
    df: pd.DataFrame,
    split_seed: int,
    n_repeats: int = 5,
    train_fractions: tuple[float, ...] = (0.5, 0.6, 0.7),
) -> dict[int, dict[int, tuple[np.ndarray, np.ndarray]]]:
    """Create time-based train/test splits for forecasting tasks.

    Instead of random cross-validation (which destroys temporal ordering),
    uses expanding-window splits where the train set is always a prefix
    of the data and the test set is the subsequent block.

    Each "repeat" uses a different train/test boundary (expanding window).
    Each "fold" within a repeat uses a different test window length.

    Returns a dict: repeat_idx -> fold_idx -> (train_indices, test_indices)
    """
    n = len(df)
    indices = np.arange(n)
    splits: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = {}

    rng = np.random.default_rng(split_seed)

    for repeat in range(n_repeats):
        splits[repeat] = {}
        for fold, train_frac in enumerate(train_fractions):
            # Add small jitter to the split point for diversity across repeats
            jitter = float(rng.uniform(-0.03, 0.03))
            frac = max(0.3, min(0.85, train_frac + jitter * repeat))
            split_idx = int(n * frac)
            split_idx = max(10, min(split_idx, n - 10))

            train_idx = indices[:split_idx]
            test_idx = indices[split_idx:]
            splits[repeat][fold] = (train_idx, test_idx)

    return splits


def _add_lagged_features(
    df: pd.DataFrame,
    target_name: str,
    features: list[dict],
    n_lags: int,
    forecast_horizon: int,
) -> pd.DataFrame:
    """Add lagged columns for forecasting tasks.

    For each continuous feature and the target, creates lag-1 through lag-k
    columns.  The target column is then shifted forward by ``forecast_horizon``
    steps so the task becomes: predict target[t+h] from features[t], lags[t].

    Rows where lags or the shifted target are undefined (first n_lags rows
    and last forecast_horizon rows) are dropped.
    """
    df = df.copy()

    # Select columns to lag: target + continuous features
    continuous_cols = [f["name"] for f in features if f["type"] == "continuous"]
    lag_cols = [target_name] + [c for c in continuous_cols if c in df.columns]
    # Limit to at most 5 columns to keep feature count manageable
    if len(lag_cols) > 6:
        lag_cols = lag_cols[:6]

    # Create lag features
    for col in lag_cols:
        for lag in range(1, n_lags + 1):
            lag_name = f"{col}_lag{lag}"
            df[lag_name] = df[col].shift(lag)

    # Shift target forward: we want to predict target at time t+h
    # using features at time t.  So the target column becomes the future value.
    if forecast_horizon > 0:
        df[target_name] = df[target_name].shift(-forecast_horizon)

    # Drop rows with NaN from lagging/shifting
    df = df.dropna().reset_index(drop=True)

    return df
