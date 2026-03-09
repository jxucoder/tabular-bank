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

from synthetic_tab.generation.dag_builder import DAGSpec, build_dag
from synthetic_tab.generation.feature_generator import generate_features
from synthetic_tab.generation.sampler import sample_dataset
from synthetic_tab.generation.seed import (
    derive_dag_seed,
    derive_dataset_seed,
    derive_feature_seed,
    derive_round_seed,
    derive_split_seed,
)
from synthetic_tab.templates.scenarios import SCENARIOS, get_scenario


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
) -> GeneratedDataset:
    """Generate a single dataset from a scenario template + seeds.

    This is the main entry point for dataset generation. It:
    1. Derives all seeds from master_secret + round_id + scenario_index
    2. Generates features procedurally
    3. Builds a random DAG
    4. Samples data from the DAG
    5. Creates cross-validation splits
    """
    template = get_scenario(scenario_index)
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
    n_samples = int(data_rng.integers(
        template["n_samples_range"][0],
        template["n_samples_range"][1] + 1,
    ))
    df = sample_dataset(data_rng, dag, features, target, n_samples)

    # Step 4: Create splits (10 repeats x 3 folds = 30 splits)
    splits = _create_splits(df, target, split_seed)

    # Build metadata
    dataset_id = f"{round_id}_{template['id']}"
    metadata = {
        "dataset_id": dataset_id,
        "scenario_id": template["id"],
        "round_id": round_id,
        "problem_type": template["problem_type"],
        "n_samples": n_samples,
        "n_features": len(features),
        "n_continuous": sum(1 for f in features if f["type"] == "continuous"),
        "n_categorical": sum(1 for f in features if f["type"] == "categorical"),
        "target_name": target["name"],
        "domain": template["domain"],
        "difficulty": template["difficulty"],
    }
    if template["problem_type"] != "regression":
        metadata["n_classes"] = template.get("n_classes", 2)

    return GeneratedDataset(
        scenario_id=template["id"],
        dataset_id=dataset_id,
        problem_type=template["problem_type"],
        target_name=target["name"],
        data=df,
        splits=splits,
        metadata=metadata,
    )


def generate_all_datasets(
    master_secret: str,
    round_id: str = "round-001",
) -> list[GeneratedDataset]:
    """Generate all benchmark datasets for a round."""
    datasets = []
    for i in range(len(SCENARIOS)):
        ds = generate_single_dataset(master_secret, round_id, i)
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
