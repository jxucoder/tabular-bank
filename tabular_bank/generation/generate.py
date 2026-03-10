"""On-demand dataset generation orchestrator.

Generates datasets into a local cache directory. Skips datasets that
already exist unless --force is specified.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from tabular_bank.generation.engine import (
    GeneratedDataset,
    generate_sampled_datasets,
)
from tabular_bank.generation.seed import get_default_cache_dir, get_master_secret

logger = logging.getLogger(__name__)


def generate_all(
    master_secret: str | None = None,
    round_id: str = "round-001",
    n_scenarios: int = 10,
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> list[Path]:
    """Generate benchmark datasets for a round by sampling from the scenario space.

    Args:
        n_scenarios: Number of scenarios to sample and generate.

    Returns list of paths to generated dataset directories.
    """
    secret = get_master_secret(master_secret, cache_dir)
    cache = Path(cache_dir) if cache_dir else get_default_cache_dir()
    round_dir = cache / round_id

    datasets = generate_sampled_datasets(secret, round_id, n_scenarios=n_scenarios)
    scenario_ids = [ds.scenario_id for ds in datasets]
    paths = []
    for i, ds in enumerate(datasets):
        ds_dir = round_dir / ds.scenario_id
        if ds_dir.exists() and not force:
            marker = ds_dir / ".complete"
            if marker.exists():
                logger.info("Skipping %s (already exists)", ds.scenario_id)
                paths.append(ds_dir)
                continue
        _save_dataset(ds, ds_dir)
        paths.append(ds_dir)
        logger.info("Generated dataset %d/%d: %s", i + 1, n_scenarios, ds_dir.name)

    _save_round_metadata(round_dir, round_id, scenario_ids)
    return paths


def generate_one(
    scenario_index: int,
    master_secret: str | None = None,
    round_id: str = "round-001",
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Generate a single sampled dataset and save to cache."""
    secret = get_master_secret(master_secret, cache_dir)
    cache = Path(cache_dir) if cache_dir else get_default_cache_dir()
    round_dir = cache / round_id

    datasets = generate_sampled_datasets(secret, round_id, n_scenarios=scenario_index + 1)
    ds = datasets[scenario_index]
    ds_dir = round_dir / ds.scenario_id

    if ds_dir.exists() and not force:
        marker = ds_dir / ".complete"
        if marker.exists():
            logger.info("Skipping %s (already exists)", ds.scenario_id)
            _save_round_metadata(round_dir, round_id, _merge_scenario_ids(round_dir, ds.scenario_id))
            return ds_dir

    _save_dataset(ds, ds_dir)
    _save_round_metadata(round_dir, round_id, _merge_scenario_ids(round_dir, ds.scenario_id))
    return ds_dir


def _save_dataset(dataset: GeneratedDataset, ds_dir: Path) -> None:
    """Save a generated dataset to disk in TabArena-compatible format."""
    ds_dir.mkdir(parents=True, exist_ok=True)

    # Save the full dataset
    dataset.data.to_csv(ds_dir / "dataset.csv", index=False)

    # Save splits as JSON-serializable format
    splits_serializable = {}
    for repeat_idx, folds in dataset.splits.items():
        splits_serializable[str(repeat_idx)] = {}
        for fold_idx, (train_idx, test_idx) in folds.items():
            splits_serializable[str(repeat_idx)][str(fold_idx)] = {
                "train": train_idx.tolist(),
                "test": test_idx.tolist(),
            }

    with open(ds_dir / "splits.json", "w") as f:
        json.dump(splits_serializable, f)

    # Save metadata
    with open(ds_dir / "metadata.json", "w") as f:
        json.dump(dataset.metadata, f, indent=2)

    # Write completion marker
    (ds_dir / ".complete").touch()


def _save_round_metadata(round_dir: Path, round_id: str, scenario_ids: list[str]) -> None:
    """Save round-level metadata."""
    round_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "round_id": round_id,
        "n_datasets": len(scenario_ids),
        "scenario_ids": scenario_ids,
    }

    with open(round_dir / "round_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def _merge_scenario_ids(round_dir: Path, scenario_id: str) -> list[str]:
    """Merge a newly generated scenario into round metadata."""
    existing: list[str] = []
    meta_path = round_dir / "round_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
        existing = list(metadata.get("scenario_ids", []))
        if not existing and metadata.get("n_datasets"):
            existing = [f"sampled_{i}" for i in range(int(metadata["n_datasets"]))]

    merged = set(existing)
    merged.add(scenario_id)
    return sorted(merged, key=_scenario_sort_key)


def _scenario_sort_key(scenario_id: str) -> tuple[str, int, str]:
    """Sort sampled scenario identifiers numerically when possible."""
    prefix, sep, suffix = scenario_id.rpartition("_")
    if sep and suffix.isdigit():
        return (prefix, int(suffix), scenario_id)
    return (scenario_id, -1, scenario_id)


def load_dataset(ds_dir: Path) -> GeneratedDataset:
    """Load a previously generated dataset from disk."""
    data = pd.read_csv(ds_dir / "dataset.csv")

    with open(ds_dir / "metadata.json") as f:
        metadata = json.load(f)

    with open(ds_dir / "splits.json") as f:
        splits_raw = json.load(f)

    # Reconstruct splits with numpy arrays
    splits: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for repeat_str, folds in splits_raw.items():
        repeat = int(repeat_str)
        splits[repeat] = {}
        for fold_str, indices in folds.items():
            fold = int(fold_str)
            splits[repeat][fold] = (
                np.array(indices["train"]),
                np.array(indices["test"]),
            )

    return GeneratedDataset(
        scenario_id=metadata["scenario_id"],
        dataset_id=metadata["dataset_id"],
        problem_type=metadata["problem_type"],
        target_name=metadata["target_name"],
        data=data,
        splits=splits,
        metadata=metadata,
    )
