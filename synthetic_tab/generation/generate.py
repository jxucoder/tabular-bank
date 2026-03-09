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

from synthetic_tab.generation.engine import (
    GeneratedDataset,
    generate_all_datasets,
    generate_single_dataset,
)
from synthetic_tab.generation.seed import get_default_cache_dir, get_master_secret
from synthetic_tab.templates.scenarios import SCENARIOS

logger = logging.getLogger(__name__)


def generate_all(
    master_secret: str | None = None,
    round_id: str = "round-001",
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> list[Path]:
    """Generate all benchmark datasets for a round and save to cache.

    Returns list of paths to generated dataset directories.
    """
    secret = get_master_secret(master_secret, cache_dir)
    cache = Path(cache_dir) if cache_dir else get_default_cache_dir()
    round_dir = cache / round_id

    paths = []
    for i in range(len(SCENARIOS)):
        ds_path = _save_dataset_if_needed(secret, round_id, i, round_dir, force)
        paths.append(ds_path)
        logger.info("Generated dataset %d/%d: %s", i + 1, len(SCENARIOS), ds_path.name)

    # Save round metadata
    _save_round_metadata(round_dir, round_id)
    return paths


def generate_one(
    scenario_index: int,
    master_secret: str | None = None,
    round_id: str = "round-001",
    cache_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Generate a single dataset and save to cache."""
    secret = get_master_secret(master_secret, cache_dir)
    cache = Path(cache_dir) if cache_dir else get_default_cache_dir()
    round_dir = cache / round_id
    return _save_dataset_if_needed(secret, round_id, scenario_index, round_dir, force)


def _save_dataset_if_needed(
    master_secret: str,
    round_id: str,
    scenario_index: int,
    round_dir: Path,
    force: bool,
) -> Path:
    """Generate and save a dataset, skipping if already exists."""
    scenario = SCENARIOS[scenario_index]
    ds_dir = round_dir / scenario["id"]

    if ds_dir.exists() and not force:
        marker = ds_dir / ".complete"
        if marker.exists():
            logger.info("Skipping %s (already exists)", scenario["id"])
            return ds_dir

    ds = generate_single_dataset(master_secret, round_id, scenario_index)
    _save_dataset(ds, ds_dir)
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


def _save_round_metadata(round_dir: Path, round_id: str) -> None:
    """Save round-level metadata."""
    round_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "round_id": round_id,
        "n_datasets": len(SCENARIOS),
        "scenarios": [s["id"] for s in SCENARIOS],
    }

    with open(round_dir / "round_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


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
