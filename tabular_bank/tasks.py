"""Task creation compatible with TabArena's UserTask format.

Creates task objects that can be used with TabArena's ExperimentBatchRunner.
When TabArena is not installed, provides a standalone task representation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class SyntheticTask:
    """A benchmark task wrapping a synthetic dataset.

    Compatible with TabArena's task interface. When TabArena is installed,
    can be converted to a UserTask via to_tabarena_task().
    """

    name: str
    dataset: pd.DataFrame
    target: str
    problem_type: str
    splits: dict[int, dict[int, tuple[np.ndarray, np.ndarray]]]
    metadata: dict = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return len(self.dataset)

    @property
    def n_features(self) -> int:
        return len(self.dataset.columns) - 1

    @property
    def feature_names(self) -> list[str]:
        return [c for c in self.dataset.columns if c != self.target]

    def get_split(self, repeat: int, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Get train/test DataFrames for a specific split."""
        train_idx, test_idx = self.splits[repeat][fold]
        return self.dataset.iloc[train_idx], self.dataset.iloc[test_idx]

    @property
    def n_repeats(self) -> int:
        return len(self.splits)

    @property
    def n_folds(self) -> int:
        if self.splits:
            return len(next(iter(self.splits.values())))
        return 0

    def to_tabarena_task(self, cache_path: Path | None = None):
        """Convert to a TabArena UserTask if TabArena is installed.

        Raises ImportError if TabArena is not installed.
        """
        try:
            from tabarena.benchmark.task.user_task import UserTask, save_local_openml_task
        except ImportError:
            raise ImportError(
                "TabArena is required for this operation. "
                "Install with: pip install tabular-bank[benchmark]"
            )

        if cache_path is None:
            import tempfile
            cache_path = Path(tempfile.mkdtemp()) / "tabular_bank_tasks"

        # Convert splits to TabArena format: repeat -> fold -> (train_idx, test_idx)
        tabarena_splits = {}
        for repeat, folds in self.splits.items():
            tabarena_splits[repeat] = {}
            for fold, (train_idx, test_idx) in folds.items():
                tabarena_splits[repeat][fold] = (train_idx, test_idx)

        task = UserTask(
            name=self.name,
            cache_path=str(cache_path),
            dataset=self.dataset,
            target=self.target,
            problem_type=self.problem_type,
        )
        save_local_openml_task(task=task, splits=tabarena_splits)
        return task


def load_tasks_from_cache(
    round_dir: Path,
    scenario_ids: list[str] | None = None,
) -> list[SyntheticTask]:
    """Load all tasks from a round's cache directory."""
    if not round_dir.exists():
        return []

    tasks = []
    if scenario_ids is not None:
        for scenario_id in scenario_ids:
            ds_dir = round_dir / scenario_id
            meta_path = ds_dir / "metadata.json"
            if ds_dir.is_dir() and meta_path.exists():
                tasks.append(_load_single_task(ds_dir))
        return tasks

    for ds_dir in sorted(round_dir.iterdir(), key=lambda path: _dataset_sort_key(path.name)):
        if ds_dir.is_dir() and (ds_dir / "metadata.json").exists():
            tasks.append(_load_single_task(ds_dir))

    return tasks


def _dataset_sort_key(name: str) -> tuple[str, int, str]:
    """Sort sampled dataset directories numerically when possible."""
    match = re.match(r"^(.*)_(\d+)$", name)
    if match:
        return (match.group(1), int(match.group(2)), name)
    return (name, -1, name)


def _load_single_task(ds_dir: Path) -> SyntheticTask:
    """Load a single task from a dataset directory."""
    data = pd.read_csv(ds_dir / "dataset.csv")

    with open(ds_dir / "metadata.json") as f:
        metadata = json.load(f)

    with open(ds_dir / "splits.json") as f:
        splits_raw = json.load(f)

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

    return SyntheticTask(
        name=metadata["dataset_id"],
        dataset=data,
        target=metadata["target_name"],
        problem_type=metadata["problem_type"],
        splits=splits,
        metadata=metadata,
    )
