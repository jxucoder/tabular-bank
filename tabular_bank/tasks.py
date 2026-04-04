"""Task creation compatible with TabArena's UserTask format.

Creates task objects that can be used with TabArena's ExperimentBatchRunner.
When TabArena is not installed, provides a standalone task representation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from tabular_bank import _scenario_sort_key


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
        """Convert to a TabArena UserTask (local OpenML task) for use with
        TabArena's ExperimentBatchRunner pipeline.

        The conversion:
        1. Creates a ``UserTask`` wrapper with a unique name
        2. Calls ``create_local_openml_task`` to build the OpenML task object
        3. Saves the task to disk so ``ExperimentBatchRunner`` can load it

        Returns the ``UserTask`` instance. The underlying OpenML task is
        accessible via ``user_task.load_local_openml_task()``.

        Raises ImportError if TabArena is not installed.
        """
        try:
            from tabarena.benchmark.task.user_task import UserTask
        except ImportError:
            raise ImportError(
                "TabArena is required for this operation. "
                "Install with: pip install tabular-bank[benchmark]"
            )

        if cache_path is None:
            import tempfile
            cache_path = Path(tempfile.gettempdir()) / "tabular_bank_tasks"
        cache_path.mkdir(parents=True, exist_ok=True)

        # Map tabular-bank problem types to TabArena's expected types
        tabarena_problem_type = (
            "classification" if self.problem_type in ("binary", "multiclass")
            else "regression"
        )

        # Convert splits to TabArena format: repeat -> fold -> (train_list, test_list)
        # TabArena expects plain Python lists, not numpy arrays.
        tabarena_splits: dict[int, dict[int, tuple[list, list]]] = {}
        for repeat, folds in self.splits.items():
            tabarena_splits[repeat] = {}
            for fold, (train_idx, test_idx) in folds.items():
                tabarena_splits[repeat][fold] = (
                    train_idx.tolist() if hasattr(train_idx, "tolist") else list(train_idx),
                    test_idx.tolist() if hasattr(test_idx, "tolist") else list(test_idx),
                )

        # Ensure categorical columns have the right dtype for TabArena/OpenML
        dataset = self.dataset.copy()
        for col in dataset.select_dtypes(include=["object"]).columns:
            dataset[col] = dataset[col].astype("category")

        user_task = UserTask(task_name=self.name, task_cache_path=cache_path)
        openml_task = user_task.create_local_openml_task(
            target_feature=self.target,
            problem_type=tabarena_problem_type,
            dataset=dataset,
            splits=tabarena_splits,
        )
        user_task.save_local_openml_task(openml_task)
        return user_task


def load_tasks_from_cache(
    round_dir: Path,
    scenario_ids: list[str] | None = None,
) -> list[SyntheticTask]:
    """Load all tasks from a round's cache directory."""
    if not round_dir.exists():
        return []

    tasks = []
    if scenario_ids is not None:
        resolved_round = round_dir.resolve()
        for scenario_id in scenario_ids:
            ds_dir = round_dir / scenario_id
            if not ds_dir.resolve().is_relative_to(resolved_round):
                continue
            meta_path = ds_dir / "metadata.json"
            if ds_dir.is_dir() and meta_path.exists():
                tasks.append(_load_single_task(ds_dir))
        return tasks

    for ds_dir in sorted(round_dir.iterdir(), key=lambda path: _dataset_sort_key(path.name)):
        if ds_dir.is_dir() and (ds_dir / "metadata.json").exists():
            tasks.append(_load_single_task(ds_dir))

    return tasks


_dataset_sort_key = _scenario_sort_key


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
