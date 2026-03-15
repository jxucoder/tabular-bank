"""TabularBankContext — the main entry point for benchmarking.

Mirrors TabArena's BenchmarkContext / TabArenaContext. Loads (or generates)
all datasets for a benchmark round and provides a unified interface for
accessing tasks, metadata, and running evaluations.
"""

from __future__ import annotations

import json
import logging
from hashlib import sha1
from pathlib import Path

import pandas as pd

from tabular_bank.generation.generate import generate_all
from tabular_bank.generation.seed import get_default_cache_dir, get_master_secret
from tabular_bank.tasks import SyntheticTask, load_tasks_from_cache

logger = logging.getLogger(__name__)


class TabularBankContext:
    """Context manager for synthetic benchmark datasets.

    Loads datasets for a benchmark round from cache. If datasets don't exist
    and auto_generate=True, generates them first (requires master_secret).

    Example:
        ctx = TabularBankContext(round_id="round-001")
        for task in ctx.get_tasks():
            print(task.name, task.n_samples, task.problem_type)
    """

    def __init__(
        self,
        round_id: str = "round-001",
        master_secret: str | None = None,
        cache_dir: str | Path | None = None,
        auto_generate: bool = True,
        n_scenarios: int = 10,
    ):
        self.round_id = round_id
        self.n_scenarios = n_scenarios
        self.cache_dir = Path(cache_dir) if cache_dir else get_default_cache_dir()
        self.round_dir = self.cache_dir / round_id
        self._tasks: list[SyntheticTask] | None = None

        # Auto-generate if needed
        if auto_generate and not self._is_round_complete():
            logger.info("Datasets for round '%s' not found. Generating...", round_id)
            secret = get_master_secret(master_secret, self.cache_dir)
            generate_all(
                master_secret=secret,
                round_id=round_id,
                n_scenarios=n_scenarios,
                cache_dir=str(self.cache_dir),
            )

    def _is_round_complete(self) -> bool:
        """Check if all datasets for this round are generated.

        Verifies that every scenario listed in round_metadata.json has a
        directory with a .complete marker, so partially written or
        corrupted caches trigger regeneration.
        """
        meta_file = self.round_dir / "round_metadata.json"
        if not meta_file.exists():
            return False
        try:
            with open(meta_file) as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            return False
        n_datasets = meta.get("n_datasets", 0)
        if n_datasets == 0 or n_datasets != self.n_scenarios:
            return False
        scenario_ids = meta.get("scenario_ids", [])
        if scenario_ids:
            return all((self.round_dir / scenario_id / ".complete").exists() for scenario_id in scenario_ids)

        complete_count = sum(
            1
            for d in self.round_dir.iterdir()
            if d.is_dir() and (d / ".complete").exists()
        )
        return complete_count >= n_datasets

    def get_tasks(self) -> list[SyntheticTask]:
        """Return list of tasks for benchmarking."""
        if self._tasks is None:
            scenario_ids = self._get_expected_scenario_ids()
            self._tasks = load_tasks_from_cache(self.round_dir, scenario_ids=scenario_ids)
        return self._tasks

    def get_task(self, name: str) -> SyntheticTask:
        """Get a specific task by name."""
        for task in self.get_tasks():
            if task.name == name:
                return task
        raise KeyError(f"Task '{name}' not found. Available: {self.get_datasets()}")

    def get_datasets(self) -> list[str]:
        """Return dataset names (opaque IDs)."""
        return [t.name for t in self.get_tasks()]

    def get_metadata(self) -> pd.DataFrame:
        """Return dataset metadata as a DataFrame."""
        rows = []
        for task in self.get_tasks():
            rows.append({
                "dataset": task.name,
                "problem_type": task.problem_type,
                "n_samples": task.n_samples,
                "n_features": task.n_features,
                "target": task.target,
                "n_repeats": task.n_repeats,
                "n_folds": task.n_folds,
                **{k: v for k, v in task.metadata.items()
                   if k not in ("dataset_id", "target_name", "problem_type", "n_samples", "n_features")},
            })
        return pd.DataFrame(rows)

    def get_tabarena_tasks(self, cache_path: Path | None = None):
        """Convert all tasks to TabArena UserTask objects.

        Returns a list of ``UserTask`` instances that can be used with
        TabArena's ``ExperimentBatchRunner``.

        Requires TabArena to be installed (pip install tabular-bank[benchmark]).
        """
        return [t.to_tabarena_task(cache_path=cache_path) for t in self.get_tasks()]

    def get_task_metadata(self) -> pd.DataFrame:
        """Build a task_metadata DataFrame compatible with TabArena's
        ``ExperimentBatchRunner``.

        The DataFrame contains one row per task with columns expected by
        TabArena: ``tid``, ``name``, ``problem_type``, ``metric``,
        ``target_name``, ``n_samples``, ``n_features``.
        """
        rows = []
        for task in self.get_tasks():
            # Map tabular-bank problem types to TabArena metric conventions
            if task.problem_type == "binary":
                metric = "roc_auc"
            elif task.problem_type == "multiclass":
                metric = "log_loss"
            else:
                metric = "rmse"

            tabarena_problem_type = (
                "classification" if task.problem_type in ("binary", "multiclass")
                else "regression"
            )

            rows.append({
                "tid": _stable_tid(task.name),
                "name": task.name,
                "problem_type": tabarena_problem_type,
                "metric": metric,
                "target_name": task.target,
                "n_samples": task.n_samples,
                "n_features": task.n_features,
            })
        return pd.DataFrame(rows)

    def _get_expected_scenario_ids(self) -> list[str] | None:
        """Return the authoritative scenario ordering for the round, if known."""
        meta_file = self.round_dir / "round_metadata.json"
        if not meta_file.exists():
            return None
        try:
            with open(meta_file) as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
        scenario_ids = meta.get("scenario_ids")
        if scenario_ids:
            return list(scenario_ids)
        n_datasets = int(meta.get("n_datasets", 0))
        if n_datasets > 0:
            return [f"sampled_{i}" for i in range(n_datasets)]
        return None


def _stable_tid(name: str) -> int:
    """Return a stable task identifier suitable for cached metadata."""
    return int(sha1(name.encode("utf-8")).hexdigest()[:12], 16)
