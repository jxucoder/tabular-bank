"""SyntheticTabContext — the main entry point for benchmarking.

Mirrors TabArena's BenchmarkContext / TabArenaContext. Loads (or generates)
all datasets for a benchmark round and provides a unified interface for
accessing tasks, metadata, and running evaluations.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from synthetic_tab.generation.generate import generate_all
from synthetic_tab.generation.seed import get_default_cache_dir, get_master_secret
from synthetic_tab.tasks import SyntheticTask, load_tasks_from_cache

logger = logging.getLogger(__name__)


class SyntheticTabContext:
    """Context manager for synthetic benchmark datasets.

    Loads datasets for a benchmark round from cache. If datasets don't exist
    and auto_generate=True, generates them first (requires master_secret).

    Example:
        ctx = SyntheticTabContext(round_id="round-001")
        for task in ctx.get_tasks():
            print(task.name, task.n_samples, task.problem_type)
    """

    def __init__(
        self,
        round_id: str = "round-001",
        master_secret: str | None = None,
        cache_dir: str | Path | None = None,
        auto_generate: bool = True,
    ):
        self.round_id = round_id
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
                cache_dir=str(self.cache_dir),
            )

    def _is_round_complete(self) -> bool:
        """Check if all datasets for this round are generated."""
        if not self.round_dir.exists():
            return False
        meta_file = self.round_dir / "round_metadata.json"
        return meta_file.exists()

    def get_tasks(self) -> list[SyntheticTask]:
        """Return list of tasks for benchmarking."""
        if self._tasks is None:
            self._tasks = load_tasks_from_cache(self.round_dir)
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

        Requires TabArena to be installed (pip install synthetic-tab[benchmark]).
        """
        return [t.to_tabarena_task(cache_path=cache_path) for t in self.get_tasks()]
