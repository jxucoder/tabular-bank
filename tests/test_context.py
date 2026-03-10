"""Tests for TabularBankContext."""

import tempfile

import pandas as pd

from tabular_bank.context import TabularBankContext
from tabular_bank.generation.generate import generate_all


SECRET = "test-secret-for-context"
ROUND = "test-round"
N_SCENARIOS = 3


def test_context_auto_generates():
    """Context auto-generates datasets when they don't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ctx = TabularBankContext(
            round_id=ROUND,
            master_secret=SECRET,
            cache_dir=tmpdir,
            auto_generate=True,
            n_scenarios=N_SCENARIOS,
        )
        tasks = ctx.get_tasks()
        assert len(tasks) == N_SCENARIOS


def test_context_get_datasets():
    """get_datasets returns list of names."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ctx = TabularBankContext(
            round_id=ROUND,
            master_secret=SECRET,
            cache_dir=tmpdir,
            n_scenarios=N_SCENARIOS,
        )
        names = ctx.get_datasets()
        assert len(names) == N_SCENARIOS
        assert all(isinstance(n, str) for n in names)


def test_context_get_metadata():
    """get_metadata returns a DataFrame with expected columns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ctx = TabularBankContext(
            round_id=ROUND,
            master_secret=SECRET,
            cache_dir=tmpdir,
            n_scenarios=N_SCENARIOS,
        )
        meta = ctx.get_metadata()
        assert isinstance(meta, pd.DataFrame)
        assert len(meta) == N_SCENARIOS
        assert "dataset" in meta.columns
        assert "problem_type" in meta.columns
        assert "n_samples" in meta.columns


def test_context_get_task_by_name():
    """Can retrieve specific tasks by name."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ctx = TabularBankContext(
            round_id=ROUND,
            master_secret=SECRET,
            cache_dir=tmpdir,
            n_scenarios=N_SCENARIOS,
        )
        names = ctx.get_datasets()
        task = ctx.get_task(names[0])
        assert task.name == names[0]


def test_context_skips_regeneration():
    """Context doesn't regenerate if data already exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # First generation
        ctx1 = TabularBankContext(
            round_id=ROUND,
            master_secret=SECRET,
            cache_dir=tmpdir,
            n_scenarios=N_SCENARIOS,
        )
        tasks1 = ctx1.get_tasks()

        # Second load (should skip generation)
        ctx2 = TabularBankContext(
            round_id=ROUND,
            master_secret=SECRET,
            cache_dir=tmpdir,
            n_scenarios=N_SCENARIOS,
        )
        tasks2 = ctx2.get_tasks()

        assert len(tasks1) == len(tasks2)
        assert [t.name for t in tasks1] == [t.name for t in tasks2]


def test_context_honors_requested_scenario_count_with_stale_cache_dirs():
    """Round metadata should decide which cached datasets are loaded."""
    with tempfile.TemporaryDirectory() as tmpdir:
        generate_all(
            master_secret=SECRET,
            round_id=ROUND,
            n_scenarios=5,
            cache_dir=tmpdir,
        )

        ctx = TabularBankContext(
            round_id=ROUND,
            master_secret=SECRET,
            cache_dir=tmpdir,
            n_scenarios=3,
        )

        tasks = ctx.get_tasks()
        assert len(tasks) == 3
        assert [task.name for task in tasks] == [
            f"{ROUND}_sampled_{i}" for i in range(3)
        ]
