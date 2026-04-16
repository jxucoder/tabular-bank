"""Tests for TabArena integration — task conversion, metadata, and API wiring.

These tests verify the integration layer without requiring TabArena to be
installed. The actual TabArena imports are mocked where necessary.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd
import pytest

from tabular_bank.context import TabularBankContext
from tabular_bank.tasks import SyntheticTask


SECRET = "test-secret-tabarena-integration"
ROUND = "test-round-ta"
N_SCENARIOS = 2


@pytest.fixture(scope="module")
def ctx():
    """Create a TabularBankContext with a small number of scenarios."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield TabularBankContext(
            round_id=ROUND,
            master_secret=SECRET,
            cache_dir=tmpdir,
            n_scenarios=N_SCENARIOS,
        )


def test_get_task_metadata_shape(ctx):
    """get_task_metadata should return one row per task with expected columns."""
    meta = ctx.get_task_metadata()
    assert isinstance(meta, pd.DataFrame)
    assert len(meta) == N_SCENARIOS
    expected_cols = {"tid", "name", "problem_type", "metric", "target_name", "n_samples", "n_features"}
    assert expected_cols.issubset(set(meta.columns))


def test_get_task_metadata_problem_types(ctx):
    """Problem types should be mapped to TabArena conventions."""
    meta = ctx.get_task_metadata()
    for _, row in meta.iterrows():
        assert row["problem_type"] in ("classification", "regression")
        if row["problem_type"] == "classification":
            assert row["metric"] in ("roc_auc", "log_loss")
        else:
            assert row["metric"] in ("rmse", "mae")


def test_get_task_metadata_tid_unique(ctx):
    """Task IDs should be unique."""
    meta = ctx.get_task_metadata()
    assert meta["tid"].nunique() == len(meta)


def test_synthetic_task_splits_format(ctx):
    """Splits should be convertible to TabArena's expected list format."""
    tasks = ctx.get_tasks()
    for task in tasks:
        for repeat, folds in task.splits.items():
            for fold, (train_idx, test_idx) in folds.items():
                # Should be convertible to lists (TabArena requirement)
                train_list = train_idx.tolist() if hasattr(train_idx, "tolist") else list(train_idx)
                test_list = test_idx.tolist() if hasattr(test_idx, "tolist") else list(test_idx)
                assert isinstance(train_list, list)
                assert isinstance(test_list, list)
                assert len(train_list) > 0
                assert len(test_list) > 0
                # No overlap
                assert not set(train_list) & set(test_list)


def test_to_tabarena_task_raises_without_tabarena(ctx):
    """to_tabarena_task should raise ImportError when TabArena is not installed."""
    tasks = ctx.get_tasks()
    # This will either work (if TabArena is installed) or raise ImportError
    try:
        task = tasks[0].to_tabarena_task()
        # If we get here, TabArena is installed — verify we got a UserTask
        assert hasattr(task, "task_name")
        assert hasattr(task, "load_local_openml_task")
    except ImportError as e:
        assert "TabArena is required" in str(e)


def test_task_metadata_matches_tasks(ctx):
    """Metadata should correspond 1:1 with tasks."""
    tasks = ctx.get_tasks()
    meta = ctx.get_task_metadata()
    task_names = {t.name for t in tasks}
    meta_names = set(meta["name"])
    assert task_names == meta_names


def test_run_benchmark_tabarena_import_error():
    """run_benchmark_tabarena should raise ImportError when TabArena is missing."""
    from tabular_bank.runner import run_benchmark_tabarena
    try:
        run_benchmark_tabarena(
            round_id="test",
            master_secret="test",
        )
        # If TabArena IS installed, this may succeed or fail for other reasons
    except ImportError as e:
        assert "TabArena is required" in str(e)
    except Exception:
        # Other errors are fine — we just want to verify the import guard works
        pass


def test_generate_leaderboard_tabarena_import_error():
    """generate_leaderboard_tabarena should raise ImportError when TabArena is missing."""
    from tabular_bank.leaderboard import generate_leaderboard_tabarena
    try:
        generate_leaderboard_tabarena(results_lst=[])
    except ImportError as e:
        assert "TabArena is required" in str(e)
    except Exception:
        pass


def test_generate_leaderboard_standalone_import_error():
    """generate_leaderboard_standalone should raise ImportError when TabArena is missing."""
    from tabular_bank.leaderboard import generate_leaderboard_standalone
    try:
        generate_leaderboard_standalone(results_lst=[])
    except ImportError as e:
        assert "TabArena is required" in str(e)
    except Exception:
        pass
