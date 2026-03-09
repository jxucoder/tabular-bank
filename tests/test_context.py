"""Tests for SyntheticTabContext."""

import tempfile
from pathlib import Path

import pandas as pd

from synthetic_tab.context import SyntheticTabContext


SECRET = "test-secret-for-context"
ROUND = "test-round"


def test_context_auto_generates():
    """Context auto-generates datasets when they don't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ctx = SyntheticTabContext(
            round_id=ROUND,
            master_secret=SECRET,
            cache_dir=tmpdir,
            auto_generate=True,
        )
        tasks = ctx.get_tasks()
        assert len(tasks) == 5


def test_context_get_datasets():
    """get_datasets returns list of names."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ctx = SyntheticTabContext(
            round_id=ROUND,
            master_secret=SECRET,
            cache_dir=tmpdir,
        )
        names = ctx.get_datasets()
        assert len(names) == 5
        assert all(isinstance(n, str) for n in names)


def test_context_get_metadata():
    """get_metadata returns a DataFrame with expected columns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ctx = SyntheticTabContext(
            round_id=ROUND,
            master_secret=SECRET,
            cache_dir=tmpdir,
        )
        meta = ctx.get_metadata()
        assert isinstance(meta, pd.DataFrame)
        assert len(meta) == 5
        assert "dataset" in meta.columns
        assert "problem_type" in meta.columns
        assert "n_samples" in meta.columns


def test_context_get_task_by_name():
    """Can retrieve specific tasks by name."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ctx = SyntheticTabContext(
            round_id=ROUND,
            master_secret=SECRET,
            cache_dir=tmpdir,
        )
        names = ctx.get_datasets()
        task = ctx.get_task(names[0])
        assert task.name == names[0]


def test_context_skips_regeneration():
    """Context doesn't regenerate if data already exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # First generation
        ctx1 = SyntheticTabContext(
            round_id=ROUND, master_secret=SECRET, cache_dir=tmpdir,
        )
        tasks1 = ctx1.get_tasks()

        # Second load (should skip generation)
        ctx2 = SyntheticTabContext(
            round_id=ROUND, master_secret=SECRET, cache_dir=tmpdir,
        )
        tasks2 = ctx2.get_tasks()

        assert len(tasks1) == len(tasks2)
        assert [t.name for t in tasks1] == [t.name for t in tasks2]
