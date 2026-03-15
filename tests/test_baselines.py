"""Tests for official baseline execution artifacts."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd

from tabular_bank.baselines import run_official_baselines
from tabular_bank.generation.generate import generate_all


def test_run_official_baselines_writes_artifacts_for_classical_track():
    with tempfile.TemporaryDirectory() as tmpdir:
        generate_all(
            master_secret="baseline-secret",
            round_id="baseline-round",
            n_scenarios=2,
            cache_dir=tmpdir,
        )

        run_manifest_path = run_official_baselines(
            round_id="baseline-round",
            cache_dir=tmpdir,
            repeats=[0],
            folds=[0],
            tracks={"classical"},
        )

        baseline_dir = Path(run_manifest_path).parent
        assert (baseline_dir / "results.csv").exists()
        assert (baseline_dir / "methods.json").exists()
        assert (baseline_dir / "summary.json").exists()

        results = pd.read_csv(baseline_dir / "results.csv")
        with open(baseline_dir / "methods.json") as f:
            methods = json.load(f)

        assert not results.empty
        assert set(results["track"]) == {"classical"}
        assert all(method["track"] == "classical" for method in methods)
        assert all(method["status"] == "ok" for method in methods)
