"""Tests for coverage-aware board artifacts."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd

from tabular_bank.baselines import run_official_baselines
from tabular_bank.board import build_board_artifacts, build_board_site
from tabular_bank.generation.generate import generate_all


def test_build_board_artifacts_excludes_partial_methods_from_overall():
    manifest = {
        "round_id": "round-1",
        "n_scenarios": 2,
        "problem_type_counts": {"binary": 2},
    }
    validation = {"status": "ok", "errors": [], "warnings": []}
    methods = [
        {
            "method_name": "full-classical",
            "track": "classical",
            "status": "ok",
            "coverage_ratio": 1.0,
            "coverage_status": "full",
        },
        {
            "method_name": "partial-foundation",
            "track": "foundation",
            "status": "ok",
            "coverage_ratio": 0.5,
            "coverage_status": "partial",
        },
    ]
    results = pd.DataFrame([
        {"model": "full-classical", "task": "t1", "score": 0.9},
        {"model": "full-classical", "task": "t2", "score": 0.8},
        {"model": "partial-foundation", "task": "t1", "score": 0.95},
    ])

    artifacts = build_board_artifacts(
        manifest=manifest,
        validation=validation,
        methods=methods,
        results=results,
    )

    assert [row["model"] for row in artifacts["overall"]["entries"]] == ["full-classical"]
    assert [row["rank"] for row in artifacts["overall"]["entries"]] == [1]
    assert [row["model"] for row in artifacts["tracks"]["foundation"]["entries"]] == ["partial-foundation"]


def test_build_board_site_writes_static_artifacts():
    with tempfile.TemporaryDirectory() as tmpdir:
        generate_all(
            master_secret="board-secret",
            round_id="board-round",
            n_scenarios=2,
            cache_dir=tmpdir,
        )
        run_official_baselines(
            round_id="board-round",
            cache_dir=tmpdir,
            repeats=[0],
            folds=[0],
            tracks={"classical"},
        )

        output_dir = f"{tmpdir}/site"
        build_board_site(
            round_id="board-round",
            cache_dir=tmpdir,
            output_dir=output_dir,
        )

        assert Path(output_dir, "index.html").exists()
        assert Path(output_dir, "rounds", "board-round", "index.html").exists()
        assert Path(output_dir, "rounds", "board-round", "leaderboard_overall.json").exists()

        with open(Path(output_dir, "rounds", "board-round", "round_summary.json")) as f:
            summary = json.load(f)
        assert summary["round_id"] == "board-round"


def test_build_board_site_preserves_multiple_rounds_in_index():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir, "site")

        for round_id in ("board-round-a", "board-round-b"):
            generate_all(
                master_secret=f"{round_id}-secret",
                round_id=round_id,
                n_scenarios=2,
                cache_dir=tmpdir,
            )
            run_official_baselines(
                round_id=round_id,
                cache_dir=tmpdir,
                repeats=[0],
                folds=[0],
                tracks={"classical"},
            )
            build_board_site(
                round_id=round_id,
                cache_dir=tmpdir,
                output_dir=output_dir,
            )

        with open(output_dir / "data" / "rounds.json") as f:
            rounds = json.load(f)

        assert [entry["round_id"] for entry in rounds] == ["board-round-a", "board-round-b"]
