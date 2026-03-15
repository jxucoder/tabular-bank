"""Tests for CLI command wiring."""

from __future__ import annotations

import tempfile
from pathlib import Path

from tabular_bank.cli import main


def test_cli_validate_run_baselines_and_build_board(capsys):
    with tempfile.TemporaryDirectory() as tmpdir:
        main([
            "generate",
            "--round", "cli-round",
            "--secret", "cli-secret",
            "--cache-dir", tmpdir,
            "--n-scenarios", "2",
        ])
        main([
            "validate",
            "--round", "cli-round",
            "--cache-dir", tmpdir,
        ])
        main([
            "run-baselines",
            "--round", "cli-round",
            "--cache-dir", tmpdir,
            "--track", "classical",
            "--repeat", "0",
            "--fold", "0",
        ])
        main([
            "build-board",
            "--round", "cli-round",
            "--cache-dir", tmpdir,
            "--output-dir", f"{tmpdir}/site",
        ])

        out = capsys.readouterr().out
        assert "Wrote validation report" in out
        assert "Wrote baseline run manifest" in out
        assert "Built static board site" in out
        assert Path(tmpdir, "site", "index.html").exists()
