"""Tests for official round manifests and validation."""

from __future__ import annotations

import json
import tempfile

from tabular_bank.context import TabularBankContext
from tabular_bank.generation.generate import generate_all
from tabular_bank.rounds import (
    ROUND_MANIFEST_NAME,
    build_round_manifest,
    validate_round,
)


SECRET = "test-secret-rounds"
ROUND = "round-test"


def test_generate_writes_round_manifest():
    with tempfile.TemporaryDirectory() as tmpdir:
        generate_all(
            master_secret=SECRET,
            round_id=ROUND,
            n_scenarios=2,
            cache_dir=tmpdir,
        )

        manifest_path = f"{tmpdir}/{ROUND}/{ROUND_MANIFEST_NAME}"
        with open(manifest_path) as f:
            manifest = json.load(f)

        assert manifest["round_id"] == ROUND
        assert manifest["n_scenarios"] == 2
        assert len(manifest["datasets"]) == 2


def test_round_manifest_is_deterministic():
    with tempfile.TemporaryDirectory() as tmpdir:
        generate_all(
            master_secret=SECRET,
            round_id=ROUND,
            n_scenarios=2,
            cache_dir=tmpdir,
        )

        manifest1 = build_round_manifest(ROUND, cache_dir=tmpdir)
        manifest2 = build_round_manifest(ROUND, cache_dir=tmpdir)

        assert manifest1 == manifest2


def test_validate_round_reports_success_for_clean_round():
    with tempfile.TemporaryDirectory() as tmpdir:
        generate_all(
            master_secret=SECRET,
            round_id=ROUND,
            n_scenarios=2,
            cache_dir=tmpdir,
        )

        report = validate_round(ROUND, cache_dir=tmpdir)

        assert report["status"] == "ok"
        assert report["errors"] == []
        assert report["summary"]["n_scenarios"] == 2


def test_context_task_metadata_ids_are_stable():
    with tempfile.TemporaryDirectory() as tmpdir:
        ctx1 = TabularBankContext(
            round_id=ROUND,
            master_secret=SECRET,
            cache_dir=tmpdir,
            n_scenarios=2,
        )
        ctx2 = TabularBankContext(
            round_id=ROUND,
            master_secret=SECRET,
            cache_dir=tmpdir,
            n_scenarios=2,
        )

        tids1 = list(ctx1.get_task_metadata()["tid"])
        tids2 = list(ctx2.get_task_metadata()["tid"])
        assert tids1 == tids2
