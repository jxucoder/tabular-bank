"""Tests for the contamination analysis module."""

import numpy as np
import pandas as pd
import pytest

from tabular_bank.evaluation.contamination import analyze_contamination, ContaminationReport


def _make_scores(models, tasks, seed=42):
    """Create a random model-by-task score matrix."""
    rng = np.random.default_rng(seed)
    data = rng.uniform(0.5, 1.0, size=(len(models), len(tasks)))
    return pd.DataFrame(data, index=models, columns=tasks)


class TestAnalyzeContamination:

    def test_returns_report(self):
        models = ["A", "B", "C", "D"]
        tasks = ["t1", "t2", "t3"]
        tbank = _make_scores(models, tasks, seed=1)
        ref = _make_scores(models, tasks, seed=2)

        report = analyze_contamination(tbank, ref)
        assert isinstance(report, ContaminationReport)
        assert report.n_common_models == 4
        assert len(report.per_model) == 4

    def test_identical_scores_perfect_concordance(self):
        models = ["A", "B", "C", "D"]
        tasks = ["t1", "t2", "t3"]
        scores = _make_scores(models, tasks)

        report = analyze_contamination(scores, scores.copy())
        assert report.overall_kendall_tau == pytest.approx(1.0)
        assert report.overall_spearman_rho == pytest.approx(1.0)
        assert all(g.rank_gap == 0 for g in report.per_model)

    def test_msi_positive_when_prone_models_better_on_reference(self):
        models = ["Classical1", "Classical2", "Classical3", "FM"]
        tasks = ["t1", "t2", "t3"]
        # On tabular-bank, FM is weak (low scores)
        tbank = pd.DataFrame(
            {
                "t1": [0.9, 0.8, 0.7, 0.5],
                "t2": [0.85, 0.75, 0.65, 0.45],
                "t3": [0.88, 0.78, 0.68, 0.48],
            },
            index=models,
        )
        # On the reference, FM jumps to the top (memorization)
        ref = pd.DataFrame(
            {
                "t1": [0.9, 0.8, 0.7, 0.95],
                "t2": [0.85, 0.75, 0.65, 0.90],
                "t3": [0.88, 0.78, 0.68, 0.92],
            },
            index=models,
        )

        report = analyze_contamination(
            tbank, ref, memorization_prone=["FM"],
        )
        assert report.msi is not None
        # FM goes from rank 4 on tbank to rank 1 on ref -> positive MSI
        assert report.msi > 0

    def test_handles_disjoint_models(self):
        tbank = _make_scores(["A", "B"], ["t1"])
        ref = _make_scores(["C", "D"], ["t1"])

        report = analyze_contamination(tbank, ref)
        assert report.n_common_models == 0
        assert np.isnan(report.overall_kendall_tau)

    def test_summary_is_string(self):
        models = ["A", "B", "C"]
        tasks = ["t1", "t2"]
        tbank = _make_scores(models, tasks, seed=1)
        ref = _make_scores(models, tasks, seed=2)

        report = analyze_contamination(tbank, ref)
        summary = report.summary()
        assert isinstance(summary, str)
        assert "Contamination Analysis" in summary
