"""Tests for scenario sampling."""

import numpy as np

from tabular_bank.templates.scenarios import sample_scenario


def test_sample_scenario_respects_tight_sample_range():
    rng = np.random.default_rng(0)
    scenario = sample_scenario(
        rng,
        scenario_space={"n_samples_range": (1200, 1400)},
    )
    lo, hi = scenario["n_samples_range"]
    assert 1200 <= lo <= hi <= 1400


def test_sample_scenario_handles_degenerate_ranges():
    rng = np.random.default_rng(1)
    scenario = sample_scenario(
        rng,
        scenario_space={"n_samples_range": (2000, 2000), "n_features_range": (9, 9)},
    )
    assert scenario["n_samples_range"] == (2000, 2000)
    assert scenario["n_features_range"] == (9, 9)


def test_sample_scenario_difficulty_bounds():
    rng = np.random.default_rng(2)
    scenario = sample_scenario(rng)
    difficulty = scenario["difficulty"]
    assert 0.0 <= difficulty["temporal_prob"] <= 0.4
    assert 0.3 <= difficulty["max_autocorr"] <= 0.95
    assert 0.0 <= difficulty["root_correlation_strength"] <= 0.8
