"""Tests for seed derivation system."""

from tabular_bank.generation.seed import (
    derive_dag_seed,
    derive_dataset_seed,
    derive_feature_seed,
    derive_round_seed,
    derive_split_seed,
)


def test_round_seed_deterministic():
    """Same inputs always produce same round seed."""
    s1 = derive_round_seed("secret", "round-001")
    s2 = derive_round_seed("secret", "round-001")
    assert s1 == s2


def test_round_seed_different_secrets():
    """Different secrets produce different round seeds."""
    s1 = derive_round_seed("secret-a", "round-001")
    s2 = derive_round_seed("secret-b", "round-001")
    assert s1 != s2


def test_round_seed_different_rounds():
    """Different round IDs produce different round seeds."""
    s1 = derive_round_seed("secret", "round-001")
    s2 = derive_round_seed("secret", "round-002")
    assert s1 != s2


def test_dataset_seeds_independent():
    """Different dataset indices produce different seeds."""
    rs = derive_round_seed("secret", "round-001")
    seeds = [derive_dataset_seed(rs, i) for i in range(10)]
    assert len(set(seeds)) == 10


def test_seed_types_independent():
    """Feature, DAG, dataset, and split seeds are all different for same index."""
    rs = derive_round_seed("secret", "round-001")
    idx = 0
    seeds = {
        derive_feature_seed(rs, idx),
        derive_dag_seed(rs, idx),
        derive_dataset_seed(rs, idx),
        derive_split_seed(rs, idx),
    }
    assert len(seeds) == 4


def test_round_seed_is_32_bytes():
    """Round seed should be 32 bytes (SHA-256 output)."""
    rs = derive_round_seed("secret", "round-001")
    assert len(rs) == 32
