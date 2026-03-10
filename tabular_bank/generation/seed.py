"""Cryptographic seed derivation for anti-contamination.

Uses HMAC-SHA256 to derive deterministic but unpredictable seeds from a
master secret + round identifier. Without the master secret, the datasets
for any round are computationally infeasible to predict.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from pathlib import Path


def derive_round_seed(master_secret: str, round_id: str) -> bytes:
    """Derive a 32-byte round seed from master secret + round identifier."""
    return hmac.new(
        master_secret.encode("utf-8"),
        round_id.encode("utf-8"),
        hashlib.sha256,
    ).digest()


def derive_dataset_seed(round_seed: bytes, dataset_index: int) -> int:
    """Derive a numpy-compatible integer seed for a specific dataset."""
    h = hmac.new(round_seed, f"dataset-{dataset_index}".encode("utf-8"), hashlib.sha256)
    return int.from_bytes(h.digest()[:4], "big")


def derive_split_seed(round_seed: bytes, dataset_index: int) -> int:
    """Derive a separate seed for split generation (independent of data seed)."""
    h = hmac.new(round_seed, f"split-{dataset_index}".encode("utf-8"), hashlib.sha256)
    return int.from_bytes(h.digest()[:4], "big")


def derive_feature_seed(round_seed: bytes, dataset_index: int) -> int:
    """Derive a seed for feature name/type generation."""
    h = hmac.new(round_seed, f"features-{dataset_index}".encode("utf-8"), hashlib.sha256)
    return int.from_bytes(h.digest()[:4], "big")


def derive_dag_seed(round_seed: bytes, dataset_index: int) -> int:
    """Derive a seed for DAG topology construction."""
    h = hmac.new(round_seed, f"dag-{dataset_index}".encode("utf-8"), hashlib.sha256)
    return int.from_bytes(h.digest()[:4], "big")


def get_master_secret(
    secret: str | None = None,
    cache_dir: str | Path | None = None,
) -> str:
    """Resolve the master secret from multiple sources.

    Priority order:
    1. Explicit `secret` parameter
    2. TABULAR_BANK_SECRET environment variable
    3. SYNTHETIC_TAB_SECRET environment variable (legacy alias)
    4. .secret file in cache_dir

    Raises ValueError if no secret can be found.
    """
    if secret is not None:
        return secret

    for env_var in ("TABULAR_BANK_SECRET", "SYNTHETIC_TAB_SECRET"):
        env_secret = os.environ.get(env_var)
        if env_secret is not None:
            return env_secret

    if cache_dir is not None:
        secret_file = Path(cache_dir) / ".secret"
        if secret_file.exists():
            return secret_file.read_text().strip()

    raise ValueError(
        "No master secret provided. Set TABULAR_BANK_SECRET env var, "
        "pass --secret on the CLI, or create a .secret file in the cache directory."
    )


def get_default_cache_dir() -> Path:
    """Return the default cache directory."""
    cache_dir = os.environ.get("TABULAR_BANK_CACHE") or os.environ.get("SYNTHETIC_TAB_CACHE")
    if cache_dir:
        return Path(cache_dir)
    return Path.home() / ".cache" / "tabular_bank"
