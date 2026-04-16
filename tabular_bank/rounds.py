"""Official round manifests and validation utilities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from tabular_bank import __version__, default_metric as _default_metric
from tabular_bank.generation.seed import get_default_cache_dir
from tabular_bank.tasks import SyntheticTask, load_tasks_from_cache


ROUND_MANIFEST_NAME = "round_manifest.json"
VALIDATION_REPORT_NAME = "validation_report.json"


def get_round_dir(round_id: str, cache_dir: str | Path | None = None) -> Path:
    """Return the cache directory for a round."""
    cache = Path(cache_dir) if cache_dir else get_default_cache_dir()
    return cache / round_id


def write_round_manifest(
    round_id: str,
    cache_dir: str | Path | None = None,
) -> Path:
    """Build and persist the authoritative manifest for a generated round."""
    round_dir = get_round_dir(round_id, cache_dir)
    manifest = build_round_manifest(round_id, cache_dir=cache_dir)
    path = round_dir / ROUND_MANIFEST_NAME
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path


def build_round_manifest(
    round_id: str,
    cache_dir: str | Path | None = None,
) -> dict:
    """Build a deterministic manifest describing a generated round."""
    round_dir = get_round_dir(round_id, cache_dir)
    round_meta = _load_round_metadata(round_dir)
    scenario_ids = _get_scenario_ids(round_meta)
    tasks = load_tasks_from_cache(round_dir, scenario_ids=scenario_ids)
    if not tasks:
        raise FileNotFoundError(f"No generated datasets found for round '{round_id}' in {round_dir}")

    dataset_entries = []
    metrics = {}
    repeat_counts = set()
    fold_counts = set()
    problem_counts: dict[str, int] = {}

    for task in tasks:
        scenario_id = str(task.metadata.get("scenario_id", task.name))
        ds_dir = round_dir / scenario_id
        repeat_counts.add(task.n_repeats)
        fold_counts.add(task.n_folds)
        metric = _default_metric(task.problem_type)
        metrics[task.problem_type] = metric
        problem_counts[task.problem_type] = problem_counts.get(task.problem_type, 0) + 1

        dataset_entries.append({
            "dataset_id": task.name,
            "scenario_id": scenario_id,
            "problem_type": task.problem_type,
            "metric": metric,
            "target_name": task.target,
            "n_samples": task.n_samples,
            "n_features": task.n_features,
            "n_repeats": task.n_repeats,
            "n_folds": task.n_folds,
            "artifacts": {
                "dataset_csv": _artifact_record(ds_dir / "dataset.csv", round_dir),
                "metadata_json": _artifact_record(ds_dir / "metadata.json", round_dir),
                "splits_json": _artifact_record(ds_dir / "splits.json", round_dir),
            },
        })

    return {
        "schema_version": 1,
        "round_id": round_id,
        "generation_version": __version__,
        "n_scenarios": len(dataset_entries),
        "scenario_ids": [entry["scenario_id"] for entry in dataset_entries],
        "problem_type_counts": problem_counts,
        "evaluation_protocol": {
            "n_repeats": _single_value_or_list(repeat_counts),
            "n_folds": _single_value_or_list(fold_counts),
            "metrics_by_problem_type": metrics,
        },
        "datasets": dataset_entries,
        "artifacts": {
            "round_metadata_json": _artifact_record(round_dir / "round_metadata.json", round_dir),
        },
        "publication": {
            "published_at": None,
        },
    }


def write_validation_report(
    round_id: str,
    cache_dir: str | Path | None = None,
) -> Path:
    """Validate a round and persist the resulting report."""
    round_dir = get_round_dir(round_id, cache_dir)
    report = validate_round(round_id, cache_dir=cache_dir)
    path = round_dir / VALIDATION_REPORT_NAME
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path


def validate_round(
    round_id: str,
    cache_dir: str | Path | None = None,
) -> dict:
    """Validate generated round artifacts and summarize their shape."""
    round_dir = get_round_dir(round_id, cache_dir)
    round_meta = _load_round_metadata(round_dir)
    scenario_ids = _get_scenario_ids(round_meta)
    tasks = load_tasks_from_cache(round_dir, scenario_ids=scenario_ids)
    if not tasks:
        raise FileNotFoundError(f"No generated datasets found for round '{round_id}' in {round_dir}")

    errors: list[str] = []
    warnings: list[str] = []
    problem_counts: dict[str, int] = {}
    repeat_counts = set()
    fold_counts = set()

    for task in tasks:
        scenario_id = str(task.metadata.get("scenario_id", task.name))
        ds_dir = round_dir / scenario_id
        complete_marker = ds_dir / ".complete"
        if not complete_marker.exists():
            errors.append(f"{scenario_id}: missing .complete marker")

        missing_files = [
            name for name in ("dataset.csv", "metadata.json", "splits.json")
            if not (ds_dir / name).exists()
        ]
        if missing_files:
            errors.append(f"{scenario_id}: missing required files: {', '.join(missing_files)}")
            continue

        problem_counts[task.problem_type] = problem_counts.get(task.problem_type, 0) + 1
        repeat_counts.add(task.n_repeats)
        fold_counts.add(task.n_folds)

        if task.n_samples <= 0:
            errors.append(f"{scenario_id}: empty dataset")
        if task.n_features <= 0:
            errors.append(f"{scenario_id}: no feature columns")

        target_unique = int(pd.Series(task.dataset[task.target]).nunique(dropna=True))
        if task.problem_type in {"binary", "multiclass"} and target_unique < 2:
            errors.append(f"{scenario_id}: target has fewer than 2 classes")
        if task.problem_type == "binary" and target_unique != 2:
            warnings.append(f"{scenario_id}: expected 2 classes, found {target_unique}")
        if task.problem_type == "forecasting" and target_unique < 5:
            warnings.append(f"{scenario_id}: forecasting target has very low cardinality ({target_unique})")

        for repeat, folds in task.splits.items():
            if not folds:
                errors.append(f"{scenario_id}: repeat {repeat} has no folds")
                continue
            for fold, (train_idx, test_idx) in folds.items():
                train_set = set(int(i) for i in train_idx)
                test_set = set(int(i) for i in test_idx)
                if not train_set:
                    errors.append(f"{scenario_id}: repeat {repeat} fold {fold} has empty train split")
                if not test_set:
                    errors.append(f"{scenario_id}: repeat {repeat} fold {fold} has empty test split")
                if train_set & test_set:
                    errors.append(f"{scenario_id}: repeat {repeat} fold {fold} overlaps train/test indices")
                if min(train_set | test_set, default=0) < 0:
                    errors.append(f"{scenario_id}: repeat {repeat} fold {fold} contains negative indices")
                if max(train_set | test_set, default=-1) >= task.n_samples:
                    errors.append(f"{scenario_id}: repeat {repeat} fold {fold} contains out-of-range indices")

    manifest = build_round_manifest(round_id, cache_dir=cache_dir)
    return {
        "schema_version": 1,
        "round_id": round_id,
        "status": "ok" if not errors else "error",
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "n_scenarios": len(tasks),
            "problem_type_counts": problem_counts,
            "n_repeats": _single_value_or_list(repeat_counts),
            "n_folds": _single_value_or_list(fold_counts),
            "manifest_sha256": _sha256_bytes(json.dumps(manifest, sort_keys=True).encode("utf-8")),
        },
    }


def _load_round_metadata(round_dir: Path) -> dict:
    meta_path = round_dir / "round_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing round metadata at {meta_path}")
    with open(meta_path) as f:
        return json.load(f)


def _get_scenario_ids(round_meta: dict) -> list[str] | None:
    scenario_ids = round_meta.get("scenario_ids")
    if scenario_ids:
        return list(scenario_ids)
    n_datasets = int(round_meta.get("n_datasets", 0))
    if n_datasets > 0:
        return [f"sampled_{i}" for i in range(n_datasets)]
    return None


def _artifact_record(path: Path, round_dir: Path) -> dict:
    return {
        "path": str(path.relative_to(round_dir)),
        "sha256": _sha256_file(path),
    }


def _single_value_or_list(values: set[int]) -> int | list[int] | None:
    if not values:
        return None
    if len(values) == 1:
        return next(iter(values))
    return sorted(values)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
