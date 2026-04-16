"""Official baseline model registry and execution helpers."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

from tabular_bank.generation.seed import get_default_cache_dir
from tabular_bank.rounds import (
    ROUND_MANIFEST_NAME,
    VALIDATION_REPORT_NAME,
    get_round_dir,
    validate_round,
    write_round_manifest,
    write_validation_report,
)
from tabular_bank.runner import run_benchmark


BASELINE_RUN_DIR = "official_baselines"


@dataclass(frozen=True)
class BaselineSpec:
    """A single official baseline method definition."""

    name: str
    track: str
    factory: Callable[[], object]
    dependencies: tuple[str, ...] = field(default_factory=tuple)
    description: str = ""

    def available(self) -> tuple[bool, str | None]:
        """Return availability status and the first missing dependency, if any."""
        for dep in self.dependencies:
            if importlib.util.find_spec(dep) is None:
                return False, dep
        return True, None


def get_official_baselines() -> list[BaselineSpec]:
    """Return the checked-in baseline registry for official runs.

    Baselines are organized into three tracks:

    - **classical**: Traditional ML models (tree ensembles, linear models,
      KNN, MLPs).  These are always available via scikit-learn.
    - **boosting**: Dedicated gradient boosting libraries (XGBoost, CatBoost,
      LightGBM).  Require optional dependencies.
    - **foundation**: Tabular foundation models (TabPFN, etc.).  Require
      optional dependencies.
    """
    return [
        # --- Classical track (sklearn, always available) ---
        BaselineSpec(
            name="RandomForest-clf",
            track="classical",
            factory=lambda: RandomForestClassifier(n_estimators=100, random_state=0),
            description="Random forest classification baseline.",
        ),
        BaselineSpec(
            name="GradientBoosting-clf",
            track="classical",
            factory=lambda: GradientBoostingClassifier(random_state=0),
            description="Gradient boosting classification baseline.",
        ),
        BaselineSpec(
            name="RandomForest-reg",
            track="classical",
            factory=lambda: RandomForestRegressor(n_estimators=100, random_state=0),
            description="Random forest regression baseline.",
        ),
        BaselineSpec(
            name="GradientBoosting-reg",
            track="classical",
            factory=lambda: GradientBoostingRegressor(random_state=0),
            description="Gradient boosting regression baseline.",
        ),
        BaselineSpec(
            name="LogisticRegression-clf",
            track="classical",
            factory=lambda: LogisticRegression(max_iter=1000, random_state=0),
            description="Logistic regression classification baseline.",
        ),
        BaselineSpec(
            name="Ridge-reg",
            track="classical",
            factory=lambda: Ridge(alpha=1.0),
            description="Ridge regression baseline.",
        ),
        BaselineSpec(
            name="KNN-clf",
            track="classical",
            factory=lambda: KNeighborsClassifier(n_neighbors=10),
            description="K-nearest neighbors classification baseline.",
        ),
        BaselineSpec(
            name="KNN-reg",
            track="classical",
            factory=lambda: KNeighborsRegressor(n_neighbors=10),
            description="K-nearest neighbors regression baseline.",
        ),
        BaselineSpec(
            name="MLP-clf",
            track="classical",
            factory=lambda: MLPClassifier(
                hidden_layer_sizes=(128, 64), max_iter=300, random_state=0,
            ),
            description="Multi-layer perceptron classification baseline.",
        ),
        BaselineSpec(
            name="MLP-reg",
            track="classical",
            factory=lambda: MLPRegressor(
                hidden_layer_sizes=(128, 64), max_iter=300, random_state=0,
            ),
            description="Multi-layer perceptron regression baseline.",
        ),
        # --- Boosting track (optional libraries) ---
        BaselineSpec(
            name="XGBoost-clf",
            track="boosting",
            factory=_make_xgboost_classifier,
            dependencies=("xgboost",),
            description="XGBoost classification baseline.",
        ),
        BaselineSpec(
            name="XGBoost-reg",
            track="boosting",
            factory=_make_xgboost_regressor,
            dependencies=("xgboost",),
            description="XGBoost regression baseline.",
        ),
        BaselineSpec(
            name="CatBoost-clf",
            track="boosting",
            factory=_make_catboost_classifier,
            dependencies=("catboost",),
            description="CatBoost classification baseline.",
        ),
        BaselineSpec(
            name="CatBoost-reg",
            track="boosting",
            factory=_make_catboost_regressor,
            dependencies=("catboost",),
            description="CatBoost regression baseline.",
        ),
        BaselineSpec(
            name="LightGBM-clf",
            track="boosting",
            factory=_make_lightgbm_classifier,
            dependencies=("lightgbm",),
            description="LightGBM classification baseline.",
        ),
        BaselineSpec(
            name="LightGBM-reg",
            track="boosting",
            factory=_make_lightgbm_regressor,
            dependencies=("lightgbm",),
            description="LightGBM regression baseline.",
        ),
        # --- Foundation track (tabular foundation models) ---
        BaselineSpec(
            name="TabPFN-clf",
            track="foundation",
            factory=_make_tabpfn_classifier,
            dependencies=("tabpfn",),
            description="TabPFN tabular foundation model (classification only).",
        ),
    ]


def run_official_baselines(
    round_id: str,
    cache_dir: str | Path | None = None,
    repeats: list[int] | None = None,
    folds: list[int] | None = None,
    tracks: set[str] | None = None,
) -> Path:
    """Run the checked-in official baseline registry for a round."""
    cache = Path(cache_dir) if cache_dir else get_default_cache_dir()
    round_dir = get_round_dir(round_id, cache)
    round_manifest_path = round_dir / ROUND_MANIFEST_NAME
    if not round_manifest_path.exists():
        round_manifest_path = write_round_manifest(round_id, cache_dir=cache)

    validation_report = validate_round(round_id, cache_dir=cache)
    write_validation_report(round_id, cache_dir=cache)
    if validation_report["status"] != "ok":
        raise ValueError(
            f"Round '{round_id}' failed validation. See {round_dir / VALIDATION_REPORT_NAME}"
        )

    with open(round_manifest_path) as f:
        round_manifest = json.load(f)

    total_tasks = int(round_manifest["n_scenarios"])
    baseline_dir = round_dir / BASELINE_RUN_DIR
    baseline_dir.mkdir(parents=True, exist_ok=True)

    registry = [
        spec for spec in get_official_baselines()
        if tracks is None or spec.track in tracks
    ]

    all_results: list[pd.DataFrame] = []
    method_records: list[dict] = []

    for spec in registry:
        available, missing_dep = spec.available()
        if not available:
            method_records.append({
                "method_name": spec.name,
                "track": spec.track,
                "status": "unavailable",
                "missing_dependency": missing_dep,
                "coverage_ratio": 0.0,
                "coverage_status": "none",
                "description": spec.description,
            })
            continue

        try:
            result = run_benchmark(
                models={spec.name: spec.factory()},
                round_id=round_id,
                cache_dir=cache,
                repeats=repeats,
                folds=folds,
                n_scenarios=total_tasks,
            )
        except Exception as exc:  # pragma: no cover - exercised via artifact status
            method_records.append({
                "method_name": spec.name,
                "track": spec.track,
                "status": "failed",
                "error": str(exc),
                "coverage_ratio": 0.0,
                "coverage_status": "none",
                "description": spec.description,
            })
            continue

        df = result.to_dataframe()
        if not df.empty:
            df["round_id"] = round_id
            df["track"] = spec.track
            df["method_name"] = spec.name
            all_results.append(df)

        task_count = int(df["task"].nunique()) if not df.empty else 0
        coverage_ratio = task_count / total_tasks if total_tasks else 0.0
        method_records.append({
            "method_name": spec.name,
            "track": spec.track,
            "status": "ok",
            "missing_dependency": None,
            "coverage_ratio": coverage_ratio,
            "coverage_status": _coverage_status(coverage_ratio),
            "tasks_evaluated": task_count,
            "description": spec.description,
        })

    results_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame(
        columns=["task", "model", "repeat", "fold", "metric", "score", "fit_time", "predict_time",
                 "round_id", "track", "method_name"]
    )
    results_path = baseline_dir / "results.csv"
    results_df.to_csv(results_path, index=False)

    methods_path = baseline_dir / "methods.json"
    with open(methods_path, "w") as f:
        json.dump(method_records, f, indent=2)

    summary_path = baseline_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(_build_summary(method_records), f, indent=2)

    run_manifest = {
        "schema_version": 1,
        "round_id": round_id,
        "run_id": _run_id(round_id),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "round_manifest": str(round_manifest_path.relative_to(round_dir)),
        "validation_report": VALIDATION_REPORT_NAME,
        "results_path": str(results_path.relative_to(round_dir)),
        "methods_path": str(methods_path.relative_to(round_dir)),
        "summary_path": str(summary_path.relative_to(round_dir)),
        "total_tasks": total_tasks,
    }
    run_manifest_path = baseline_dir / "run_manifest.json"
    with open(run_manifest_path, "w") as f:
        json.dump(run_manifest, f, indent=2)

    return run_manifest_path


def _build_summary(method_records: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "method_counts_by_status": _count_by(method_records, "status"),
        "method_counts_by_track": _count_by(method_records, "track"),
    }


def _count_by(records: list[dict], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(key))
        counts[value] = counts.get(value, 0) + 1
    return counts


def _coverage_status(ratio: float) -> str:
    if ratio <= 0:
        return "none"
    if ratio >= 1.0:
        return "full"
    return "partial"


def _run_id(round_id: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{round_id}-{timestamp}"


def _make_tabpfn_classifier():
    from tabpfn import TabPFNClassifier

    return TabPFNClassifier()


def _make_xgboost_classifier():
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=0, use_label_encoder=False, eval_metric="logloss",
    )


def _make_xgboost_regressor():
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.1, random_state=0,
    )


def _make_catboost_classifier():
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        iterations=100, depth=6, learning_rate=0.1,
        random_seed=0, verbose=0,
    )


def _make_catboost_regressor():
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        iterations=100, depth=6, learning_rate=0.1,
        random_seed=0, verbose=0,
    )


def _make_lightgbm_classifier():
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=0, verbose=-1,
    )


def _make_lightgbm_regressor():
    from lightgbm import LGBMRegressor

    return LGBMRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=0, verbose=-1,
    )
