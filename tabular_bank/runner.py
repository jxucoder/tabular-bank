"""Benchmark runner — wraps TabArena's evaluation pipeline.

Provides both a standalone evaluation mode (no TabArena dependency) and
full TabArena integration mode for official benchmarking.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import is_classifier, is_regressor
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_squared_error,
    roc_auc_score,
)

from tabular_bank.context import TabularBankContext
from tabular_bank.tasks import SyntheticTask

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Result of evaluating a model on a single task split."""

    task_name: str
    model_name: str
    repeat: int
    fold: int
    metric_name: str
    metric_value: float
    fit_time: float
    predict_time: float


@dataclass
class BenchmarkResult:
    """Aggregated results from a full benchmark run."""

    results: list[TaskResult] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a DataFrame."""
        return pd.DataFrame([
            {
                "task": r.task_name,
                "model": r.model_name,
                "repeat": r.repeat,
                "fold": r.fold,
                "metric": r.metric_name,
                "score": r.metric_value,
                "fit_time": r.fit_time,
                "predict_time": r.predict_time,
            }
            for r in self.results
        ])

    def summary(self) -> pd.DataFrame:
        """Compute mean score per model per task."""
        df = self.to_dataframe()
        if df.empty:
            return df
        return (
            df.groupby(["model", "task", "metric"])["score"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )


def evaluate_model(
    model,
    task: SyntheticTask,
    model_name: str | None = None,
    repeats: list[int] | None = None,
    folds: list[int] | None = None,
) -> list[TaskResult]:
    """Evaluate a sklearn-compatible model on a single task.

    Args:
        model: A model with fit() and predict()/predict_proba() methods.
        task: The benchmark task.
        model_name: Name for results. Defaults to class name.
        repeats: Which repeats to run. Defaults to all.
        folds: Which folds to run. Defaults to all.

    Returns:
        List of TaskResult for each repeat/fold combination.
    """
    if model_name is None:
        model_name = type(model).__name__

    if repeats is None:
        repeats = list(range(task.n_repeats))
    if folds is None:
        folds = list(range(task.n_folds))

    results = []
    metric_name = _default_metric(task.problem_type)

    for repeat in repeats:
        for fold in folds:
            train_df, test_df = task.get_split(repeat, fold)

            X_train = train_df[task.feature_names]
            y_train = train_df[task.target]
            X_test = test_df[task.feature_names]
            y_test = test_df[task.target]

            # Encode categoricals for sklearn models
            X_train_enc, X_test_enc = _encode_features(X_train, X_test)

            # Fit
            model_copy = copy.deepcopy(model)
            t0 = time.time()
            model_copy.fit(X_train_enc, y_train)
            fit_time = time.time() - t0

            # Predict
            t0 = time.time()
            score = _evaluate_metric(model_copy, X_test_enc, y_test, task.problem_type, metric_name)
            predict_time = time.time() - t0

            results.append(TaskResult(
                task_name=task.name,
                model_name=model_name,
                repeat=repeat,
                fold=fold,
                metric_name=metric_name,
                metric_value=score,
                fit_time=fit_time,
                predict_time=predict_time,
            ))

            logger.debug(
                "%s on %s repeat=%d fold=%d: %s=%.4f",
                model_name, task.name, repeat, fold, metric_name, score,
            )

    return results


def run_benchmark(
    models: dict[str, object],
    round_id: str = "round-001",
    master_secret: str | None = None,
    cache_dir: str | Path | None = None,
    repeats: list[int] | None = None,
    folds: list[int] | None = None,
    n_scenarios: int = 10,
) -> BenchmarkResult:
    """Run a full benchmark with multiple models across all tasks.

    Models are routed to compatible tasks by problem type. A model is
    considered a classifier if it has a ``_estimator_type`` attribute set
    to ``"classifier"`` (the scikit-learn convention), and a regressor if
    it is set to ``"regressor"``.  Models without that attribute are
    evaluated on every task.

    Args:
        models: Dict of model_name -> sklearn-compatible model instance.
        round_id: Benchmark round identifier.
        master_secret: Secret for dataset generation.
        cache_dir: Cache directory.
        repeats: Which repeats to run. Defaults to all.
        folds: Which folds to run. Defaults to all.
        n_scenarios: Expected number of scenarios in the round cache.

    Returns:
        BenchmarkResult with all results.
    """
    ctx = TabularBankContext(
        round_id=round_id,
        master_secret=master_secret,
        cache_dir=cache_dir,
        n_scenarios=n_scenarios,
    )

    benchmark = BenchmarkResult()

    for task in ctx.get_tasks():
        logger.info("Running task: %s (type=%s)", task.name, task.problem_type)
        for model_name, model in models.items():
            if not _is_compatible(model, task.problem_type):
                logger.info(
                    "  Skipping %s — incompatible with %s task",
                    model_name, task.problem_type,
                )
                continue
            logger.info("  Model: %s", model_name)
            task_results = evaluate_model(
                model=model,
                task=task,
                model_name=model_name,
                repeats=repeats,
                folds=folds,
            )
            benchmark.results.extend(task_results)

    return benchmark


# Task types that are classification problems
_CLASSIFICATION_TYPES = {"binary", "multiclass"}
# Task types that are regression problems
_REGRESSION_TYPES = {"regression"}


def _is_compatible(model: object, problem_type: str) -> bool:
    """Check whether a model is compatible with a task's problem type."""
    try:
        if is_classifier(model):
            return problem_type in _CLASSIFICATION_TYPES
        if is_regressor(model):
            return problem_type in _REGRESSION_TYPES
    except (AttributeError, TypeError):
        pass
    # Not a recognized sklearn estimator — assume the caller knows what
    # they're doing and allow it through.
    return True


def run_benchmark_tabarena(
    round_id: str = "round-001",
    master_secret: str | None = None,
    cache_dir: str | Path | None = None,
    experiments: list | None = None,
    expname: str | None = None,
    datasets: list[str] | None = None,
    folds: list[int] | None = None,
    ignore_cache: bool = False,
    n_scenarios: int = 10,
):
    """Run benchmark using TabArena's full pipeline.

    Generates contamination-proof datasets and evaluates them using
    TabArena's ``ExperimentBatchRunner`` with 8-fold bagging, standardized
    HPO, and full result serialization.

    Args:
        round_id: Benchmark round identifier.
        master_secret: Secret for dataset generation.
        cache_dir: Cache directory for generated datasets.
        experiments: List of TabArena experiment objects (e.g.,
            ``AGModelBagExperiment``).  If None, a default set of
            LightGBM + RandomForest experiments is used.
        expname: Directory path for saving experiment artifacts.
            Defaults to ``<cache_dir>/<round_id>/tabarena_experiments``.
        datasets: Subset of dataset names to evaluate. Defaults to all.
        folds: Which folds to evaluate. Defaults to ``[0]`` (first fold).
        ignore_cache: If True, re-run experiments from scratch.
        n_scenarios: Number of scenarios to generate.

    Returns:
        A list of raw result dicts from ``ExperimentBatchRunner.run()``,
        suitable for passing to ``EndToEnd.from_raw()`` for leaderboard
        generation.

    Requires TabArena to be installed (pip install tabular-bank[benchmark]).
    """
    try:
        from tabarena.benchmark.experiment import (
            AGModelBagExperiment,
            ExperimentBatchRunner,
        )
    except ImportError:
        raise ImportError(
            "TabArena is required for this operation. "
            "Install with: pip install tabular-bank[benchmark]"
        )

    ctx = TabularBankContext(
        round_id=round_id,
        master_secret=master_secret,
        cache_dir=cache_dir,
        n_scenarios=n_scenarios,
    )

    # Convert tasks to TabArena format
    user_tasks = ctx.get_tabarena_tasks()
    task_metadata = ctx.get_task_metadata()

    # Default experiments: LightGBM and RandomForest with 8-fold bagging
    if experiments is None:
        try:
            from autogluon.tabular.models import LGBModel, RFModel
            experiments = [
                AGModelBagExperiment(
                    name="LightGBM_BAG_L1",
                    model_cls=LGBModel,
                    model_hyperparameters={},
                    num_bag_folds=8,
                    time_limit=3600,
                ),
                AGModelBagExperiment(
                    name="RandomForest_BAG_L1",
                    model_cls=RFModel,
                    model_hyperparameters={},
                    num_bag_folds=8,
                    time_limit=3600,
                ),
            ]
        except ImportError:
            raise ImportError(
                "AutoGluon is required for default experiments. "
                "Install with: pip install tabular-bank[benchmark] "
                "or pass custom experiments via the `experiments` parameter."
            )

    if expname is None:
        exp_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "tabular_bank"
        expname = str(exp_dir / round_id / "tabarena_experiments")

    if datasets is None:
        datasets = [ut.task_name for ut in user_tasks]

    if folds is None:
        folds = [0]

    exp_batch_runner = ExperimentBatchRunner(
        expname=expname,
        task_metadata=task_metadata,
    )

    results_lst = exp_batch_runner.run(
        datasets=datasets,
        folds=folds,
        methods=experiments,
        ignore_cache=ignore_cache,
    )

    logger.info(
        "TabArena benchmark complete: %d result entries for %d datasets",
        len(results_lst), len(datasets),
    )

    return results_lst


def _default_metric(problem_type: str) -> str:
    """Return the default metric for a problem type."""
    if problem_type == "binary":
        return "roc_auc"
    elif problem_type == "multiclass":
        return "log_loss"
    else:
        return "rmse"


def _evaluate_metric(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    problem_type: str,
    metric_name: str,
) -> float:
    """Compute a metric for a fitted model.

    All returned scores follow the "higher is better" convention so that
    leaderboard ranking is consistent across metric types.  For metrics
    that are naturally "lower is better" (log_loss, RMSE) the value is
    negated before being returned.
    """
    if metric_name == "roc_auc":
        # ROC-AUC is undefined when the evaluation split has only one class.
        # Fall back to accuracy so benchmarking continues deterministically.
        if pd.Series(y_test).nunique(dropna=True) < 2:
            import warnings
            warnings.warn(
                "ROC-AUC is undefined for a single-class test split — "
                "falling back to accuracy. This may indicate a data "
                "distribution issue (e.g., extreme class imbalance or "
                "too-small test fold).",
                stacklevel=2,
            )
            y_pred = model.predict(X_test)
            return float(accuracy_score(y_test, y_pred))
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            if y_proba.ndim == 2:
                y_proba = y_proba[:, 1]
            return float(roc_auc_score(y_test, y_proba))
        else:
            y_pred = model.predict(X_test)
            return float(roc_auc_score(y_test, y_pred))
    elif metric_name == "log_loss":
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            return float(-log_loss(y_test, y_proba))  # Negate so higher is better
        else:
            # log_loss requires probabilities; hard labels from predict()
            # will crash for multiclass targets.  Fall back to accuracy.
            import warnings
            warnings.warn(
                "Model lacks predict_proba — falling back from log_loss "
                "to accuracy. Scores may not be comparable with "
                "probabilistic models.",
                stacklevel=2,
            )
            y_pred = model.predict(X_test)
            return float(accuracy_score(y_test, y_pred))
    elif metric_name == "rmse":
        y_pred = model.predict(X_test)
        return float(-np.sqrt(mean_squared_error(y_test, y_pred)))  # Negate so higher is better
    elif metric_name == "accuracy":
        y_pred = model.predict(X_test)
        return float(accuracy_score(y_test, y_pred))
    else:
        raise ValueError(f"Unknown metric: {metric_name}")


def _encode_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simple categorical encoding for sklearn models.

    Encodes categoricals with train-set label maps and imputes numeric NaN
    values with the train median so pure-numeric datasets with missingness
    remain compatible with basic sklearn estimators.

    Limitation: categorical columns are integer-label-encoded, which imposes
    an artificial ordinal relationship.  This is fine for tree-based models
    but can bias linear or distance-based models.  Callers needing
    one-hot encoding should pre-process their data before calling
    ``evaluate_model``.
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        categories = sorted(X_train[col].dropna().unique())
        cat_map = {c: i for i, c in enumerate(categories)}
        X_train[col] = X_train[col].map(cat_map).fillna(-1).astype(int)
        X_test[col] = X_test[col].map(cat_map).fillna(-1).astype(int)

    # Fill remaining NaN in numeric columns with the train-set median.
    numeric_cols = X_train.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        if X_train[col].isna().any() or X_test[col].isna().any():
            fill_val = X_train[col].median()
            if pd.isna(fill_val):
                fill_val = 0.0
            X_train[col] = X_train[col].fillna(fill_val)
            X_test[col] = X_test[col].fillna(fill_val)

    return X_train, X_test
