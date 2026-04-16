"""Forecasting baseline models with sklearn-compatible interface.

These models operate on tabular-formatted forecasting tasks (lagged features
as columns) and implement fit()/predict() so they plug into the standard
runner pipeline.  They serve as sanity-check baselines — any model that
cannot beat LastValue or RollingMean is not learning from the features.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

try:
    from sklearn.utils._tags import Tags, RegressorTags, TargetTags

    def _regressor_tags():
        return Tags(
            estimator_type="regressor",
            regressor_tags=RegressorTags(),
            target_tags=TargetTags(required=True),
        )

    _HAS_NEW_TAGS = True
except ImportError:
    _HAS_NEW_TAGS = False


class _ForecastingBase(BaseEstimator, RegressorMixin):
    """Base class that ensures sklearn recognises these as regressors."""

    _estimator_type = "regressor"

    if _HAS_NEW_TAGS:
        def __sklearn_tags__(self):
            return _regressor_tags()


class LastValue(_ForecastingBase):
    """Predict the most recent lagged target value.

    Looks for a column named ``target_lag1`` (or the first ``*_lag1``
    column) and returns it as the prediction.  This is the simplest
    possible forecasting baseline.
    """

    def __init__(self):
        self.lag_col_idx_: int | None = None

    def fit(self, X, y=None):
        # Find the lag-1 target column
        if hasattr(X, "columns"):
            cols = list(X.columns)
        else:
            cols = [f"col_{i}" for i in range(X.shape[1])]

        for i, col in enumerate(cols):
            if col == "target_lag1":
                self.lag_col_idx_ = i
                return self
        # Fall back to first *_lag1 column
        for i, col in enumerate(cols):
            if col.endswith("_lag1"):
                self.lag_col_idx_ = i
                return self
        # No lag column found — use the last column
        self.lag_col_idx_ = X.shape[1] - 1
        return self

    def predict(self, X):
        if hasattr(X, "iloc"):
            return np.asarray(X.iloc[:, self.lag_col_idx_], dtype=float)
        return np.asarray(X[:, self.lag_col_idx_], dtype=float)


class RollingMean(_ForecastingBase):
    """Predict the mean of all available lagged target values.

    Averages all ``target_lag*`` columns to produce the forecast.
    """

    def __init__(self):
        self.lag_col_indices_: list[int] = []

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            cols = list(X.columns)
        else:
            cols = [f"col_{i}" for i in range(X.shape[1])]

        self.lag_col_indices_ = [
            i for i, col in enumerate(cols) if "target_lag" in col
        ]
        if not self.lag_col_indices_:
            # Fall back: use all *_lag* columns
            self.lag_col_indices_ = [
                i for i, col in enumerate(cols) if "_lag" in col
            ]
        if not self.lag_col_indices_:
            self.lag_col_indices_ = list(range(X.shape[1]))
        return self

    def predict(self, X):
        if hasattr(X, "iloc"):
            subset = X.iloc[:, self.lag_col_indices_].values.astype(float)
        else:
            subset = np.asarray(X[:, self.lag_col_indices_], dtype=float)
        return np.nanmean(subset, axis=1)


class LinearTrend(_ForecastingBase):
    """Fit a linear trend through the lagged target values and extrapolate.

    For each row, fits a line through [target_lag_k, ..., target_lag_1]
    and extrapolates one step ahead.  This captures simple trends that
    LastValue and RollingMean miss.
    """

    def __init__(self):
        self.lag_col_indices_: list[int] = []

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            cols = list(X.columns)
        else:
            cols = [f"col_{i}" for i in range(X.shape[1])]

        # Find target_lag columns sorted by lag number
        lag_cols = []
        for i, col in enumerate(cols):
            if "target_lag" in col:
                # Extract lag number
                parts = col.split("lag")
                if len(parts) == 2 and parts[1].isdigit():
                    lag_cols.append((int(parts[1]), i))

        if lag_cols:
            # Sort by lag (highest lag = oldest value comes first)
            lag_cols.sort(key=lambda x: -x[0])
            self.lag_col_indices_ = [idx for _, idx in lag_cols]
        else:
            # Fall back to all _lag columns
            self.lag_col_indices_ = [
                i for i, col in enumerate(cols) if "_lag" in col
            ]
        return self

    def predict(self, X):
        if hasattr(X, "iloc"):
            data = X.iloc[:, self.lag_col_indices_].values.astype(float)
        else:
            data = np.asarray(X[:, self.lag_col_indices_], dtype=float)

        n_rows, n_lags = data.shape
        predictions = np.zeros(n_rows)

        if n_lags < 2:
            # Can't fit a trend with < 2 points; fall back to last value or zero
            if n_lags == 1:
                return data[:, 0]
            return predictions

        # Time axis: 0 = oldest lag, n_lags-1 = most recent
        t = np.arange(n_lags, dtype=float)
        t_next = float(n_lags)  # extrapolation point

        for i in range(n_rows):
            row = data[i]
            valid = ~np.isnan(row)
            if valid.sum() < 2:
                predictions[i] = np.nanmean(row)
                continue
            # Simple least-squares linear fit
            t_valid = t[valid]
            y_valid = row[valid]
            slope, intercept = np.polyfit(t_valid, y_valid, 1)
            predictions[i] = slope * t_next + intercept

        return predictions
