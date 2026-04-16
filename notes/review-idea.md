# Review: Evaluation Framework, Forecasting, and Scaling Infrastructure

**Branch:** `claude/review-idea-T1POB`
**Reviewer:** Claude
**Date:** 2026-04-16
**Test suite:** 147/147 passing

---

## Overview

This branch adds ~3,500 lines across 34 files, introducing:

1. **Contamination analysis** — detect memorization by comparing rankings on tabular-bank vs. a reference benchmark
2. **Ground-truth feature importance** — leverage the known causal DAG to evaluate model explanations
3. **Dimension-aware diagnostics** — profile model performance along difficulty axes (noise, nonlinearity, confounders, etc.)
4. **IRT analysis** — 2-parameter Item Response Theory for per-task difficulty/discrimination estimation
5. **Scaling analysis** — bootstrap-based ranking stability and minimum-scenario-count determination
6. **Distribution shift** — covariate shift, concept drift, and temporal splits for robustness testing
7. **Forecasting tasks** — time-series generation from the same DAG framework with lag features and temporal splits
8. **Expanded baselines** — registry-based baseline system with classical, boosting, forecasting, and foundation tracks
9. **Website updates** — new sections for baselines, contamination, and scaling documentation
10. **README overhaul** — comprehensive documentation of all new APIs

The scope is ambitious and well-motivated. The evaluation framework fills genuine gaps — most synthetic benchmark papers don't ship tools to evaluate *benchmark quality itself*. The contamination analysis is directly aligned with the project's core thesis.

---

## Strengths

### Conceptually sound architecture
The evaluation modules are well-separated: contamination, diagnostics, feature_importance, scaling, and meta_eval each address a distinct question about benchmark quality. The dataclass-based result types (`ContaminationReport`, `DiagnosticReport`, `IRTResult`, etc.) give a clean, inspectable API.

### Good use of the procedural generation advantage
Ground-truth feature importance is a killer feature — because you *know* the DAG, you can compute exact importance and measure how well models recover it. This is something no real-world benchmark can do. The `FeatureImportanceProfile` with path-based importance, mechanism tracking, and noise detection is well-designed.

### Forecasting from the same DAG framework
Extending the existing generation engine to produce forecasting tasks (via AR(1) autocorrelation, lagged features, and temporal splits) is a clean design that avoids building a separate generation pipeline. The sklearn-compatible baseline models (LastValue, RollingMean, LinearTrend) follow good conventions.

### Comprehensive test coverage
147 tests all pass. New test files cover contamination analysis, diagnostics, feature importance, forecasting, IRT, scaling, and distribution shift. Test data is generated synthetically with helpers, avoiding coupling to real data.

### Clean public API
The `evaluation/__init__.py` exports are well-curated. The README shows clear usage patterns for every new feature. The `run_meta_eval()` / `analyze_contamination()` / `run_diagnostics()` entry points follow a consistent pattern.

---

## Issues Found

### High Severity

#### 1. IRT convergence is never validated (`meta_eval.py:372-416`)
The Gaussian IRT fit runs exactly 15 iterations with no convergence check, then unconditionally sets `converged=True`. If the alternating least-squares hasn't converged, downstream consumers will trust bogus difficulty/discrimination estimates.

**Recommendation:** Track the objective (reconstruction error) across iterations. Set `converged=True` only if the change drops below a tolerance. Alternatively, expose `max_iterations` and report the final residual.

#### 2. MAPE returns 0.0 when all targets are near zero (`runner.py:417-421`)
When `mask.sum() == 0` (all `|y_true| < 1e-10`), the function returns `0.0`, meaning "perfect prediction." This is semantically wrong and could hide data quality issues. Should return `float('nan')` or raise.

```python
# Current (misleading):
if mask.sum() == 0:
    return 0.0

# Suggested:
if mask.sum() == 0:
    return float("nan")  # MAPE undefined when all true values ≈ 0
```

#### 3. Unused imports in `meta_eval.py`
`scipy.optimize.minimize` (line 22) and `scipy.special.expit` (line 23) are imported but never used. These are heavy imports that slow down module loading and signal dead code.

#### 4. Feature importance noise detection edge case (`feature_importance.py:249-251`)
When `n_noise == len(common)` (all features are noise), the else-branch sets `precision = 0.0` and `recall = 0.0`. But if every feature is noise, a model that ranks them all at the bottom should get perfect recall. The condition conflates "no noise features" with "all noise features."

```python
# Current:
else:
    precision = 1.0 if n_noise == 0 else 0.0
    recall = 1.0 if n_noise == 0 else 0.0

# This else-branch fires when n_noise == 0 OR n_noise == len(common).
# When n_noise == len(common): all features are noise, and bottom_k == common,
# so precision and recall should both be 1.0.
```

### Medium Severity

#### 5. Concept drift crashes on single-class targets (`shift.py:135-140`)
In classification concept drift, if a row's current class is the *only* class, `others` becomes empty and `rng.choice(others)` raises `ValueError`. This can happen with extreme imbalance in small test splits.

```python
# Line 139-140:
others = [c for c in classes if c != current]
test_df.at[i, target] = rng.choice(others)  # crashes if others is empty
```

**Fix:** Guard with `if others:` before the choice.

#### 6. `str()` wrapping of `rng.choice()` result (`shift.py:117`)
```python
drift_feature = str(rng.choice(numeric_features))
```
`rng.choice()` already returns an element from the list. The `str()` conversion is unnecessary and could produce unexpected results if feature names aren't strings (e.g., integer column indices in some pandas workflows).

#### 7. Hardcoded lag column limit (`engine.py:299-301`)
```python
if len(lag_cols) > 6:
    lag_cols = lag_cols[:6]
```
The comment says "at most 5 columns" but the code checks `> 6` and keeps 6. This is unexplained, uncommented, and should be configurable via the template (e.g., `max_lag_features`).

#### 8. Broad exception handling in baselines (`baselines.py:~257-276`)
`except Exception` catches *all* exceptions including `TypeError`, `AttributeError`, and other programming errors, silently recording them as "failed" method entries. This makes debugging hard. Should catch specific expected exceptions (`ImportError`, `ValueError`, `RuntimeError`).

#### 9. Silent skip when no numeric features for concept drift (`shift.py:109-110`)
```python
if not numeric_features:
    return train_df, test_df
```
Returns unmodified data without logging a warning. Callers may not realize no drift was applied.

#### 10. Categorical encoding collision (`runner.py:459`)
```python
X_train[col] = X_train[col].map(cat_map).fillna(-1).astype(int)
```
If `-1` is a legitimate category value, this collides with the missing sentinel. Consider using a value guaranteed to be outside the mapping range (e.g., `max(cat_map.values()) + 1`).

### Low Severity

#### 11. Inefficient dimension detection (`diagnostics.py:244`)
```python
sample = task_metadata["difficulty"].dropna().iloc[0] if not task_metadata["difficulty"].dropna().empty else None
```
Calls `.dropna()` twice. Minor, but easy to fix by assigning to a variable.

#### 12. LinearTrend edge case (`forecasting.py:159`)
When `n_lags < 2`, the fallback returns `data[:, -1]`. If `n_lags == 0` (shouldn't happen but isn't validated), this fails with `IndexError`. Add a guard or document the n_lags >= 1 precondition.

#### 13. Hardcoded empty DataFrame columns (`baselines.py:~299`)
The fallback `pd.DataFrame(columns=[...])` hardcodes column names. If the result schema changes, this will silently produce mismatched DataFrames.

#### 14. Incomplete Pareto test (`test_diagnostics.py:~113`)
The `test_identifies_pareto_optimal` test function ends without asserting the actual content of the Pareto set. It constructs the data but doesn't verify the result.

#### 15. `directional_accuracy` returns 0.5 for single samples (`runner.py:426-427`)
Returning 0.5 (coin-flip) when `len(y_true) < 2` is arbitrary. The metric is undefined for single samples; `float('nan')` would be more honest.

---

## Design Observations

### The `generate_single_dataset()` breaking change (`engine.py:75-78`)
`template_override` is now *required* — calling without it raises `ValueError`. This is a breaking change for any code that previously called this function. The error message helpfully points to `generate_sampled_datasets()`, but existing callers will break silently. Since this is pre-1.0 software, this may be acceptable, but it should be noted in a changelog.

### Difficulty presets vs. parametric sampling
`DIFFICULTY_PRESETS` are defined in `templates/scenarios.py` but `sample_scenario()` ignores them entirely, sampling each axis independently. The presets appear to be legacy code. Either remove them or make `sample_scenario()` optionally use them (e.g., `sample_scenario(preset="hard")` narrows the ranges).

### Forecasting parameter forcing (`templates/scenarios.py:214-218`)
For forecasting tasks, `temporal_prob` is silently forced to 1.0 and `max_autocorr` is floored at 0.5. This makes sense physically but overrides user-specified `scenario_space` without warning.

### The `expit` and `minimize` imports
Both are imported from scipy in `meta_eval.py` but unused. They look like remnants of a planned logistic IRT model that was replaced with the current Gaussian alternating-least-squares approach. Clean them up.

---

## Test Quality Assessment

**Strengths:**
- Good coverage of happy paths and basic edge cases
- Synthetic data helpers avoid coupling to real datasets
- IRT tests verify structural invariants (discrimination > 0, ability correlation with scores)
- Contamination tests check both identical-score and disjoint-model edge cases

**Gaps:**
- Boundary conditions are undertested (single model, single task, empty data)
- Several tests verify structure but not numerical correctness (e.g., shift tests check output shape but not that distributions actually shifted by the requested magnitude)
- `test_diagnostics.py` Pareto test is incomplete (no final assertion)
- Floating-point equality checks (e.g., `rank_gap == 0`) could fail under rounding
- No tests for IRT convergence behavior or stability across different score distributions

---

## Summary Verdict

**The idea is strong and well-executed.** The evaluation framework (contamination analysis, IRT, diagnostics, scaling) is the kind of meta-tooling that distinguishes a serious benchmark project from a toy one. Ground-truth feature importance is a genuine differentiator. Forecasting integration is clean.

**Ship-blocking issues (fix before merge):**
1. IRT `converged=True` without convergence checking — misleading downstream
2. MAPE returning 0.0 for degenerate inputs — semantically wrong
3. Unused `expit`/`minimize` imports — dead code

**Should-fix (soon after merge):**
4. Concept drift crash on single-class targets
5. Feature importance noise detection edge case
6. Hardcoded lag column limit needs documentation or configurability
7. Broad `except Exception` in baselines
8. Incomplete Pareto test assertion

**Nice-to-fix (low urgency):**
9-15 from the list above

Overall: solid work that significantly advances the project's evaluation story. The three blocking issues are straightforward fixes.
