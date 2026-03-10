"""Generate Kaggle-style example problem bundles.

Run:
    python examples/scripts/export_kaggle_style_problems.py

Writes a small collection of themed competition folders under
``examples/problems/``. Each bundle includes:
  - ``train.csv``
  - ``test.csv``
  - ``sample_submission.csv``
  - ``solution/test_labels.csv``
  - ``data_dictionary.csv``
  - ``metadata.json``
  - ``starter.py``
  - ``README.md``
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from tabular_bank.generation.engine import GeneratedDataset, generate_single_dataset


EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
PROBLEMS_ROOT = EXAMPLES_ROOT / "problems"
ROUND_ID = "examples-kaggle-v1"
MASTER_SECRET = "examples-kaggle-secret-v1"


@dataclass(frozen=True)
class ProblemSpec:
    slug: str
    title: str
    subtitle: str
    problem_type: str
    target_name: str
    metric: str
    metric_blurb: str
    submission_hint: str
    template: dict


PROBLEM_SPECS = [
    ProblemSpec(
        slug="customer-retention-risk",
        title="Customer Retention Risk",
        subtitle="Predict whether an account will churn in the next billing cycle.",
        problem_type="binary",
        target_name="will_churn",
        metric="roc_auc",
        metric_blurb="Submissions are ranked by ROC AUC on the positive class.",
        submission_hint="Submit a probability between 0 and 1 for each row.",
        template={
            "id": "customer_retention_risk",
            "domain": "telecom",
            "problem_type": "binary",
            "n_features_range": (14, 14),
            "n_samples_range": (420, 420),
            "categorical_ratio": 0.35,
            "noise_feature_ratio": 0.15,
            "missing_rate": 0.08,
            "missing_mechanism": "MAR",
            "imbalance_ratio": 0.22,
            "difficulty": "medium",
        },
    ),
    ProblemSpec(
        slug="store-demand-forecast",
        title="Store Demand Forecast",
        subtitle="Forecast normalized demand for the next replenishment cycle.",
        problem_type="regression",
        target_name="future_demand",
        metric="rmse",
        metric_blurb="Submissions are ranked by root mean squared error.",
        submission_hint="Submit one numeric forecast per row.",
        template={
            "id": "store_demand_forecast",
            "domain": "retail",
            "problem_type": "regression",
            "n_features_range": (16, 16),
            "n_samples_range": (480, 480),
            "categorical_ratio": 0.25,
            "noise_feature_ratio": 0.2,
            "missing_rate": 0.05,
            "missing_mechanism": "MCAR",
            "difficulty": "medium",
        },
    ),
    ProblemSpec(
        slug="ticket-priority-routing",
        title="Ticket Priority Routing",
        subtitle="Assign each incoming support ticket to the correct priority band.",
        problem_type="multiclass",
        target_name="priority_band",
        metric="log_loss",
        metric_blurb="Submissions are ranked by multiclass log loss.",
        submission_hint="Submit a probability for every class on each row.",
        template={
            "id": "ticket_priority_routing",
            "domain": "support",
            "problem_type": "multiclass",
            "n_features_range": (12, 12),
            "n_samples_range": (450, 450),
            "categorical_ratio": 0.4,
            "noise_feature_ratio": 0.1,
            "missing_rate": 0.09,
            "missing_mechanism": "MNAR",
            "n_classes": 5,
            "difficulty": "hard",
        },
    ),
]


def _feature_sort_key(name: str) -> tuple[str, int]:
    prefix, sep, suffix = name.partition("_")
    if prefix == "f" and sep and suffix.isdigit():
        return (prefix, int(suffix))
    return (name, -1)


def _metric_for_display(metric: str) -> str:
    return {
        "roc_auc": "ROC AUC",
        "rmse": "RMSE",
        "log_loss": "Log Loss",
    }.get(metric, metric)


def _prepare_frames(dataset: GeneratedDataset, spec: ProblemSpec) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = dataset.data.copy().reset_index(drop=True)
    frame.insert(0, "row_id", np.arange(len(frame), dtype=int))

    feature_cols = sorted(dataset.feature_names, key=_feature_sort_key)
    ordered_cols = ["row_id"] + feature_cols + [dataset.target_name]
    frame = frame[ordered_cols].rename(columns={dataset.target_name: spec.target_name})

    train_idx, test_idx = dataset.splits[0][0]
    train_df = frame.iloc[train_idx].reset_index(drop=True)
    test_df = frame.iloc[test_idx].reset_index(drop=True)
    labels_df = test_df[["row_id", spec.target_name]].copy()
    public_test = test_df.drop(columns=[spec.target_name])
    return train_df, public_test, labels_df


def _make_sample_submission(spec: ProblemSpec, labels_df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    submission = pd.DataFrame({"row_id": labels_df["row_id"]})
    if spec.problem_type == "regression":
        submission[spec.target_name] = 0.0
        return submission
    if spec.problem_type == "binary":
        submission[spec.target_name] = 0.5
        return submission

    n_classes = int(metadata["n_classes"])
    uniform_prob = 1.0 / n_classes
    for class_id in range(n_classes):
        submission[f"class_{class_id}"] = uniform_prob
    return submission


def _make_data_dictionary(train_df: pd.DataFrame, spec: ProblemSpec) -> pd.DataFrame:
    rows = []
    for col in train_df.columns:
        if col == "row_id":
            role = "id"
            description = "Stable row identifier for train/test/submission joins."
        elif col == spec.target_name:
            role = "target"
            description = f"Competition target column for {spec.title.lower()}."
        else:
            role = "feature"
            description = "Synthetic input feature available at prediction time."
        dtype = str(train_df[col].dtype)
        kind = "categorical" if dtype in {"object", "string"} else "numeric"
        rows.append({
            "column": col,
            "role": role,
            "dtype": dtype,
            "kind": kind,
            "description": description,
        })
    return pd.DataFrame(rows)


def _render_problem_readme(
    spec: ProblemSpec,
    metadata: dict,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> str:
    return dedent(
        f"""\
        # {spec.title}

        {spec.subtitle}

        ## Overview

        This is a fully synthetic tabular prediction problem exported from
        `tabular-bank` in a Kaggle-style layout. The competition framing is
        human-readable; the underlying rows and features are procedurally
        generated.

        ## Files

        - `train.csv`: {len(train_df)} labeled rows
        - `test.csv`: {len(test_df)} unlabeled rows for submission
        - `sample_submission.csv`: submission template
        - `solution/test_labels.csv`: held-out labels for local scoring
        - `data_dictionary.csv`: column roles and dtypes
        - `starter.py`: minimal sklearn baseline that writes `submission.csv`
        - `metadata.json`: generation metadata and benchmark context

        ## Evaluation

        Metric: **{_metric_for_display(spec.metric)}**

        {spec.metric_blurb}
        {spec.submission_hint}

        ## Dataset Snapshot

        - Problem type: `{spec.problem_type}`
        - Target column: `{spec.target_name}`
        - Total observed features: `{metadata["n_features"]}`
        - Informative features: `{metadata["n_informative_features"]}`
        - Noise features: `{metadata["n_noise_features"]}`
        - Missingness mechanism: `{metadata.get("missing_mechanism", "none")}`
        - Domain tag: `{metadata["domain"]}`

        ## Reproducibility

        Generated from:

        - round id: `{ROUND_ID}`
        - scenario id: `{metadata["scenario_id"]}`
        - dataset id: `{metadata["dataset_id"]}`

        In a real competition, `solution/test_labels.csv` would remain hidden.
        It is included here only so the example package is self-contained.
        """
    )


def _render_starter(spec: ProblemSpec, metadata: dict) -> str:
    if spec.problem_type == "regression":
        model_import = "from sklearn.ensemble import RandomForestRegressor"
        model_ctor = "RandomForestRegressor(n_estimators=200, random_state=0)"
        predict_lines = [
            "pred = model.predict(X_test)",
            "submission = test[[ID_COL]].copy()",
            "submission[TARGET_COL] = pred",
        ]
    elif spec.problem_type == "binary":
        model_import = "from sklearn.ensemble import RandomForestClassifier"
        model_ctor = "RandomForestClassifier(n_estimators=200, random_state=0)"
        predict_lines = [
            "pred = model.predict_proba(X_test)[:, 1]",
            "submission = test[[ID_COL]].copy()",
            "submission[TARGET_COL] = pred",
        ]
    else:
        model_import = "from sklearn.ensemble import RandomForestClassifier"
        model_ctor = "RandomForestClassifier(n_estimators=300, random_state=0)"
        predict_lines = [
            "pred = model.predict_proba(X_test)",
            "submission = test[[ID_COL]].copy()",
        ]
        predict_lines.extend(
            f"submission['class_{class_id}'] = pred[:, {class_id}]"
            for class_id in range(int(metadata["n_classes"]))
        )

    lines = [
        "from __future__ import annotations",
        "",
        "import pandas as pd",
        model_import,
        "",
        "",
        'TRAIN_PATH = "train.csv"',
        'TEST_PATH = "test.csv"',
        'OUTPUT_PATH = "submission.csv"',
        'ID_COL = "row_id"',
        f'TARGET_COL = "{spec.target_name}"',
        "",
        "",
        "def _encode(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:",
        "    train_enc = pd.get_dummies(train_df, dummy_na=True)",
        "    test_enc = pd.get_dummies(test_df, dummy_na=True)",
        "    train_enc, test_enc = train_enc.align(test_enc, join=\"outer\", axis=1, fill_value=0)",
        "    return train_enc, test_enc",
        "",
        "",
        "train = pd.read_csv(TRAIN_PATH)",
        "test = pd.read_csv(TEST_PATH)",
        "",
        "X_train = train.drop(columns=[ID_COL, TARGET_COL])",
        "y_train = train[TARGET_COL]",
        "X_test = test.drop(columns=[ID_COL])",
        "X_train, X_test = _encode(X_train, X_test)",
        "",
        f"model = {model_ctor}",
        "model.fit(X_train, y_train)",
        "",
        *predict_lines,
        "submission.to_csv(OUTPUT_PATH, index=False)",
        "print(f\"Wrote {OUTPUT_PATH}\")",
        "",
    ]
    return "\n".join(lines)


def _write_problem_bundle(spec: ProblemSpec, dataset: GeneratedDataset) -> dict:
    bundle_dir = PROBLEMS_ROOT / spec.slug
    solution_dir = bundle_dir / "solution"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    solution_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df, labels_df = _prepare_frames(dataset, spec)
    sample_submission = _make_sample_submission(spec, labels_df, dataset.metadata)
    data_dictionary = _make_data_dictionary(train_df, spec)

    public_metadata = {
        "title": spec.title,
        "slug": spec.slug,
        "round_id": ROUND_ID,
        "dataset_id": dataset.dataset_id,
        "scenario_id": dataset.scenario_id,
        "problem_type": spec.problem_type,
        "target_name": spec.target_name,
        "metric": spec.metric,
        "metric_display": _metric_for_display(spec.metric),
        "n_train_rows": len(train_df),
        "n_test_rows": len(test_df),
        "missing_rate": spec.template.get("missing_rate", 0.0),
        "missing_mechanism": spec.template.get("missing_mechanism", "none"),
        **dataset.metadata,
    }
    public_metadata["target_name"] = spec.target_name

    train_df.to_csv(bundle_dir / "train.csv", index=False)
    test_df.to_csv(bundle_dir / "test.csv", index=False)
    sample_submission.to_csv(bundle_dir / "sample_submission.csv", index=False)
    labels_df.to_csv(solution_dir / "test_labels.csv", index=False)
    data_dictionary.to_csv(bundle_dir / "data_dictionary.csv", index=False)
    (bundle_dir / "README.md").write_text(
        _render_problem_readme(spec, public_metadata, train_df, test_df),
        encoding="utf-8",
    )
    (bundle_dir / "starter.py").write_text(
        _render_starter(spec, public_metadata),
        encoding="utf-8",
    )
    with open(bundle_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(public_metadata, f, indent=2)

    return {
        "slug": spec.slug,
        "title": spec.title,
        "problem_type": spec.problem_type,
        "metric": spec.metric,
        "n_train_rows": len(train_df),
        "n_test_rows": len(test_df),
    }


def _write_root_readme(manifest: list[dict]) -> None:
    lines = [
        "# Kaggle-Style Example Problems",
        "",
        "These folders are generated examples that package synthetic `tabular-bank`",
        "datasets in a competition-style layout.",
        "",
        "Regenerate them with:",
        "",
        "```bash",
        "python examples/scripts/export_kaggle_style_problems.py",
        "```",
        "",
        "| Problem | Type | Metric | Train rows | Test rows |",
        "|---------|------|--------|------------|-----------|",
    ]
    for item in manifest:
        lines.append(
            f"| `{item['slug']}` | `{item['problem_type']}` | `{item['metric']}` | "
            f"{item['n_train_rows']} | {item['n_test_rows']} |"
        )
    lines.extend([
        "",
        "Each problem directory includes train/test CSVs, a sample submission,",
        "held-out labels for local evaluation, a simple starter baseline, and",
        "metadata describing the generated task.",
        "",
    ])
    (PROBLEMS_ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    with open(PROBLEMS_ROOT / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def main() -> None:
    PROBLEMS_ROOT.mkdir(parents=True, exist_ok=True)

    manifest = []
    for scenario_index, spec in enumerate(PROBLEM_SPECS):
        dataset = generate_single_dataset(
            master_secret=MASTER_SECRET,
            round_id=ROUND_ID,
            scenario_index=scenario_index,
            template_override=spec.template,
        )
        manifest.append(_write_problem_bundle(spec, dataset))

    _write_root_readme(manifest)
    print(f"Wrote {len(manifest)} problem bundles to {PROBLEMS_ROOT}")


if __name__ == "__main__":
    main()
