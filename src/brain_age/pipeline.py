"""Train and evaluate age-regression models with participant-level separation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV or TSV feature table."""
    separator = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    frame = pd.read_csv(path, sep=separator)
    if frame.empty:
        raise ValueError(f"No rows found in input table: {path}")
    return frame


def select_features(
    frame: pd.DataFrame,
    target: str,
    group_column: str,
    excluded: Sequence[str],
    prefixes: Sequence[str],
) -> list[str]:
    """Select numeric predictors while excluding labels and participant metadata."""
    forbidden = {target, group_column, *excluded}
    candidates = [
        column
        for column in frame.columns
        if column not in forbidden and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if prefixes:
        candidates = [name for name in candidates if name.startswith(tuple(prefixes))]
    if not candidates:
        raise ValueError(
            "No numeric feature columns selected. Check the input schema or provide "
            "valid --feature-prefix values."
        )
    return candidates


def make_models(seed: int) -> dict[str, object]:
    """Return simple baselines and nonlinear regressors as sklearn pipelines."""
    return {
        "median_baseline": make_pipeline(SimpleImputer(strategy="median"), DummyRegressor(strategy="median")),
        "ridge": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0)
        ),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=400,
                min_samples_leaf=2,
                max_features=0.7,
                random_state=seed,
                n_jobs=-1,
            ),
        ),
        "hist_gradient_boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(
                max_iter=200, l2_regularization=1.0, random_state=seed
            ),
        ),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    frame = read_table(args.input)
    required = {args.target, args.group_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Required column(s) missing from input: {', '.join(missing)}")

    frame = frame.dropna(subset=[args.target, args.group_column]).copy()
    frame[args.target] = pd.to_numeric(frame[args.target], errors="coerce")
    frame = frame.dropna(subset=[args.target]).reset_index(drop=True)
    features = select_features(
        frame,
        args.target,
        args.group_column,
        args.exclude,
        args.feature_prefix,
    )

    X = frame[features].replace([np.inf, -np.inf], np.nan)
    y = frame[args.target].astype(float)
    groups = frame[args.group_column].astype(str)
    unique_groups = groups.nunique()
    if unique_groups < 3:
        raise ValueError("At least three distinct participants are required.")

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=args.test_size, random_state=args.seed
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    if set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]):
        raise RuntimeError("Participant leakage detected between training and test sets.")
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]

    n_splits = min(args.cv_folds, groups_train.nunique())
    if n_splits < 2:
        raise ValueError("At least two training participants are required for CV.")
    cv = GroupKFold(n_splits=n_splits)
    models = make_models(args.seed)
    summary: dict[str, object] = {
        "input_file": args.input.name,
        "target": args.target,
        "group_column": args.group_column,
        "n_rows": int(len(frame)),
        "n_participants": int(unique_groups),
        "n_features": int(len(features)),
        "features": features,
        "age_min": float(y.min()),
        "age_max": float(y.max()),
        "seed": int(args.seed),
        "cv_folds": int(n_splits),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_participants": int(groups.iloc[train_idx].nunique()),
        "test_participants": int(groups.iloc[test_idx].nunique()),
        "models": {},
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame(
        {
            args.group_column: groups.iloc[test_idx].to_numpy(),
            "age_true": y_test.to_numpy(),
        }
    )

    for name, model in models.items():
        cv_result = cross_validate(
            model,
            X_train,
            y_train,
            groups=groups_train,
            cv=cv,
            scoring={"mae": "neg_mean_absolute_error", "r2": "r2"},
            n_jobs=1,
            error_score="raise",
        )
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        summary["models"][name] = {
            "cv_mae_mean": float(-cv_result["test_mae"].mean()),
            "cv_mae_std": float(cv_result["test_mae"].std()),
            "cv_r2_mean": float(cv_result["test_r2"].mean()),
            "test_mae": float(mean_absolute_error(y_test, predicted)),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, predicted))),
            "test_r2": float(r2_score(y_test, predicted)),
        }
        predictions[f"{name}_predicted_age"] = predicted
        if name != "median_baseline":
            joblib.dump(
                {"model": model, "features": features, "target": args.target},
                args.output_dir / f"{name}.joblib",
            )

    predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV/TSV table of MRI features")
    parser.add_argument("--target", default="age", help="Chronological age column")
    parser.add_argument(
        "--group-column",
        default="participant_id",
        help="Participant ID used to keep repeated visits in one split",
    )
    parser.add_argument(
        "--feature-prefix",
        action="append",
        default=[],
        help="Optional feature prefix; repeat to include multiple prefixes, e.g. lh_ and rh_",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Numeric metadata column to exclude from predictors; may be repeated",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = run(arguments)
    print(json.dumps(result, indent=2))
