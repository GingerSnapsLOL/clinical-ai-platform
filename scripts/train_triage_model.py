#!/usr/bin/env python3
"""Train a baseline triage-severity classifier from ``triage_train.parquet``.

Writes ``models/triage_severity/model.pkl`` (joblib) and ``models/triage_severity/metrics.json``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LABEL_MAPPING: dict[str, int] = {"low": 0, "medium": 1, "high": 2}
LABEL_ORDER: list[int] = [0, 1, 2]
LABEL_NAMES: list[str] = ["low", "medium", "high"]


def _build_estimator(model_type: str, random_state: int) -> Any:
    if model_type == "logreg":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        multi_class="multinomial",
                        solver="lbfgs",
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise SystemExit(
                "XGBoost is not installed. Install with `uv add xgboost` or use --model logreg."
            ) from exc
        return XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            max_depth=5,
            n_estimators=200,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            n_jobs=-1,
            eval_metric="mlogloss",
        )
    raise ValueError(f"unknown model type: {model_type!r}")


def _feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c != "label_int"]
    if not cols:
        raise SystemExit("No feature columns found (expected everything except label_int).")
    return cols


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path("data/processed/triage_train.parquet"),
        help="Training parquet (default: data/processed/triage_train.parquet)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/triage_severity"),
        help="Output directory (default: models/triage_severity)",
    )
    parser.add_argument(
        "--model",
        choices=("logreg", "xgboost"),
        default="logreg",
        help="Estimator type (default: logreg). xgboost requires the xgboost package.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Validation fraction (default: 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()
    in_path: Path = args.input
    model_dir: Path = args.model_dir
    rs = int(args.random_state)

    if not in_path.is_file():
        raise SystemExit(f"Input not found: {in_path}")

    df = pd.read_parquet(in_path)
    if "label_int" not in df.columns:
        raise SystemExit("Parquet must contain label_int column.")

    feature_names = _feature_columns(df)
    X = df[feature_names].to_numpy(dtype=np.float64)
    y = df["label_int"].to_numpy(dtype=np.int64)

    unique_labels = np.unique(y)
    if not np.array_equal(np.sort(unique_labels), np.arange(3)):
        print(
            f"warning: label_int values are {unique_labels.tolist()}; expected 0,1,2",
            file=sys.stderr,
        )

    counts = np.bincount(y.astype(np.int64), minlength=3)
    stratify = y if len(y) >= 6 and int(counts.min()) >= 2 else None
    if stratify is None and len(y) >= 10:
        print(
            "warning: stratified split disabled (too few samples per class)",
            file=sys.stderr,
        )

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=float(args.test_size),
        random_state=rs,
        stratify=stratify,
    )

    estimator = _build_estimator(args.model, random_state=rs)
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_val)

    acc = float(accuracy_score(y_val, y_pred))
    macro_f1 = float(f1_score(y_val, y_pred, average="macro", labels=LABEL_ORDER, zero_division=0))
    cm = confusion_matrix(y_val, y_pred, labels=LABEL_ORDER)

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pkl"
    metrics_path = model_dir / "metrics.json"

    artifact: dict[str, Any] = {
        "model": estimator,
        "feature_names": feature_names,
        "label_mapping": dict(LABEL_MAPPING),
        "label_names": list(LABEL_NAMES),
        "model_type": args.model,
        "random_state": rs,
    }
    joblib.dump(artifact, model_path)

    metrics: dict[str, Any] = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": LABEL_NAMES,
        "label_mapping": dict(LABEL_MAPPING),
        "feature_names": feature_names,
        "model_type": args.model,
        "train_rows": int(len(y_train)),
        "val_rows": int(len(y_val)),
        "total_rows": int(len(y)),
        "random_state": rs,
        "test_size": float(args.test_size),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    spec_src = Path("data/processed/feature_spec.json")
    spec_dst = model_dir / "feature_spec.json"
    if spec_src.is_file():
        shutil.copy(spec_src, spec_dst)
        print(f"Copied feature spec to {spec_dst}", flush=True)

    print(
        f"Saved model to {model_path} ({args.model}), "
        f"val accuracy={acc:.4f} macro_f1={macro_f1:.4f}",
        flush=True,
    )
    print(f"Metrics written to {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
